import numpy as np
import os
from pydrake.all import (
    DiagramBuilder,
    MeshcatVisualizer,
    Simulator,
    StartMeshcat,
    TrajectorySource,
    RigidTransform,
    RotationMatrix,
    Integrator,
    MultibodyPlant,
    Parser,
    AbstractValue
)

from manipulation.station import MakeHardwareStation, Scenario
from pydrake.common.yaml import yaml_load_typed

import helper_func as hf

def register_package(parser: Parser):
    """
    Callback to register the current directory as the 'tower_of_hanoi_drake' package.
    This allows the YAML file to use 'package://tower_of_hanoi_drake/...'
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.package_map().Add("tower_of_hanoi_drake", current_dir)

def run():
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    yaml_file = "load_scene.yaml"
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"File {yaml_file} not found.")
        
    data = open(yaml_file, "r").read()
    scenario = yaml_load_typed(schema=Scenario, data=data)

    station = MakeHardwareStation(
        scenario, 
        meshcat=meshcat,
        parser_preload_callback=register_package
    )
    builder.AddSystem(station)
    plant = station.GetSubsystemByName("plant")
    plant_context = plant.CreateDefaultContext()
    
    # get initial pose for disk_1 (smallest disk)
    disk1_instance = plant.GetModelInstanceByName("disk_1")
    X_WO_Disk1 = hf.get_initial_pose(
        plant, 
        "torus_link", 
        disk1_instance, 
        plant_context
    )
    
    # get initial pose of gripper
    wsg_instance = plant.GetModelInstanceByName("wsg")
    wsg_body = plant.GetBodyByName("body", wsg_instance)
    X_WG_Initial = plant.EvalBodyPoseInWorld(plant_context, wsg_body)

    # 4. Define Grasp Parameters
    # We must explicitly define how we want to grasp the object (Offset and Rotation)
    
    # Position: Grasp at the center of the torus (adjust Z if needed)
    p_Grasp = np.array([0.0, 0.17, 0.0]) 
    
    # Orientation: We want the gripper fingers to point DOWN (-Z in World).
    # In standard WSG frames, X is often the approach direction. 
    # Rotating Y by +90 degrees aligns the gripper X with World -Z.
    # R_Grasp = RotationMatrix.MakeXRotation(np.pi / 2)
    R_Grasp = np.array([
        [0, 0, 1],
        [0, -1, 0],
        [1, 0, 0]
    ])

    # 5. Build Trajectory Keyframes
    # Goal: Move Disk 1 to Peg 1
    
    # A. Calculate Pick Poses
    X_OG, X_WG_Pick = hf.design_grasp_pose(X_WO_Disk1, p_Grasp, R_Grasp)
    X_WG_PrePick = hf.design_approach_pose(X_WG_Pick, approach_distance=0.1)
    
    # B. Calculate Place Poses (Targeting Peg 1)
    # Peg 1 location from YAML: [0.65, -0.75, 0.35]
    # We add a small offset to Z so we don't crash into the peg base immediately
    p_WO_Goal = np.array([0.65, -0.75, 0.4]) 
    R_WO_Goal = X_WO_Disk1.rotation() # Keep the disk flat
    
    # Use helper to get the gripper pose needed to achieve this object pose
    # Note: design_goal_pose returns a tuple
    X_WG_Place = hf.design_goal_pose(X_OG, p_WO_Goal, R_WO_Goal)
    X_WG_PrePlace = hf.design_approach_pose(X_WG_Place, approach_distance=0.1)

    # C. Assemble Sequence
    opened = 0.107
    closed = 0.01 # Close, but not to zero (avoids physics issues if collision)
    
    # Format: (Gripper Pose, Finger Width)
    keyframes = [
        (X_WG_Initial, opened),
        (X_WG_PrePick, opened),   # Move above disk
        (X_WG_Pick, opened),      # Move down
        (X_WG_Pick, closed)      # Grasp
        # (X_WG_PrePick, closed),   # Lift up
        # (X_WG_PrePlace, closed),  # Move above goal peg
        # (X_WG_Place, closed),     # Lower
        # (X_WG_Place, opened),     # Release
        # (X_WG_PrePlace, opened),  # Retreat up
        # (X_WG_Initial, opened)    # Return home
    ]

    # 6. Generate Trajectories
    gripper_poses = [k[0] for k in keyframes]
    finger_states = np.asarray([k[1] for k in keyframes]).reshape(1, -1)
    
    # Timing: 2 seconds per keyframe
    sample_times = [2.0 * i for i in range(len(gripper_poses))]
    
    traj_V_G, traj_wsg_command = hf.make_trajectory(
        gripper_poses, 
        finger_states, 
        sample_times
    )

    # 7. Add Controllers and Connect Diagram
    V_G_source = builder.AddSystem(TrajectorySource(traj_V_G))
    wsg_source = builder.AddSystem(TrajectorySource(traj_wsg_command))
    
    # Controller from helper_func.py
    controller = builder.AddSystem(hf.PseudoInverseController(plant))
    
    # Integrator to convert Velocity commands -> Position commands
    integrator = builder.AddSystem(Integrator(7))

    # -- Wiring --
    # 1. Trajectory -> Controller (Desired Velocity)
    builder.Connect(V_G_source.get_output_port(), controller.GetInputPort("V_WG"))
    
    # 2. Controller -> Integrator (Calculated Velocity -> Position)
    builder.Connect(controller.get_output_port(), integrator.get_input_port())
    
    # 3. Integrator -> Station (Position Command)
    builder.Connect(integrator.get_output_port(), station.GetInputPort("iiwa.position"))
    
    # 4. Station -> Controller (Feedback for Jacobian calculation)
    builder.Connect(station.GetOutputPort("iiwa.position_measured"), controller.GetInputPort("iiwa.position"))
    
    # 5. Finger Trajectory -> Station (Gripper Command)
    builder.Connect(wsg_source.get_output_port(), station.GetInputPort("wsg.position"))

    # 8. Build and Run Simulation
    diagram = builder.Build()
    simulator = Simulator(diagram)
    
    # Initialize the Integrator with the robot's starting configuration
    iiwa_model = plant.GetModelInstanceByName("iiwa")
    q0 = plant.GetPositions(plant_context, iiwa_model)
    
    context = simulator.get_mutable_context()
    station_context = station.GetMyContextFromRoot(context)
    integrator.set_integral_value(
        integrator.GetMyContextFromRoot(context),
        plant.GetPositions(
            plant.GetMyContextFromRoot(context),
            plant.GetModelInstanceByName("iiwa"),
        ),
    )
    diagram.ForcedPublish(context)
    print(f"sanity check, simulation will run for {traj_V_G.end_time()} seconds")
    
    simulator.Initialize()
    
    print("Starting simulation...")
    meshcat.StartRecording()
    simulator.AdvanceTo(traj_V_G.end_time())
    meshcat.StopRecording()
    meshcat.PublishRecording()
    print("Simulation complete.")
    
    return True

if __name__ == "__main__":
    run()