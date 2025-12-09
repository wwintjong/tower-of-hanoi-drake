import numpy as np
import os
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LoadModelDirectives,
    MeshcatVisualizer,
    Parser,
    ProcessModelDirectives,
    Simulator,
    StartMeshcat,
    InverseKinematics,
    Solve,
    PiecewisePolynomial,
    TrajectorySource,
    RigidTransform,
    RotationMatrix,
    InverseDynamicsController,
    ConstantVectorSource,
    MultibodyPlant
)

# --- CONFIGURATION ---
PEG_X = 0.5 
SAFE_Z = 0.7  # High clearance for crane arm
TABLE_SURFACE_Z = 0.0 
DISK_HEIGHT = 0.1 

# User Targets (Fingertip Goals)
PICK_TARGET = [0.4, 0.0, 0.4]   
PLACE_TARGET = [0.4, 0.5, 0.4]

# Offset from WSG Body (palm) to Fingertips
# Standard WSG50 is approx 12cm from body center to tips
GRIPPER_FINGER_OFFSET = 0.12 

def calculate_ik(plant, context, target_pose):
    ik = InverseKinematics(plant, context)
    ik.AddPositionConstraint(
        frameB=plant.GetFrameByName("body", plant.GetModelInstanceByName("wsg")),
        p_BQ=[0, 0, 0],
        frameA=plant.world_frame(),
        p_AQ_lower=target_pose.translation() - 0.005,
        p_AQ_upper=target_pose.translation() + 0.005,
    )
    
    ik.AddOrientationConstraint(
        frameAbar=plant.world_frame(),
        R_AbarA=target_pose.rotation(),
        frameBbar=plant.GetFrameByName("body", plant.GetModelInstanceByName("wsg")),
        R_BbarB=RotationMatrix(),
        theta_bound=0.05
    )
    
    q_full = plant.GetPositions(context)
    prog = ik.get_mutable_prog()
    prog.SetInitialGuess(ik.q(), q_full)
    
    result = Solve(prog)
    iiwa_model = plant.GetModelInstanceByName("iiwa")
    q_current_iiwa = plant.GetPositions(context, iiwa_model)
    
    if not result.is_success():
        print("IK Failed for pose:", target_pose.translation())
        return q_current_iiwa
    
    q_solution = result.GetSolution(ik.q())
    start_idx = plant.GetJointByName("iiwa_joint_1", iiwa_model).position_start()
    return q_solution[start_idx : start_idx+7]

def make_traj(q_start, q_end, duration):
    breaks = [0.0, duration]
    samples = np.column_stack((q_start, q_end))
    v_zero = np.zeros_like(q_start)
    samples_dot = np.column_stack((v_zero, v_zero))
    return PiecewisePolynomial.CubicHermite(breaks, samples, samples_dot)

def create_controller_plant(time_step=1e-3):
    controller_plant = MultibodyPlant(time_step=time_step)
    parser = Parser(controller_plant)
    
    iiwa_file = "package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
    iiwa = parser.AddModelsFromUrl(iiwa_file)[0]
    
    controller_plant.WeldFrames(
        controller_plant.world_frame(),
        controller_plant.GetFrameByName("iiwa_link_0", iiwa),
        RigidTransform(RotationMatrix(), [0, 0, 0.05])
    )
    controller_plant.Finalize()
    return controller_plant

def reset_disks(plant, context):
    """ Places disks at the calculated stack heights on Peg 0 (Middle). """
    disk_names = ["disk_4", "disk_3", "disk_2", "disk_1"]
    
    # Calculate base height assuming stacking from z=0
    # Disk center is at height/2 + index * height
    base_z = DISK_HEIGHT / 2.0 

    for i, name in enumerate(disk_names):
        try:
            model = plant.GetModelInstanceByName(name)
            body = plant.GetBodyByName("torus_link", model)
            z_center = base_z + (i * DISK_HEIGHT)
            X_World_Disk = RigidTransform(RotationMatrix(), [PEG_X, 0.0, z_center])
            plant.SetFreeBodyPose(context, body, X_World_Disk)
        except Exception as e:
            print(f"Could not reset {name}: {e}")

def run_test_simulation():
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
    parser = Parser(plant)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.package_map().Add("tower_of_hanoi_drake", current_dir) 
    
    directives = LoadModelDirectives(os.path.join(current_dir, "load_scene.yaml"))
    ProcessModelDirectives(directives, plant, parser)
    plant.Finalize()
    
    controller_plant = create_controller_plant(time_step=1e-3)
    iiwa_model = plant.GetModelInstanceByName("iiwa")
    
    kp = [100]*7
    ki = [1]*7
    kd = [20]*7
    arm_controller = builder.AddSystem(InverseDynamicsController(controller_plant, kp, ki, kd, False))
    
    builder.Connect(plant.get_state_output_port(iiwa_model), arm_controller.get_input_port_estimated_state())
    builder.Connect(arm_controller.get_output_port_control(), plant.get_actuation_input_port(iiwa_model))

    wsg_model = plant.GetModelInstanceByName("wsg")
    wsg_command = builder.AddSystem(ConstantVectorSource([0.1, 0.1])) 
    builder.Connect(wsg_command.get_output_port(), plant.get_actuation_input_port(wsg_model))

    dummy_traj = PiecewisePolynomial.ZeroOrderHold([0, 0.1], np.zeros((14, 2)))
    traj_source = builder.AddSystem(TrajectorySource(dummy_traj))
    builder.Connect(traj_source.get_output_port(), arm_controller.get_input_port_desired_state())

    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)
    wsg_context = wsg_command.GetMyContextFromRoot(context)
    
    reset_disks(plant, plant_context)
    
    q0 = plant.GetPositions(plant_context, iiwa_model)
    
    # 2. Settle Phase
    settle_time = 2.0
    full_state_traj = PiecewisePolynomial.FirstOrderHold(
        [0, settle_time + 0.1], 
        np.column_stack((np.concatenate((q0, np.zeros(7))), np.concatenate((q0, np.zeros(7)))))
    )
    traj_source.UpdateTrajectory(full_state_traj)
    print(f"Settling for {settle_time}s...")
    simulator.AdvanceTo(settle_time)
    
    # --- CRANE GRASP ORIENTATION (Downwards) ---
    # We want Gripper +Z (Approach) to align with World -Z.
    # We want Gripper +Y (Fingers) to align with World +Y (Tangential pinch).
    # R_WG columns: [Gx, Gy, Gz]
    # Gz = (0, 0, -1) [Down]
    # Gy = (0, 1, 0)  [Y axis]
    # Gx = Gy cross Gz = (0, 1, 0) x (0, 0, -1) = (-1, 0, 0)
    
    R_Crane = RotationMatrix(np.column_stack((
        [-1, 0, 0], # Gx
        [0, 1, 0],  # Gy
        [0, 0, -1]  # Gz (Approach Down)
    )))

    # --- CALCULATE POSES ---
    # The IK solves for the Body frame. We must offset the Z target by finger length.
    
    # Pre-Pick (Hover above target)
    p_pre_pick = np.array(PICK_TARGET) + [0, 0, 0.2] # 20cm above target
    pose_pre_pick = RigidTransform(R_Crane, p_pre_pick + [0, 0, GRIPPER_FINGER_OFFSET])
    
    # Pick (Descend to target)
    # Target is 0.4. We go slightly lower (0.35) to grasp the "sides" of the disk.
    p_pick = np.array(PICK_TARGET) + [0, 0, -0.05] 
    pose_pick = RigidTransform(R_Crane, p_pick + [0, 0, GRIPPER_FINGER_OFFSET])
    
    # Pre-Place (Hover above place target)
    p_pre_place = np.array(PLACE_TARGET) + [0, 0, 0.2]
    pose_pre_place = RigidTransform(R_Crane, p_pre_place + [0, 0, GRIPPER_FINGER_OFFSET])
    
    # Place (Descend to place target)
    p_place = np.array(PLACE_TARGET) + [0, 0, -0.05]
    pose_place = RigidTransform(R_Crane, p_place + [0, 0, GRIPPER_FINGER_OFFSET])

    def move_arm(target_pose, duration=2.5):
        start_time = context.get_time()
        q_now = plant.GetPositions(plant_context, iiwa_model)
        q_next = calculate_ik(plant, plant_context, target_pose)
        
        q_traj = make_traj(q_now, q_next, duration)
        v_traj = q_traj.MakeDerivative()
        
        num_samples = 100
        times = np.linspace(start_time, start_time + duration, num_samples)
        qs = np.array([q_traj.value(t - start_time).flatten() for t in times])
        vs = np.array([v_traj.value(t - start_time).flatten() for t in times])
        full_data = np.vstack((qs.T, vs.T))
        
        full_traj = PiecewisePolynomial.CubicShapePreserving(times, full_data)
        traj_source.UpdateTrajectory(full_traj)
        simulator.AdvanceTo(start_time + duration)

    def set_gripper(width, force=40.0, wait_time=1.0):
        wsg_command.get_mutable_source_value(wsg_context).set_value([width, force])
        current_time = context.get_time()
        simulator.AdvanceTo(current_time + wait_time)

    # --- EXECUTE CRANE SEQUENCE ---
    print("Moving to Pre-Pick (Hover)...")
    move_arm(pose_pre_pick, duration=3.0)
    
    print("Opening Gripper...")
    set_gripper(0.12) # Open wide enough for the ring chord
    
    print("Descending to Pick...")
    move_arm(pose_pick, duration=2.0)
    
    print("Grasping...")
    set_gripper(0.05) # Close fingers
    
    print("Lifting...")
    move_arm(pose_pre_pick, duration=2.0)
    
    print("Moving to Pre-Place...")
    move_arm(pose_pre_place, duration=3.0)
    
    print("Lowering...")
    move_arm(pose_place, duration=3.0)
    
    print("Releasing...")
    set_gripper(0.12)
    
    print("Retreating...")
    move_arm(pose_pre_place, duration=2.0)
    
    print("Done.")

if __name__ == "__main__":
    run_test_simulation()