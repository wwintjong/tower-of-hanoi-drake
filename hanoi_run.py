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

PEG_X = 0.4
SAFE_Z = 0.5
TABLE_SURFACE_Z = 0.1 
TUBE_RADIUS = 0.075 
DISK_HEIGHT = 2 * TUBE_RADIUS 

PICK_TARGET = [0.4, 0.0, 0.55]   
PLACE_TARGET = [0.4, 0.75, 0.55] 

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
    """
    Creates a plant for the controller. 
    """
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
    """ Places disks at the calculated stack heights on Peg 0. """
    disk_names = ["disk_3", "disk_2", "disk_1"]
    for i, name in enumerate(disk_names):
        try:
            model = plant.GetModelInstanceByName(name)
            body = plant.GetBodyByName("torus_link", model)
            z_center = TABLE_SURFACE_Z + TUBE_RADIUS + (i * DISK_HEIGHT)
            X_World_Disk = RigidTransform(RotationMatrix(), [PEG_X, 0.0, z_center])
            plant.SetFreeBodyPose(context, body, X_World_Disk)
        except:
            pass

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
    full_state_traj = PiecewisePolynomial.FirstOrderHold(
        [0, 2.0], 
        np.column_stack((np.concatenate((q0, np.zeros(7))), np.concatenate((q0, np.zeros(7)))))
    )
    traj_source.UpdateTrajectory(full_state_traj)
    
    simulator.AdvanceTo(1.0)
    R_WG = RotationMatrix.MakeXRotation(np.pi)

    pose_pre_pick = RigidTransform(R_WG, [PICK_TARGET[0], PICK_TARGET[1], SAFE_Z])
    pose_pick = RigidTransform(R_WG, PICK_TARGET)
    pose_pre_place = RigidTransform(R_WG, [PLACE_TARGET[0], PLACE_TARGET[1], SAFE_Z])
    pose_place = RigidTransform(R_WG, PLACE_TARGET)
    
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

    def set_gripper(width_force, wait_time=1.0):
        wsg_command.get_mutable_source_value(wsg_context).set_value([width_force, width_force])
        current_time = context.get_time()
        simulator.AdvanceTo(current_time + wait_time)

    # TRAJECTORY
    move_arm(pose_pre_pick, duration=3.0)
    set_gripper(0.15, wait_time=0.5) 
    move_arm(pose_pick, duration=3.0)
    set_gripper(0.15, wait_time=0.5)
    set_gripper(-40.0, wait_time=1.5)
    move_arm(pose_pre_pick, duration=2.0)
    move_arm(pose_pre_place, duration=3.0)
    move_arm(pose_place, duration=3.0)
    set_gripper(-40.0, wait_time=0.5)
    set_gripper(0.11, wait_time=1.0) 
    move_arm(pose_pre_place, duration=2.0)
    simulator.AdvanceTo(context.get_time() + 3.0)

if __name__ == "__main__":
    run_test_simulation()