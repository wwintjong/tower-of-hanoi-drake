import numpy as np
import os
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LoadModelDirectives,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
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
    MultibodyPlant,
    LeafSystem
)

# --- CUSTOM CONTROLLER ---
class GripperController(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.DeclareVectorInputPort("command", 2) 
        self.DeclareVectorInputPort("state", 4)   
        self.DeclareVectorOutputPort("force", 2, self.CalcOutput)
        self.kp = 2000.0
        self.kd = 5.0

    def CalcOutput(self, context, output):
        command = self.get_input_port(0).Eval(context)
        state = self.get_input_port(1).Eval(context)
        width_des, force_limit = command[0], command[1]
        
        # State: [q_left, q_right, v_left, v_right]
        q_left, q_right = state[0], state[1]
        v_left, v_right = state[2], state[3]
        
        # Target: Centered grasp
        q_left_des = -width_des / 2.0
        q_right_des = width_des / 2.0
        
        # PD Calculation
        f_left = self.kp * (q_left_des - q_left) - self.kd * v_left
        f_right = self.kp * (q_right_des - q_right) - self.kd * v_right
        
        # Force Limits
        f_left = np.clip(f_left, -force_limit, force_limit)
        f_right = np.clip(f_right, -force_limit, force_limit)
        
        output.SetFromVector([f_left, f_right])

# --- CONFIGURATION ---
PEG_X = 0.5 
SAFE_Z = 0.65 
TABLE_SURFACE_Z = 0.0 
DISK_HEIGHT = 0.1 

# User Targets
PICK_XY = [0.32, 0.0]   
PLACE_XY = [0.32, 0.75]

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
        print(f"IK Failed for {target_pose.translation()}")
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

    # --- UPDATED GRIPPER SETUP ---
    wsg_model = plant.GetModelInstanceByName("wsg")
    wsg_command = builder.AddSystem(ConstantVectorSource([0.11, 200.0])) 
    wsg_controller = builder.AddSystem(GripperController())
    
    builder.Connect(wsg_command.get_output_port(), wsg_controller.get_input_port(0))
    builder.Connect(plant.get_state_output_port(wsg_model), wsg_controller.get_input_port(1))
    builder.Connect(wsg_controller.get_output_port(0), plant.get_actuation_input_port(wsg_model))

    dummy_traj = PiecewisePolynomial.ZeroOrderHold([0, 0.1], np.zeros((14, 2)))
    traj_source = builder.AddSystem(TrajectorySource(dummy_traj))
    builder.Connect(traj_source.get_output_port(), arm_controller.get_input_port_desired_state())

    meshcat_params = MeshcatVisualizerParams()
    meshcat_params.publish_period = 0.01 
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, meshcat_params)
    
    diagram = builder.Build()
    
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)
    wsg_context = wsg_command.GetMyContextFromRoot(context)
    
    reset_disks(plant, plant_context)
    q0 = plant.GetPositions(plant_context, iiwa_model)
    
    # Settle
    settle_time = 3.0
    full_state_traj = PiecewisePolynomial.FirstOrderHold(
        [0, settle_time],
        np.column_stack((np.concatenate((q0, np.zeros(7))), np.concatenate((q0, np.zeros(7)))))
    )
    traj_source.UpdateTrajectory(full_state_traj)
    print(f"Settling for {settle_time}s...")
    simulator.AdvanceTo(settle_time)
    
    R_Crane = RotationMatrix(np.column_stack(([1, 0, 0], [0, 0, -1], [0, 1, 0])))

    # Waypoints
    p_pre_pick = np.array([PICK_XY[0], PICK_XY[1], 0.55]) 
    pose_pre_pick = RigidTransform(R_Crane, p_pre_pick + [0, 0, GRIPPER_FINGER_OFFSET])
    
    p_pick = np.array([PICK_XY[0], PICK_XY[1], 0.225]) 
    pose_pick = RigidTransform(R_Crane, p_pick + [0, 0, GRIPPER_FINGER_OFFSET])
    
    p_pre_place = np.array([PLACE_XY[0], PLACE_XY[1], 0.55])
    pose_pre_place = RigidTransform(R_Crane, p_pre_place + [0, 0, GRIPPER_FINGER_OFFSET])
    
    p_place = np.array([PLACE_XY[0], PLACE_XY[1], 0.40])
    pose_place = RigidTransform(R_Crane, p_place + [0, 0, GRIPPER_FINGER_OFFSET])

    # IK
    print("Calculating IK solutions...")
    waypoints = []
    q_seed_forward = np.array([0.0, 0.6, 0.0, -1.2, 0.0, 1.6, 0.0])
    
    plant.SetPositions(plant_context, iiwa_model, q_seed_forward)
    q_pre_pick = calculate_ik(plant, plant_context, pose_pre_pick)
    waypoints.append(q_pre_pick)
    
    plant.SetPositions(plant_context, iiwa_model, q_pre_pick)
    q_pick = calculate_ik(plant, plant_context, pose_pick)
    waypoints.append(q_pick)
    
    plant.SetPositions(plant_context, iiwa_model, q_pick)
    q_lift = calculate_ik(plant, plant_context, pose_pre_pick)
    waypoints.append(q_lift)
    
    plant.SetPositions(plant_context, iiwa_model, q_lift)
    q_pre_place = calculate_ik(plant, plant_context, pose_pre_place)
    waypoints.append(q_pre_place)
    
    plant.SetPositions(plant_context, iiwa_model, q_pre_place)
    q_place = calculate_ik(plant, plant_context, pose_place)
    waypoints.append(q_place)
    
    plant.SetPositions(plant_context, iiwa_model, q_place)
    q_retreat = calculate_ik(plant, plant_context, pose_pre_place)
    waypoints.append(q_retreat)
    
    # Trajectory
    print("Building continuous trajectory...")
    start_time = settle_time
    durations = [3.0, 2.5, 2.0, 3.0, 2.5, 2.0]
    gripper_waits = [2.2, 2.2, 0.5, 0.5, 2.2, 0.5]
    
    times = [start_time]
    positions = [q0]
    velocities = [np.zeros(7)]
    
    current_time = start_time
    current_q = q0
    event_times = {}
    
    for i, (q_target, duration, wait) in enumerate(zip(waypoints, durations, gripper_waits)):
        segment_traj = make_traj(current_q, q_target, duration)
        segment_vel = segment_traj.MakeDerivative()
        dt = 0.02
        segment_times = np.arange(0, duration, dt)
        if segment_times[-1] < duration - dt/2: segment_times = np.append(segment_times, duration)
        
        for t in segment_times[1:]:
            times.append(current_time + t)
            positions.append(segment_traj.value(t).flatten())
            velocities.append(segment_vel.value(t).flatten())
        
        current_time += duration
        arrival_time = current_time
        
        if i == 0: event_times['ensure_open_prepick'] = arrival_time + 0.5
        elif i == 1: event_times['close_gripper'] = arrival_time + 0.5
        elif i == 2: event_times['ensure_closed_lift'] = arrival_time + 0.1
        elif i == 3: event_times['ensure_closed_preplace'] = arrival_time + 0.1
        elif i == 4: event_times['open_gripper_place'] = arrival_time + 0.5
        elif i == 5: event_times['ensure_open_retreat'] = arrival_time + 0.1
        
        current_q = q_target
        times.append(current_time + wait)
        positions.append(current_q)
        velocities.append(np.zeros(7))
        current_time += wait
    
    times = np.array(times)
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    full_state = np.vstack((positions.T, velocities.T))
    continuous_traj = PiecewisePolynomial.FirstOrderHold(times, full_state)
    traj_source.UpdateTrajectory(continuous_traj)
    
    # Gripper Sequence
    gripper_actions = [
        (event_times['ensure_open_prepick'], 0.15, 200.0, "Ensure OPEN"),
        (event_times['close_gripper'], 0.01, 200.0, "CLOSE gripper"),
        (event_times['ensure_closed_lift'], 0.01, 200.0, "Hold CLOSE"),
        (event_times['ensure_closed_preplace'], 0.01, 200.0, "Hold CLOSE"),
        (event_times['open_gripper_place'], 0.15, 200.0, "OPEN gripper"),
        (event_times['ensure_open_retreat'], 0.15, 200.0, "Hold OPEN"),
    ]
    gripper_actions.sort(key=lambda x: x[0])
    
    print("\n=== EXECUTION ===")
    for action_time, width, force, desc in gripper_actions:
        simulator.AdvanceTo(action_time)
        print(f"[{action_time:.2f}s] {desc}: w={width}, f={force}")
        wsg_command.get_mutable_source_value(wsg_context).set_value([width, force])
    
    simulator.AdvanceTo(times[-1])
    print("Complete.")

if __name__ == "__main__":
    run_test_simulation()