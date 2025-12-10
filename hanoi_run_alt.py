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
        self.kp = 100.0 
        self.kd = 5.0

    def CalcOutput(self, context, output):
        command = self.get_input_port(0).Eval(context)
        state = self.get_input_port(1).Eval(context)
        width_des, force_limit = command[0], command[1]
        
        q_left, q_right = state[0], state[1]
        v_left, v_right = state[2], state[3]
        
        q_left_des = -width_des / 2.0
        q_right_des = width_des / 2.0
        
        f_left = self.kp * (q_left_des - q_left) - self.kd * v_left
        f_right = self.kp * (q_right_des - q_right) - self.kd * v_right
        
        f_left = np.clip(f_left, -force_limit, force_limit)
        f_right = np.clip(f_right, -force_limit, force_limit)
        
        output.SetFromVector([f_left, f_right])

# --- CONFIGURATION ---
# COORDINATES MATCHING YAML
PEG_X = 0.5 
SAFE_Z = 0.75  # High clearance
RETRACT_X = 0.35 # Safe X distance closer to robot base for traversing

# PHYSICS
DISK_HEIGHT = 0.03 
GRIPPER_FINGER_OFFSET = 0.12 

# TOWER COORDINATES (Must match load_scene.yaml)
TOWER_COORDS = {
    1: [PEG_X, -0.75], # Left/Aux
    2: [PEG_X, 0.0],   # Middle/Source
    3: [PEG_X, 0.75]   # Right/Target
}

def calculate_ik(plant, context, target_pose, seed_q):
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
    
    iiwa_model = plant.GetModelInstanceByName("iiwa")
    plant.SetPositions(context, iiwa_model, seed_q)
    full_seed = plant.GetPositions(context)
    
    prog = ik.get_mutable_prog()
    prog.SetInitialGuess(ik.q(), full_seed)
    
    result = Solve(prog)
    
    if not result.is_success():
        print(f"IK Failed for {target_pose.translation()}")
        return seed_q
    
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
    """ Places disks at the calculated stack heights on Tower 2. """
    disk_names = ["disk_4", "disk_3", "disk_2", "disk_1"]
    base_z = 0.05 

    start_tower = 2
    start_xy = TOWER_COORDS[start_tower]

    for i, name in enumerate(disk_names):
        try:
            model = plant.GetModelInstanceByName(name)
            body = plant.GetBodyByName("torus_link", model)
            z_center = base_z + (i * DISK_HEIGHT)
            X_World_Disk = RigidTransform(RotationMatrix(), [start_xy[0], start_xy[1], z_center])
            plant.SetFreeBodyPose(context, body, X_World_Disk)
        except Exception as e:
            print(f"Could not reset {name}: {e}")

def solve_hanoi(n, source, target, aux):
    if n == 1:
        return [(source, target)]
    else:
        moves = solve_hanoi(n - 1, source, aux, target)
        moves.append((source, target))
        moves += solve_hanoi(n - 1, aux, target, source)
        return moves

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

    # --- PLAN HANOI MOVES ---
    stacks = {1: [], 2: [4, 3, 2, 1], 3: []}
    moves = solve_hanoi(4, 2, 3, 1)
    
    # Timing
    durations = [3.0, 2.5, 2.0, 2.0, 2.0, 2.5, 2.0] # Extra steps for retract/extend
    gripper_waits = [1.5, 1.5, 0.0, 0.0, 0.0, 1.5, 0.5]
    
    trajectory_times = [settle_time]
    trajectory_positions = [q0]
    trajectory_velocities = [np.zeros(7)]
    
    gripper_schedule = []
    current_time = settle_time
    current_q = q0
    q_seed = np.array([0.0, 0.6, 0.0, -1.2, 0.0, 1.6, 0.0])
    
    for move_idx, (src, dst) in enumerate(moves):
        print(f"Move {move_idx+1}/15: {src} -> {dst}")
        
        src_xy = TOWER_COORDS[src]
        dst_xy = TOWER_COORDS[dst]
        
        z_pick = 0.05 + (len(stacks[src]) - 1) * DISK_HEIGHT
        z_place = 0.05 + (len(stacks[dst])) * DISK_HEIGHT
        
        # Update Stacks
        disk = stacks[src].pop()
        stacks[dst].append(disk)
        
        # --- POSES WITH RETRACTION ---
        # 1. Pre-Pick (Above Source)
        p_pre_pick = np.array([src_xy[0], src_xy[1], SAFE_Z]) 
        pose_pre_pick = RigidTransform(R_Crane, p_pre_pick + [0, 0, GRIPPER_FINGER_OFFSET])
        
        # 2. Pick (At Disk)
        p_pick = np.array([src_xy[0], src_xy[1], z_pick]) 
        pose_pick = RigidTransform(R_Crane, p_pick + [0, 0, GRIPPER_FINGER_OFFSET])
        
        # 3. Retract (Safe X, Source Y, Safe Z) - Pull back to avoid center peg collision
        p_retract = np.array([RETRACT_X, src_xy[1], SAFE_Z])
        pose_retract = RigidTransform(R_Crane, p_retract + [0, 0, GRIPPER_FINGER_OFFSET])
        
        # 4. Traverse (Safe X, Target Y, Safe Z) - Move sideways
        p_traverse = np.array([RETRACT_X, dst_xy[1], SAFE_Z])
        pose_traverse = RigidTransform(R_Crane, p_traverse + [0, 0, GRIPPER_FINGER_OFFSET])
        
        # 5. Extend (Target X, Target Y, Safe Z) - Move forward to target
        p_pre_place = np.array([dst_xy[0], dst_xy[1], SAFE_Z])
        pose_pre_place = RigidTransform(R_Crane, p_pre_place + [0, 0, GRIPPER_FINGER_OFFSET])
        
        # 6. Place (At Disk)
        p_place = np.array([dst_xy[0], dst_xy[1], z_place])
        pose_place = RigidTransform(R_Crane, p_place + [0, 0, GRIPPER_FINGER_OFFSET])
        
        # --- IK ---
        waypoints = []
        
        # To Pre-Pick
        plant.SetPositions(plant_context, iiwa_model, current_q)
        q_pre_pick = calculate_ik(plant, plant_context, pose_pre_pick, q_seed)
        waypoints.append(q_pre_pick)
        
        # To Pick
        plant.SetPositions(plant_context, iiwa_model, q_pre_pick)
        q_pick = calculate_ik(plant, plant_context, pose_pick, q_pre_pick)
        waypoints.append(q_pick)
        
        # Lift (To Pre-Pick)
        plant.SetPositions(plant_context, iiwa_model, q_pick)
        q_lift = calculate_ik(plant, plant_context, pose_pre_pick, q_pick)
        waypoints.append(q_lift)
        
        # Retract
        plant.SetPositions(plant_context, iiwa_model, q_lift)
        q_retract = calculate_ik(plant, plant_context, pose_retract, q_lift)
        waypoints.append(q_retract)
        
        # Traverse
        plant.SetPositions(plant_context, iiwa_model, q_retract)
        q_traverse = calculate_ik(plant, plant_context, pose_traverse, q_retract)
        waypoints.append(q_traverse)
        
        # Extend
        plant.SetPositions(plant_context, iiwa_model, q_traverse)
        q_extend = calculate_ik(plant, plant_context, pose_pre_place, q_traverse)
        waypoints.append(q_extend)
        
        # Descend
        plant.SetPositions(plant_context, iiwa_model, q_extend)
        q_place = calculate_ik(plant, plant_context, pose_place, q_extend)
        waypoints.append(q_place)
        
        # Retreat (To Pre-Place)
        plant.SetPositions(plant_context, iiwa_model, q_place)
        q_retreat = calculate_ik(plant, plant_context, pose_pre_place, q_place)
        waypoints.append(q_retreat)
        
        # --- BUILD SEGMENTS ---
        # 0: PrePick, 1: Pick, 2: Lift, 3: Retract, 4: Traverse, 5: Extend, 6: Place, 7: Retreat
        # Wait logic applied at: PrePick (Open), Pick (Close), Place (Open)
        
        segment_durations = [3.0, 2.5, 2.0, 2.0, 3.0, 2.0, 2.5, 2.0]
        segment_waits =     [1.5, 1.5, 0.0, 0.0, 0.0, 0.0, 1.5, 0.5]
        
        for k, (q_target, duration, wait) in enumerate(zip(waypoints, segment_durations, segment_waits)):
            segment_traj = make_traj(current_q, q_target, duration)
            segment_vel = segment_traj.MakeDerivative()
            
            dt = 0.05
            segment_times = np.arange(0, duration, dt)
            if segment_times[-1] < duration - dt/2: segment_times = np.append(segment_times, duration)
            
            for t in segment_times[1:]:
                trajectory_times.append(current_time + t)
                trajectory_positions.append(segment_traj.value(t).flatten())
                trajectory_velocities.append(segment_vel.value(t).flatten())
            
            current_time += duration
            arrival_time = current_time
            
            # Events
            trigger_time = arrival_time + 0.5
            if k == 0: # PrePick
                gripper_schedule.append((trigger_time, 0.15, 200.0, f"M{move_idx}: Open"))
            elif k == 1: # Pick
                gripper_schedule.append((trigger_time, 0.01, 200.0, f"M{move_idx}: Grasp"))
            elif k == 6: # Place
                gripper_schedule.append((trigger_time, 0.15, 200.0, f"M{move_idx}: Release"))
            
            current_q = q_target
            trajectory_times.append(current_time + wait)
            trajectory_positions.append(current_q)
            trajectory_velocities.append(np.zeros(7))
            current_time += wait

    # --- COMPILE ---
    print(f"Total time: {current_time:.2f}s")
    times_np = np.array(trajectory_times)
    pos_np = np.array(trajectory_positions)
    vel_np = np.array(trajectory_velocities)
    
    full_state = np.vstack((pos_np.T, vel_np.T))
    continuous_traj = PiecewisePolynomial.FirstOrderHold(times_np, full_state)
    traj_source.UpdateTrajectory(continuous_traj)
    
    # --- EXECUTE ---
    print("Executing...")
    wsg_command.get_mutable_source_value(wsg_context).set_value([0.01, 200.0])
    gripper_schedule.sort(key=lambda x: x[0])
    
    for action_time, width, force, desc in gripper_schedule:
        if action_time > times_np[-1]: break
        simulator.AdvanceTo(action_time)
        print(f"[{action_time:.2f}s] {desc}")
        wsg_command.get_mutable_source_value(wsg_context).set_value([width, force])
    
    simulator.AdvanceTo(times_np[-1])
    print("Complete.")

if __name__ == "__main__":
    run_test_simulation()