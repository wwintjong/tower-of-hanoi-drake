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
        self.DeclareDiscreteState([0.11])
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1e-3, 
            offset_sec=0.0, 
            update=self.UpdateState
        )
        self.kp = 500.0
        self.kd = 10.0
        self.closing_speed = 0.2
        self.grasp_threshold = 40.0
        self.squeeze_penetration = 0.002

    def UpdateState(self, context, discrete_state):
        command = self.get_input_port(0).Eval(context)
        state = self.get_input_port(1).Eval(context)
        target_width = command[0]
        current_ramped_width = context.get_discrete_state(0).get_value()[0]
        q_left, v_left = state[0], state[2]
        q_left_des = -current_ramped_width / 2.0
        f_current = abs(self.kp * (q_left_des - q_left) - self.kd * v_left)
        dt = 1e-3
        step = self.closing_speed * dt
        is_trying_to_close = target_width < current_ramped_width
        
        if is_trying_to_close:
            if f_current > self.grasp_threshold:
                actual_width = -2.0 * q_left
                next_width = actual_width - self.squeeze_penetration
                next_width = min(next_width, current_ramped_width)
            else:
                next_width = max(target_width, current_ramped_width - step)
        else:
            next_width = min(target_width, current_ramped_width + step)
            
        discrete_state.get_mutable_vector().SetFromVector([next_width])

    def CalcOutput(self, context, output):
        command = self.get_input_port(0).Eval(context)
        state = self.get_input_port(1).Eval(context)
        force_limit = command[1]
        ramped_width = context.get_discrete_state(0).get_value()[0]
        q_left, q_right = state[0], state[1]
        v_left, v_right = state[2], state[3]
        q_left_des = -ramped_width / 2.0
        q_right_des = ramped_width / 2.0
        f_left = self.kp * (q_left_des - q_left) - self.kd * v_left
        f_right = self.kp * (q_right_des - q_right) - self.kd * v_right
        f_left = np.clip(f_left, -force_limit, force_limit)
        f_right = np.clip(f_right, -force_limit, force_limit)
        output.SetFromVector([f_left, f_right])

# --- CONFIGURATION ---
PEG_X = 0.5 
DISK_HEIGHT = 0.1 
GRIPPER_FINGER_OFFSET = 0.12 

# Physical dimensions
TABLE_SURFACE_Z = 0.05  # Ground level / table surface
PLATFORM_HEIGHT = 0.15  # Platform thickness
PLATFORM_TOP_Z = TABLE_SURFACE_Z + PLATFORM_HEIGHT  # 0.20m - where disks sit
DISK_THICKNESS = 0.06   # Height of each torus
# Grasp heights for 3 disks (accounting for platform elevation)
GRASP_HEIGHTS = [0.20, 0.26, 0.32]  # z-heights for positions 0, 1, 2 on a tower

# Tower configurations (grasp points)
TOWERS = {
    1: np.array([0.36, -0.5]),   # Tower 1 (left)
    2: np.array([0.36, 0.0]),    # Tower 2 (middle)
    3: np.array([0.36, 0.5])    # Tower 3 (right)
}

# Movement parameters
PARAMS = {
    'safe_height': 0.55,           # Height for moving between towers (above all disks)
    'move_duration': 3.0,          # Time for horizontal moves
    'vertical_duration': 2.5,      # Time for vertical moves
    'gripper_wait': 2.2,           # Time to wait after gripper action
    'quick_wait': 0.5,             # Shorter wait for mid-motion checks
    'gripper_open': 0.15,          # Gripper width when open
    'gripper_closed': 0.01,        # Gripper width when closed
    'gripper_force': 200.0         # Gripper force limit
}

def get_grasp_height(position_on_tower):
    """Get the z-height for grasping a disk at given position on tower (0=bottom, 1=second, etc.)"""
    return GRASP_HEIGHTS[position_on_tower]

def calculate_disk_height(disk_position_on_tower):
    """Calculate z-coordinate for a disk at given position (0=bottom, 1=second, etc.)
    This is an alias for get_grasp_height for backward compatibility"""
    return get_grasp_height(disk_position_on_tower)

def calculate_ik(plant, context, target_pose):
    """Calculate inverse kinematics for a target pose"""
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

def make_pose(xy_position, z_height):
    """Create a pose for the gripper at given xy position and height"""
    R_Crane = RotationMatrix(np.column_stack(([1, 0, 0], [0, 0, -1], [0, 1, 0])))
    position = np.array([xy_position[0], xy_position[1], z_height])
    position = position + np.array([0, 0, GRIPPER_FINGER_OFFSET])
    return RigidTransform(R_Crane, position)

def make_traj(q_start, q_end, duration):
    """Create a smooth trajectory between two joint configurations"""
    breaks = [0.0, duration]
    samples = np.column_stack((q_start, q_end))
    v_zero = np.zeros_like(q_start)
    samples_dot = np.column_stack((v_zero, v_zero))
    return PiecewisePolynomial.CubicHermite(breaks, samples, samples_dot)

def create_controller_plant(time_step=1e-3):
    """Create a plant for the inverse dynamics controller"""
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
    """Places 3 disks at tower 2 (middle tower) in initial configuration"""
    # Only using 3 disks now: disk_3, disk_2, disk_1
    disk_names = ["disk_3", "disk_2", "disk_1"]
    
    for i, name in enumerate(disk_names):
        try:
            model = plant.GetModelInstanceByName(name)
            body = plant.GetBodyByName("torus_link", model)
            # Use the actual grasp heights for initial placement
            # Disk center is at grasp height (midpoint of torus)
            z_center = GRASP_HEIGHTS[i]
            X_World_Disk = RigidTransform(RotationMatrix(), [PEG_X, 0.0, z_center])
            plant.SetFreeBodyPose(context, body, X_World_Disk)
        except Exception as e:
            print(f"Could not reset {name}: {e}")

class TrajectoryBuilder:
    """Helper class to build complex trajectories from simple movements"""
    
    def __init__(self, plant, plant_context, iiwa_model, start_time, start_q):
        self.plant = plant
        self.plant_context = plant_context
        self.iiwa_model = iiwa_model
        self.current_time = start_time
        self.current_q = start_q
        self.times = [start_time]
        self.positions = [start_q]
        self.velocities = [np.zeros(7)]
        self.gripper_events = []
        
    def add_waypoint(self, pose, duration, wait_after=0.0, gripper_action=None):
        """Add a waypoint with optional gripper action"""
        # Calculate IK - the plant context should already have the right seed from previous call
        q_target = calculate_ik(self.plant, self.plant_context, pose)
        
        # After IK, update plant to this new solution (becomes seed for next IK)
        self.plant.SetPositions(self.plant_context, self.iiwa_model, q_target)
        
        # Create trajectory segment
        segment_traj = make_traj(self.current_q, q_target, duration)
        segment_vel = segment_traj.MakeDerivative()
        
        # Sample the trajectory
        dt = 0.02
        segment_times = np.arange(0, duration, dt)
        if segment_times[-1] < duration - dt/2:
            segment_times = np.append(segment_times, duration)
        
        for t in segment_times[1:]:
            self.times.append(self.current_time + t)
            self.positions.append(segment_traj.value(t).flatten())
            self.velocities.append(segment_vel.value(t).flatten())
        
        self.current_time += duration
        
        # Add gripper action if specified
        if gripper_action:
            action_time = self.current_time + 0.5
            self.gripper_events.append((action_time, gripper_action['width'], 
                                       gripper_action['force'], gripper_action['description']))
        
        # Add wait time and update current configuration
        self.current_q = q_target
        if wait_after > 0:
            self.times.append(self.current_time + wait_after)
            self.positions.append(self.current_q)
            self.velocities.append(np.zeros(7))
            self.current_time += wait_after
            
        return self
    
    def pick_disk(self, tower_num, grasp_z_height):
        """Execute a pick operation at specified tower and z-height"""
        tower_xy = TOWERS[tower_num]
        
        print(f"    Planning pick at tower {tower_num}: xy={tower_xy}, grasp_z={grasp_z_height:.3f}")
        
        # Move to safe height above tower
        pose_above = make_pose(tower_xy, PARAMS['safe_height'])
        self.add_waypoint(pose_above, PARAMS['move_duration'], PARAMS['quick_wait'],
                         gripper_action={'width': PARAMS['gripper_open'], 
                                       'force': PARAMS['gripper_force'],
                                       'description': f"Ensure OPEN above tower {tower_num}"})
        
        # Descend to grasp height
        pose_pick = make_pose(tower_xy, grasp_z_height)
        print(f"    Pick pose: [{tower_xy[0]:.3f}, {tower_xy[1]:.3f}, {grasp_z_height + GRIPPER_FINGER_OFFSET:.3f}]")
        self.add_waypoint(pose_pick, PARAMS['vertical_duration'], PARAMS['gripper_wait'],
                         gripper_action={'width': PARAMS['gripper_closed'], 
                                       'force': PARAMS['gripper_force'],
                                       'description': f"CLOSE gripper on tower {tower_num}"})
        
        # Lift back to safe height
        self.add_waypoint(pose_above, PARAMS['vertical_duration'], PARAMS['quick_wait'],
                         gripper_action={'width': PARAMS['gripper_closed'], 
                                       'force': PARAMS['gripper_force'],
                                       'description': "Hold CLOSED while lifting"})
        
        return self
    
    def place_disk(self, tower_num, grasp_z_height):
        """Execute a place operation at specified tower and z-height"""
        tower_xy = TOWERS[tower_num]
        
        print(f"    Planning place at tower {tower_num}: xy={tower_xy}, grasp_z={grasp_z_height:.3f}")
        
        # Move to safe height above target tower
        pose_above = make_pose(tower_xy, PARAMS['safe_height'])
        self.add_waypoint(pose_above, PARAMS['move_duration'], PARAMS['quick_wait'],
                         gripper_action={'width': PARAMS['gripper_closed'], 
                                       'force': PARAMS['gripper_force'],
                                       'description': f"Hold CLOSED above tower {tower_num}"})
        
        # Descend to place height
        pose_place = make_pose(tower_xy, grasp_z_height)
        print(f"    Place pose: [{tower_xy[0]:.3f}, {tower_xy[1]:.3f}, {grasp_z_height + GRIPPER_FINGER_OFFSET:.3f}]")
        self.add_waypoint(pose_place, PARAMS['vertical_duration'], PARAMS['gripper_wait'],
                         gripper_action={'width': PARAMS['gripper_open'], 
                                       'force': PARAMS['gripper_force'],
                                       'description': f"OPEN gripper on tower {tower_num}"})
        
        # Retreat to safe height
        self.add_waypoint(pose_above, PARAMS['vertical_duration'], PARAMS['quick_wait'],
                         gripper_action={'width': PARAMS['gripper_open'], 
                                       'force': PARAMS['gripper_force'],
                                       'description': "Hold OPEN while retreating"})
        
        return self
    
    def build(self):
        """Build the final trajectory and gripper events"""
        times = np.array(self.times)
        positions = np.array(self.positions)
        velocities = np.array(self.velocities)
        full_state = np.vstack((positions.T, velocities.T))
        continuous_traj = PiecewisePolynomial.FirstOrderHold(times, full_state)
        self.gripper_events.sort(key=lambda x: x[0])
        return continuous_traj, self.gripper_events, times[-1]

def solve_hanoi(n, source, target, auxiliary):
    """Generate the sequence of moves to solve Tower of Hanoi
    Returns list of (from_tower, to_tower) tuples"""
    moves = []
    
    def hanoi_recursive(n, source, target, auxiliary):
        if n == 1:
            moves.append((source, target))
        else:
            hanoi_recursive(n - 1, source, auxiliary, target)
            moves.append((source, target))
            hanoi_recursive(n - 1, auxiliary, target, source)
    
    hanoi_recursive(n, source, target, auxiliary)
    return moves

def run_simulation():
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
    
    # =================================================================
    # BUILD TRAJECTORY FOR TOWER OF HANOI
    # =================================================================
    
    # Solve Tower of Hanoi for 3 disks from tower 2 to tower 3
    moves = solve_hanoi(3, source=2, target=3, auxiliary=1)
    print(f"\nTower of Hanoi solution ({len(moves)} moves):")
    for i, (from_t, to_t) in enumerate(moves, 1):
        print(f"  Move {i}: Tower {from_t} → Tower {to_t}")
    
    # Track disk positions on each tower (bottom to top)
    # Each entry is (disk_number, current_z_height)
    # Now using 3 disks: disk_3 (largest), disk_2 (medium), disk_1 (smallest)
    tower_state = {
        1: [],  # Empty
        2: [(3, GRASP_HEIGHTS[0]), (2, GRASP_HEIGHTS[1]), (1, GRASP_HEIGHTS[2])],  # 3 disks on tower 2
        3: []   # Empty
    }
    
    # IMPORTANT: Use seed configuration for first IK solution like original code
    q_seed_forward = np.array([0.0, 0.6, 0.0, -1.2, 0.0, 1.6, 0.0])
    plant.SetPositions(plant_context, iiwa_model, q_seed_forward)
    
    # Build trajectory starting from settled position
    traj_builder = TrajectoryBuilder(plant, plant_context, iiwa_model, settle_time, q0)
    
    for move_num, (from_tower, to_tower) in enumerate(moves, 1):
        # Get the top disk from source tower
        disk_num, current_z_height = tower_state[from_tower][-1]  # Top disk with its z-height
        
        # Determine new z-height on destination tower
        # New position is at the bottom-most available spot
        destination_position = len(tower_state[to_tower])  # 0 if empty, 1 if one disk, etc.
        new_z_height = GRASP_HEIGHTS[destination_position]
        
        print(f"\n--- Move {move_num}/{len(moves)}: Disk {disk_num} from Tower {from_tower} to Tower {to_tower} ---")
        print(f"    Current z-height: {current_z_height:.3f}m, New z-height: {new_z_height:.3f}m")
        
        # Execute pick and place
        traj_builder.pick_disk(from_tower, current_z_height)
        traj_builder.place_disk(to_tower, new_z_height)
        
        # Update tower state
        tower_state[from_tower].pop()
        tower_state[to_tower].append((disk_num, new_z_height))
    
    # Build final trajectory
    continuous_traj, gripper_events, end_time = traj_builder.build()
    traj_source.UpdateTrajectory(continuous_traj)
    
    # Execute with gripper commands
    print("\n=== EXECUTION ===")
    for action_time, width, force, desc in gripper_events:
        simulator.AdvanceTo(action_time)
        print(f"[{action_time:.2f}s] {desc}: w={width:.3f}, f={force:.1f}")
        wsg_command.get_mutable_source_value(wsg_context).set_value([width, force])
    
    simulator.AdvanceTo(end_time)
    print("\n🎉 Tower of Hanoi complete!")
    print(f"Final tower state: {tower_state}")

if __name__ == "__main__":
    run_simulation()
