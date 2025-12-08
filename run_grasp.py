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
    RigidTransform,
    RotationMatrix,
    InverseDynamicsController,
    ConstantVectorSource,
    MultibodyPlant,
    PiecewisePolynomial,
    InputPort
)

# --- CONFIGURATION ---
SAFE_Z_HEIGHT = 0.55  # Height to travel between pegs (clearance)
GRIPPER_OPEN = 0.11   # Meters
GRIPPER_CLOSE = 0.005 # Meters (basically closed, stops when hits object)
MOVE_TIME = 2.0       # Seconds per segment

class HanoiStateMachine:
    def __init__(self, plant, context, internal_plant):
        self.plant = plant
        self.context = context
        self.internal_plant = internal_plant # Lightweight plant for IK
        
        # State Tracking
        self.current_state = "INIT"
        self.start_time = 0.0
        self.trajectory = None
        self.gripper_target = GRIPPER_OPEN
        
        # We need to know where the pegs are. 
        # Based on your YAML: Peg 2 (Start) is at Y=0.0, Peg 3 (Goal) is at Y=1.0
        self.start_peg_loc = np.array([0.75, 0.0, 0.35]) 
        self.goal_peg_loc = np.array([0.75, 1.0, 0.35]) 

    def get_current_robot_q(self):
        iiwa = self.plant.GetModelInstanceByName("iiwa")
        return self.plant.GetPositions(self.context, iiwa)

    def solve_ik(self, target_pose):
        """ Calculates joint angles for a specific Cartesian pose """
        ik = InverseKinematics(self.internal_plant)
        gripper_frame = self.internal_plant.GetBodyByName("body").body_frame()
        
        # Position Constraint
        ik.AddPositionConstraint(
            frameB=gripper_frame, p_BQ=[0,0,0],
            frameA=self.internal_plant.world_frame(), 
            p_AQ_lower=target_pose.translation() - 0.005, 
            p_AQ_upper=target_pose.translation() + 0.005
        )
        
        # Orientation Constraint (Fingers pointing DOWN)
        ik.AddOrientationConstraint(
            frameA=self.internal_plant.world_frame(), R_Abar=target_pose.rotation(),
            frameB=gripper_frame, R_Bbar=RotationMatrix(),
            theta_bound=0.05
        )
        
        # Seed solver with current robot position for smoothness
        prog = ik.get_mutable_prog()
        prog.SetInitialGuess(ik.q(), self.get_current_robot_q())
        
        result = Solve(prog)
        if not result.is_success():
            print(f"[ERROR] IK Failed for target: {target_pose.translation()}")
            return self.get_current_robot_q() # Fail safe: stay put
        return result.GetSolution(ik.q())

    def make_traj(self, end_q, now, duration):
        """ Creates a Cubic Spline from current q to target q """
        start_q = self.get_current_robot_q()
        times = [now, now + duration]
        knots = np.vstack([start_q, end_q]).T
        return PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            times, knots, np.zeros(7), np.zeros(7)
        )

    def update(self, now):
        """ 
        Main FSM Loop. 
        Returns: (q_command, v_command, gripper_width_command) 
        """
        
        # If we have an active trajectory, just follow it
        if self.trajectory and now < self.trajectory.end_time():
            q_d = self.trajectory.value(now).flatten()
            v_d = self.trajectory.derivative(1).value(now).flatten()
            return q_d, v_d, self.gripper_target

        # If trajectory finished, or we are in INIT, switch states
        # FSM Logic Sequence
        
        if self.current_state == "INIT":
            # Wait for disks to settle
            if now > 2.0:
                print(">>> TRANSITION: INIT -> PRE_PICK")
                self.current_state = "PRE_PICK"
                
                # Calculate Pick Pose based on actual disk location
                disk_body = self.plant.GetBodyByName("disk_1")
                X_WD = self.plant.EvalBodyPoseInWorld(self.plant.GetMyContextFromRoot(self.context), disk_body)
                
                # Pre-Pick: High above disk
                R_down = RotationMatrix.MakeXRotation(np.pi)
                target_pose = RigidTransform(R_down, X_WD.translation() + [0, 0, 0.25])
                
                q_target = self.solve_ik(target_pose)
                self.trajectory = self.make_traj(q_target, now, MOVE_TIME)

        elif self.current_state == "PRE_PICK":
            print(">>> TRANSITION: PRE_PICK -> PICK")
            self.current_state = "PICK"
            
            # Recalculate precise disk location
            disk_body = self.plant.GetBodyByName("disk_1")
            X_WD = self.plant.EvalBodyPoseInWorld(self.plant.GetMyContextFromRoot(self.context), disk_body)
            
            # Pick: Encompassing disk (Z + 0.02)
            R_down = RotationMatrix.MakeXRotation(np.pi)
            target_pose = RigidTransform(R_down, X_WD.translation() + [0, 0, 0.02])
            
            q_target = self.solve_ik(target_pose)
            self.trajectory = self.make_traj(q_target, now, MOVE_TIME)

        elif self.current_state == "PICK":
            print(">>> TRANSITION: PICK -> CLOSE_GRIPPER")
            self.current_state = "CLOSE_GRIPPER"
            self.gripper_target = GRIPPER_CLOSE
            # Wait 1 second for gripper to close physically
            self.trajectory = self.make_traj(self.get_current_robot_q(), now, 1.0) 

        elif self.current_state == "CLOSE_GRIPPER":
            print(">>> TRANSITION: CLOSE_GRIPPER -> LIFT")
            self.current_state = "LIFT"
            
            # Lift straight up to safe Z
            current_xyz = self.plant.EvalBodyPoseInWorld(
                self.plant.GetMyContextFromRoot(self.context), 
                self.plant.GetBodyByName("iiwa_link_7")
            ).translation()
            
            target_xyz = np.array([current_xyz[0], current_xyz[1], SAFE_Z_HEIGHT])
            R_down = RotationMatrix.MakeXRotation(np.pi)
            target_pose = RigidTransform(R_down, target_xyz)
            
            q_target = self.solve_ik(target_pose)
            self.trajectory = self.make_traj(q_target, now, MOVE_TIME)

        elif self.current_state == "LIFT":
            print(">>> TRANSITION: LIFT -> MOVE_OVER_PEG")
            self.current_state = "MOVE_OVER_PEG"
            
            # Move horizontally to Goal Peg (Peg 3)
            # Peg 3 is at self.goal_peg_loc
            target_xyz = np.array([self.goal_peg_loc[0], self.goal_peg_loc[1], SAFE_Z_HEIGHT])
            R_down = RotationMatrix.MakeXRotation(np.pi)
            target_pose = RigidTransform(R_down, target_xyz)
            
            q_target = self.solve_ik(target_pose)
            self.trajectory = self.make_traj(q_target, now, MOVE_TIME)

        elif self.current_state == "MOVE_OVER_PEG":
            print(">>> TRANSITION: MOVE_OVER_PEG -> LOWER_PLACE")
            self.current_state = "LOWER_PLACE"
            
            # Lower onto Peg 3
            # Target height: Peg Base Z + peg height/padding
            # Peg base is 0.35. Let's aim for 0.40 to be safe
            target_xyz = np.array([self.goal_peg_loc[0], self.goal_peg_loc[1], 0.40])
            R_down = RotationMatrix.MakeXRotation(np.pi)
            target_pose = RigidTransform(R_down, target_xyz)
            
            q_target = self.solve_ik(target_pose)
            self.trajectory = self.make_traj(q_target, now, MOVE_TIME)
            
        elif self.current_state == "LOWER_PLACE":
            print(">>> TRANSITION: LOWER_PLACE -> OPEN_GRIPPER")
            self.current_state = "OPEN_GRIPPER"
            self.gripper_target = GRIPPER_OPEN
            # Wait 1 second to release
            self.trajectory = self.make_traj(self.get_current_robot_q(), now, 1.0)
            
        elif self.current_state == "OPEN_GRIPPER":
            print(">>> TRANSITION: OPEN_GRIPPER -> RETREAT")
            self.current_state = "RETREAT"
            
            # Move back up to Safe Z
            target_xyz = np.array([self.goal_peg_loc[0], self.goal_peg_loc[1], SAFE_Z_HEIGHT])
            R_down = RotationMatrix.MakeXRotation(np.pi)
            target_pose = RigidTransform(R_down, target_xyz)
            
            q_target = self.solve_ik(target_pose)
            self.trajectory = self.make_traj(q_target, now, MOVE_TIME)
            
        elif self.current_state == "RETREAT":
            # Done with one move
            pass

        # Return current state if we just switched
        if self.trajectory:
            return self.trajectory.value(now).flatten(), self.trajectory.derivative(1).value(now).flatten(), self.gripper_target
        
        # Idle/Hold
        return self.get_current_robot_q(), np.zeros(7), self.gripper_target


# --- HELPER: Create Internal Plant for Controller/IK ---
def create_internal_plant():
    plant = MultibodyPlant(time_step=1e-3)
    parser = Parser(plant)
    # IIWA
    parser.AddModels("package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf")
    plant.WeldFrames(plant.world_frame(), plant.GetBodyByName("iiwa_link_0").body_frame(), RigidTransform([0, 0, 0.05]))
    # WSG
    parser.AddModels("package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf")
    plant.WeldFrames(plant.GetBodyByName("iiwa_link_7").body_frame(), plant.GetBodyByName("body").body_frame(), 
                     RigidTransform(RotationMatrix.MakeYRotation(np.pi/2).multiply(RotationMatrix.MakeZRotation(np.pi/2)), [0,0,0.09]))
    plant.Finalize()
    return plant

# --- MAIN EXECUTION ---
def run_hanoi_fsm():
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
    parser = Parser(plant)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.package_map().Add("tower_of_hanoi_drake", current_dir) 
    ProcessModelDirectives(LoadModelDirectives(os.path.join(current_dir, "load_scene.yaml")), plant, parser)
    plant.Finalize()
    
    # Controllers
    # 1. IIWA
    iiwa_plant = create_internal_plant() # This internal plant has BOTH robot and gripper kinematic chain for IK
    iiwa_controller = builder.AddSystem(InverseDynamicsController(iiwa_plant, [100]*7, [1]*7, [20]*7, False))
    iiwa_input = builder.AddSystem(ConstantVectorSource(np.zeros(14))) # 7 pos, 7 vel
    
    # We need to map the full plant state to the controller state
    # This requires a Multiplexer or custom logic because the internal plant assumes just robot+gripper
    # For simplicity in this script, we will bypass the "Perfect" wiring and just wire the IIWA part.
    # Note: Correct wiring in Drake is complex. Here we assume the state indices align for the IIWA instance.
    
    builder.Connect(iiwa_input.get_output_port(), iiwa_controller.get_input_port_desired_state())
    builder.Connect(plant.get_state_output_port(plant.GetModelInstanceByName("iiwa")), iiwa_controller.get_input_port_estimated_state())
    builder.Connect(iiwa_controller.get_output_port_control(), plant.get_actuation_input_port(plant.GetModelInstanceByName("iiwa")))

    # 2. WSG
    wsg_controller = builder.AddSystem(InverseDynamicsController(plant, [100]*2, [1]*2, [20]*2, False)) # Using full plant for simple gripper PID is usually okay
    wsg_input = builder.AddSystem(ConstantVectorSource([0.05, -0.05, 0, 0]))
    builder.Connect(wsg_input.get_output_port(), wsg_controller.get_input_port_desired_state())
    builder.Connect(plant.get_state_output_port(plant.GetModelInstanceByName("wsg")), wsg_controller.get_input_port_estimated_state())
    builder.Connect(wsg_controller.get_output_port_control(), plant.get_actuation_input_port(plant.GetModelInstanceByName("wsg")))

    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    context = simulator.get_mutable_context()
    
    # Initialize FSM
    # Note: We pass the 'iiwa_plant' (internal) to the FSM for IK calculations
    fsm = HanoiStateMachine(plant, context, iiwa_plant)
    
    print(f"Open browser to: {meshcat.web_url()}")
    
    # --- SIMULATION LOOP ---
    sim_time = 0.0
    dt = 0.01 # 100Hz control update
    
    while sim_time < 20.0:
        # 1. Query FSM
        q_cmd, v_cmd, wsg_width = fsm.update(sim_time)
        
        # 2. Send commands to controllers
        # IIWA (Position + Velocity)
        iiwa_input.get_mutable_parameters().SetFromVector(np.concatenate([q_cmd, v_cmd]))
        
        # WSG (Width -> Position for two fingers)
        # Finger 1 moves +half_width, Finger 2 moves -half_width
        w_half = wsg_width / 2.0
        wsg_input.get_mutable_parameters().SetFromVector([w_half, -w_half, 0, 0])
        
        # 3. Step Sim
        sim_time += dt
        simulator.AdvanceTo(sim_time)

if __name__ == "__main__":
    run_hanoi_fsm()