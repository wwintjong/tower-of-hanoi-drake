#!/usr/bin/env python3
"""
iiwa_hanoi_meshcat.py

Simulate a KUKA iiwa14 with a WSG gripper solving Tower of Hanoi (3 disks)
moving from middle peg to right peg, visualized in MeshCat.

Notes:
- Requires Drake / pydrake with SceneGraph, Meshcat, InverseKinematics, etc.
- Model URIs assume you have drake_models available on the package path; adjust
  package:// URIs if needed for your environment.
- You may need to adjust frame names depending on your URDF/SDF content.
"""
import os
import time
import tempfile
import numpy as np

from pydrake.all import (
    DiagramBuilder, MultibodyPlant, SceneGraph, Parser,
    RigidTransform, RollPitchYaw, AddMultibodyPlantSceneGraph,
    InverseKinematics, PositionConstraint, OrientationConstraint,
    Quaternion, Solve, PiecewisePolynomial, TrajectorySource,
    Simulator, MeshcatVisualizer, MeshcatVisualizerParams,
    JacobianWrtVariable, RotationMatrix
)

# ---- Helper: write small URDF discs to temp files ----
def make_disk_urdf(radius, thickness, mass, name):
    """
    Returns path to a temporary URDF file that describes a simple
    cylindrical disk collision+visual geometry (base link only).
    """
    urdf = f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="{name}_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia  ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{radius}" length="{thickness}"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.9 1"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{radius}" length="{thickness}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
    fd, path = tempfile.mkstemp(suffix=".urdf", text=True)
    with os.fdopen(fd, "w") as f:
        f.write(urdf)
    return path

# ---- Build world + iiwa + wsg + disks ----
def build_plant():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

    parser = Parser(plant)

    # Add iiwa
    iiwa_uri = "package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
    iiwa_model = parser.AddModelsFromUrl(iiwa_uri, model_name="iiwa")
    # set default positions later via named positions / initial context

    # Weld iiwa base to world as requested (iiwa_link_0)
    # We'll weld using frames by name after finalizing (but can do now via WeldFrames)
    # Add WSG
    wsg_uri = "package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf"
    wsg_model = parser.AddModelsFromUrl(wsg_uri, model_name="wsg")

    # Add simple ground plane (optional)
    # parser.AddModelFromFile("package://drake_models/ground_plane/plane.sdf")

    # Add three disk models (small cylinders)
    disks = []
    disk_paths = []
    radii = [0.035, 0.03, 0.025]  # large -> small
    thickness = 0.015
    masses = [0.05, 0.04, 0.03]
    for i, (r, m) in enumerate(zip(radii, masses)):
        name = f"disk_{i}"
        p = make_disk_urdf(r, thickness, m, name)
        disk_paths.append(p)
        disks.append(parser.AddModelFromFile(p, model_name=name))

    # Now finalize welds and positions.
    # We must finalize when all models are added.
    # But we need to perform welds by specifying frames. Use plant.GetBodyFrameByName etc after Finalize.
    plant.Finalize()

    # Set default joint positions for iiwa:
    # Provided defaults (dictionary) per your YAML:
    default_positions = {
        "iiwa_joint_1": [-1.57],
        "iiwa_joint_2": [0.1],
        "iiwa_joint_3": [0.0],
        "iiwa_joint_4": [-1.2],
        "iiwa_joint_5": [0.0],
        "iiwa_joint_6": [1.6],
        "iiwa_joint_7": [0.0],
    }
    # Apply them to the default context - we will do this later on the simulator's context.

    # Weld iiwa::iiwa_link_0 to the world with small z offset
    try:
        world_frame = plant.world_frame()
        iiwa_base_frame = plant.GetFrameByName("iiwa_link_0", model_instance_name="iiwa")
    except Exception:
        # Fallback: try plain name
        iiwa_base_frame = plant.GetFrameByName("iiwa_link_0")
    X_WP = RigidTransform(RollPitchYaw(0, 0, 0), [0, 0, 0.05])
    plant.WeldFrames(world_frame, iiwa_base_frame, X_WP)

    # Weld wsg body to iiwa_link_7 with the requested transform
    try:
        iiwa_link_7 = plant.GetFrameByName("iiwa_link_7", model_instance_name="iiwa")
    except Exception:
        iiwa_link_7 = plant.GetFrameByName("iiwa_link_7")
    try:
        wsg_body = plant.GetFrameByName("body", model_instance_name="wsg")
    except Exception:
        # some SDFs use different frame names; attempt common ones
        wsg_body = plant.GetFrameByName("body")
    X_PC = RigidTransform(RollPitchYaw(np.deg2rad(90), 0, np.deg2rad(90)), [0, 0, 0.09])
    plant.WeldFrames(iiwa_link_7, wsg_body, X_PC)

    # Place disks on middle peg: We'll create WeldJoint-like placements by adding a Joint? Simpler: we'll set their poses directly in the Simulator initial context.
    # But we still keep their bodies in the plant.

    # Return builder, plant, scene_graph and useful info
    return builder, plant, scene_graph, disk_paths, default_positions

# ---- Inverse kinematics helpers ----
def compute_ik_for_pose(plant, context, end_effector_frame, X_WT_desired, q_initial):
    """
    Solve IK to find joint positions q that bring end_effector_frame to X_WT_desired.
    Returns q (numpy array).
    """
    ik = InverseKinematics(plant, context)
    # Position constraint: end_effector origin (p_Bo) to desired translation
    p_Bo = end_effector_frame.CalcPose(context, plant.world_frame()).translation()
    # But better to use frame_point_in_child_frame = [0,0,0]
    # Add position constraint
    tolerance = 1e-3
    ik.AddPositionConstraint(
        frameA=plant.world_frame(),
        p_BQ=[0, 0, 0],  # world frame point (we will express via SetFrame)
        frameB=end_effector_frame,
        p_BQ_lower=X_WT_desired.translation() - tolerance,
        p_BQ_upper=X_WT_desired.translation() + tolerance,
    )
    # Orientation constraint (quaternion)
    R = X_WT_desired.rotation().matrix()
    # orientation constraint takes axis-angle tolerances; provide small tolerance
    # Use OrientationConstraint convenience:
    ik.AddOrientationConstraint(
        frameA=plant.world_frame(),
        frameAbar_R_frameB=R,
        frameB=end_effector_frame,
        angle_tol=0.05
    )

    prog = ik.prog()
    q_vars = ik.q()  # convenience variable
    prog.SetInitialGuess(q_vars, q_initial)
    result = Solve(prog)
    if not result.is_success():
        raise RuntimeError("IK failed")
    q_sol = result.GetSolution(q_vars)
    return q_sol

# ---- Build the trajectory through the pick/place waypoints ----
def plan_hanoi_joint_trajectory(plant, context, iiwa_frame, q0, peg_positions):
    """
    Given peg end-effector poses (a dict with names->RigidTransform targets for the 'grasp')
    and initial joint positions q0, compute joint-space waypoints for the classic
    Hanoi solution for 3 disks (middle -> right). We'll return a PiecewisePolynomial
    joint-space trajectory that cycles through the waypoint joint values.
    """
    # For simplicity, we'll define a small set of named poses:
    # pre_grasp_above (0.08 m above disk), grasp (at disk height),
    # lift (0.12 m above), move_above_target, place (at disk), retreat.
    def make_poses_for_peg(peg_x, peg_y, disk_z):
        # base height of peg is disk_z for top of current disk
        poses = {}
        # grasp pose: keep end effector pointing down (z axis of end-effector aligns -z in world).
        # We'll build RPY such that gripper points down: rotation of -90 deg about x to point down.
        r = RollPitchYaw(np.pi, 0, 0)  # maybe flips; this is heuristic and may need tuning
        # position: slightly above peg center
        grasp_tf = RigidTransform(RotationMatrix.MakeFromRollPitchYaw(0, np.pi/2, 0), [peg_x, peg_y, disk_z + 0.02,])
        poses["grasp"] = grasp_tf
        poses["pre_grasp"] = RigidTransform(grasp_tf.rotation(), [peg_x, peg_y, disk_z + 0.08])
        poses["lift"] = RigidTransform(grasp_tf.rotation(), [peg_x, peg_y, disk_z + 0.15])
        poses["place"] = RigidTransform(grasp_tf.rotation(), [peg_x, peg_y, disk_z + 0.02])
        poses["retreat"] = RigidTransform(grasp_tf.rotation(), [peg_x, peg_y, disk_z + 0.08])
        return poses

    # We'll define peg_positions as dict: { 'left': (x,y), 'mid':..., 'right': ... } and a stack heights per move
    # Build the sequence of moves for Tower of Hanoi with three disks to move stack middle -> right.
    # Hardcode the minimal sequence of 7 moves (for 3 disks). Each move is (src, dst).
    moves = [
        ('mid', 'right'),
        ('mid', 'left'),
        ('right', 'left'),
        ('mid', 'right'),
        ('left', 'mid'),
        ('left', 'right'),
        ('mid', 'right')
    ]
    # For a correct sequence we need to decide which disk height to pick/place each time.
    # We will simulate a simple stack height map that decrements/increments per move.
    # Initialize stack heights: left:0, mid:3, right:0 (count of disks)
    stacks = {'left': 0, 'mid': 3, 'right': 0}
    # Prepare joint waypoints
    q_waypoints = [q0.copy()]
    times = [0.0]
    t = 0.0
    dt_move = 2.0   # seconds per primitive motion (pre_grasp -> grasp -> lift -> move -> place -> retreat)
    # For each move, build subposes
    for move in moves:
        src, dst = move
        # disk is top disk of src (height index = stacks[src]-1)
        src_height = stacks[src]
        dst_height = stacks[dst]
        # compute disk top Z for src: assume peg top at z=0.0 and each disk thickness ~ 0.015
        disk_thickness = 0.015
        disk_z_src = 0.05 + (src_height - 1) * disk_thickness if src_height > 0 else 0.05
        disk_z_dst = 0.05 + dst_height * disk_thickness  # new top after place
        # create poses for src and dst
        peg_coords_src = peg_positions[src]
        peg_coords_dst = peg_positions[dst]
        poses_src = make_poses_for_peg(peg_coords_src[0], peg_coords_src[1], disk_z_src)
        poses_dst = make_poses_for_peg(peg_coords_dst[0], peg_coords_dst[1], disk_z_dst)

        # For each primitive pose (pre_grasp_src -> grasp_src -> lift_src -> move_above_dst -> place_dst -> retreat_dst)
        primitives = [
            poses_src["pre_grasp"],
            poses_src["grasp"],
            poses_src["lift"],
            poses_dst["lift"],
            poses_dst["place"],
            poses_dst["retreat"]
        ]
        for prim in primitives:
            # Solve IK for prim
            try:
                q_sol = compute_ik_for_pose(plant, context, iiwa_frame, prim, q_waypoints[-1])
            except Exception as e:
                # if IK fails, try using last q as fallback
                print("IK failed for a pose; using last joint value as fallback:", str(e))
                q_sol = q_waypoints[-1]
            t += dt_move / len(primitives)
            q_waypoints.append(q_sol)
            times.append(t)
        # update stacks
        stacks[src] -= 1
        stacks[dst] += 1

    # Build piecewise cubic trajectory through q_waypoints
    q_array = np.column_stack(q_waypoints)  # shape (nq, n_knots)
    traj = PiecewisePolynomial.Cubic(times, q_array)
    return traj

# ---- Main: assemble diagram & run simulator ----
def main():
    builder, plant, scene_graph, disk_paths, default_positions = build_plant()

    # Add Meshcat visualizer to builder (will create meshcat server)
    meshcat = None
    try:
        # Use default Meshcat instance
        meshcat = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, zmq_url=None, open_browser=True)
        # AddToBuilder returns a system in older Drake; in newer Drake use ConnectMeshcatVisualizer
    except Exception:
        # fallback method
        from pydrake.geometry import MeshcatVisualizer as MCV
        meshcat_vis = MCV.AddToBuilder(builder, scene_graph)
        meshcat = meshcat_vis

    diagram = builder.Build()
    simulator = Simulator(diagram)
    diagram_context = diagram.GetMutableSubsystemContext(plant, simulator.get_mutable_context())
    # We need to set initial joint positions for the iiwa joints as requested
    # Find the model instance for iiwa and set positions in plant context
    iiwa_instance = plant.GetModelInstanceByName("iiwa")
    joint_names = [
        "iiwa_joint_1", "iiwa_joint_2", "iiwa_joint_3",
        "iiwa_joint_4", "iiwa_joint_5", "iiwa_joint_6", "iiwa_joint_7"
    ]
    for name, val in default_positions.items():
        try:
            joint = plant.GetJointByName(name, iiwa_instance)
            # Each val is a single-element list
            joint.set_angle(diagram.GetMutableSubsystemContext(plant, simulator.get_mutable_context()), float(val[0]))
        except Exception as e:
            # fallback: try by searching for joint globally and setting in plant context
            try:
                joint = plant.GetJointByName(name)
                joint.set_angle(diagram.GetMutableSubsystemContext(plant, simulator.get_mutable_context()), float(val[0]))
            except Exception:
                print(f"Warning: could not set default position for joint {name}: {e}")

    # Place disks on middle peg in the world by setting body poses in the plant context.
    # We'll assume peg coordinates:
    peg_positions = {
        'left': (-0.25, 0.35),
        'mid': (0.0, 0.35),
        'right': (0.25, 0.35)
    }
    # Disk starting heights: bottom disk z ~ 0.05, next at +0.015, etc.
    base_z = 0.05
    disk_thickness = 0.015
    for i, path in enumerate(disk_paths):
        model_name = f"disk_{i}"
        instance = plant.GetModelInstanceByName(model_name)
        # Compute z for disk i (largest at bottom -> disk 0 bottom)
        z = base_z + i * disk_thickness
        # Place all disks on middle peg
        px, py = peg_positions['mid']
        X_WB = RigidTransform([px, py, z])
        # Set the world pose for the single body in the disk model
        # Find body by model instance name (assumes single body)
        bodies = plant.GetBodyIndices(instance)
        if len(bodies) > 0:
            body = plant.get_body(bodies[0])
            plant.SetFreeBodyPose(diagram.GetMutableSubsystemContext(plant, simulator.get_mutable_context()), body, X_WB)

    # Now we plan robot joint trajectory for the Hanoi moves
    # Identify iiwa end-effector frame (we used iiwa_link_7)
    iiwa_frame = plant.GetFrameByName("iiwa_link_7", plant.GetModelInstanceByName("iiwa"))
    # Extract initial joint positions q0 from the current plant context
    q0 = plant.GetPositions(diagram.GetMutableSubsystemContext(plant, simulator.get_mutable_context()),
                             plant.GetJointIndices(plant.GetModelInstanceByName("iiwa")))
    # The above call is a bit fragile depending on API; as fallback get joint positions per-joint
    try:
        q0 = np.array([plant.GetJointByName(n).get_angle(diagram.GetMutableSubsystemContext(plant, simulator.get_mutable_context())) for n in joint_names])
    except Exception:
        # fallback zeros
        q0 = np.zeros(7)

    # plan the trajectory
    print("Planning joint trajectory (IK may take a few seconds)...")
    traj = plan_hanoi_joint_trajectory(plant, diagram.GetMutableSubsystemContext(plant, simulator.get_mutable_context()), iiwa_frame, q0, peg_positions)

    # Create a TrajectorySource to feed joint position references to controller/noise-free plant
    traj_source = TrajectorySource(traj)
    builder = DiagramBuilder()  # Rebuild minimal diagram: plant already built earlier; for simplicity, we just simulate diagram rather than wiring controllers.
    # NOTE: Wiring a full controller is complex and version-dependent. As a pragmatic approach we'll drive plant positions open-loop by stepping simulation
    # while directly setting joint positions from the trajectory at each simulation step (teleporting joints).
    # This is simpler to implement and fine for visualization/demo purposes (not a dynamic torque-controlled sim).

    # Run the simulation loop and at each timestep set positions to the desired trajectory evaluation.
    sim = Simulator(diagram)
    sim.set_target_realtime_rate(1.0)
    context = sim.get_mutable_context()
    t_final = traj.end_time()
    dt = 0.02
    t = 0.0
    print("Starting simulation. MeshCat viewer should open in your browser (or at the MeshCat URL printed).")
    # If MeshCatVisualizer provided a meshcat instance we can print URL (older/newer APIs differ)
    try:
        # Try to access meshcat websocket URL
        mc_url = meshcat.web_url
        print("MeshCat URL:", mc_url)
    except Exception:
        pass

    while t <= t_final + 1e-6:
        q_des = traj.value(t).flatten()
        # Set iiwa joint positions directly in plant context (teleport)
        for j, name in enumerate(joint_names):
            try:
                joint = plant.GetJointByName(name, plant.GetModelInstanceByName("iiwa"))
            except Exception:
                joint = plant.GetJointByName(name)
            # set_angle in context:
            try:
                joint.set_angle(diagram.GetMutableSubsystemContext(plant, sim.get_mutable_context()), float(q_des[j]))
            except Exception:
                # some APIs: SetPositions
                try:
                    idx = plant.GetJointIndices(plant.GetModelInstanceByName("iiwa"))[j]
                    # Directly set positions vector
                except Exception:
                    pass
        # Advance a little
        sim.AdvanceTo(min(t + dt, t_final))
        t += dt

    print("Simulation finished.")
    # done

if __name__ == "__main__":
    main()
