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
from pydrake.math import RigidTransform, RotationMatrix

################################################################################
#                               HELPER FUNCTIONS
################################################################################

def design_grasp_pose(X_WO: RigidTransform) -> tuple[RigidTransform, RigidTransform]:
    """Return (X_OG, X_WG) where X_OG is gripper-in-object-frame and X_WG is world-frame."""
    p_OG = np.array([0.0, 0.17, 0.0])
    R_OG = np.array([
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    X_OG = RigidTransform(RotationMatrix(R_OG), p_OG)
    X_WG = X_WO.multiply(X_OG)
    return X_OG, X_WG


def design_goal_poses(
    X_WO1: RigidTransform, X_WO2: RigidTransform, X_OG: RigidTransform
) -> tuple[RigidTransform, RigidTransform]:
    """Goal gripper poses for objects 1 and 2."""
    p_WO1_goal = X_WO2.translation() + np.array([-0.4, 0.0, 0.0])
    R_WO1_goal = X_WO1.rotation()
    X_WO1_goal = RigidTransform(R_WO1_goal, p_WO1_goal)

    p_WO2_goal = X_WO2.translation() + np.array([0.2, 0.0, 0.0])
    R_WO2_goal = X_WO2.rotation()
    X_WO2_goal = RigidTransform(RotationMatrix(R_WO2_goal.matrix()), p_WO2_goal)

    X_WG1goal = X_WO1_goal.multiply(X_OG)
    X_WG2goal = X_WO2_goal.multiply(X_OG)
    return X_WG1goal, X_WG2goal


def approach_pose(X_WG: RigidTransform) -> RigidTransform:
    """Move 10cm along gripper +Z."""
    p = X_WG.translation()
    R = X_WG.rotation()
    z = R.col(2)
    return RigidTransform(R, p + 0.1 * z)


def make_trajectory(
    X_Gs: list[RigidTransform], finger_values: np.ndarray, sample_times: list[float]
):
    """Return (V_G trajectory, WSG trajectory)."""
    pose_traj = PiecewisePose.MakeLinear(sample_times, X_Gs)
    vel_traj = pose_traj.MakeDerivative()

    if finger_values.ndim == 1:
        finger_values = finger_values.reshape(1, -1)

    wsg_traj = PiecewisePolynomial.FirstOrderHold(sample_times, finger_values)
    return vel_traj, wsg_traj


def get_initial_pose(
    plant: MultibodyPlant,
    body_name: str,
    model_instance: ModelInstanceIndex,
    plant_context,
) -> RigidTransform:
    """Return world->COM pose."""
    body = plant.GetBodyByName(body_name, model_instance)
    X_WS = plant.EvalBodyPoseInWorld(plant_context, body)
    com = body.default_spatial_inertia().get_com()
    X_SO = RigidTransform(com)
    return X_WS.multiply(X_SO)


################################################################################
#                           PSEUDO INVERSE CONTROLLER
################################################################################

class PseudoInverseController(LeafSystem):
    def __init__(self, plant: MultibodyPlant):
        super().__init__()
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()

        self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)
        self.q_port = self.DeclareVectorInputPort("iiwa.position", 7)

        self.DeclareVectorOutputPort("iiwa.velocity", 7, self.CalcOutput)

        first_joint = plant.GetJointByName("iiwa_joint_1")
        self.iiwa_start = first_joint.velocity_start()

    def CalcOutput(self, context, output: BasicVector):
        V_WG_desired = self.V_G_port.Eval(context)
        q = self.q_port.Eval(context)

        self._plant.SetPositions(self._plant_context, self._iiwa, q)

        J = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,
            JacobianWrtVariable.kV,
            self._G,
            np.zeros(3),
            self._W,
            self._W
        )
        J_iiwa = J[:, self.iiwa_start:self.iiwa_start+7]

        v = np.linalg.pinv(J_iiwa) @ V_WG_desired
        output.SetFromVector(v)


################################################################################
#                           MAIN HANOI SIMULATION
################################################################################

def run_hanoi_simulation():
    meshcat = StartMeshcat()

    # Load scene YAML
    from pydrake.all import LoadScenario
    scenario = LoadScenario("load_scene.yaml")

    builder = DiagramBuilder()
    station = MakeHardwareStation(scenario, meshcat=meshcat)
    builder.AddSystem(station)
    plant = station.GetSubsystemByName("plant")

    # Get object initial poses
    temp_context = station.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(temp_context)

    X_WGinitial = plant.EvalBodyPoseInWorld(
        plant_context, plant.GetBodyByName("body")
    )

    model_instance_initial1 = plant.GetModelInstanceByName("first_initial")
    model_instance_initial2 = plant.GetModelInstanceByName("last_initial")

    initials = ["disk_2", "disk_3"]  # EDIT to match your objects
    X_WO1initial = get_initial_pose(
        plant, f"{initials[0]}_body_link", model_instance_initial1, plant_context
    )
    X_WO2initial = get_initial_pose(
        plant, f"{initials[1]}_body_link", model_instance_initial2, plant_context
    )

    # Build trajectory keyframes
    X_OG, X_WG2pick = design_grasp_pose(X_WO2initial)
    _, X_WG1pick = design_grasp_pose(X_WO1initial)

    X_WG1prepick = approach_pose(X_WG1pick)
    X_WG2prepick = approach_pose(X_WG2pick)

    X_WG1goal, X_WG2goal = design_goal_poses(X_WO1initial, X_WO2initial, X_OG)

    X_WG1pregoal = approach_pose(X_WG1goal)
    X_WG2pregoal = approach_pose(X_WG2goal)

    opened = 0.107
    closed = 0.0

    keyframes = [
        (X_WGinitial, opened),
        (X_WG2prepick, opened),
        (X_WG2pick, opened),
        (X_WG2pick, closed),
        (X_WGinitial, closed),
        (X_WG2pregoal, closed),
        (X_WG2goal, closed),
        (X_WG2goal, closed),
        (X_WG2goal, opened),
        (X_WGinitial, opened),
        (X_WG1prepick, opened),
        (X_WG1pick, opened),
        (X_WG1pick, closed),
        (X_WGinitial, closed),
        (X_WG1pregoal, closed),
        (X_WG1goal, closed),
        (X_WG1goal, opened),
        (X_WGinitial, opened),
    ]

    gripper_poses = [p for p,_ in keyframes]
    finger_states = np.array([f for _,f in keyframes]).reshape(1,-1)
    sample_times = [2*i for i in range(len(keyframes))]

    traj_V_G, traj_wsg_command = make_trajectory(
        gripper_poses, finger_states, sample_times
    )

    # Build the controller diagram
    V_G_source = builder.AddSystem(TrajectorySource(traj_V_G))
    controller  = builder.AddSystem(PseudoInverseController(plant))
    integrator  = builder.AddSystem(Integrator(7))
    wsg_source  = builder.AddSystem(TrajectorySource(traj_wsg_command))

    builder.Connect(V_G_source.get_output_port(), controller.GetInputPort("V_WG"))
    builder.Connect(controller.get_output_port(), integrator.get_input_port())
    builder.Connect(integrator.get_output_port(), station.GetInputPort("iiwa.position"))
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        controller.GetInputPort("iiwa.position")
    )
    builder.Connect(wsg_source.get_output_port(), station.GetInputPort("wsg.position"))

    # Add debug axes
    scenegraph = station.GetSubsystemByName("scene_graph")
    AddFrameTriadIllustration(scenegraph, plant.GetBodyByName("body"), length=0.1)

    # Build full diagram
    diagram = builder.Build()

    # Run simulation
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    station_context = station.GetMyContextFromRoot(context)
    plant_context_real = plant.GetMyContextFromRoot(context)

    # set integrator to current joint angles
    q0 = plant.GetPositions(plant_context_real, plant.GetModelInstanceByName("iiwa"))
    integrator.set_integral_value(integrator.GetMyContextFromRoot(context), q0)

    print(f"Running for {traj_V_G.end_time()} seconds")
    meshcat.StartRecording()

    simulator.AdvanceTo(traj_V_G.end_time())

    meshcat.StopRecording()
    meshcat.PublishRecording()


if __name__ == "__main__":
    run_hanoi_simulation()
