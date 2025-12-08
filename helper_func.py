from pathlib import Path

import mpld3
import numpy as np
from pydrake.all import (
    AddFrameTriadIllustration,
    BasicVector,
    Context,
    DiagramBuilder,
    Integrator,
    JacobianWrtVariable,
    LeafSystem,
    ModelInstanceIndex,
    MultibodyPlant,
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    Trajectory,
    TrajectorySource,
)

def design_grasp_pose(
        X_WO: RigidTransform, 
        p_OG: np.ndarray,
        R_OG: np.ndarray
) -> tuple[RigidTransform, RigidTransform]:
    X_OG = RigidTransform(RotationMatrix(R_OG), p_OG)
    X_WG = X_WO.multiply(X_OG)
    return X_OG, X_WG

def design_goal_pose(
        X_OG: RigidTransform,
        p_WO_goal: np.ndarray,
        R_WO_goal: np.ndarray
) -> tuple[RigidTransform]:
    X_WO_goal = RigidTransform(R_WO_goal, p_WO_goal)
    X_WGgoal = X_WO_goal.multiply(X_OG)
    return X_WGgoal

def design_approach_pose(
        X_WG: RigidTransform, 
        approach_distance: float = 0.1
) -> RigidTransform:
    p_WG = X_WG.translation()
    R_WG = X_WG.rotation()
    gripper_z_axis = np.array([0, 0, 1])
    p_WGApproach = p_WG + (approach_distance * gripper_z_axis)
    X_WGApproach = RigidTransform(R_WG, p_WGApproach)
    return X_WGApproach

def make_trajectory(
    X_Gs: list[RigidTransform], 
    finger_values: np.ndarray, 
    sample_times: list[float]
) -> tuple[Trajectory, PiecewisePolynomial]:
    robot_velocity_trajectory = None
    traj_wsg_command = None

    pose_trajectory = PiecewisePose.MakeLinear(sample_times, X_Gs)
    robot_velocity_trajectory = pose_trajectory.MakeDerivative()
    
    if finger_values.ndim == 1:
        finger_values = finger_values.reshape(1, -1)
    
    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(sample_times, finger_values)

    return robot_velocity_trajectory, traj_wsg_command

def get_initial_pose(
    plant: MultibodyPlant,
    body_name: str,
    model_instance: ModelInstanceIndex,
    plant_context: Context,
) -> RigidTransform:
    body = plant.GetBodyByName(body_name, model_instance)
    X_WS = plant.EvalBodyPoseInWorld(plant_context, body)
    X_SO = RigidTransform(body.default_spatial_inertia().get_com())
    return X_WS @ X_SO

class PseudoInverseController(LeafSystem):
    def __init__(self, plant: MultibodyPlant):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self._G = plant.GetBodyByName("body").body_frame()
        self._W = plant.world_frame()

        self.V_G_port = self.DeclareVectorInputPort("V_WG", 6)
        self.q_port = self.DeclareVectorInputPort("iiwa.position", 7)
        self.DeclareVectorOutputPort("iiwa.velocity", 7, self.CalcOutput)
        self.iiwa_start = plant.GetJointByName("iiwa_joint_1").velocity_start()
        self.iiwa_end = plant.GetJointByName("iiwa_joint_7").velocity_start()

    def CalcOutput(self, context: Context, output: BasicVector):
        """
        fill in our code below.
        """
        V_WG_desired = self.V_G_port.Eval(context)
        q_current = self.q_port.Eval(context)

        self._plant.SetPositions(self._plant_context, self._iiwa, q_current)

        J_WG = self._plant.CalcJacobianSpatialVelocity(
            self._plant_context,                     # context
            JacobianWrtVariable.kV,                  # with_respect_to (velocities)
            self._G,                                 # frame_B (gripper frame)
            np.zeros(3),
            self._W,                                 # frame_A (world frame)
            self._W                                  # frame_E (express result in world frame)
        )
        J_WG_iiwa = J_WG[:, self.iiwa_start:self.iiwa_start + 7]

        J_pinv = np.linalg.pinv(J_WG_iiwa)  # 7 x 6 pseudo-inverse
        v = J_pinv @ V_WG_desired
        output.SetFromVector(v)