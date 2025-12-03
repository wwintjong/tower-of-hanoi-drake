from pathlib import Path

import mpld3
import numpy as np
from pydrake.all import (
    AddFrameTriadIllustration,
    AddMultibodyPlantSceneGraph,
    BasicVector,
    Context,
    Diagram,
    DiagramBuilder,
    Integrator,
    JacobianWrtVariable,
    LeafSystem,
    MeshcatVisualizer,
    ModelInstanceIndex,
    MultibodyPlant,
    Parser,
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
    RobotDiagram,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    Trajectory,
    TrajectorySource,
)

from manipulation import running_as_notebook
from manipulation.exercises.grader import Grader
from manipulation.exercises.pick.test_pickplace_initials import TestPickPlacePoses
from manipulation.letter_generation import create_sdf_asset_from_letter
from manipulation.station import LoadScenario, MakeHardwareStation

if running_as_notebook:
    mpld3.enable_notebook()

meshcat = StartMeshcat()