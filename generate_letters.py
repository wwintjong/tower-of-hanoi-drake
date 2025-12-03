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

from manipulation import running_as_notebook
from manipulation.letter_generation import create_sdf_asset_from_letter
from manipulation.station import LoadScenario, MakeHardwareStation

if running_as_notebook:
    mpld3.enable_notebook()

initials = 'O'

output_dir = Path("assets/")
for letter in initials:
    create_sdf_asset_from_letter(
        text=letter,
        font_name="DejaVu Sans",
        letter_height_meters=0.2,
        extrusion_depth_meters=0.04,
        output_dir=output_dir / f"{letter}_model",
        use_bbox_collision_geometry=True,
        mass=0.1,
    )
letter_sdfs = [
    f"{Path.cwd()}/assets/{letter}_model/{letter}.sdf" for letter in initials
]