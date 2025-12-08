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

from manipulation.station import MakeHardwareStation, Scenario
from pydrake.common.yaml import yaml_load_typed

import helper_func as hf

def run():
    meshcat = StartMeshcat()
    builder = DiagramBuilder()

    yaml_file = "load_scene.yaml"
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"File doesn't exist")
    data = open(yaml_file, "r").read()
    
    scenario = yaml_load_typed(schema=Scenario, data=data)

    station = MakeHardwareStation(scenario, meshcat=meshcat)
    builder.AddSystem(station)
    plant = station.GetSubsystemByName("plant")
    return True

if __name__ == "__main__":
    run()