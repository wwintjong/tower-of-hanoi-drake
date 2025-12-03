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
)

def run_yaml_scence():
    meshcat = StartMeshcat()
    
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
    parser = Parser(plant)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(current_dir, "assets")
    parser.package_map().Add("tower_of_hanoi_drake", current_dir) 
    
    yaml_path = os.path.join(current_dir, "load_scene.yaml")
    directives = LoadModelDirectives(yaml_path)
    ProcessModelDirectives(directives, plant, parser)
    
    plant.Finalize()
    
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    
    print(f"Loading scene from: {yaml_path}")
    print(f"Open your browser to: {meshcat.web_url()}")
    
    simulator.AdvanceTo(np.inf)

if __name__ == "__main__":
    run_yaml_scence()