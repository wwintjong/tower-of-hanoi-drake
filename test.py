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
    # 1. Start Meshcat
    meshcat = StartMeshcat()
    
    # 2. Build the Diagram
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
    parser = Parser(plant)

    # 3. Load the YAML Directives
    # We get the absolute path to ensure Drake finds the file easily
    current_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(current_dir, "assets")
    parser.package_map().Add("tower_of_hanoi_drake", current_dir)  # Add the root directory
    
    yaml_path = os.path.join(current_dir, "scene.yaml")
    directives = LoadModelDirectives(yaml_path)
    ProcessModelDirectives(directives, plant, parser)
    
    # 4. Finalize the plant
    plant.Finalize()
    
    # 5. Add Visualization
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    
    # 6. Build and Run
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    
    print(f"Loading scene from: {yaml_path}")
    print(f"Open your browser to: {meshcat.web_url()}")
    
    # Run indefinitely so you can look at it
    simulator.AdvanceTo(np.inf)

if __name__ == "__main__":
    run_yaml_scence()