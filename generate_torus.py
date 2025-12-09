import numpy as np
import os

def generate_torus(filename, disk_name, radius, thickness, color):
    """
    Because sdformat does not have the primitive for torus object, and loading the mesh does not work as expected, a workaround for this
    is to take a primitive object and create several duplicates at different angles to forma torus-like object.
    """
    sdf = f"""
    <?xml version='1.0'?>
    <sdf version='1.7'>
        <model name='{disk_name}'>
            <link name='torus_link'>
            <inertial>
                <mass>0.2</mass>
                <inertia>
                <ixx>0.001</ixx> <iyy>0.001</iyy> <izz>0.002</izz>
                <ixy>0</ixy> <ixz>0</ixz> <iyz>0</iyz>
                </inertia>
            </inertial>

            <visual name="visual">
                <geometry>
                <mesh>
                    <uri>objects/{disk_name}.obj</uri>
                </mesh>
                </geometry>
                <material>
                <diffuse>0.2 0.2 0.8 1.0</diffuse>
                </material>
            </visual>
    """
    n_segments = 64
    for i in range(n_segments):
        angle = 2 * np.pi * i / n_segments
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        yaw = angle + (np.pi / 2)
        seg_len = (2 * np.pi * radius) / n_segments * 1.1 
        
        pose = f"{x:.4f} {y:.4f} 0 1.5708 0 {yaw:.4f}"

        sdf += f"""
        <collision name='col_{i}'>
            <pose>{pose}</pose>
            <geometry>
            <capsule> <radius>{thickness}</radius> <length>{seg_len}</length> </capsule>
            </geometry>
        </collision>
        """

    sdf += """
            </link>
        </model>
    </sdf>
    """
    
    path = os.path.join("assets", filename)
    with open(path, "w") as f:
        f.write(sdf)
    print(f"Created {filename} in {path}")

if __name__ == "__main__":
    if not os.path.exists("assets"):
        os.makedirs("assets")

    generate_torus("disk_1.sdf", "disk_1", radius=0.1, thickness=0.05, color="0 0 1 1")
    generate_torus("disk_2.sdf", "disk_2", radius=0.125, thickness=0.05, color="0 0 1 1")
    generate_torus("disk_3.sdf", "disk_3", radius=0.15, thickness=0.05, color="0 0 1 1")
    generate_torus("disk_4.sdf", "disk_4", radius=0.175, thickness=0.05, color="0 0 1 1")