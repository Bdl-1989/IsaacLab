#

"""This script spawns conveyor into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p /mnt/sdb2/Omniverse/pick_place_env/my_project/spawn_conveyor.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse
from isaaclab.app import AppLauncher

# For parsing and launching app do not modify unless you need custom arguments
parser = argparse.ArgumentParser(description="Spawn a conveyor belt")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

#Imports
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
import omni.usd
from pxr import UsdPhysics, PhysxSchema, Usd, Gf
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# from isaaclab.assets import RigidObject, RigidObjectCfg
import time
import torch
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
    DeformableObject,
    DeformableObjectCfg
)

import random
from isaaclab.sim import SimulationContext

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR,ISAAC_NUCLEUS_DIR


pancake_cfg =  RigidObjectCfg(
        spawn=sim_utils.CylinderCfg(
                radius=0.04,
                height=0.005,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )



pancake_cfg_dict = {}

CONVEYOR_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(size=[8.0, 1.5, 3],
                              collision_props=sim_utils.CollisionPropertiesCfg(),
                              mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                              rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                  kinematic_enabled=True,
                              ),
                              ),
)

# CONVEYOR_CFG = RigidObjectCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Conveyors/ConveyorBelt_A09.usd",
#         # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
#         # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
#         collision_props=sim_utils.CollisionPropertiesCfg(),
#     ),
# )

def spawn_object(i):
    pancake_cfg_dict = {}
    #This is for spawning objects onto the conveyor.
    potential_y = [-0.40, -0.30, -0.18, -0.06, 0.06, 0.18, 0.30, 0.40] # idea make a map of potential y's and spawn them randomly

    for index, value in enumerate(potential_y):
        spawn_location = [-3.5, value, 1.8]
        pancake = pancake_cfg.copy()
        pancake.prim_path = "{ENV_REGEX_NS}/pancake_" + str(i+1) + "_" + str(index+1)

        # pancake.init_state = RigidObjectCfg.InitialStateCfg(pos=spawn_location) 

        key = f'pancake_{i+1}_{index+1}'
        pancake_cfg_dict[key] = pancake.replace(prim_path="/World/envs/env_.*/"+key)

  
    return pancake_cfg_dict

combined_dic = {}

# Loop from 0 to 100 (inclusive)
for i in range(20):
    # Spawn the object and get the dictionary
    current_dic = spawn_object(i)
    # Combine it with the existing dictionary
    combined_dic.update(current_dic)

@configclass
class PancakeSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""


    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
 

    conveyor: RigidObjectCfg = CONVEYOR_CFG.replace(prim_path="{ENV_REGEX_NS}/conveyor")

    pancake: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/pancake",spawn=sim_utils.CylinderCfg(
                radius=0.04,
                height=0.005,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
            ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 3.0)),)



    pancake_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(rigid_objects=combined_dic)


def move_conveyor():
    stage = omni.usd.get_context().get_stage()
    conveyor_prim = stage.GetPrimAtPath("/World/envs/env_0/conveyor")
    if conveyor_prim.IsValid():
        velocity_attr = conveyor_prim.GetAttribute("physics:velocity")
        velocity_attr.Set((0.075,0,0)) #meters per second
        print("Velocity set!")
    else:
        print("Conveyor or conveyor velocity not found!")



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""

    move_conveyor()
    # conveyor_status = scene['conveyor'].data.default_root_state.clone()
    # conveyor_status[:,:3] = conveyor_status[:,:3] - scene.env_origins
    # conveyor_status[:,7] = 10
    # scene['conveyor'].write_root_state_to_sim(conveyor_status)



    pancake_objects = scene['pancake_collection'].cfg.rigid_objects

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    i = 0
    batch = 8 
    # Simulate physics
    while simulation_app.is_running():
        # reset

        if i >len(pancake_objects):
            # reset counters
            sim_time = 0.0
            count = 0
            i = 0
            # reset root state
            # pancakes_status=initial_status.clone()
            # reset buffers 

            print("----------------------------------------")
            print("[INFO]: Resetting object state...")
            scene['pancake_collection'].reset()
            scene['conveyor'].reset()
            
        # apply sim data 
        scene['pancake_collection'].reset()
        scene['conveyor'].reset()

        if count % 200 ==0:
            pancakes_status = scene['pancake_collection'].data.object_state_w.clone() 
            potential_y = [-0.40, -0.30, -0.18, -0.06, 0.06, 0.18, 0.30, 0.40] # idea make a map of potential y's and spawn them randomly
            indices = []
            scene['pancake_collection'].reset()
            for index, key in enumerate(pancake_objects.keys()):

                if index >= i and index < i + batch:
                    pancakes_status[:,index,0] = -3.5
                    pancakes_status[:,index,1] = potential_y[index - i]
                    pancakes_status[:,index,2] = 1.8
                    pancakes_status[:,index,3] = 1
                    pancakes_status[:,index,4] = 0
                    pancakes_status[:,index,5] = 0
                    pancakes_status[:,index,6] = 0
                    pancakes_status[:,index,7] = 0
                    pancakes_status[:,index,8] = 0
                    pancakes_status[:,index,9] = 0
                    pancakes_status[:,index,10] = 0
                    pancakes_status[:,index,11] = 0
                    pancakes_status[:,index,12] = 0
                    indices.append(index)
                    pancakes_status[:,index,:3] = pancakes_status[:,index,:3] + scene.env_origins
 
            i += batch
            if len(indices) > 0:
                
                print("----------------------------------------")
                print("[INFO]: Spawn pancakes...")
                scene['pancake_collection'].write_object_state_to_sim(pancakes_status, scene['pancake_collection']._ALL_ENV_INDICES, scene['pancake_collection']._ALL_OBJ_INDICES )
                print('write')
        # scene.write_data_to_sim()    
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)
 




def main():
    """Main function."""
 

    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = PancakeSceneCfg(num_envs=2, env_spacing=10.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


 

if __name__ == "__main__":
    main()
    simulation_app.close()