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
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Conveyors/ConveyorBelt_A09.usd",
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
)



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
    pancake_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(rigid_objects={})


def spawn_object(i):
    pancake_cfg_dict = {}
    #This is for spawning objects onto the conveyor.
    potential_y = [-0.40, -0.30, -0.18, -0.06, 0.06, 0.18, 0.30, 0.40] # idea make a map of potential y's and spawn them randomly

    for index, value in enumerate(potential_y):
        spawn_location = [-3.5, value, 1.8]
        pancake = pancake_cfg.copy()
        pancake.prim_path = "{ENV_REGEX_NS}/pancake_" + str(i+1) + "_" + str(index+1)

        pancake.init_state = RigidObjectCfg.InitialStateCfg(pos=spawn_location) 

        key = f'pancake_{i+1}_{index+1}'
        pancake_cfg_dict[key] = pancake.replace(prim_path="/World/envs/env_.*/"+key)

  
    return pancake_cfg_dict


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
 
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    i = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            i = 0
            # reset root state
  
            # reset buffers 

            print("----------------------------------------")
            print("[INFO]: Resetting object state...")
        # apply sim data 


        if count % 4 ==0:
            dict = spawn_object(i)
            for key, value in dict.items():
                scene['pancake_collection'].cfg.rigid_objects[key] = value 
            i += 1
            print("----------------------------------------")
            print("[INFO]: Spawn pancakes...")

        # scene.write_data_to_sim()    
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        # scene.update(sim_dt)
 




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