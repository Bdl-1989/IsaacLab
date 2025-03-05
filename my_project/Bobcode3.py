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


deltaT = 0.01
infeedVelocity = 0.0467
outfeedVelocity = 0.1333
infeed_y_offset = -0.5
outfeed_y_offset = 0.1
pancakes_per_container = 6
infeed_gen_dist = 0.095 
outfeed_gen_dist = 0.230
potential_y = [-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4] # idea make a map of potential y's and spawn them randomly

device = "cuda:0"


pancake_cfg =  RigidObjectCfg(
        spawn=sim_utils.CylinderCfg(
                radius=0.045,
                height=0.010,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

container_cfg =  RigidObjectCfg(
        spawn=sim_utils.CuboidCfg(
                size=(0.195,0.125 , 0.005),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=6.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0.2),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )


pancake_cfg_dict = {}

INFEED_CONVEYOR_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(size=[8.0, 1, 0.9],
                              collision_props=sim_utils.CollisionPropertiesCfg(),
                              mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                              rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                  kinematic_enabled=True,
                              ),
                              ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0, infeed_y_offset, 0.45)),
)
OUTFEED_CONVEYOR_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(size=[8.0, 0.2, 0.8],
                              collision_props=sim_utils.CollisionPropertiesCfg(),
                              mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                              rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                  kinematic_enabled=True,
                              ),
                              ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0, outfeed_y_offset, 0.4)),
)
 

def spawn_object(i):
    pancake_cfg_dict = {}
    #This is for spawning objects onto the conveyor.
 
    for index in range(i):
        spawn_location = [-3.5, infeed_y_offset - 2, 0]
        pancake = pancake_cfg.copy()
        # pancake.prim_path = "{ENV_REGEX_NS}/pancake_" + str(i+1) + "_" + str(index+1)

        pancake.init_state = RigidObjectCfg.InitialStateCfg(pos=spawn_location) 

        key = f'pancake_{index+1}'
        pancake_cfg_dict[key] = pancake.replace(prim_path="/World/envs/env_.*/"+key)

    return pancake_cfg_dict
import math

def spawn_container(pancake_num):
    container_cfg_dict={}
    num = math.ceil(pancake_num / pancakes_per_container)
    for i in range(num):
        spawn_location = [-3.5, outfeed_y_offset + 1, 0]
        container = container_cfg.copy()
        container.init_state = RigidObjectCfg.InitialStateCfg(pos=spawn_location) 

        key = f'container_{i+1}'
        container_cfg_dict[key] = container.replace(prim_path="/World/envs/env_.*/"+key)

    return container_cfg_dict





combined_dic = {}
container_dic = {}
 
    # Spawn the object and get the dictionary
combined_dic = spawn_object(20 * len(potential_y) )
    # Combine it with the existing dictionary 
# spawn container based on the pancakes

total_pancakes = len(combined_dic.keys())

container_dic = spawn_container(total_pancakes)

@configclass
class PancakeSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""


    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
 

    infeed_conveyor: RigidObjectCfg = INFEED_CONVEYOR_CFG.replace(prim_path="{ENV_REGEX_NS}/infeed_conveyor")
    outfeed_conveyor: RigidObjectCfg = OUTFEED_CONVEYOR_CFG.replace(prim_path="{ENV_REGEX_NS}/outfeed_conveyor")


    pancake_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(rigid_objects=combined_dic)
    container_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(rigid_objects=container_dic)


def move_conveyor():
    stage = omni.usd.get_context().get_stage()
    infeed_conveyor_prim = stage.GetPrimAtPath("/World/envs/env_0/infeed_conveyor")
    if infeed_conveyor_prim.IsValid():
        velocity_attr = infeed_conveyor_prim.GetAttribute("physics:velocity")
        velocity_attr.Set((infeedVelocity,0,0)) #meters per second
        print("Velocity set!")
    else:
        print("Infeed conveyor or infeed conveyor velocity not found!")
    outfeed_conveyor_prim = stage.GetPrimAtPath("/World/envs/env_0/outfeed_conveyor")
    if outfeed_conveyor_prim.IsValid():
        velocity_attr = outfeed_conveyor_prim.GetAttribute("physics:velocity")
        velocity_attr.Set((outfeedVelocity,0,0)) #meters per second
        print("Velocity set!")
    else:
        print("Outfeed conveyor or outfeed conveyor velocity not found!")

import numpy as np

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
    container_index = 0
    batch = len(potential_y) 
    # Simulate physics
    while simulation_app.is_running():
        # reset

        if i >len(pancake_objects):
            # reset counters
            sim_time = 0.0
            count = 0
            i = 0
            container_index = 0 
            print("----------------------------------------")
            print("[INFO]: Resetting object state...")
            scene['pancake_collection'].reset()
            scene['infeed_conveyor'].reset()
            
 
        if (outfeedVelocity * deltaT * count >= outfeed_gen_dist * container_index) and (outfeedVelocity * deltaT * (count-1) < outfeed_gen_dist * container_index):
            containers_status = scene['container_collection'].data.object_state_w.clone() 
            deltaX_container = outfeedVelocity * deltaT * count - outfeed_gen_dist * container_index  # need to suppliment the distance during increasing the time step
            print(f"[INFO]: Spawn container when {deltaT * count}..and {deltaX_container =}.")
 
            container_initial_status = [
                    -3.5 + deltaX_container, outfeed_y_offset, 0.8 + 0.003, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ]
            container_initial_status_tensor = torch.zeros(len(scene['container_collection']._ALL_ENV_INDICES), 13, device=device)
            container_initial_status_tensor += torch.tensor(container_initial_status, device=device)
            container_initial_status_tensor[:,:3] += scene.env_origins

            container_index +=1
            scene.reset()
            scene['container_collection'].write_object_com_state_to_sim(container_initial_status_tensor.unsqueeze(1),None,scene['container_collection']._ALL_OBJ_INDICES[container_index].unsqueeze(0))
            
            print("----------------------------------------")






        
        if (infeedVelocity * deltaT * count >= infeed_gen_dist * i / batch) and (infeedVelocity * deltaT * (count -1 ) > infeed_gen_dist * i /batch ):

 
            deltaX = infeedVelocity * deltaT * count - infeed_gen_dist * i / batch  # need to suppliment the distance during increasing the time step
            print(f"[INFO]: Spawn pancake when {deltaT * count}..and {deltaX =}.")
 
            pancakes_initial_status = torch.zeros(batch, 13, device=device)
            pancakes_initial_status[:, 0] = -3.5 + deltaX  # 广播机制
            pancakes_initial_status[:, 1] = torch.tensor(potential_y) + infeed_y_offset  # 直接加法
            pancakes_initial_status[:, 2] = 0.9 + 0.005  # 广播机制
            pancakes_initial_status[:, 3] = 1.0  # 广播机制

            # 利用广播机制将 scene.env_origins 扩展到 [2, batch, 3]
            expanded_env_origins = scene.env_origins[:, None, :]  # 形状变为 [2, 1, 3]
            # 创建一个 [2, batch, 13] 的张量，前 3 列是 expanded_env_origins，其余为 0
            final_env_origins = torch.zeros(len(scene['container_collection']._ALL_ENV_INDICES), batch, 13, device=device)
            final_env_origins[:, :, :3] = expanded_env_origins  # 广播机制会自动扩展
            pancakes_initial_status_tensor =  final_env_origins + pancakes_initial_status

            object_start_index = i
            object_end_index = i + batch if i + batch <= total_pancakes - 1 else total_pancakes - 1
 
            i += batch
            scene.reset()
            scene['pancake_collection'].write_object_com_state_to_sim(pancakes_initial_status_tensor,None,scene['pancake_collection']._ALL_OBJ_INDICES[object_start_index:object_end_index] ) 
            print("----------------------------------------")


 


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
    # sim_cfg = sim_utils.SimulationCfg(dt=0.005,device=args_cli.device)
    sim_cfg = sim_utils.SimulationCfg(dt=deltaT,device=device)

    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 8.0], [0.0, 0.0, 3.0])
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