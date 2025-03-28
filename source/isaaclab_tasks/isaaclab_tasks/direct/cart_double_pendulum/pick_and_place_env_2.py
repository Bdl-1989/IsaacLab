# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab_assets.robots.cart_double_pendulum import CART_DOUBLE_PENDULUM_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg, 
)






infeedVelocity = 0.0467
outfeedVelocity = 0.1333
infeed_y_offset = -0.5
outfeed_y_offset = 0.1+0.001
pancakes_per_container = 6
infeed_gen_dist = 0.095 
outfeed_gen_dist = 0.230
potential_y = [-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4] # idea make a map of potential y's and spawn them randomly

num_containers = math.ceil(4 / (outfeed_gen_dist)) 
num_pancake_row = math.ceil(4/(infeed_gen_dist))

pick_height = 0.020 
place_height = 0.020  + 0.1

device = "cuda:1"


pancake_radius = 0.045
pancake_height = 0.010


pancake_cfg =  RigidObjectCfg(
        spawn=sim_utils.CylinderCfg(
                radius=pancake_radius,
                height=pancake_height,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled= True,
                                                             disable_gravity=True,
                                                             kinematic_enabled=False,
                                                             enable_gyroscopic_forces=False,
                                                             retain_accelerations=False,
                                                             solver_position_iteration_count=4, 
                                                             solver_velocity_iteration_count=0,
                                                             linear_damping=0.0,
                                                             angular_damping=0.0,
                                                             max_linear_velocity=1000.0,
                                                             max_angular_velocity=1000.0,
                                                             max_depenetration_velocity=1.0,
                                                             sleep_threshold=0.05,
                                                             stabilization_threshold=0.01,),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

container_cfg =  RigidObjectCfg(
        spawn=sim_utils.CuboidCfg(
                size=(0.195,0.125 , 0.005),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled= True,
                                                             disable_gravity=True, 
                                                             kinematic_enabled=False,
                                                             enable_gyroscopic_forces=False,
                                                             retain_accelerations=False,
                                                             solver_position_iteration_count=4, 
                                                             solver_velocity_iteration_count=0,
                                                             linear_damping=0.0,
                                                             angular_damping=0.0,
                                                             max_linear_velocity=1000.0,
                                                             max_angular_velocity=1000.0,
                                                             max_depenetration_velocity=1.0,
                                                             sleep_threshold=0.05,
                                                             stabilization_threshold=0.01,),
                mass_props=sim_utils.MassPropertiesCfg(mass=6.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0.2),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

# TCP_TOP_CFG = RigidObjectCfg(
#     spawn=sim_utils.ShapeCfg(radius = 0.03,
#                               collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
#                               visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), metallic=0.2),
#                               mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
#                               rigid_props=sim_utils.RigidBodyPropertiesCfg(
#                                     rigid_body_enabled= True,
#                                     disable_gravity=True, 
#                                     kinematic_enabled=False,
#                               ),
#                               ),
#     init_state=RigidObjectCfg.InitialStateCfg(pos=(-2, infeed_y_offset, 0.45)),
# ) 

INFEED_CONVEYOR_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(size=[4.0, 1, 0.9],
                              collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                              mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                              rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                  kinematic_enabled=True,
                                  linear_damping=10.0,
                              ),
                              ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-2, infeed_y_offset, 0.45)),
)
OUTFEED_CONVEYOR_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(size=[4.0, 0.2, 0.8],
                              collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                              mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                              rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                  kinematic_enabled=True,
                                  linear_damping=10.0,
                              ),
                              ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-2, outfeed_y_offset, 0.4)),
)

PLACE_WORKAREA_1= RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(size=[0.5, 0.2, 0.002],
                              collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                              mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                              visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0), metallic=0.2,opacity=1),
                              rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                  kinematic_enabled=True,
                              ),
                              ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.75, outfeed_y_offset, 0.8)),
    debug_vis = True
)
PICK_WORKAREA_1= RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(size=[0.5, 1, 0.002],
                              collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                              mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                              visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.4, 1.0), metallic=0.2,opacity=1),
                              rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                  kinematic_enabled=True,
                              ),
                              ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.75, infeed_y_offset, 0.9)),
    debug_vis = True
)
 



def spawn_object(i):
    pancake_cfg_dict = {}
    #This is for spawning objects onto the conveyor.
 
    for index in range(i):
        spawn_location = [-3.5, infeed_y_offset - 2, index* 0.01 +0.01]
        pancake = pancake_cfg.copy()
        # pancake.prim_path = "{ENV_REGEX_NS}/pancake_" + str(i+1) + "_" + str(index+1)

        pancake.init_state = RigidObjectCfg.InitialStateCfg(pos=spawn_location) 

        key = f'pancake_{index+1}'
        pancake_cfg_dict[key] = pancake.replace(prim_path="/World/envs/env_.*/"+key)

    return pancake_cfg_dict


def spawn_container(container_num):
    container_cfg_dict={}
    num = math.ceil(container_num)
    for i in range(num):
        spawn_location = [-3.5, outfeed_y_offset + 1, 0.005 * i +0.1]
        container = container_cfg.copy()
        container.init_state = RigidObjectCfg.InitialStateCfg(pos=spawn_location) 

        key = f'container_{i+1}'
        container_cfg_dict[key] = container.replace(prim_path="/World/envs/env_.*/"+key)

    return container_cfg_dict



combined_dic = {}
container_dic = {}
 
    # Spawn the object and get the dictionary
combined_dic = spawn_object(num_pancake_row* len(potential_y) )
total_pancakes = len(combined_dic.keys())
container_dic = spawn_container(num_containers)
total_containers = len(container_dic.keys())

import omni.usd

def move_conveyor(i):
    stage = omni.usd.get_context().get_stage()
    infeed_conveyor_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/infeed_conveyor")
    if infeed_conveyor_prim.IsValid():
        velocity_attr = infeed_conveyor_prim.GetAttribute("physics:velocity")
        velocity_attr.Set((infeedVelocity,0,0)) #meters per second
        print("Velocity set!")
    else:
        print("Infeed conveyor or infeed conveyor velocity not found!")
    outfeed_conveyor_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/outfeed_conveyor")
    if outfeed_conveyor_prim.IsValid():
        velocity_attr = outfeed_conveyor_prim.GetAttribute("physics:velocity")
        velocity_attr.Set((outfeedVelocity,0,0)) #meters per second
        print("Velocity set!")
    else:
        print("Outfeed conveyor or outfeed conveyor velocity not found!")
 

pick_place_time_consumption = 0.8 # s, commented by Micheal, 3/13/2025
deltaT = pick_place_time_consumption 

# value only from 0 to 1
# final target, 2 in container, 3 for stacked
# stack_matrix = torch.zeros(2,3) 
# test structure, 2 in container, 1 for stacked
stack_matrix = torch.zeros(2,1) 

from stack_transform import generate_pancake_coordinates

@configclass
class PickAndPlaceEnvCfg(DirectMARLEnvCfg):
    # env
    decimation =16
    episode_length_s = 5.0
    possible_agents = ["cart", "pendulum","pickmaster"]
    action_spaces = {"cart": 1, "pendulum": 1,"pickmaster":6}  # pick_x, pick_y,  place_x, place_y, placeorwait
    observation_spaces = {"cart": 4, "pendulum": 3,"pickmaster":64}
    # possible_agents = ["cart", "pendulum"]
    # action_spaces = {"cart": 1, "pendulum": 1}
    # observation_spaces = {"cart": 4, "pendulum": 3}
    state_space = -1

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=deltaT/decimation,device = device, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = CART_DOUBLE_PENDULUM_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    pendulum_dof_name = "pole_to_pendulum"

    infeed_conveyor_cfg: RigidObjectCfg = INFEED_CONVEYOR_CFG.replace(prim_path="/World/envs/env_.*/infeed_conveyor")
    outfeed_conveyor_cfg: RigidObjectCfg = OUTFEED_CONVEYOR_CFG.replace(prim_path="/World/envs/env_.*/outfeed_conveyor")
    place_work_area_1_cfg: RigidObjectCfg = PLACE_WORKAREA_1.replace(prim_path="/World/envs/env_.*/place_work_area_1")
    pick_work_area_1_cfg: RigidObjectCfg = PICK_WORKAREA_1.replace(prim_path="/World/envs/env_.*/pick_work_area_1")
    pancake_collection_cfg: RigidObjectCollectionCfg = RigidObjectCollectionCfg(rigid_objects=combined_dic)
    container_collection_cfg: RigidObjectCollectionCfg = RigidObjectCollectionCfg(rigid_objects=container_dic) 

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=10.0, replicate_physics=True)

    # reset
    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]
    initial_pendulum_angle_range = [-0.25, 0.25]  # the range in which the pendulum angle is sampled from on reset [rad]

    # action scales
    cart_action_scale = 100.0  # [N]
    pendulum_action_scale = 50.0  # [Nm]

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_cart_pos = 0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_pos = -1.0
    rew_scale_pole_vel = -0.01
    rew_scale_pendulum_pos = -1.0
    rew_scale_pendulum_vel = -0.01


from gym import spaces
import numpy as np


class PickAndPlaceEnv(DirectMARLEnv):
    cfg: PickAndPlaceEnvCfg

    def __init__(self, cfg: PickAndPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.count = 0
        self.i = 0
        self.container_index = 0
        self.batch = len(potential_y) 
        self.pancake_offset_count = 0
        self.container_offset_count = 0
        self.pick_workarea_1_movement = torch.zeros(self.scene.num_envs, 2)
        self.pick_workarea_1_movement[:,1] -= 1 # for holding pancake index

        
        # y
        # ^
        # |
        # |                       
        # |  +--------------------+  y = 0.2 (place_workarea_1 top)
        # |  |                    |
        # |  |  Place Workarea 1  |
        # |  |                    |
        # |  +--------------------+  y = 0   (place_workarea_1 bottom / pick_workarea_1 top)
        # |  |                    |
        # |  |  Pick Workarea 1   |
        # |  |                    |
        # |  +--------------------+  y = -1  (pick_workarea_1 bottom)
        # |
        # |-------------------------------> x
        #     x = -3               x = -2.5
        self.pick_workarea_1_boundaries = spaces.Box(
            low=np.array([-3, -1]), 
            high=np.array([-2.5, 0]), 
            dtype=np.float32
        )
 
        self.place_workarea_1_boundaries = spaces.Box(
            low=np.array([-3, 0]), 
            high=np.array([-2.5, 0.2]), 
            dtype=np.float32
        )
 

        self.pick_encoder = PointNetEncoder(feature_dim=64, device = device \
                                                            , x_min=self.pick_workarea_1_boundaries.low[0] \
                                                            , x_max=self.pick_workarea_1_boundaries.high[0] \
                                                            , y_min=self.pick_workarea_1_boundaries.low[1] \
                                                            , y_max=self.pick_workarea_1_boundaries.high[1])

        self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)
        self._pendulum_dof_idx, _ = self.robot.find_joints(self.cfg.pendulum_dof_name)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.pancakes_per_container_dict = torch.zeros(self.scene.num_envs, self.scene['container_collection'].num_objects)






    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
 
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # add rigidbody
        self.infeed_conveyor = RigidObject(self.cfg.infeed_conveyor_cfg)
        self.outfeed_conveyor = RigidObject(self.cfg.outfeed_conveyor_cfg)
        self.place_work_area_1 = RigidObject(self.cfg.place_work_area_1_cfg)
        self.pick_work_area_1 = RigidObject(self.cfg.pick_work_area_1_cfg)
        self.pancake_collection = RigidObjectCollection(self.cfg.pancake_collection_cfg)
        self.container_collection = RigidObjectCollection(self.cfg.container_collection_cfg)
        # add rigid to scene
        self.scene.rigid_objects["infeed_conveyor"] = self.infeed_conveyor
        self.scene.rigid_objects["outfeed_conveyor"] = self.outfeed_conveyor
        self.scene.rigid_objects["place_work_area_1"] = self.place_work_area_1
        self.scene.rigid_objects["pick_work_area_1"] = self.pick_work_area_1
        self.scene.rigid_object_collections["pancake_collection"] = self.pancake_collection
        self.scene.rigid_object_collections["container_collection"] = self.container_collection

    def _generator_objects(self):
        pancake_delta_count = self.count - self.pancake_offset_count
        if (infeedVelocity * deltaT * pancake_delta_count >= infeed_gen_dist * self.i / self.batch) and (infeedVelocity * deltaT * (pancake_delta_count -1 ) > infeed_gen_dist * self.i /self.batch ):
            deltaX = infeedVelocity * deltaT * pancake_delta_count - infeed_gen_dist * self.i / self.batch  # need to suppliment the distance during increasing the time step
            print(f"[INFO]: Spawn pancake when {deltaT * pancake_delta_count}..and {deltaX =}.")
 
            if self.i >= total_pancakes:

                self.i = 0
                self.pancake_offset_count = self.count 
                print("----------------------------------------")
                print("[INFO]: Resetting pancakes state...") 
                # scene['infeed_conveyor'].reset()
            pancakes_initial_status = torch.zeros(self.batch, 13, device=device)
            pancakes_initial_status[:, 0] = -3.5 + deltaX  # 广播机制
            pancakes_initial_status[:, 1] = torch.tensor(potential_y) + infeed_y_offset  # 直接加法
            pancakes_initial_status[:, 2] = 0.9 +0.005  # 广播机制
            pancakes_initial_status[:, 3] = 1.0  # 广播机制
            pancakes_initial_status[:, 7] = infeedVelocity  # 广播机制
            # 利用广播机制将 scene.env_origins 扩展到 [2, batch, 3]
            expanded_env_origins = self.scene.env_origins[:, None, :]  # 形状变为 [2, 1, 3]
            # 创建一个 [2, batch, 13] 的张量，前 3 列是 expanded_env_origins，其余为 0
            final_env_origins = torch.zeros(self.scene.num_envs, self.batch, 13, device=device)
            final_env_origins[:, :, :3] = expanded_env_origins  # 广播机制会自动扩展
            pancakes_initial_status_tensor =  final_env_origins + pancakes_initial_status
            object_start_index = self.i
            object_end_index = self.i + self.batch if self.i + self.batch <= total_pancakes - 1 else total_pancakes 
            self.scene['pancake_collection'].write_object_com_state_to_sim(pancakes_initial_status_tensor[:,object_start_index - self.i:object_end_index - self.i,:],None,self.scene['pancake_collection']._ALL_OBJ_INDICES[object_start_index:object_end_index] ) 
            self.scene['pancake_collection'].reset()
            # self.scene['pancake_collection'].update(sim_dt)
            print("----------------------------------------")
            self.i += self.batch

        # cycle
        container_delta_count = self.count - self.container_offset_count
        if (outfeedVelocity * deltaT * container_delta_count >= outfeed_gen_dist * self.container_index) and (outfeedVelocity * deltaT * (container_delta_count-1) < outfeed_gen_dist * self.container_index):
            deltaX_container = outfeedVelocity * deltaT * container_delta_count - outfeed_gen_dist * self.container_index  # need to suppliment the distance during increasing the time step
            print(f"[INFO]: Spawn container when {deltaT * container_delta_count}..and {deltaX_container =}.")
            if self.container_index >= total_containers:
                self.container_index = 0 
                self.container_offset_count = self.count
                print("----------------------------------------")
                print("[INFO]: Resetting containers state...")

            container_initial_status = [
                    -3.5 + deltaX_container, outfeed_y_offset, 0.8   , 1.0, 0.0, 0.0, 0.0, outfeedVelocity, 0.0, 0.0, 0.0, 0.0, 0.0
                ]
            container_initial_status_tensor = torch.zeros(self.scene.num_envs, 13, device=device)
            container_initial_status_tensor += torch.tensor(container_initial_status, device=device)
            container_initial_status_tensor[:,:3] += self.scene.env_origins

            self.scene['container_collection'].write_object_com_state_to_sim(container_initial_status_tensor.view(self.scene.num_envs,1, -1),None,self.scene['container_collection']._ALL_OBJ_INDICES[self.container_index].view(1))
            self.scene['container_collection'].reset() 
            print("----------------------------------------")
            self.container_index +=1


    def _reset_objects(self):
        # reset
        pick_reset_area = (self.scene['pancake_collection'].data.object_com_pos_w[:, :, 0] -  self.scene.env_origins[:, None, 0])> 0 
        if pick_reset_area.any(): 
            for env_i in range(self.scene.num_envs):
                true_indices = torch.nonzero(pick_reset_area[env_i])[:, 0]  # 取当前维度的索引
                for true_i in true_indices:
                    object_default_state = self.scene['pancake_collection'].data.default_object_state[env_i,true_i,:].clone() 
                    object_default_state[:3] += self.scene.env_origins[env_i]
                    self.scene['pancake_collection'].write_object_com_state_to_sim(object_default_state.view(1,1,-1), \
                                                                            self.scene['pancake_collection']._ALL_ENV_INDICES[env_i].view(1), \
                                                                            self.scene['pancake_collection']._ALL_OBJ_INDICES[true_i].view(1))
                    self.scene['pancake_collection'].reset() 
        container_reset_area = (self.scene['container_collection'].data.object_com_pos_w[:, :, 0]-  self.scene.env_origins[:, None, 0] )   > 0
        if container_reset_area.any(): 
            for env_i in range(self.scene.num_envs):
                true_indices = torch.nonzero(container_reset_area[env_i])[:, 0]  # 取当前维度的索引
                for true_i in true_indices:
                    object_default_state = self.scene['container_collection'].data.default_object_state[env_i,true_i,:].clone() 
                    object_default_state[:3] += self.scene.env_origins[env_i]
                    self.pancakes_per_container_dict[env_i,true_i] = 0
                    self.scene['container_collection'].write_object_com_state_to_sim(object_default_state.view(1,1,-1), \
                                                                            self.scene['container_collection']._ALL_ENV_INDICES[env_i].view(1), \
                                                                            self.scene['container_collection']._ALL_OBJ_INDICES[true_i].view(1))
                    self.scene['container_collection'].reset() 

    def _pick_and_place(self):
        # pick # place
        self.pick_workarea_1_movement[:, 0] -= self.physics_dt 

        pancakes_xy_pos = self.scene['pancake_collection'].data.object_com_pos_w[:,:,:2]
        pancakes_xy_pos -= self.scene.env_origins[:, None, :2]
        containers_xy_pos = self.scene['container_collection'].data.object_com_pos_w[:,:,:2]
        containers_xy_pos -= self.scene.env_origins[:, None, :2]
        
        # 使用 self.pick_workarea_1_boundaries 的边界值
        pick_workarea_1 = (
            (pancakes_xy_pos[:, :, 0] > self.pick_workarea_1_boundaries.low[0]) & 
            (pancakes_xy_pos[:, :, 0] < self.pick_workarea_1_boundaries.high[0]) & 
            (pancakes_xy_pos[:, :, 1] > self.pick_workarea_1_boundaries.low[1]) & 
            (pancakes_xy_pos[:, :, 1] < self.pick_workarea_1_boundaries.high[1])
        )

        # 使用 self.place_workarea_1_boundaries 的边界值
        place_workarea_1 = (
            (containers_xy_pos[:, :, 0] > self.place_workarea_1_boundaries.low[0]) & 
            (containers_xy_pos[:, :, 0] < self.place_workarea_1_boundaries.high[0]) & 
            (containers_xy_pos[:, :, 1] > self.place_workarea_1_boundaries.low[1]) & 
            (containers_xy_pos[:, :, 1] < self.place_workarea_1_boundaries.high[1])
        )
        for env_i in range(self.scene.num_envs):
            if self.pick_workarea_1_movement[env_i, 0].item() <= 0 and self.pick_workarea_1_movement[env_i, 1].item() >= 0:
                self.pick_workarea_1_movement[env_i, 1] = -1

            if self.pick_workarea_1_movement[env_i, 0].item() <= 0 and place_workarea_1.any() and self.pick_workarea_1_movement[env_i, 1].item() < 0:
                # pick a pancake index
                if pick_workarea_1.any(): 
                    pancakes_true_indices = torch.nonzero(pick_workarea_1[env_i])[:, 0]  # 取当前维度的索引
                    if len(pancakes_true_indices) > 0:
                        pancake_random_index = pancakes_true_indices[torch.randint(0, len(pancakes_true_indices), (1,))].item()
                        print(f"env_{env_i},随机选择pancake的索引:", pancake_random_index)
                        pancake_object_state = self.scene['pancake_collection'].data.object_com_state_w[env_i,pancake_random_index,:].clone()
                        self.pick_workarea_1_movement[env_i, 0] = pick_place_time_consumption
                        self.pick_workarea_1_movement[env_i, 1] = pancake_random_index
                        pancake_object_state[2] += 0.01
                        pancake_object_state[7] = 0
                        pancake_object_state[:3] += self.scene.env_origins[env_i]
                        self.scene['pancake_collection'].write_object_com_state_to_sim(pancake_object_state.view(1,1,-1), \
                                                                                self.scene['pancake_collection']._ALL_ENV_INDICES[env_i].view(1), \
                                                                                self.scene['pancake_collection']._ALL_OBJ_INDICES[pancake_random_index].view(1))
                        self.scene['pancake_collection'].reset() 
                else:
                    print(f"env_{env_i},没有满足条件的pancake点。")


            elif abs(self.pick_workarea_1_movement[env_i, 0].item() - pick_place_time_consumption/2.) <= self.physics_dt and self.pick_workarea_1_movement[env_i, 1].item() >= 0:
                # find a container to place
                hold_index = int(self.pick_workarea_1_movement[env_i, 1].item())
                if place_workarea_1.any(): 
                    containers_true_indices = torch.nonzero(place_workarea_1[env_i])[:, 0]  # 取当前维度的索引
                    container_random_index = containers_true_indices[torch.randint(0, len(containers_true_indices), (1,))].item()
                    print(f"env_{env_i},随机选择container的索引:", container_random_index)
                    pancake_object_state = self.scene['container_collection'].data.object_com_state_w[env_i,container_random_index,:].clone()
                    self.pancakes_per_container_dict[env_i, int(container_random_index)] += 1
                    pancake_object_state[2] += self.pancakes_per_container_dict[env_i, int(container_random_index)] * 0.01 - 0.005
                    pancake_object_state[:3] += self.scene.env_origins[env_i]
                    self.scene['pancake_collection'].write_object_com_state_to_sim(pancake_object_state.view(1,1,-1), \
                                                                            self.scene['pancake_collection']._ALL_ENV_INDICES[env_i].view(1), \
                                                                            self.scene['pancake_collection']._ALL_OBJ_INDICES[hold_index].view(1))
                    self.scene['pancake_collection'].reset()  
                    self.pick_workarea_1_movement[env_i, 1] = -1
            else:
                # change the status
                hold_index = int(self.pick_workarea_1_movement[env_i, 1].item())
                if self.pick_workarea_1_movement[env_i, 1].item() >= 0:

                    pancake_object_state = self.scene['pancake_collection'].data.object_com_state_w[env_i,hold_index,:].clone()
                    pancake_object_state[2] += 0.01
                    pancake_object_state[7] = 0
                    pancake_object_state[:3] += self.scene.env_origins[env_i]
                    self.scene['pancake_collection'].write_object_com_state_to_sim(pancake_object_state.view(1,1,-1), \
                                                                            self.scene['pancake_collection']._ALL_ENV_INDICES[env_i].view(1), \
                                                                            self.scene['pancake_collection']._ALL_OBJ_INDICES[hold_index].view(1))
                    self.scene['pancake_collection'].reset() 


    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions

        self._generator_objects()
        self._reset_objects()
        self.count += 1


    def _apply_action(self) -> None:

        # self._pick_and_place()


        self.robot.set_joint_effort_target(
            self.actions["cart"] * self.cfg.cart_action_scale, joint_ids=self._cart_dof_idx
        )
        self.robot.set_joint_effort_target(
            self.actions["pendulum"] * self.cfg.pendulum_action_scale, joint_ids=self._pendulum_dof_idx
        )

 

    def _get_observations(self) -> dict[str, torch.Tensor]:


        # whether the obs need the future deltaT???
        # it seems no, next loop will immediately happen


        pancakes_xy_pos = self.scene['pancake_collection'].data.object_com_pos_w[:,:,:2].clone()
        pancakes_xy_pos -= self.scene.env_origins[:, None, :2]
 
        
        # 使用 self.pick_workarea_1_boundaries 的边界值
        pick_workarea_1 = (
            (pancakes_xy_pos[:, :, 0] > self.pick_workarea_1_boundaries.low[0]) & 
            (pancakes_xy_pos[:, :, 0] < self.pick_workarea_1_boundaries.high[0]) & 
            (pancakes_xy_pos[:, :, 1] > self.pick_workarea_1_boundaries.low[1]) & 
            (pancakes_xy_pos[:, :, 1] < self.pick_workarea_1_boundaries.high[1])
        )
        pick_features = torch.zeros(self.scene.num_envs,self.pick_encoder.get_feature_dim(),device=device)
        for env_i in range(self.scene.num_envs):
            indices=torch.nonzero(pick_workarea_1[env_i]).squeeze()
            #     # 随机丢弃一项
            # if len(indices) > 1:  # 确保至少有两项可以丢弃
            #     drop_index = torch.randint(0, len(indices), (1,)).item()
            #     indices = torch.cat([indices[:drop_index], indices[drop_index+1:]])
            
                    
            positions = pancakes_xy_pos[env_i,indices].clone()
            pick_features[env_i] = self.pick_encoder(positions)




        pole_joint_pos = normalize_angle(self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1))
        pendulum_joint_pos = normalize_angle(self.joint_pos[:, self._pendulum_dof_idx[0]].unsqueeze(dim=1))
        observations = {
            "cart": torch.cat(
                (
                    self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                    self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                    pole_joint_pos,
                    self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                ),
                dim=-1,
            ),
            "pendulum": torch.cat(
                (
                    pole_joint_pos + pendulum_joint_pos,
                    pendulum_joint_pos,
                    self.joint_vel[:, self._pendulum_dof_idx[0]].unsqueeze(dim=1),
                ),
                dim=-1,
            ),
            "pickmaster": pick_features
        }
        return observations

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_cart_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_pole_vel,
            self.cfg.rew_scale_pendulum_pos,
            self.cfg.rew_scale_pendulum_vel,
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            normalize_angle(self.joint_pos[:, self._pole_dof_idx[0]]),
            self.joint_vel[:, self._pole_dof_idx[0]],
            normalize_angle(self.joint_pos[:, self._pendulum_dof_idx[0]]),
            self.joint_vel[:, self._pendulum_dof_idx[0]],
            math.prod(self.terminated_dict.values()),
        )
        return total_reward

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:

 



        # self.joint_pos = self.robot.data.joint_pos
        # self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        # out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)

        # terminated = {agent: out_of_bounds for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return time_outs, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.scene.num_envs
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_pos[:, self._pendulum_dof_idx] += sample_uniform(
            self.cfg.initial_pendulum_angle_range[0] * math.pi,
            self.cfg.initial_pendulum_angle_range[1] * math.pi,
            joint_pos[:, self._pendulum_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_cart_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_pos: float,
    rew_scale_pole_vel: float,
    rew_scale_pendulum_pos: float,
    rew_scale_pendulum_vel: float,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    pendulum_pos: torch.Tensor,
    pendulum_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_pendulum_pos = rew_scale_pendulum_pos * torch.sum(
        torch.square(pole_pos + pendulum_pos).unsqueeze(dim=1), dim=-1
    )
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    rew_pendulum_vel = rew_scale_pendulum_vel * torch.sum(torch.abs(pendulum_vel).unsqueeze(dim=1), dim=-1)

    total_reward = {
        "cart": rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel,
        "pendulum": rew_alive + rew_termination + rew_pendulum_pos + rew_pendulum_vel,
        "pickmaster": 0
    }
    return total_reward




import torch.nn as nn


class PointNetEncoder(nn.Module):
    def __init__(self, feature_dim=64, device='cuda:0', x_min=0, x_max=1, y_min=0, y_max=1):
        super(PointNetEncoder, self).__init__()

        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.feature_dim = feature_dim

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),  # Input is 2D coordinates (x, y)
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Max Pooling
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Output layer
        self.output_layer = nn.Linear(64, feature_dim)

        # Move to device
        self.to(device)




    def forward(self, x):
        """
        输入:
        x: Tensor, 形状为 (num_points, 2)
        输出:
        features: Tensor, 形状为 (feature_dim,)
        """
        if len(x) == 0:
            # 如果 x 为空列表，返回形状为 (feature_dim,) 的全零张量
            return torch.zeros(self.feature_dim)
        num_points, _ = x.shape


        # Min-Max 归一化
        x[:, 0] = (x[:, 0] - self.x_min) / (self.x_max - self.x_min)
        x[:, 1] = (x[:, 1] - self.y_min) / (self.y_max - self.y_min)

        x = x.view(-1, 2)  # 展平为 (num_points, 2)
        x = self.mlp(x)  # 形状为 (num_points, 64)
        x = x.view(1, num_points, -1)  # 增加批次维度，形状为 (1, num_points, 64)
        x = x.permute(0, 2, 1)  # 形状为 (1, 64, num_points)
        x = self.pool(x)  # 形状为 (1, 64, 1)
        x = x.squeeze(-1)  # 形状为 (1, 64)
        x = self.output_layer(x)  # 形状为 (1, feature_dim)
        x = x.squeeze(0)  # 移除批次维度，形状为 (feature_dim,)

        return x
    
    def get_feature_dim(self):
        return self.feature_dim