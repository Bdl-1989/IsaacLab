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

deltaT = 0.02
infeedVelocity = 0.0467
outfeedVelocity = 0.1333
infeed_y_offset = -0.5
outfeed_y_offset = 0.1+0.001
pancakes_per_container = 6
infeed_gen_dist = 0.095 
outfeed_gen_dist = 0.230
potential_y = [-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4] # idea make a map of potential y's and spawn them randomly

item_veloctiy = 0.5 # m/s

num_containers = math.ceil(4 / (outfeed_gen_dist)) 
num_pancake_row = math.ceil(4/(infeed_gen_dist))

pick_height = 0.020 
place_height = 0.020  + 0.1

device = "cuda:0"


pancake_cfg =  RigidObjectCfg(
        spawn=sim_utils.CylinderCfg(
                radius=0.045,
                height=0.010,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled= True,  
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
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

container_cfg =  RigidObjectCfg(
        spawn=sim_utils.CuboidCfg(
                size=(0.195,0.125 , 0.005),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled= True,  
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
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0.2),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

 

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


@configclass
class PancakeSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""


    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg(
        physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1, dynamic_friction=1,restitution= 1)
    ))

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
 

    infeed_conveyor: RigidObjectCfg = INFEED_CONVEYOR_CFG.replace(prim_path="{ENV_REGEX_NS}/infeed_conveyor")
    outfeed_conveyor: RigidObjectCfg = OUTFEED_CONVEYOR_CFG.replace(prim_path="{ENV_REGEX_NS}/outfeed_conveyor")

    place_work_area_1: RigidObjectCfg = PLACE_WORKAREA_1.replace(prim_path="{ENV_REGEX_NS}/place_work_area_1")
    pick_work_area_1: RigidObjectCfg = PICK_WORKAREA_1.replace(prim_path="{ENV_REGEX_NS}/pick_work_area_1")

    pancake_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(rigid_objects=combined_dic)
    container_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(rigid_objects=container_dic) 


@configclass
class PickAndPlaceEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 1
    episode_length_s = 5.0
    possible_agents = ["actor"]
    action_spaces = {"actor": 1}
    observation_spaces = {"actor": 4}
    state_space = -1  #?

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=deltaT, device=device, render_interval=decimation)

    # robot
    # robot_cfg: ArticulationCfg = CART_DOUBLE_PENDULUM_CFG.replace(prim_path="/World/envs/env_.*/Robot")
 

    # scene
    scene: InteractiveSceneCfg = PancakeSceneCfg(num_envs=2, env_spacing=10, replicate_physics=True)

    # reset
    # max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    # initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]
    # initial_pendulum_angle_range = [-0.25, 0.25]  # the range in which the pendulum angle is sampled from on reset [rad]

    # action scales
    # cart_action_scale = 100.0  # [N]
    # pendulum_action_scale = 50.0  # [Nm]

    # reward scales
    # rew_scale_alive = 1.0
    # rew_scale_terminated = -2.0
    # rew_scale_cart_pos = 0
    # rew_scale_cart_vel = -0.01
    # rew_scale_pole_pos = -1.0
    # rew_scale_pole_vel = -0.01
    # rew_scale_pendulum_pos = -1.0
    # rew_scale_pendulum_vel = -0.01


class PickAndPlaceEnv(DirectMARLEnv):
    cfg: PickAndPlaceEnvCfg

    def __init__(self, cfg: PickAndPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        pass
        # self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        # self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)
        # self._pendulum_dof_idx, _ = self.robot.find_joints(self.cfg.pendulum_dof_name)

        # self.joint_pos = self.robot.data.joint_pos
        # self.joint_vel = self.robot.data.joint_vel

    def _setup_scene(self):
 
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
 
        # add lights
 

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions

    def _apply_action(self) -> None:
        # self.robot.set_joint_effort_target(
        #     self.actions["cart"] * self.cfg.cart_action_scale, joint_ids=self._cart_dof_idx
        # )
        # self.robot.set_joint_effort_target(
        #     self.actions["pendulum"] * self.cfg.pendulum_action_scale, joint_ids=self._pendulum_dof_idx
        # )
        pass

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # pole_joint_pos = normalize_angle(self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1))
        # pendulum_joint_pos = normalize_angle(self.joint_pos[:, self._pendulum_dof_idx[0]].unsqueeze(dim=1))
        observations = {
            # "cart": torch.cat(
            #     (
            #         self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            #         self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            #         pole_joint_pos,
            #         self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
            #     ),
            #     dim=-1,
            # ),
            # "pendulum": torch.cat(
            #     (
            #         pole_joint_pos + pendulum_joint_pos,
            #         pendulum_joint_pos,
            #         self.joint_vel[:, self._pendulum_dof_idx[0]].unsqueeze(dim=1),
            #     ),
            #     dim=-1,
            # ),
        }
        return observations

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        total_reward = compute_rewards(
        )
        return total_reward

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
 
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        # out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)

        terminated = {agent: torch.Tensor() for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.scene.num_envs
        super()._reset_idx(env_ids)
        pass
        # joint_pos = self.robot.data.default_joint_pos[env_ids]
        # joint_pos[:, self._pole_dof_idx] += sample_uniform(
        #     self.cfg.initial_pole_angle_range[0] * math.pi,
        #     self.cfg.initial_pole_angle_range[1] * math.pi,
        #     joint_pos[:, self._pole_dof_idx].shape,
        #     joint_pos.device,
        # )
        # joint_pos[:, self._pendulum_dof_idx] += sample_uniform(
        #     self.cfg.initial_pendulum_angle_range[0] * math.pi,
        #     self.cfg.initial_pendulum_angle_range[1] * math.pi,
        #     joint_pos[:, self._pendulum_dof_idx].shape,
        #     joint_pos.device,
        # )
        # joint_vel = self.robot.data.default_joint_vel[env_ids]

        # default_root_state = self.robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # self.joint_pos[env_ids] = joint_pos
        # self.joint_vel[env_ids] = joint_vel

        # self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def compute_rewards():
 
    total_reward = {
        "actor":0 
    }
    return total_reward
