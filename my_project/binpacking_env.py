# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence


from omni.isaac.lab_assets.bins import bins_cfg_dict, PALLET_CFG,boxes,nBoxes,initPositions,nBoxesPerSKUs,dimPerSKUs
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg,RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.sim import PhysxCfg, SimulationCfg

import numpy as np
from gym import spaces



# Nash: the external paras is not here
# import argparse
# def add_binpacking_args(parser: argparse.ArgumentParser):
#     """Add Binpacking arguments to the parser.

#     Args:
#         parser: The parser to add the arguments to.
#     """
#     # create a new argument group
#     arg_group = parser.add_argument_group("binpacking", description="Arguments for Binpacking envs")
#     # -- experiment arguments
#     arg_group.add_argument(
#         "--reward_paras", type=float, default=1.0, help="The parameter of the reward function"
#     )
#     # arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
#     # # -- load arguments
#     # arg_group.add_argument("--resume", type=bool, default=None, help="Whether to resume from a checkpoint.")
#     # arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
#     # arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
#     # # -- logger arguments
#     # arg_group.add_argument(
#     #     "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
#     # )
#     # arg_group.add_argument(
#     #     "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
#     # )


# The dic to define the paras of the envs, mainly the l, w, h
binpacking_cfg = {
            'pallet':{  
                'x_min': 0,  
                'x_max': 5,  
                'y_min': 0,  
                'y_max': 5,  
                'z_min': 0,  
                'z_max': 20  
            },
            'env':{
                'numObservations':0,
                'numActions':0,
                'envSpacing': 5.0,
                'clipObservations': 1.0,
                'clipActions': 1.0
            }
        }

SPATIAL_DELTA = 1e-3
MOTION_DELTA = 1e-4

# ALL PARAS should be defined here
@configclass
class BinpackingEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 10 # number of simulation steps per environment step, check direct_rl_env.py for more details
    
    episode_length_s = 5.0
    action_scale = 100.0  # [N]
    num_actions = 2  # 2
    num_observations = 12 *  nBoxes #  12 * 6 (6 bins)
    num_states = 0
    # num_Envs = 2
    # simulation
    sim: SimulationCfg = SimulationCfg(
        # why in W&B it shows GPU 3 is used
    device = "cuda:0", # can be "cpu", "cuda", "cuda:<device_id>"
    dt=1 / 120,
    render_interval=10,
    # decimation will be set in the task config
    # up axis will always be Z in isaac sim
    # use_gpu_pipeline is deduced from the device
    gravity=(0.0, 0.0, -9.81),
    physx = PhysxCfg(
        # num_threads is no longer needed
        solver_type=1,
        # use_gpu is deduced from the device
        max_position_iteration_count=4,
        max_velocity_iteration_count=0,
        # moved to actor config
        # moved to actor config
        bounce_threshold_velocity=0.2,
        # moved to actor config
        # default_buffer_size_multiplier is no longer needed
        gpu_max_rigid_contact_count=2**23
        # num_subscenes is no longer needed
        # contact_collection is no longer needed
    ))
    # robot
    # robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # cart_dof_name = "slider_to_cart"
    # pole_dof_name = "cart_to_pole"

    # pallet
    pallet_cfg = PALLET_CFG.replace(prim_path="/World/envs/env_.*/Pallet")
    # bin dict
    bins_cfg_dict = bins_cfg_dict

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2, env_spacing=20, replicate_physics=True)

    # reset
    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]

    # reward scales
    # rew_scale_alive = 1.0
    # rew_scale_terminated = -2.0
    # rew_scale_pole_pos = -1.0
    # rew_scale_cart_vel = -0.01
    # rew_scale_pole_vel = -0.005


    x_pallet_min = binpacking_cfg["pallet"]["x_min"]
    x_pallet_max = binpacking_cfg["pallet"]["x_max"]
    y_pallet_min = binpacking_cfg["pallet"]["y_min"]
    y_pallet_max = binpacking_cfg["pallet"]["y_max"]
    z_pallet_min = binpacking_cfg["pallet"]["z_min"]
    z_pallet_max = binpacking_cfg["pallet"]["z_max"]

    # Nash:external args
    # reward_paras = args.reward_paras


# The entry point for the envs, the function and interaction of the isaac sim
class BinpackingEnv(DirectRLEnv):
    cfg: BinpackingEnvCfg

    def __init__(self, cfg: BinpackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


        self.x_pallet_min = self.cfg.x_pallet_min
        self.x_pallet_max = self.cfg.x_pallet_max
        self.y_pallet_min = self.cfg.y_pallet_min
        self.y_pallet_max = self.cfg.y_pallet_max
        self.z_pallet_min = self.cfg.z_pallet_min
        self.z_pallet_max = self.cfg.z_pallet_max

        # self.nEnvs = self.cfg.num_Envs
        

        self.env_boundaries = spaces.Box(low=np.array([self.x_pallet_min, self.y_pallet_min, self.z_pallet_min]), high=np.array([self.x_pallet_max, self.y_pallet_max, self.z_pallet_max]), dtype=np.float32)

        # Casue the action only consdiers the x and y
        self.action_bias = [float(self.x_pallet_min + self.x_pallet_max)/2, float(self.y_pallet_min + self.y_pallet_max)/2]
        self.action_scale = [float(self.x_pallet_max - self.x_pallet_min), float(self.y_pallet_max - self.y_pallet_min)]

        self.queued = [{} for _ in range(self.num_envs)] #boxes to be stacked
        self.nQueued = [0 for _ in range(self.num_envs)] #number of boxes waiting to be stacked
        self.stacked = [[] for _ in range(self.num_envs)] #boxes already stacked, recording indices
        self.xyAreas = [[] for _ in range(self.num_envs)] #xyArea of each group // in version 1.0, this determines the stacking order (larger first)
        self.currBoxIdx = [-1 for _ in range(self.num_envs)] #index of currently handling boxes (used for state calculation & stacked record)
                
        self.InitSKUInfo()

        #For reward calculation
        self.totalStackedVol = torch.zeros(self.num_envs, device=self.device)
        self.occupiedVol = torch.zeros(self.num_envs, 6, device=self.device)
        self.occupiedVol[:, :3] = 1.0
        '''
        Addition
        '''
        self.stacked_z_vals = [{} for _ in range(self.num_envs)]
        self.stacked_z_upper_pivot_coord = [{} for _ in range(self.num_envs)]
        self.delta_gap = torch.tensor([1e-3/self.action_scale[0], 1e-3/self.action_scale[1]], device=self.device)
        self.stacked_z_max = torch.zeros(self.num_envs, device=self.device)

        #Initial reset
        # self.reset_idx(torch.arange(self.nEnvs, device=self.device))
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.reset_buf[torch.arange(self.nEnvs, device=self.device)] = 0
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf[torch.arange(self.num_envs, device=self.device)] = 0

        #Finally, override the the observation space with box [0, +1]
        # self.obs_space = spaces.Box(low=0.0, high=1.0, shape=(12*nBoxes,))
        # self.state_space = spaces.Box(low=0.0, high=1.0, shape=(12*nBoxes,))
        root_states = [] 
        dim_list = []

        for key, _ in self.cfg.bins_cfg_dict.items():
            root_state = getattr(self,key).data.default_root_state.clone()
            root_states.append(root_state.unsqueeze(1)) 
            dim_list.append(getattr(self,key).cfg.spawn.size)
 
        # 使用torch.cat()函数将所有root_state合并到一个tensor中，新的维度是第二个维度（维度索引为1）  
        self._root_states = torch.cat(root_states, 1).to(device=self.device) 
        # self._root_states shape should be env, 6, 13
        self._root_pos = self._root_states[..., 0:3]
        self._root_quat = self._root_states[..., 3:7]
        self._root_lin_vel = self._root_states[..., 7:10]
        self._root_ang_vel = self._root_states[..., 10:13]
        self._dim_list = torch.tensor(dim_list)
 


        self._reset_idx(torch.arange(self.num_envs, device=self.device))

 

    def _setup_scene(self): 
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
 

        # Comments now, seems some probs
        # self.scene.rigid_objects["pallet"] = RigidObject(self.cfg.pallet_cfg) 


        # Add attributes for each key-value pair in bins_cfg_dict  
 
        for key, value in self.cfg.bins_cfg_dict.items():
            setattr(self, key, RigidObject(value.replace(prim_path="/World/envs/env_.*/"+key))) 
            self.scene.rigid_objects[key] =  getattr(self,key) 

 

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_buf[env_ids] = 0
            self.progress_buf[env_ids] = 0


        self.rew_buf = torch.zeros(self.num_envs, device=self.device)

        self.actions =   actions.clone()
        x = self.actions[:, 0]
        y = self.actions[:, 1]        
        positions = torch.zeros(self.num_envs, 3, device=self.device)
        quats = torch.zeros(self.num_envs, 4, device=self.device)
        quats[:, -1] = 1.0
        positions[:, 0] = self.action_bias[0] + x*(self.action_scale[0] - self.obs_buf_bins[:, -3]*self.action_scale[0])/2
        positions[:, 1] = self.action_bias[1] + y*(self.action_scale[1] - self.obs_buf_bins[:, -2]*self.action_scale[1])/2
        # positions[:, 0] = self.action_bias[0] -  self.action_scale[0]* (x - 1/2)
        # positions[:, 1] = self.action_bias[1] -  self.action_scale[1]* (y - 1/2)
        positions[:, 2] = self.updateZcoord(positions[:, 0], positions[:, 1])


        #root state vectors update for simulation
        indices = torch.tensor(self.currBoxIdx, device=self.device)
        self._root_pos[:, indices, :] = positions
        self._root_quat[:, indices,  :] = quats
        self._root_lin_vel[:, indices, :] = torch.zeros(self.num_envs,  3, device=self.device)
        self._root_ang_vel[:, indices, :] = torch.zeros(self.num_envs, 3, device=self.device)
 
        #For motion validity check at post_physics_step
        self.prePositions = [self.currBoxIdx, self.stacked, self._root_pos.clone()]

    def updateZcoord(self, x, y):
        '''
        Given the action values (x, y), find the valid z value
        The idea is to first assess if the 2D area (xy) required for the current box is already occupied or not.
        If it is, then find the maximum value of the z_max of the box(es) that occupies this area.
        The current item will be placed at max(z_max) + curr_box_z_dim

        Otherwise, z = curr_box_z_dim + self.env_boundaries.low[2]

        Returns
        ________
        Tensor [numEnv, ]: z value
        '''

        z_action_scale = (self.env_boundaries.high[2] - self.env_boundaries.low[2])
        z_max = torch.zeros(self.num_envs, device=self.device) + self.env_boundaries.low[2]
        for n in range(self.num_envs):
            curr_x_min = x[n] - self.action_scale[0]*self.obs_buf_bins[n, -3]/2
            curr_x_max = x[n] + self.action_scale[0]*self.obs_buf_bins[n, -3]/2
            curr_y_min = y[n] - self.action_scale[1]*self.obs_buf_bins[n, -2]/2
            curr_y_max = y[n] + self.action_scale[1]*self.obs_buf_bins[n, -2]/2

            for b in self.stacked[n]:
                stacked_x_min = self.obs_buf_bins[n, 12*b + 0]*self.action_scale[0] + self.env_boundaries.low[0]
                stacked_x_max = self.obs_buf_bins[n, 12*b + 6]*self.action_scale[0] + self.env_boundaries.low[0]
                stacked_y_min = self.obs_buf_bins[n, 12*b + 1]*self.action_scale[1] + self.env_boundaries.low[1]
                stacked_y_max = self.obs_buf_bins[n, 12*b + 7]*self.action_scale[1] + self.env_boundaries.low[1]

                x2 = min(curr_x_max, stacked_x_max)
                x1 = max(curr_x_min, stacked_x_min)
                y2 = min(curr_y_max, stacked_y_max)
                y1 = max(curr_y_min, stacked_y_min)

                if(x2-x1>0 and y2-y1>0):
                    stacked_z_max = self.obs_buf_bins[n, 12*b+8]*(z_action_scale) + self.env_boundaries.low[2]
                    if(stacked_z_max > z_max[n]):
                        z_max[n] = stacked_z_max

            #final validity check
            if(z_max[n] > self.env_boundaries.high[2] - z_action_scale*self.obs_buf_bins[n, -1]):
                #Place it at max valid z coord for current box in this case
                #This will create collision with the already stacked box(es), and isStaionary will return False
                z_max[n] = self.env_boundaries.high[2] - z_action_scale*self.obs_buf_bins[n, -1]

        return z_max + z_action_scale*self.obs_buf_bins[:, -1]/2

    def _apply_action(self) -> None:

        for index, (key, _) in enumerate(self.cfg.bins_cfg_dict.items()):
            selected_root_states = self._root_states[:, index, :].clone() 
            env_ids = getattr(self,key)._ALL_INDICES
            selected_root_states[:, :3] += self.scene.env_origins[env_ids]
            getattr(self,key).write_root_pose_to_sim(selected_root_states[:, :7],env_ids)
            getattr(self,key).write_root_velocity_to_sim(selected_root_states[:, 7:],env_ids)
 
        

    def _get_observations(self) -> dict:
 
        observations = {"policy": self.obs_buf_bins}
        return observations

    def _get_rewards(self) -> torch.Tensor:

 
        return self.rew_buf

    def compute_reward(self, valid_ids):
        '''
        The self.rew_buf is updated based on a stacked volume efficiency
        '''

        self.rew_buf[valid_ids] += compute_binpacking_reward(self.totalStackedVol, self.occupiedVol, valid_ids)

        return


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
 
 
        self.progress_buf += 1
        

        root_states = []  
        for key, _ in self.cfg.bins_cfg_dict.items():
            root_state = getattr(self,key).data.root_state_w.clone()
            env_ids = getattr(self,key)._ALL_INDICES
            root_state[:,:3] -= self.scene.env_origins[env_ids]
            root_states.append(root_state.unsqueeze(1))  
 
        # 使用torch.cat()函数将所有root_state合并到一个tensor中，新的维度是第二个维度（维度索引为1）  
        self._root_states = torch.cat(root_states, 1).to(device=self.device) 
        # self._root_states shape should be 2, 6, 13
        self._root_pos = self._root_states[..., 0:3]
        self._root_quat = self._root_states[..., 3:7]
        self._root_lin_vel = self._root_states[..., 7:10]
        self._root_ang_vel = self._root_states[..., 10:13]


        self.isNonStationary()

        '''
        Based on self.reset_buf, the self.obs_buf_bins is updated accordingly.
            -> If self.reset_buf for a particular env is not zero, self.obs_buf_bins is reset 
                [Final state's next trasition is masked. So update it to the init state] => Seems like IssacGym Env doesn't call reset outside of the step func
            -> Otherwise, self.obs_buf_bins is updated with newly stacked positions of the currently handling item, and dimensions of a newly extracted item
        '''
        valid_ids = torch.arange(self.num_envs, device=self.device)
        invalid_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        mask = self.reset_buf.eq(0)
        valid_ids = torch.masked_select(valid_ids, mask)
        final_ids= []
        non_final_ids = []


        # need check the father do this again
        if len(invalid_ids) > 0:
            self._reset_idx(invalid_ids)

        if len(valid_ids) > 0:
            non_final_ids, final_ids = self.updateState(valid_ids)
            #self.compute_reward(valid_ids)
            
            #Turn the lists to tensors
            non_final_ids = torch.tensor(non_final_ids, device=self.device).to(dtype=torch.long)
            final_ids = torch.tensor(final_ids, device=self.device).to(dtype=torch.long)

            if(len(non_final_ids)>0):
                #deque for non_final environments
                self.extractItem(non_final_ids)
                self.rew_buf[non_final_ids] = 0.01

            if len(final_ids) > 0:
                self.compute_reward(final_ids)
                #self.rew_buf[final_ids] += 10.0*(1 - self.stacked_z_max[valid_ids])
                self._reset_idx(final_ids)
                self.reset_buf[final_ids] = 1.0
 
        self.reset_timeout_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  
        # need check
        return self.reset_buf , self.reset_timeout_buf

    def isNonStationary(self):
        '''
        This determins if the physics simulation creates any motion for currently handling item and/or stacked items.
        Based on self.prePosition that is assigned at each pre_sim call, it compares any position changes (due to collision, gravity, external forces, etc.)
        If the positional changes are above the certain threshold, that particular environment will be assigned "non stationary"
        And, based on the non-stationarity, self.reset_buf is updated for self.obs_buf_bins and self.rew_buf calculations later on
        '''
        
        bNonStationary = torch.zeros(self.num_envs, device=self.device).to(dtype=torch.bool)
        for n in range(self.num_envs):
            currNonStationary = False
            currIdx = self.prePositions[0][n] #single scalar
            compIdx = self.prePositions[1][n].copy()
            compIdx.append(currIdx)
            # compIdx = 1 + torch.tensor(compIdx, device=self.device, dtype=torch.int64) #scalar array
            prevPosVectors = self.prePositions[2][n, compIdx] #[nBoxes, 3] vector
            currPosVectors = self._root_pos[n, compIdx] #[nBoxes, 3] vector

            diffPos = currPosVectors - prevPosVectors
            diffPos = torch.sqrt(torch.pow(diffPos[:, 0], 2) + torch.pow(diffPos[:, 1], 2) + torch.pow(diffPos[:, 2], 2)) > MOTION_DELTA
            bNonStationary[n] = torch.any(diffPos)
            
        #???need check
        self.reset_buf[bNonStationary] = 1.0
        self.rew_buf[bNonStationary] = -2.25


    def updateState(self, valid_ids):

        final_ids= []
        non_final_ids = []
            
        #obs_buf update 
        currBoxIdx = torch.tensor(self.currBoxIdx, device=self.device)
        currBoxIdx = currBoxIdx[valid_ids]
        x = self._root_pos[valid_ids, currBoxIdx, 0].clone()
        y = self._root_pos[valid_ids, currBoxIdx, 1].clone()
        z = self._root_pos[valid_ids, currBoxIdx, 2].clone()
        z_scale = (self.env_boundaries.high[2] - self.env_boundaries.low[2])

        x_min = x - self.action_scale[0]*self.obs_buf_bins[valid_ids, -3]/2
        x_max = x + self.action_scale[0]*self.obs_buf_bins[valid_ids, -3]/2
        y_min = y - self.action_scale[1]*self.obs_buf_bins[valid_ids, -2]/2
        y_max = y + self.action_scale[1]*self.obs_buf_bins[valid_ids, -2]/2
        z_min = z - z_scale*self.obs_buf_bins[valid_ids, -1]/2
        z_max = z + z_scale*self.obs_buf_bins[valid_ids, -1]/2
        
        self.obs_buf_bins[valid_ids, 12*currBoxIdx + 0] = (x_min - self.env_boundaries.low[0])/self.action_scale[0]
        self.obs_buf_bins[valid_ids, 12*currBoxIdx + 1] = (y_min - self.env_boundaries.low[1])/self.action_scale[1]
        self.obs_buf_bins[valid_ids, 12*currBoxIdx + 2] = (z_min - self.env_boundaries.low[2])/z_scale

        self.obs_buf_bins[valid_ids, 12*currBoxIdx + 6] = (x_max - self.env_boundaries.low[0])/self.action_scale[0]
        self.obs_buf_bins[valid_ids, 12*currBoxIdx + 7] = (y_max - self.env_boundaries.low[1])/self.action_scale[1]
        self.obs_buf_bins[valid_ids, 12*currBoxIdx + 8] = (z_max - self.env_boundaries.low[2])/z_scale
        self.stacked_z_max[valid_ids] = torch.max(self.stacked_z_max[valid_ids], self.obs_buf_bins[valid_ids, 12*currBoxIdx + 8])
                      
        self.obs_buf_bins[valid_ids, :] = torch.clamp(self.obs_buf_bins[valid_ids, :], min=0.0, max=1.0) 
        self.rew_buf[valid_ids] = 1.0 - self.obs_buf_bins[valid_ids, 12*currBoxIdx + 2] 

        #for reward calculation
        self.totalStackedVol[valid_ids] += (self.obs_buf_bins[valid_ids, 12*currBoxIdx + 6] - self.obs_buf_bins[valid_ids, 12*currBoxIdx + 0])*(self.obs_buf_bins[valid_ids, 12*currBoxIdx + 7]-self.obs_buf_bins[valid_ids, 12*currBoxIdx + 1])*(self.obs_buf_bins[valid_ids, 12*currBoxIdx + 8]-self.obs_buf_bins[valid_ids, 12*currBoxIdx + 2])
        self.occupiedVol[valid_ids, 0] = torch.min(torch.vstack([self.occupiedVol[valid_ids, 0], self.obs_buf_bins[valid_ids, 12*currBoxIdx + 0]]).t(), dim=1).values
        self.occupiedVol[valid_ids, 1] = torch.min(torch.vstack([self.occupiedVol[valid_ids, 1], self.obs_buf_bins[valid_ids, 12*currBoxIdx + 1]]).t(), dim=1).values
        self.occupiedVol[valid_ids, 2] = torch.min(torch.vstack([self.occupiedVol[valid_ids, 2], self.obs_buf_bins[valid_ids, 12*currBoxIdx + 2]]).t(), dim=1).values
        self.occupiedVol[valid_ids, 3] = torch.max(torch.vstack([self.occupiedVol[valid_ids, 3], self.obs_buf_bins[valid_ids, 12*currBoxIdx + 6]]).t(), dim=1).values
        self.occupiedVol[valid_ids, 4] = torch.max(torch.vstack([self.occupiedVol[valid_ids, 4], self.obs_buf_bins[valid_ids, 12*currBoxIdx + 7]]).t(), dim=1).values
        self.occupiedVol[valid_ids, 5] = torch.max(torch.vstack([self.occupiedVol[valid_ids, 5], self.obs_buf_bins[valid_ids, 12*currBoxIdx + 8]]).t(), dim=1).values

        #stacked boxes recording
        ptr = 0
        for idx, stacked in enumerate(self.stacked):
            if(ptr==len(valid_ids)):
                break
            if(idx==valid_ids[ptr]):
                ptr += 1
                stacked.append(self.currBoxIdx[idx])

                if(self.nQueued[idx]>0):
                    non_final_ids.append(idx)
                else:
                    #final state
                    final_ids.append(idx)
                
                
                
                

        return non_final_ids, final_ids

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # if env_ids is None:
        #     env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)
 
       #print("reset call", len(env_ids))

        ###########################################################
        ## Boxes and SKU Info reset
        ########################################################### 
        for idx in range(len(env_ids)):
            self.stacked[env_ids[idx]] = []
            self.nQueued[env_ids[idx]] = 0
            self.xyAreas[env_ids[idx]] = []
            self.stacked_z_vals[env_ids[idx]] = {}
            self.stacked_z_upper_pivot_coord [env_ids[idx]] = {}
            for jdx, b_entry in enumerate(boxes):
                self.queued[env_ids[idx]][b_entry] = boxes[b_entry]["num"]
                self.nQueued[env_ids[idx]] += boxes[b_entry]["num"]
                self.xyAreas[env_ids[idx]].append([boxes[b_entry]["dim"][0]*boxes[b_entry]["dim"][1], jdx])

            self.xyAreas[env_ids[idx]].sort(reverse=True)
            self.currBoxIdx[env_ids[idx]] = -1

        self.totalStackedVol[env_ids] = 0.0
        self.occupiedVol[env_ids] = torch.zeros(len(env_ids), 6, device=self.device)
        self.occupiedVol[env_ids, :3] = 1.0
        self.occupiedVol[env_ids, 3:] = 0.0
        self.stacked_z_max[env_ids] = 0.0
        ###########################################################
        ## RL observation space reset
        ########################################################### 
        # create obs buffer
        self.obs_buf_bins = torch.zeros(
            (self.num_envs, self.cfg.num_observations), device=self.device, dtype=torch.float)


        #update currently handling box's dims in obs_buf, and update currBoxIdx
        self.extractItem(env_ids)

        currBoxIdx = torch.tensor(self.currBoxIdx, device=self.device)
        currBoxIdx = currBoxIdx[env_ids]

        ###########################################################
        ## Issac Gym reset
        ########################################################### 

        positions = torch.zeros(len(env_ids), nBoxes, 3, device=self.device)
        quats = torch.zeros(len(env_ids), nBoxes, 4, device=self.device)

        for b in range(nBoxes):

            positions[:, b, 0] = initPositions[b][0]
            positions[:, b, 1] = initPositions[b][1]
            positions[:, b, 2] = initPositions[b][2]
            quats[:, b, 3] = 1.0


        self._root_pos[env_ids, :, :] = positions
        self._root_quat[env_ids, :,  :] = quats
        self._root_lin_vel[env_ids, :, :] = torch.zeros(len(env_ids), nBoxes, 3, device=self.device)
        self._root_ang_vel[env_ids, :, :] = torch.zeros(len(env_ids), nBoxes, 3, device=self.device)

        for index, (key, _) in enumerate(self.cfg.bins_cfg_dict.items()):
            selected_root_states = self._root_states[:, index, :].clone() 
            each_env_ids = getattr(self,key)._ALL_INDICES
            selected_root_states[:, :3] += self.scene.env_origins[each_env_ids]
            getattr(self,key).write_root_pose_to_sim(selected_root_states[:, :7], each_env_ids)
            getattr(self,key).write_root_velocity_to_sim(selected_root_states[:, 7:], each_env_ids)
    
    
    def InitSKUInfo(self):

        for idx, b_entry in enumerate(boxes):
            # self.nBoxes += boxes[b_entry]["num"]
            # self.nBoxesPerSKUs.append(boxes[b_entry]["num"])
            # self.dimPerSKUs[b_entry] = boxes[b_entry]["dim"]
            # self.maxDims[0] = max(self.maxDims[0], self.dimPerSKUs[b_entry][0])
            # self.maxDims[1] = max(self.maxDims[1], self.dimPerSKUs[b_entry][1])
            # self.maxDims[2] = max(self.maxDims[2], self.dimPerSKUs[b_entry][2])
            for n in range(self.num_envs):
                self.queued[n][b_entry] = boxes[b_entry]["num"]
                self.nQueued[n] += boxes[b_entry]["num"]
                self.xyAreas[n].append([boxes[b_entry]["dim"][0]*boxes[b_entry]["dim"][1], idx])

        for n in range(self.num_envs):
            self.xyAreas[n].sort(reverse=True)


    def extractItem(self, env_ids):
        '''
        Item extraction module:
        The idea is that based on the extraction policy (In version 1.0, it's simply bigger xy area first), the index of a box that is getting stacked is extracted

        The extracted box index is recorded to self.currBoxIdx,
        and the box's dimensions are recorded to self.obs_buf_bins
        '''

        for idx in range(len(env_ids)):
            currSKU_idx = self.xyAreas[env_ids[idx]][0][1]
            b_entry = "SKU"+str(currSKU_idx+1)
            self.queued[env_ids[idx]][b_entry] -= 1
            self.nQueued[env_ids[idx]] -= 1
            if(self.queued[env_ids[idx]][b_entry]==0):
                self.xyAreas[env_ids[idx]].pop(0)

            currBoxIdx = 0
            for box_entry in boxes:
                if(box_entry==b_entry):
                    currBoxIdx += self.queued[env_ids[idx]][b_entry] #extracting the box from top
                    break
                else:
                    currBoxIdx += nBoxesPerSKUs[int(box_entry[-1])-1]

            #update the obs_buf with normalized curr box dims
            for b in range(nBoxes):
                self.obs_buf_bins[env_ids[idx], 12*b + 3] = dimPerSKUs[b_entry][0]/(self.action_scale[0])
                self.obs_buf_bins[env_ids[idx], 12*b + 4] = dimPerSKUs[b_entry][1]/(self.action_scale[1])
                self.obs_buf_bins[env_ids[idx], 12*b + 5] = dimPerSKUs[b_entry][2]/(self.env_boundaries.high[2] - self.env_boundaries.low[2])

                self.obs_buf_bins[env_ids[idx], 12*b + 9] = dimPerSKUs[b_entry][0]/(self.action_scale[0])
                self.obs_buf_bins[env_ids[idx], 12*b + 10] = dimPerSKUs[b_entry][1]/(self.action_scale[1])
                self.obs_buf_bins[env_ids[idx], 12*b + 11] = dimPerSKUs[b_entry][2]/(self.env_boundaries.high[2] - self.env_boundaries.low[2])
            #print(currBoxIdx)

            self.currBoxIdx[env_ids[idx]] = currBoxIdx

@torch.jit.script
def compute_binpacking_reward(total_stacked_vol, occupiedVol, valid_ids):
    '''
    This calculates the final reward when box(es) are stacked
    The reward will be assigned based on the overall volume efficiency of the current stacking
        
    Returns
    ________
    Tensor [numEnv (valid), ]: reward
    '''  

    curr_occupied_volume = torch.abs((occupiedVol[valid_ids,3] - occupiedVol[valid_ids,0])*(occupiedVol[valid_ids,4] - occupiedVol[valid_ids,1])*(occupiedVol[valid_ids,5] - occupiedVol[valid_ids,2]))
    reward = 10*total_stacked_vol[valid_ids]/curr_occupied_volume

    return reward
