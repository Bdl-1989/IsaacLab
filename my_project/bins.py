# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot."""


import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg,AssetBaseCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
import omni.isaac.core.utils.prims as prims_utils
##
# Configuration
##
 

import numpy as np

binpacking_cfg = {
            "boxes":{  
                'nSKU': 4,  
                'SKU1': {  
                    'num': 1,  
                    'dim': [2.0, 1.0, 1.0]  
                },  
                'SKU2': {  
                    'num': 1,  
                    'dim': [1.5, 1.0, 1.0]  
                },  
                'SKU3': {  
                    'num': 2,  
                    'dim': [1.0, 1.0, 1.0]  
                },  
                'SKU4': {  
                    'num': 2,  
                    'dim': [2.0, 2.0, 2.0]  
                }  
            },

            'pallet':{  
                'x_min': 0,  
                'x_max': 5,  
                'y_min': 0,  
                'y_max': 5,  
                'z_min': 0.4,  
                'z_max': 20.4  
            },

            'base':{  
                'x_size': 5,  
                'y_size': 5,  
                'z_size': 0.4  
            },

            'env':{
                'numObservations':0,
                'numActions':0,
                'envSpacing': 5.0,
                'clipObservations': 1.0,
                'clipActions': 1.0
            }
        }



x_pallet_min = binpacking_cfg["pallet"]["x_min"]
x_pallet_max = binpacking_cfg["pallet"]["x_max"]
y_pallet_min = binpacking_cfg["pallet"]["y_min"]
y_pallet_max = binpacking_cfg["pallet"]["y_max"]
z_pallet_min = binpacking_cfg["pallet"]["z_min"]
z_pallet_max = binpacking_cfg["pallet"]["z_max"]   
# env_boundaries = spaces.Box(low=np.array([x_pallet_min, y_pallet_min, z_pallet_min]), high=np.array([x_pallet_max, y_pallet_max, z_pallet_max]), dtype=np.float32)
action_bias = [float(x_pallet_min + x_pallet_max)/2, float(y_pallet_min + y_pallet_max)/2]
action_scale = [float(x_pallet_max - x_pallet_min), float(y_pallet_max - y_pallet_min)]



    # Define and place the pallet in each environment
# usd_path = "/mnt/sdb2/Omniverse/IsaacLab/source/standalone/tutorials/06_binpack/pallet_modified.usd"
usd_path = "/mnt/sdb2/Omniverse/IsaacLab/source/standalone/tutorials/06_binpack/pal.usd"

# PALLET_CFG = RigidObjectCfg(
#     prim_path="{ENV_REGEX_NS}/Pallet",
#     spawn=sim_utils.UsdFileCfg(
#             usd_path=usd_path,
#             rigid_props=sim_utils.RigidBodyPropertiesCfg(),
#             mass_props=sim_utils.MassPropertiesCfg(mass=10),
#             collision_props=sim_utils.CollisionPropertiesCfg(),
#     ),
#     init_state=RigidObjectCfg.InitialStateCfg(
#         pos=[2.0, 2.95, 1.5],
#         rot=[float(np.sqrt(2)/2), 0.0, 0.0, float(np.sqrt(2)/2)]
#     ),
# )



PALLET_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(
            size = (binpacking_cfg["base"]["x_size"],
                    binpacking_cfg["base"]["y_size"],
                    binpacking_cfg["base"]["z_size"]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=10),
            collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=[2.5, 2.5, 0.2],
        rot=[float(np.sqrt(2)/2), 0.0, 0.0, float(np.sqrt(2)/2)]
    ),
)

 

spacing=5

nSKU = binpacking_cfg["boxes"]["nSKU"]
boxes = dict()
for i in range(nSKU):
    boxes["SKU"+str(i+1)] = binpacking_cfg["boxes"]["SKU"+str(i+1)]

nBoxes = 0 #number of total boxes
dimPerSKUs = {} #dimensions of each SKU 
maxDims = [0.0, 0.0, 0.0] #max size of the dimensions
nBoxesPerSKUs = [] #number of boxes for each SKU
    
for idx, b_entry in enumerate(boxes):
    nBoxes += boxes[b_entry]["num"]
    nBoxesPerSKUs.append(boxes[b_entry]["num"])
    dimPerSKUs[b_entry] = boxes[b_entry]["dim"]
    maxDims[0] = max(maxDims[0], dimPerSKUs[b_entry][0])
    maxDims[1] = max(maxDims[1], dimPerSKUs[b_entry][1])
    maxDims[2] = max(maxDims[2], dimPerSKUs[b_entry][2])

bins_cfg_dict={}
initPositions = []
# bins_cfg = sim_utils.CuboidCfg()

for i, nBox in enumerate(nBoxesPerSKUs):
    curr_SKU_size = (dimPerSKUs["SKU"+str(i+1)][0], dimPerSKUs["SKU"+str(i+1)][1], dimPerSKUs["SKU"+str(i+1)][2])
    
    color = np.clip(np.abs(np.random.normal(loc=0.5, scale=0.5, size=2)), 0.0, 1.0)
    color = [float(c) for c in color] 
    curr_SKU_color = (i/nSKU, color[0], color[1])
    
    
 
    
    bin_cfg = RigidObjectCfg(
        spawn=sim_utils.CuboidCfg(
                size = curr_SKU_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=curr_SKU_color, metallic=0.2),
            ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    # bin_object = RigidObject(cfg=bin_cfg)
 
    for b in range(nBox):
        initPosition = (-spacing + curr_SKU_size[0]/2 + i*maxDims[0], -spacing + curr_SKU_size[1]/2, curr_SKU_size[2]/2 + b*curr_SKU_size[2])
        
        initPositions.append(initPosition)
        bin = bin_cfg.copy()
        bin.prim_path = "{ENV_REGEX_NS}/SKU" + str(i+1) + "_" + str(b+1)
        bin.init_state = RigidObjectCfg.InitialStateCfg(pos=initPosition)


        bins_cfg_dict[f'SKU{i+1}_{b+1}'] = bin


