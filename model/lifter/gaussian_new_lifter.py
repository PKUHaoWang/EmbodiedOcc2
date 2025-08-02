import numpy as np, math
import torch
from torch import nn
import numpy as np
from PIL import Image
from mmengine import MODELS
import torchvision.transforms as transforms
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import torch.nn.functional as F
from ..encoder.gaussianformer.utils import safe_sigmoid
from ..encoder.gaussianformer.utils import safe_get_quaternion, batch_quaternion_multiply, get_rotation_matrix, safe_sigmoid

LOGIT_MAX = 0.99

def depth2occ(points_world, vox_origin, scene_size):
    vox_near = vox_origin
    vox_far = vox_origin + scene_size
    delta = 1e-3
    points_inroom_mask = (points_world[..., 0] > (vox_near[0]+delta)) & (points_world[..., 0] < (vox_far[0]-delta)) & (points_world[..., 1] > (vox_near[1]+delta)) & (points_world[..., 1] < (vox_far[1]-delta)) & (points_world[..., 2] > (vox_near[2]+delta)) & (points_world[..., 2] < (vox_far[2]-delta))
    points_inroom = points_world[points_inroom_mask]
    grid_size = 0.08
    points_idx = ((points_inroom - vox_near) / grid_size).long()
    occ_label = torch.zeros(60, 60, 36, dtype=torch.float32).to(points_world.device)
    occ_label[points_idx[:, 0], points_idx[:, 1], points_idx[:, 2]] = 1
    return occ_label


def safe_inverse_sigmoid(tensor): # 逆 Sigmoid 函数
    tensor = torch.clamp(tensor, 1 - LOGIT_MAX, LOGIT_MAX)
    return torch.log(tensor / (1 - tensor))

def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False):
    """
    Converts depth map into bin indices
    Args:
        depth_map [torch.Tensor(H, W)]: Depth Map
        mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min [float]: Minimum depth value
        depth_max [float]: Maximum depth value
        num_bins [int]: Number of depth bins
        target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
    Returns:
        indices [torch.Tensor(H, W)]: Depth bin indices
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = (depth_map - depth_min) / bin_size
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = (
            num_bins
            * (torch.log(1 + depth_map) - math.log(1 + depth_min))
            / (math.log(1 + depth_max) - math.log(1 + depth_min))
        )
    else:
        raise NotImplementedError

    if target:
        # Remove indicies outside of bounds (-2, -1, 0, 1, ..., num_bins, num_bins +1) --> (num_bins, num_bins, 0, 1, ..., num_bins, num_bins)
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices[mask] = num_bins

        # Convert to integer
        indices = indices.type(torch.int64)
    return indices.long()

def sample_3d_feature(feature_3d, pix_xy, pix_z, fov_mask):
    """
    Args:
        feature_3d (torch.tensor): 3D feature, shape (C, D, H, W).
        pix_xy (torch.tensor): Projected pix coordinate, shape (N, 2).
        pix_z (torch.tensor): Projected pix depth coordinate, shape (N,).
    
    Returns:
        torch.tensor: Sampled feature, shape (N, C)
    """
    pix_x, pix_y = pix_xy[:, 0][fov_mask], pix_xy[:, 1][fov_mask]
    pix_z = pix_z[fov_mask].to(pix_y.dtype)
    ret = feature_3d[:, pix_z, pix_y, pix_x].T
    return ret

class DepthAwareLayer(nn.Module):
    def __init__(self, embed_dim):
        super(DepthAwareLayer, self).__init__()
        self.fc1 = nn.Linear(2, 64)   
        self.fc2 = nn.Linear(64, 128) 
        self.fc3 = nn.Linear(128, embed_dim) 
        self.relu = nn.ReLU()       

    def forward(self, x):
        x = self.relu(self.fc1(x)) 
        x = self.relu(self.fc2(x))  
        x = self.fc3(x)            
        return x


@MODELS.register_module()
class GaussianNewLifter(nn.Module):
    def __init__(
        self,
        embed_dims, # 96
        num_anchor=25600, # 21600
        anchor=None,
        anchor_grad=False, 
        feat_grad=False,
        semantic_dim=0, # 13
        include_opa=True,
        include_v=False,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        elif anchor is None:
            total_anchor = num_anchor
            xyz = torch.rand(num_anchor, 3, dtype=torch.float)
            assert xyz.shape[0] == num_anchor
            xyz = safe_inverse_sigmoid(xyz)
    
            scale = torch.rand_like(xyz)
            scale = safe_inverse_sigmoid(scale)
            rots = torch.zeros(num_anchor, 4, dtype=torch.float)
            rots[:, 0] = 1
            opacity = safe_inverse_sigmoid(0.1 * torch.ones((
                num_anchor, int(include_opa)), dtype=torch.float))
            semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float)
            self.semantic_dim = semantic_dim
            
            anchor = torch.cat([xyz, scale, rots, opacity, semantic], dim=-1)

        self.num_anchor = total_anchor
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        
        self.instance_feature_layer = nn.Linear(
            3 + 3 + 4 + int(include_opa) + semantic_dim, embed_dims)
        
        self.depth_aware_layer = DepthAwareLayer(embed_dims)
        

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
    
    def forward(self, flag_depthbranch, flag_depthanything_as_gt, depthnet_output, mlvl_img_feats, metas):
        
        batch_size = mlvl_img_feats[0].shape[0]
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1)) # 1, 16200, 23
        world_near = metas[0]['vox_origin']
        world_far = metas[0]['vox_origin'] + metas[0]['scene_size']
        anchor_xyz_logits = anchor[:, :, :3]
        anchor_xyz_01 = safe_sigmoid(anchor_xyz_logits)
        anchor_xyz_world = anchor_xyz_01 * (world_far - world_near) + world_near
        anchor_xyz_world = anchor_xyz_world.squeeze(0)
        world2cam = metas[0]['world2cam'].to(torch.float32)
        anchor_xyz_world_ = torch.cat((anchor_xyz_world, torch.ones((anchor_xyz_world.shape[0], 1), device=anchor_xyz_world.device)), dim=1).to(torch.float32)
        anchor_xyz_cam_ = (world2cam @ anchor_xyz_world_.unsqueeze(-1)).squeeze(-1) # 16200, 3
        anchor_xyz_cam = anchor_xyz_cam_[:, :3]
        
        f_l_x = torch.tensor(metas[0]['cam_k'][0, 0]).cuda()
        f_l_y = torch.tensor(metas[0]['cam_k'][1, 1]).cuda()
        c_x = torch.tensor(metas[0]['cam_k'][0, 2]).cuda()
        c_y = torch.tensor(metas[0]['cam_k'][1, 2]).cuda()    
        anchor_pix_x = f_l_x * anchor_xyz_cam[:, 0] / anchor_xyz_cam[:, 2] + c_x
        anchor_pix_y = f_l_y * anchor_xyz_cam[:, 1] / anchor_xyz_cam[:, 2] + c_y
        
        if flag_depthbranch:
            if flag_depthanything_as_gt:
                z = depthnet_output # 480, 640
            else:
                z = metas[0]['depth_gt']
                
        anchor_pix_x = torch.clamp(anchor_pix_x, 0, 639)
        anchor_pix_y = torch.clamp(anchor_pix_y, 0, 479)
        anchor_pix_x = anchor_pix_x.long()
        anchor_pix_y = anchor_pix_y.long()
        anchor_depth_from_z = z[anchor_pix_y, anchor_pix_x]
        anchor_depth_real = anchor_xyz_cam[:, 2]
        anchor_depth_feature = torch.stack((anchor_depth_from_z, anchor_depth_real), dim=-1)
        anchor_depth_feature = self.depth_aware_layer(anchor_depth_feature)
          
        points_cam = anchor_xyz_cam
        nyu_pc_range = metas[0]['cam_vox_range']
        points_cam = torch.clamp(points_cam, nyu_pc_range[:3], nyu_pc_range[3:])
        points_cam = (points_cam - nyu_pc_range[:3]) / (nyu_pc_range[3:] - nyu_pc_range[:3]) # 0-1
        
        anchor_points = points_cam
        anchor_points = anchor_points.float().unsqueeze(0).to(anchor.device)
        anchor_points_ = anchor[:, :, 3:].clone()
        anchor_rots = anchor_points_[:, :, 3:7]
        w2c_rot = metas[0]['world2cam'][:3, :3].to(torch.float32)
        w2c_quat = safe_get_quaternion(w2c_rot.unsqueeze(0)).squeeze(0)
        anchor_rots_cam = batch_quaternion_multiply(w2c_quat, anchor_rots.squeeze(0)).unsqueeze(0)
        anchor_points_[:, :, 3:7] = anchor_rots_cam
        
        anchor_points = torch.cat([
            safe_inverse_sigmoid(torch.clamp(anchor_points, 0.001, 0.999)),
            anchor_points_
        ], dim=-1)
        
        anchor = anchor_points
        
        instance_feature = self.instance_feature_layer(anchor)
        instance_feature = instance_feature + anchor_depth_feature.unsqueeze(0)
        
        return anchor, instance_feature, None, None, None