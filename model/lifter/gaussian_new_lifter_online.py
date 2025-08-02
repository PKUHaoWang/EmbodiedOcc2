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

from ..encoder.gaussianformer.utils import safe_get_quaternion, batch_quaternion_multiply, get_rotation_matrix, safe_sigmoid

LOGIT_MAX = 0.99

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
    
class TagAwareLayer(nn.Module):
    def __init__(self):
        super(TagAwareLayer, self).__init__()
        self.fc1 = nn.Linear(24, 32)   
        self.fc2 = nn.Linear(32, 16) 
        self.fc3 = nn.Linear(16, 1) 
        self.relu = nn.ReLU()       

    def forward(self, x):
        x = self.relu(self.fc1(x)) 
        x = self.relu(self.fc2(x))  
        x = self.fc3(x)            
        return x

@MODELS.register_module()
class GaussianNewLifterOnline(nn.Module):
    def __init__(
        self,
        reuse_instance_feature,
        embed_dims, # 96
        num_anchor=25600, # 21600
        anchor=None,
        anchor_grad=False, 
        feat_grad=False,
        semantic_dim=0, # 13
        include_opa=True,
        include_v=False,
        with_unc=True,  
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
        self.reuse_instance_feature = reuse_instance_feature
        # self.tag_aware_layer = TagAwareLayer()
        self.with_unc = with_unc

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
    
    def forward(self, scenemeta, gaussian_pool, instance_feature_pool, global_mask_thistime, flag_depthbranch, flag_depthanything_as_gt, depthnet_output, mlvl_img_feats, metas):
        
        batch_size = mlvl_img_feats[0].shape[0]
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))
        
        gaussian_pool_old = gaussian_pool
        instance_feature_pool_old = instance_feature_pool
        
        if gaussian_pool.shape[1] > 0:
            # get anchor from the pool
            gaussian_pool_old = gaussian_pool_old.squeeze(0)
            if self.reuse_instance_feature:
                instance_feature_pool_old = instance_feature_pool_old.squeeze(0)
            gaussian_pool_xyz = gaussian_pool_old[:, :3] # world coord
            
            world2cam = metas[0]['world2cam'].to(torch.float32)
            gaussian_pool_xyz_ = torch.cat([gaussian_pool_xyz, torch.ones((gaussian_pool_xyz.shape[0], 1), device=gaussian_pool_xyz.device)], dim=1).to(torch.float32)
            gaussian_pool_cam_ = (world2cam @ gaussian_pool_xyz_.unsqueeze(-1)).squeeze(-1)
            gaussian_pool_cam = gaussian_pool_cam_[:, :3]
            cam_k = metas[0]['cam_k'].to(torch.float32)
            gaussian_pool_cam_x = gaussian_pool_cam[:, 0]
            gaussian_pool_cam_y = gaussian_pool_cam[:, 1]
            gaussian_pool_cam_z = gaussian_pool_cam[:, 2]
            mask1 = gaussian_pool_cam_z > 1e-6
            gaussian_pool_cam_z[~mask1] = 1e-6
            gaussian_pool_cam_x = gaussian_pool_cam_x / gaussian_pool_cam_z
            gaussian_pool_cam_y = gaussian_pool_cam_y / gaussian_pool_cam_z
            gaussian_pool_pix_x = torch.floor(cam_k[0, 0] * gaussian_pool_cam_x + cam_k[0, 2]).to(torch.int32)
            gaussian_pool_pix_y = torch.floor(cam_k[1, 1] * gaussian_pool_cam_y + cam_k[1, 2]).to(torch.int32)
            mask2 = (gaussian_pool_pix_x >= 0) & (gaussian_pool_pix_x < 640) & (gaussian_pool_pix_y >= 0) & (gaussian_pool_pix_y < 480)
            mask_all = mask1 & mask2
            
            vox_near_world = metas[0]['vox_origin']
            vox_far_world = metas[0]['vox_origin'] + metas[0]['scene_size']
            epsilon = 1e-3
            gaussian_pool_mask = (gaussian_pool_xyz[:, 0] > (vox_near_world[0]+epsilon)) & (gaussian_pool_xyz[:, 0] < (vox_far_world[0]-epsilon)) & (gaussian_pool_xyz[:, 1] > (vox_near_world[1]+epsilon)) & (gaussian_pool_xyz[:, 1] < (vox_far_world[1]-epsilon)) & (gaussian_pool_xyz[:, 2] > (vox_near_world[2]+epsilon)) & (gaussian_pool_xyz[:, 2] < (vox_far_world[2]-epsilon))
            gaussian_pool_mask_detach = mask_all & gaussian_pool_mask
            
            pool_splat_tag = gaussian_pool_old[..., 24]
            pool_splat_tag[gaussian_pool_mask] = 1
            gaussian_pool_old[..., 24] = pool_splat_tag
            tag_mask = torch.zeros(gaussian_pool_old.shape[0], device=gaussian_pool.device)
            pool_tag = gaussian_pool_old[..., 23]
            if self.with_unc:
                tag_mask[pool_tag > 0] = pool_tag[pool_tag > 0]
            else:
                tag_mask[pool_tag == 1] = 0.5
            # tag_mask[pool_tag == 1] = 0.7
            # tag_mask[pool_tag == 1] = 0.3
            tag_mask[~gaussian_pool_mask_detach] = 1
            tag = tag_mask[gaussian_pool_mask]
            # tag = tag_mask[gaussian_pool_mask_detach]
            
            gaussian_reused = gaussian_pool_old[gaussian_pool_mask]
            # gaussian_reused = gaussian_pool_old[gaussian_pool_mask_detach]
            # gaussian_reused = gaussian_pool_old[gaussian_pool_mask]
            # gaussian_unchange = gaussian_pool_old[~gaussian_pool_mask]
            gaussian_unchange = gaussian_pool_old[~gaussian_pool_mask_detach]
            gaussian_pool_new = gaussian_unchange.unsqueeze(0)
            if self.reuse_instance_feature:
                instance_feature_reused = instance_feature_pool_old[gaussian_pool_mask]
                # instance_feature_reused = instance_feature_pool_old[gaussian_pool_mask_detach]
                instance_feature_unchange = instance_feature_pool_old[~gaussian_pool_mask_detach]
                instance_feature_pool_new = instance_feature_unchange.unsqueeze(0)
            else:
                instance_feature_pool_new = instance_feature_pool_old
            
            # gaussian_reused_tag = gaussian_reused[..., 23]
            gaussian_reused_tag = tag
            # worldgaussian to camanchor
            gaussian_reused = gaussian_reused[..., :-2]
            gaussian_means_world = gaussian_reused[:, :3]
            gaussian_scales = gaussian_reused[:, 3:6]
            gaussian_rotations_world = gaussian_reused[:, 6:10]
            gaussian_opacities = gaussian_reused[:, 10:11]
            gaussian_semantics = gaussian_reused[:, 11:]
            
            gaussian_means_world_ = torch.cat([gaussian_means_world, torch.ones((gaussian_means_world.shape[0], 1), device=gaussian_means_world.device)], dim=1)
            world2cam = metas[0]['world2cam'].to(torch.float32)
            gaussian_means_cam = (world2cam @ gaussian_means_world_.unsqueeze(-1)).squeeze(-1)
            gaussian_means_cam = gaussian_means_cam[:, :3]
            nyu_pc_range = metas[0]['cam_vox_range']
            gaussian_mask = (gaussian_means_cam[:, 0] >= nyu_pc_range[0]) & (gaussian_means_cam[:, 0] <= nyu_pc_range[3]) & (gaussian_means_cam[:, 1] >= nyu_pc_range[1]) & (gaussian_means_cam[:, 1] <= nyu_pc_range[4]) & (gaussian_means_cam[:, 2] >= nyu_pc_range[2]) & (gaussian_means_cam[:, 2] <= nyu_pc_range[5])
            gaussian_means_cam = gaussian_means_cam[gaussian_mask]
            gaussian_scales = gaussian_scales[gaussian_mask]
            gaussian_rotations_world = gaussian_rotations_world[gaussian_mask]
            gaussian_opacities = gaussian_opacities[gaussian_mask]
            gaussian_semantics = gaussian_semantics[gaussian_mask]
            
            w2c_rot = metas[0]['world2cam'][:3, :3].to(torch.float32)
            w2c_quat = safe_get_quaternion(w2c_rot.unsqueeze(0)).squeeze(0)
            gaussian_rotations_cam = batch_quaternion_multiply(w2c_quat, gaussian_rotations_world)
            
            f_l_x = torch.tensor(metas[0]['cam_k'][0, 0]).cuda()
            f_l_y = torch.tensor(metas[0]['cam_k'][1, 1]).cuda()
            c_x = torch.tensor(metas[0]['cam_k'][0, 2]).cuda()
            c_y = torch.tensor(metas[0]['cam_k'][1, 2]).cuda()
            gaussian_pix_x = f_l_x * gaussian_means_cam[:, 0] / gaussian_means_cam[:, 2] + c_x
            gaussian_pix_y = f_l_y * gaussian_means_cam[:, 1] / gaussian_means_cam[:, 2] + c_y
            anchor_depth_real = gaussian_means_cam[:, 2]
            
            anchor_means_cam = (gaussian_means_cam - nyu_pc_range[:3]) / (nyu_pc_range[3:] - nyu_pc_range[:3]) # 0-1
            anchor_reused = torch.cat([anchor_means_cam, gaussian_scales, gaussian_rotations_cam, gaussian_opacities, gaussian_semantics], dim=-1).unsqueeze(0)
            
        
        if flag_depthbranch:
            if flag_depthanything_as_gt:
                z = depthnet_output
            else:
                z = metas[0]['depth_gt']
            
        anchor_pix_x = torch.clamp(gaussian_pix_x, 0, 639)
        anchor_pix_y = torch.clamp(gaussian_pix_y, 0, 479)
        anchor_pix_x = anchor_pix_x.long()
        anchor_pix_y = anchor_pix_y.long()
        anchor_depth_from_z = z[anchor_pix_y, anchor_pix_x]
        anchor_depth_feature = torch.stack((anchor_depth_from_z, anchor_depth_real), dim=-1)    
        anchor_depth_feature = self.depth_aware_layer(anchor_depth_feature)  
        
        if anchor_reused.shape[1] > 0:
            anchor_reused = anchor_reused.squeeze(0)
            anchor_reused_xyz = safe_inverse_sigmoid(torch.clamp(anchor_reused[..., :3], 0.001, 0.999))
            anchor_reused_scale = safe_inverse_sigmoid(anchor_reused[..., 3:6])
            anchor_reused_rot = anchor_reused[..., 6:10]
            anchor_reused_opa = safe_inverse_sigmoid(anchor_reused[..., 10:11])
            anchor_reused_sem = anchor_reused[..., 11:]
            anchor_reused = torch.cat([anchor_reused_xyz, anchor_reused_scale, anchor_reused_rot, anchor_reused_opa, anchor_reused_sem], dim=-1).unsqueeze(0)
            anchor_reused_tag = gaussian_reused_tag.unsqueeze(0).unsqueeze(-1)
            
        
        anchor = anchor_reused
        anchor_tag = anchor_reused_tag
        
        instance_feature = self.instance_feature_layer(anchor)
        # if self.reuse_instance_feature:
        #     instance_mask = (anchor_tag == 0.5).squeeze(-1)
        #     instance_feature_reused = instance_feature_reused.unsqueeze(0)
        #     instance_feature[instance_mask] = instance_feature_reused[instance_mask]
        
        instance_feature = instance_feature + anchor_depth_feature.unsqueeze(0)
        
        return anchor, instance_feature, None, None, None, gaussian_pool_new, anchor_tag, instance_feature_pool_new