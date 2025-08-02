from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import Linear, Scale

from .utils import linear_relu_ln, safe_sigmoid, GaussianPrediction
import torch, torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from pytorch3d.ops.knn import knn_points


@MODELS.register_module()
class OnlineRefineLayer(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        pc_range=None,
        scale_range=None,
        restrict_xyz=False,  # True
        unit_xyz=None,  # [4.0, 4.0, 1.0]
        refine_manual=None,  # [0, 1, 2]
        semantic_dim=0,  # 13
        semantics_activation='softmax',  # Identity
        include_opa=True,
        include_v=False,
        # Basic normal constraint
        use_normal_constraint=True,
        normal_weight=1.0,
        # Depth-guided normal constraint parameters
        use_depth_guided_normal=False,
        dist_threshold_near=0.1,
        dist_threshold_far=0.25,
        k_neighbors=10,
        # Kappa guidance parameters
        use_kappa_guidance=False,
        min_normal_weight=0.0,
        max_normal_weight=1.0,
        kappa_threshold_low=5.0,
        kappa_threshold_high=20.0,
        # Fusion guidance parameters
        use_fusion_guidance=True,
        fusion_strategy='product',  
        fusion_depth_weight=0.7,
        fusion_kappa_weight=0.3,
        region_adaptive=False,
        fusion_confidence_scale=2.0,
        # Online parameters
        if_frozen=True,
        threshold=0.0
    ):
        super(OnlineRefineLayer, self).__init__()
        self.embed_dims = embed_dims
        self.output_dim = 10 + int(include_opa) + semantic_dim + int(include_v) * 2
        self.semantic_start = 10 + int(include_opa)
        self.semantic_dim = semantic_dim
        self.include_opa = include_opa
        self.semantics_activation = semantics_activation
        self.pc_range = pc_range
        self.scale_range = scale_range
        self.restrict_xyz = restrict_xyz
        self.unit_xyz = unit_xyz
        self.if_frozen = if_frozen
        
        if restrict_xyz:
            assert unit_xyz is not None
            unit_prob = [unit_xyz[i] / (pc_range[i + 3] - pc_range[i]) for i in range(3)]
            unit_sigmoid = [4 * unit_prob[i] for i in range(3)]
            self.unit_sigmoid = unit_sigmoid
        
        assert isinstance(refine_manual, list)
        self.refine_state = refine_manual
        assert all([self.refine_state[i] == i for i in range(len(self.refine_state))])

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )
        
        # Basic normal constraint parameters
        self.use_normal_constraint = use_normal_constraint
        self.normal_weight = normal_weight
        
        # Depth-guided normal constraint parameters
        self.use_depth_guided_normal = use_depth_guided_normal
        self.dist_threshold_near = dist_threshold_near
        self.dist_threshold_far = dist_threshold_far
        self.k_neighbors = k_neighbors
        
        # Kappa guidance parameters
        self.use_kappa_guidance = use_kappa_guidance
        self.min_normal_weight = min_normal_weight
        self.max_normal_weight = max_normal_weight
        self.kappa_threshold_low = kappa_threshold_low
        self.kappa_threshold_high = kappa_threshold_high
        
        # Fusion guidance parameters
        self.use_fusion_guidance = use_fusion_guidance
        self.fusion_strategy = fusion_strategy
        self.fusion_depth_weight = fusion_depth_weight
        self.fusion_kappa_weight = fusion_kappa_weight
        self.region_adaptive = region_adaptive
        self.fusion_confidence_scale = fusion_confidence_scale
        
        self.threshold = threshold
        
    def depth_to_point_cloud(self, depth_map, cam_k, cam2world=None):
        """
        Convert depth map to 3D point cloud
        
        Args:
            depth_map: [H, W] depth map
            cam_k: [3, 3] camera intrinsic matrix
            cam2world: [4, 4] camera to world transformation matrix, if None returns points in camera coordinates
        
        Returns:
            point_cloud: [N, 3] point cloud coordinates (world coordinates if cam2world is provided)
            valid_mask: [N] mask of valid points
        """
        height, width = depth_map.shape
        device = depth_map.device
        
        # Create pixel coordinate grid
        v, u = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        
        # Flatten to 1D coordinates
        u_flat = u.reshape(-1).float()
        v_flat = v.reshape(-1).float()
        depth_flat = depth_map.reshape(-1)
        
        # Camera intrinsics
        fx = cam_k[0, 0]
        fy = cam_k[1, 1]
        cx = cam_k[0, 2]
        cy = cam_k[1, 2]
        
        # Compute 3D points in camera coordinates
        z = depth_flat
        x = (u_flat - cx) * z / fx
        y = (v_flat - cy) * z / fy
        
        # Create point cloud in camera coordinates
        points_cam = torch.stack([x, y, z], dim=-1)
        
        # Valid mask (depth > 0)
        valid_mask = depth_flat > 0
        
        # Convert to world coordinates if needed
        if cam2world is not None:            
            # Add homogeneous coordinate
            points_cam_homo = torch.cat([
                points_cam,
                torch.ones((points_cam.shape[0], 1), device=device)
            ], dim=1).to(torch.float32)
            
            # Transform to world coordinates
            points_world_homo = (cam2world @ points_cam_homo.unsqueeze(-1)).squeeze(-1)
            points_world = points_world_homo[:, :3]
            
            return points_world, valid_mask
        
        return points_cam, valid_mask
    
    def compute_normal_weights(self, xyz_world, metas):
        """
        Compute normal constraint weights for each Gaussian point
        
        Args:
            xyz_world: [B, N, 3] Gaussian points coordinates in world space
            metas: dictionary containing camera parameters and depth information
        
        Returns:
            normal_weights: [B, N] normal constraint weights for each point
            valid_mask: [B, N] mask of valid points
        """
        B, N = xyz_world.shape[:2]
        device = xyz_world.device
        
        batch_weights = []
        batch_masks = []
        
        for b in range(B):
            # Get depth map and camera parameters
            depth_map = metas[b]['depth_pred']  # [H, W]
            cam_k = torch.tensor(metas[b]['cam_k']).to(device)
            world2cam = metas[b]['world2cam'].to(torch.float32)
            cam2world = metas[b]['cam2world'].to(torch.float32)
            
            # Convert Gaussian points from camera-centered to global world coordinates
            cam_position = cam2world[:3, 3]  # Camera position in world coordinates
            
            # Transform points from camera-centered to world coordinates
            xyz_global = torch.zeros_like(xyz_world[b])
            xyz_global = torch.matmul(xyz_world[b], cam2world[:3, :3].T)
            xyz_global = xyz_global + cam_position
            
            # Convert depth map to point cloud (in world coordinates)
            depth_points, valid_depth_mask = self.depth_to_point_cloud(
                depth_map, 
                cam_k, 
                cam2world
            )  # [H*W, 3], [H*W]
            
            # Keep only valid depth points
            depth_points_valid = depth_points[valid_depth_mask]
            
            if depth_points_valid.shape[0] == 0:
                # Return zero weights if no valid depth points
                weights = torch.zeros(N, device=device)
                masks = torch.zeros(N, dtype=torch.bool, device=device)
                batch_weights.append(weights)
                batch_masks.append(masks)
                continue
            
            # Compute KNN (both sets of points are in world coordinates)
            dists, idx, _ = knn_points(
                xyz_global.unsqueeze(0),  # [1, N, 3]
                depth_points_valid.unsqueeze(0),  # [1, M, 3]
                K=self.k_neighbors
            )
            
            # Get nearest neighbor distance
            min_dists = dists.squeeze(0)[:, 0]  # [N]
            
            # Compute weights (linear interpolation)
            weights = torch.clamp(
                (self.dist_threshold_far - min_dists) / (self.dist_threshold_far - self.dist_threshold_near), 
                0.0, 
                1.0
            )
            
            # Create valid mask (points with distance < far threshold)
            masks = min_dists < self.dist_threshold_far
            
            batch_weights.append(weights)
            batch_masks.append(masks)
        
        # Stack batch results
        normal_weights = torch.stack(batch_weights)  # [B, N]
        valid_mask = torch.stack(batch_masks)  # [B, N]
        
        return normal_weights, valid_mask
    
    def project_points_to_image(self, points_xyz_logits, metas):
        """
        Project 3D points to 2D image plane
        
        Args:
            points_xyz_logits: [N, 3] xyz coordinates before sigmoid
            metas: dictionary containing camera parameters
        
        Returns:
            pixel_coords: [N, 2] projected pixel coordinates (x, y)
            valid_mask: [N] validity mask
        """
        # 1. Convert logits to world coordinates
        world_near = metas[0]['vox_origin']
        world_far = metas[0]['vox_origin'] + metas[0]['scene_size']
        
        # Sigmoid and scale to world coordinates
        points_xyz_01 = safe_sigmoid(points_xyz_logits)
        points_xyz_world = points_xyz_01 * (world_far - world_near) + world_near  # [N, 3]

        # 2. World to camera coordinates
        world2cam = metas[0]['world2cam'].to(torch.float32)
        
        # Add homogeneous coordinate
        points_xyz_world_homo = torch.cat([
            points_xyz_world, 
            torch.ones((points_xyz_world.shape[0], 1), device=points_xyz_world.device)
        ], dim=1).to(torch.float32)  # [N, 4]
        
        # Apply world2cam transform
        points_xyz_cam = (world2cam @ points_xyz_world_homo.unsqueeze(-1)).squeeze(-1)  # [N, 4]
        points_xyz_cam = points_xyz_cam[:, :3]  # [N, 3]

        # 3. Project camera coordinates to pixel coordinates
        # Get camera intrinsics
        f_l_x = torch.tensor(metas[0]['cam_k'][0, 0]).to(points_xyz_logits.device)
        f_l_y = torch.tensor(metas[0]['cam_k'][1, 1]).to(points_xyz_logits.device)
        c_x = torch.tensor(metas[0]['cam_k'][0, 2]).to(points_xyz_logits.device)
        c_y = torch.tensor(metas[0]['cam_k'][1, 2]).to(points_xyz_logits.device)

        # Compute pixel coordinates
        pixel_x = f_l_x * points_xyz_cam[:, 0] / points_xyz_cam[:, 2] + c_x
        pixel_y = f_l_y * points_xyz_cam[:, 1] / points_xyz_cam[:, 2] + c_y

        # 4. Constrain pixel coordinates to image bounds
        img_h, img_w = 480, 640
        
        in_image_mask = (
            (pixel_x >= 0) & (pixel_x < img_w) &
            (pixel_y >= 0) & (pixel_y < img_h)
        )

        # 5. Create validity mask
        depth_positive_mask = (points_xyz_cam[:, 2] > 0)
        valid_mask = in_image_mask & depth_positive_mask
        
        pixel_coords = torch.stack([pixel_x, pixel_y], dim=-1)  # [N, 2]
        
        return pixel_coords, valid_mask
    
    def get_projected_normals(self, xyz_logits, normals, metas):
        """
        Get normal directions for 3D points
        
        Args:
            xyz_logits: [B, N, 3] xyz coordinates before sigmoid
            normals: [B, 3, H, W] predicted normal map
            metas: camera parameters
        
        Returns:
            point_normals: [B, N, 3] normal directions for each point
        """
        B, N = xyz_logits.shape[:2]
        point_normals = []
        
        for b in range(B):
            # Project to pixel coordinates
            pixel_coords, valid_mask = self.project_points_to_image(
                xyz_logits[b], 
                metas
            )  # [N, 2], [N]
            
            # Get normals
            batch_normals = []
            for n in range(N):
                if valid_mask[n]:
                    x = int(pixel_coords[n, 0])
                    y = int(pixel_coords[n, 1])
                    normal = normals[b, :, y, x]
                else:
                    normal = torch.zeros(3, device=xyz_logits.device)
                batch_normals.append(normal)
                
            point_normals.append(torch.stack(batch_normals))
            
        return torch.stack(point_normals)  # [B, N, 3]

    def apply_normal_constraint(self, delta_xyz, normals, weights=None, valid_mask=None):
        """Apply normal constraint to position updates
        
        Args:
            delta_xyz: [B, N, 3] original position updates
            normals: [B, N, 3] corresponding normal directions
            weights: [B, N] normal constraint weights per point, None uses global weight
            valid_mask: [B, N] valid point mask, None means all points are valid
        
        Returns:
            constrained_delta: [B, N, 3] position updates with normal constraint
        """
        B, N = delta_xyz.shape[:2]
        device = delta_xyz.device
        
        # Use global weight if not provided
        if weights is None:
            weights = torch.ones(B, N, device=device) * self.normal_weight
        
        # All points valid if mask not provided
        if valid_mask is None:
            valid_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        
        # Decompose delta_xyz into normal and tangent components
        normal_component = (delta_xyz * normals).sum(dim=-1, keepdim=True) * normals
        tangent_component = delta_xyz - normal_component
        
        # Expand weights for broadcasting
        weights = weights.unsqueeze(-1)  # [B, N, 1]
        valid_mask = valid_mask.unsqueeze(-1)  # [B, N, 1]
        
        # Mix components based on weights, only apply to valid points
        constrained_delta = torch.where(
            valid_mask,
            weights * tangent_component + (1 - weights) * delta_xyz,
            delta_xyz
        )
        
        return constrained_delta
    
    def get_kappa_guided_weight(self, xyz_logits, kappa_map, metas):
        """Adjust normal_weight dynamically based on kappa values
        
        Args:
            xyz_logits: [B, N, 3] xyz coordinates before sigmoid
            kappa_map: [B, 1, H, W] predicted kappa map
            metas: camera parameters
        
        Returns:
            weights: [B, N] normal_weight for each point
        """
        B, N = xyz_logits.shape[:2]
        point_weights = []
        
        for b in range(B):
            # Project to pixel coordinates
            pixel_coords, valid_mask = self.project_points_to_image(
                xyz_logits[b], 
                metas
            )  # [N, 2], [N]
            
            # Get kappa value for each point
            batch_weights = []
            for n in range(N):
                if valid_mask[n]:
                    x = int(pixel_coords[n, 0])
                    y = int(pixel_coords[n, 1])
                    kappa_value = kappa_map[b, 0, y, x].item()
                    
                    # Compute normal_weight based on kappa
                    if kappa_value <= self.kappa_threshold_low:
                        weight = self.min_normal_weight
                    elif kappa_value >= self.kappa_threshold_high:
                        weight = self.max_normal_weight
                    else:
                        # Linear interpolation
                        ratio = (kappa_value - self.kappa_threshold_low) / (self.kappa_threshold_high - self.kappa_threshold_low)
                        weight = self.min_normal_weight + ratio * (self.max_normal_weight - self.min_normal_weight)
                else:
                    weight = self.min_normal_weight  # Use min weight for invalid points
                    
                batch_weights.append(weight)
                
            point_weights.append(torch.tensor(batch_weights, device=xyz_logits.device))
            
        return torch.stack(point_weights)  # [B, N]
    
    def get_fused_normal_weights(self, xyz_logits, kappa_map, metas, point_semantics=None):
        """Fuse depth and kappa information to compute normal constraint weights
        
        Args:
            xyz_logits: [B, N, 3] xyz coordinates before sigmoid
            kappa_map: [B, 1, H, W] predicted kappa map
            metas: camera parameters
            point_semantics: [B, N, C] point semantic information
        
        Returns:
            fused_weights: [B, N] fused weights
            valid_mask: [B, N] valid point mask
        """
        B, N = xyz_logits.shape[:2]
        device = xyz_logits.device
        
        # 1. Get depth-guided weights
        batch_xyz = []
        for b in range(B):
            nyu_pc_range = metas[b]['cam_vox_range'].to(device)
            xyz = safe_sigmoid(xyz_logits[b])
            x = xyz[:, 0] * (nyu_pc_range[3] - nyu_pc_range[0]) + nyu_pc_range[0]
            y = xyz[:, 1] * (nyu_pc_range[4] - nyu_pc_range[1]) + nyu_pc_range[1]
            z = xyz[:, 2] * (nyu_pc_range[5] - nyu_pc_range[2]) + nyu_pc_range[2]
            xyz_cam = torch.stack([x, y, z], dim=1)
            
            # Convert to world coordinates
            cam2world = metas[b]['cam2world'].to(torch.float32)
            xyz_cam_homo = torch.cat([
                xyz_cam, 
                torch.ones((xyz_cam.shape[0], 1), device=device)
            ], dim=1)
            xyz_world_homo = (cam2world @ xyz_cam_homo.unsqueeze(-1)).squeeze(-1)
            xyz_world = xyz_world_homo[:, :3]
            batch_xyz.append(xyz_world)
        
        xyz_world = torch.stack(batch_xyz)  # [B, N, 3]
        
        # Compute depth-guided weights
        depth_weights, depth_valid_mask = self.compute_normal_weights(xyz_world, metas)
        
        # 2. Get kappa-guided weights
        kappa_weights = self.get_kappa_guided_weight(xyz_logits, kappa_map, metas)
        
        # 3. Fuse weights using product strategy
        fused_weights = depth_weights * kappa_weights
        
        # 4. Ensure weights are in valid range
        fused_weights = torch.clamp(fused_weights, 0.0, 1.0)
        
        return fused_weights, depth_valid_mask
    
    
    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        metas,
        anchor_confidence=None,
        mc_samples=3
    ):
        output = self.layers(instance_feature + anchor_embed)  # [B, N, output_dim]
        anchor_update_ratio = 1 - anchor_confidence
        if self.if_frozen:
            anchor_update_ratio[anchor_update_ratio != 1] = 0
        else:
            anchor_update_ratio[anchor_update_ratio <= self.threshold] = 0
        
        # Get initial delta_xyz
        if self.restrict_xyz:
            delta_xyz_sigmoid = output[..., :3]
            delta_xyz_prob = 2 * safe_sigmoid(delta_xyz_sigmoid) - 1
            delta_xyz = torch.stack([
                delta_xyz_prob[..., 0] * self.unit_sigmoid[0],
                delta_xyz_prob[..., 1] * self.unit_sigmoid[1],
                delta_xyz_prob[..., 2] * self.unit_sigmoid[2]
            ], dim=-1)
        else:
            delta_xyz = output[..., :3]

        # Apply normal constraint if enabled
        if self.use_normal_constraint and 'normals' in metas[0] and metas[0]['normals'] is not None:
            # Get normal predictions
            pred_norm = metas[0]['normals']  # [B, 3, H, W]
            
            # Get current xyz logits
            xyz_logits = anchor[..., :3]
            
            # Get corresponding normals
            point_normals = self.get_projected_normals(
                xyz_logits,
                pred_norm,
                metas
            )
            
            # Initialize weights and valid mask
            weights = None
            normal_valid_mask = None

            # Fusion mode: use both depth and kappa guidance
            if (self.use_fusion_guidance and 
                'kappas' in metas[0] and metas[0]['kappas'] is not None and
                'depth_pred' in metas[0] and metas[0]['depth_pred'] is not None):
                
                # Get kappa map
                kappa_map = metas[0]['kappas']  # [B, 1, H, W]
                
                # Fuse depth and kappa guidance
                weights, normal_valid_mask = self.get_fused_normal_weights(
                    xyz_logits, 
                    kappa_map,
                    metas,
                )
            
            # Apply normal constraint
            delta_xyz = self.apply_normal_constraint(
                delta_xyz, 
                point_normals,
                weights=weights,
                valid_mask=normal_valid_mask
            )
        
        # Concatenate constrained delta_xyz with other outputs
        output = torch.cat([delta_xyz, output[..., 3:]], dim=-1)
        
        if len(self.refine_state) > 0:
            refined_part_output = output[..., self.refine_state] * anchor_update_ratio + anchor[..., self.refine_state]
            output = torch.cat([refined_part_output, output[..., len(self.refine_state):]], dim=-1)
        
        delta_scale = output[..., 3:6]
        scale_final = anchor[..., 3:6] + delta_scale * anchor_update_ratio 
        output = torch.cat([output[..., :3], scale_final, output[..., 6:]], dim=-1)

        rot = torch.nn.functional.normalize(output[..., 6:10], dim=-1)
        delta_w1, delta_x1, delta_y1, delta_z1 = rot[..., 0], rot[..., 1], rot[..., 2], rot[..., 3]
        
        if anchor_confidence is not None:
            if self.if_frozen:
                delta_w1[anchor_confidence.squeeze(-1) > 0.1] = 1
                delta_x1[anchor_confidence.squeeze(-1) > 0.1] = 0
                delta_y1[anchor_confidence.squeeze(-1) > 0.1] = 0
                delta_z1[anchor_confidence.squeeze(-1) > 0.1] = 0
            else:
                delta_w1[anchor_confidence.squeeze(-1) > 0.8] = 1
                delta_x1[anchor_confidence.squeeze(-1) > 0.8] = 0
                delta_y1[anchor_confidence.squeeze(-1) > 0.8] = 0
                delta_z1[anchor_confidence.squeeze(-1) > 0.8] = 0
        
        w1, x1, y1, z1 = anchor[..., 6], anchor[..., 7], anchor[..., 8], anchor[..., 9]
        w_final = delta_w1 * w1 - delta_x1 * x1 - delta_y1 * y1 - delta_z1 * z1
        x_final = delta_w1 * x1 + delta_x1 * w1 + delta_y1 * z1 - delta_z1 * y1
        y_final = delta_w1 * y1 - delta_x1 * z1 + delta_y1 * w1 + delta_z1 * x1
        z_final = delta_w1 * z1 + delta_x1 * y1 - delta_y1 * x1 + delta_z1 * w1
        w_final = w_final.unsqueeze(-1)
        x_final = x_final.unsqueeze(-1)
        y_final = y_final.unsqueeze(-1)
        z_final = z_final.unsqueeze(-1)
        rot_final = torch.cat([w_final, x_final, y_final, z_final], dim=-1)
        rot_final = torch.nn.functional.normalize(rot_final, dim=-1)
        output = torch.cat([output[..., :6], rot_final, output[..., 10:]], dim=-1)
        
        delta_opa = output[..., 10: (10 + int(self.include_opa))]
        opa_final = anchor[..., 10: (10 + int(self.include_opa))] + delta_opa * anchor_update_ratio 
        output = torch.cat([output[..., :10], opa_final, output[..., (10 + int(self.include_opa)):]], dim=-1)
        
        delta_semantics = output[..., self.semantic_start: (self.semantic_start + self.semantic_dim)]
        semantics_final = anchor[..., self.semantic_start: (self.semantic_start + self.semantic_dim)] + delta_semantics * anchor_update_ratio 
        output = torch.cat([output[..., :self.semantic_start], semantics_final, output[..., (self.semantic_start + self.semantic_dim):]], dim=-1)
        
        # Convert output to format needed by renderer
        nyu_pc_range = metas[0]['cam_vox_range'].to(output.device)
        xyz = safe_sigmoid(output[..., :3])
        xxx = xyz[..., 0] * (nyu_pc_range[3] - nyu_pc_range[0]) + nyu_pc_range[0]
        yyy = xyz[..., 1] * (nyu_pc_range[4] - nyu_pc_range[1]) + nyu_pc_range[1]
        zzz = xyz[..., 2] * (nyu_pc_range[5] - nyu_pc_range[2]) + nyu_pc_range[2]
        xyz = torch.stack([xxx, yyy, zzz], dim=-1)

        gs_scales = safe_sigmoid(output[..., 3:6])
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales

        opas = safe_sigmoid(output[..., 10: (10 + int(self.include_opa))])

        shs = torch.zeros(*instance_feature.shape[:-1], 0, 
            device=instance_feature.device, dtype=instance_feature.dtype)
        
        # Store multiple semantic samples
        semantic_samples = []
        
        # Monte Carlo sampling
        for _ in range(mc_samples):
            x = instance_feature + anchor_embed
            for i, layer in enumerate(self.layers[:-2]):  # Except last Linear and Scale
                x = layer(x)
                if isinstance(layer, nn.ReLU):  # Apply dropout after ReLU
                    x = F.dropout(x, p=0.1, training=True)  # Always True for Monte Carlo sampling
            x = self.layers[-2](x)  # Linear
            semantic_output = self.layers[-1](x)  # Scale
            
            # Extract semantic part
            delta_semantics = semantic_output[..., self.semantic_start: (self.semantic_start + self.semantic_dim)]
            
            # Apply activation function
            if self.semantics_activation == 'softmax':
                semantics = delta_semantics.softmax(dim=-1)
            elif self.semantics_activation == 'softplus':
                semantics = F.softplus(delta_semantics)
            else:
                semantics = delta_semantics
                
            semantic_samples.append(semantics)
        
        # Stack all semantic samples
        stacked_semantics = torch.stack(semantic_samples, dim=0)  # [mc_samples, batch, N, semantic_dim]
        
        # Compute mean semantics
        mean_semantics = torch.mean(stacked_semantics, dim=0)
        
        # Compute class distribution for each sample
        probs = stacked_semantics.mean(dim=0)  # [batch, N, semantic_dim]
        # Ensure probabilities sum to 1 and no negative values
        probs = F.softmax(probs, dim=-1)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
        # Normalize to [0,1] range
        normalized_uncertainty = entropy / torch.log(torch.tensor(self.semantic_dim, device=entropy.device))
        
        semantics = mean_semantics
        
        if self.semantics_activation == 'softmax':
            semantics = semantics.softmax(dim=-1)
        elif self.semantics_activation == 'softplus':
            semantics = F.softplus(semantics)
            
        gaussian = GaussianPrediction(
            means=xyz,
            scales=gs_scales,
            rotations=rot,
            harmonics=shs.unflatten(-1, (3, -1)),
            opacities=opas,
            semantics=semantics
        )
        
        if self.if_frozen:
            return output, gaussian, semantics
        else:
            return output, gaussian, semantics, normalized_uncertainty