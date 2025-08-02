#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch.nn as nn
import torch
import torch.nn.functional as F
from . import _C


class _LocalAggregate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pts,
        points_int,
        means3D,
        means3D_int,
        opacities,
        semantics,
        radii,
        cov3D,
        H, W, D
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            pts,
            points_int,
            means3D,
            means3D_int,
            opacities,
            semantics,
            radii,
            cov3D,
            H, W, D
        )
        # Invoke C++/CUDA rasterizer
        num_rendered, logits, geomBuffer, binningBuffer, imgBuffer = _C.local_aggregate(*args) # todo
        
        # Keep relevant tensors for backward
        ctx.num_rendered = num_rendered
        ctx.H = H
        ctx.W = W
        ctx.D = D
        ctx.save_for_backward(
            geomBuffer, 
            binningBuffer, 
            imgBuffer, 
            means3D,
            pts,
            points_int,
            cov3D,
            opacities,
            semantics
        )
        return logits

    @staticmethod # todo
    def backward(ctx, out_grad):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        H = ctx.H
        W = ctx.W
        D = ctx.D
        geomBuffer, binningBuffer, imgBuffer, means3D, pts, points_int, cov3D, opacities, semantics = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            geomBuffer,
            binningBuffer,
            imgBuffer,
            H, W, D,
            num_rendered,
            means3D,
            pts,
            points_int,
            cov3D,
            opacities,
            semantics,
            out_grad)

        # Compute gradients for relevant tensors by invoking backward method
        means3D_grad, opacity_grad, semantics_grad, cov3D_grad = _C.local_aggregate_backward(*args)

        grads = (
            None,
            None,
            means3D_grad,
            None,
            opacity_grad,
            semantics_grad,
            None,
            cov3D_grad,
            None, None, None
        )

        return grads

class LocalAggregator(nn.Module):
    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size, inv_softmax=False):
        super().__init__()
        self.scale_multiplier = scale_multiplier
        self.H = H
        self.W = W
        self.D = D
        self.register_buffer('pc_min', torch.tensor(pc_min, dtype=torch.float).unsqueeze(0))
        self.grid_size = grid_size
        self.inv_softmax = inv_softmax

    def forward(
        self, 
        pts,
        means3D, 
        opacities, 
        semantics, 
        scales, 
        cov3D,
        metas,
        origin_use): 
        
        assert pts.shape[0] == 1
        pts = pts.squeeze(0)
        assert not pts.requires_grad
        means3D = means3D.squeeze(0)
        opacities = opacities.squeeze(0)
        semantics = semantics.squeeze(0)
        scales = scales.detach().squeeze(0)
        cov3D = cov3D.squeeze(0) # n, 3, 3

        # nyu_pc_min = torch.tensor(metas[0]['vox_origin']).to(pts.device)
        nyu_pc_min = origin_use
        points_int = ((pts - nyu_pc_min) / self.grid_size).to(torch.int)
        assert points_int.min() >= 0 and points_int[:, 0].max() < self.H and points_int[:, 1].max() < self.W and points_int[:, 2].max() < self.D
        means3D_int = ((means3D.detach() - nyu_pc_min) / self.grid_size).to(torch.int)
        # assert means3D_int.min() >= 0 and means3D_int[:, 0].max() < self.H and means3D_int[:, 1].max() < self.W and means3D_int[:, 2].max() < self.D
        assert means3D_int.min() >= 0
        assert means3D_int[:, 0].max() < self.H
        assert means3D_int[:, 1].max() < self.W
        assert means3D_int[:, 2].max() < self.D
        radii = torch.ceil(scales.max(dim=-1)[0] * self.scale_multiplier / self.grid_size).to(torch.int)
        assert radii.min() >= 1
        cov3D = cov3D.flatten(1)[:, [0, 4, 8, 1, 5, 2]]

        # Invoke C++/CUDA rasterization routine
        logits = _LocalAggregate.apply(
            pts,
            points_int,
            means3D,
            means3D_int,
            opacities,
            semantics,
            radii,
            cov3D,
            self.H, self.W, self.D
        )

        if not self.inv_softmax:
            return logits # n, c
        else:
            assert False
            
    def get_influential_gaussians(self, pts, means3D, opacities, scales, cov3D, metas, origin_use, influence_threshold=0.0):
        """
        获取每个体素受到影响的高斯点，按影响程度排序
        
        参数:
            pts: 查询点坐标 [1, N, 3]，每个点是体素的中心点
            means3D: 高斯点中心坐标 [1, P, 3]
            opacities: 高斯点不透明度 [1, P]
            scales: 高斯点尺度 [1, P, 3]
            cov3D: 高斯点协方差矩阵 [1, P, 3, 3]
            metas: 元数据
            origin_use: 坐标系原点
            influence_threshold: 影响程度阈值，低于此值的高斯点将被忽略
            
        返回:
            voxel_gaussians: 字典，键为体素索引，值为(高斯点索引, 影响程度)的列表，按影响程度降序排列
        """
        if pts.shape[0] != 1:
            print(f"Warning: Expected batch size 1, got {pts.shape[0]}. Using first batch.")
        
        # 去掉批次维度
        pts = pts.squeeze(0)
        means3D = means3D.squeeze(0)
        opacities = opacities.squeeze(0)
        scales = scales.detach().squeeze(0)
        cov3D = cov3D.squeeze(0) # n, 3, 3

        print(f"查询点数量: {pts.shape[0]}, 高斯点数量: {means3D.shape[0]}")
        print(f"查询点范围: [{pts.min().item():.2f}, {pts.max().item():.2f}]")
        print(f"高斯点范围: [{means3D.min().item():.2f}, {means3D.max().item():.2f}]")
        print(f"尺度范围: [{scales.min().item():.4f}, {scales.max().item():.4f}]")

        # 坐标转换
        nyu_pc_min = origin_use
        print(f"坐标系原点: {nyu_pc_min}")
        print(f"体素网格大小: H={self.H}, W={self.W}, D={self.D}")
        print(f"体素大小: {self.grid_size}")
        
        # 由于pts是体素中心点，我们需要先将其转换到体素索引
        # 体素中心点的坐标 = 体素索引 * grid_size + pc_min + grid_size/2
        # 所以体素索引 = (坐标 - pc_min - grid_size/2) / grid_size
        points_int = ((pts - nyu_pc_min) / self.grid_size).to(torch.int)
        
        # 检查点云坐标范围（只对查询点进行范围限制）
        if points_int.min() < 0 or points_int[:, 0].max() >= self.H or points_int[:, 1].max() >= self.W or points_int[:, 2].max() >= self.D:
            print("Warning: points_int out of range!")
            print(f"pts range: [{pts.min():.2f}, {pts.max():.2f}]")
            print(f"points_int range: [{points_int.min()}, {points_int.max()}]")
            print(f"X range: [{points_int[:, 0].min()}, {points_int[:, 0].max()}], expected [0, {self.H})")
            print(f"Y range: [{points_int[:, 1].min()}, {points_int[:, 1].max()}], expected [0, {self.W})")
            print(f"Z range: [{points_int[:, 2].min()}, {points_int[:, 2].max()}], expected [0, {self.D})")
            print(f"pc_min: {nyu_pc_min}, grid_size: {self.grid_size}")
            
            # 裁剪坐标到有效范围
            points_int = torch.clamp(points_int, 0, torch.tensor([self.H-1, self.W-1, self.D-1]))
            print(f"Clamped points_int range: [{points_int.min()}, {points_int.max()}]")
        
        # 高斯点坐标转换（不进行范围限制）
        means3D_int = ((means3D.detach() - nyu_pc_min) / self.grid_size).to(torch.int)
        print(f"高斯点整数坐标范围: [{means3D_int.min().item()}, {means3D_int.max().item()}]")
        
        # 计算半径
        radii = torch.ceil(scales.max(dim=-1)[0] * self.scale_multiplier / self.grid_size).to(torch.int)
        print(f"半径范围: [{radii.min().item()}, {radii.max().item()}]")
        
        # 检查半径
        if radii.min() < 1:
            print(f"Warning: radii too small! Min radius: {radii.min()}, setting to 1")
            radii = torch.clamp(radii, min=1)
        
        # 转换协方差矩阵格式
        cov3D = cov3D.flatten(1)[:, [0, 4, 8, 1, 5, 2]]

        # 调用C++/CUDA函数获取按影响程度排序的高斯点
        try:
            total_pairs, voxel_indices, gaussian_indices, influence_values = _C.GetInfluentialGaussians(
                pts,
                points_int,
                means3D,
                means3D_int,
                opacities,
                radii,
                cov3D,
                self.H, self.W, self.D
            )
            print(f"找到 {total_pairs} 对有效的高斯点-体素对")
            if total_pairs > 0:
                print(f"影响值范围: [{influence_values.min().item():.6f}, {influence_values.max().item():.6f}]")
                print(f"体素索引范围: [{voxel_indices.min().item()}, {voxel_indices.max().item()}]")
                print(f"高斯点索引范围: [{gaussian_indices.min().item()}, {gaussian_indices.max().item()}]")
        except Exception as e:
            print(f"Error calling GetInfluentialGaussians: {e}")
            return {}
        
        # 将结果转换为字典格式，同时确保不包含重复项
        voxel_gaussians = {}
        valid_pairs = 0
        
        # 先按体素索引对结果进行分组
        for i in range(total_pairs):
            voxel_idx = voxel_indices[i].item()
            gaussian_idx = gaussian_indices[i].item()
            influence = influence_values[i].item()
            
            if influence < influence_threshold:
                continue
                
            valid_pairs += 1
            if voxel_idx not in voxel_gaussians:
                voxel_gaussians[voxel_idx] = []
            
            # 检查是否已经添加了相同的高斯点
            already_added = False
            for idx, (g_idx, _) in enumerate(voxel_gaussians[voxel_idx]):
                if g_idx == gaussian_idx:
                    already_added = True
                    # 如果当前影响值更大，则更新
                    if influence > voxel_gaussians[voxel_idx][idx][1]:
                        voxel_gaussians[voxel_idx][idx] = (gaussian_idx, influence)
                    break
            
            if not already_added:
                voxel_gaussians[voxel_idx].append((gaussian_idx, influence))
        
        # 对每个体素的高斯点列表按影响程度降序排序
        for voxel_idx in voxel_gaussians:
            voxel_gaussians[voxel_idx].sort(key=lambda x: x[1], reverse=True)
            
            # 限制每个体素最多返回20个高斯点
            if len(voxel_gaussians[voxel_idx]) > 20:
                voxel_gaussians[voxel_idx] = voxel_gaussians[voxel_idx][:20]
        
        print(f"有效高斯点-体素对数量: {valid_pairs} (阈值: {influence_threshold})")
        print(f"有影响的体素数量: {len(voxel_gaussians)}")
        
        # 抽样打印一些体素的高斯点列表，用于调试
        if len(voxel_gaussians) > 0:
            sample_voxel_idx = list(voxel_gaussians.keys())[0]
            print(f"示例体素 {sample_voxel_idx} 的高斯点列表:")
            for i, (g_idx, inf) in enumerate(voxel_gaussians[sample_voxel_idx][:5]):
                print(f"  {i+1}. 高斯点 {g_idx}: 影响值 = {inf:.6f}")
        
        return voxel_gaussians