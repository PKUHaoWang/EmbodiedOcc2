import torch
import numpy as np


def get_cross_view_ref_points(tpv_h, tpv_w, tpv_z, num_points_in_pillar):
    # ref points generating target: (#query)hw+zh+wz, (#level)3, #p, 2
    # generate points for hw and level 1
    h_ranges = torch.linspace(0.5, tpv_h-0.5, tpv_h) / tpv_h
    w_ranges = torch.linspace(0.5, tpv_w-0.5, tpv_w) / tpv_w
    h_ranges = h_ranges.unsqueeze(-1).expand(-1, tpv_w).flatten()
    w_ranges = w_ranges.unsqueeze(0).expand(tpv_h, -1).flatten()
    hw_hw = torch.stack([w_ranges, h_ranges], dim=-1) # hw, 2
    hw_hw = hw_hw.unsqueeze(1).expand(-1, num_points_in_pillar[2], -1) # hw, #p, 2
    # generate points for hw and level 2
    z_ranges = torch.linspace(0.5, tpv_z-0.5, num_points_in_pillar[2]) / tpv_z # #p
    z_ranges = z_ranges.unsqueeze(0).expand(tpv_h*tpv_w, -1) # hw, #p
    h_ranges = torch.linspace(0.5, tpv_h-0.5, tpv_h) / tpv_h
    h_ranges = h_ranges.reshape(-1, 1, 1).expand(-1, tpv_w, num_points_in_pillar[2]).flatten(0, 1)
    hw_zh = torch.stack([h_ranges, z_ranges], dim=-1) # hw, #p, 2
    # generate points for hw and level 3
    z_ranges = torch.linspace(0.5, tpv_z-0.5, num_points_in_pillar[2]) / tpv_z # #p
    z_ranges = z_ranges.unsqueeze(0).expand(tpv_h*tpv_w, -1) # hw, #p
    w_ranges = torch.linspace(0.5, tpv_w-0.5, tpv_w) / tpv_w
    w_ranges = w_ranges.reshape(1, -1, 1).expand(tpv_h, -1, num_points_in_pillar[2]).flatten(0, 1)
    hw_wz = torch.stack([z_ranges, w_ranges], dim=-1) # hw, #p, 2
    
    # generate points for zh and level 1
    w_ranges = torch.linspace(0.5, tpv_w-0.5, num_points_in_pillar[1]) / tpv_w
    w_ranges = w_ranges.unsqueeze(0).expand(tpv_z*tpv_h, -1)
    h_ranges = torch.linspace(0.5, tpv_h-0.5, tpv_h) / tpv_h
    h_ranges = h_ranges.reshape(1, -1, 1).expand(tpv_z, -1, num_points_in_pillar[1]).flatten(0, 1)
    zh_hw = torch.stack([w_ranges, h_ranges], dim=-1)
    # generate points for zh and level 2
    z_ranges = torch.linspace(0.5, tpv_z-0.5, tpv_z) / tpv_z
    z_ranges = z_ranges.reshape(-1, 1, 1).expand(-1, tpv_h, num_points_in_pillar[1]).flatten(0, 1)
    h_ranges = torch.linspace(0.5, tpv_h-0.5, tpv_h) / tpv_h
    h_ranges = h_ranges.reshape(1, -1, 1).expand(tpv_z, -1, num_points_in_pillar[1]).flatten(0, 1)
    zh_zh = torch.stack([h_ranges, z_ranges], dim=-1) # zh, #p, 2
    # generate points for zh and level 3
    w_ranges = torch.linspace(0.5, tpv_w-0.5, num_points_in_pillar[1]) / tpv_w
    w_ranges = w_ranges.unsqueeze(0).expand(tpv_z*tpv_h, -1)
    z_ranges = torch.linspace(0.5, tpv_z-0.5, tpv_z) / tpv_z
    z_ranges = z_ranges.reshape(-1, 1, 1).expand(-1, tpv_h, num_points_in_pillar[1]).flatten(0, 1)
    zh_wz = torch.stack([z_ranges, w_ranges], dim=-1)

    # generate points for wz and level 1
    h_ranges = torch.linspace(0.5, tpv_h-0.5, num_points_in_pillar[0]) / tpv_h
    h_ranges = h_ranges.unsqueeze(0).expand(tpv_w*tpv_z, -1)
    w_ranges = torch.linspace(0.5, tpv_w-0.5, tpv_w) / tpv_w
    w_ranges = w_ranges.reshape(-1, 1, 1).expand(-1, tpv_z, num_points_in_pillar[0]).flatten(0, 1)
    wz_hw = torch.stack([w_ranges, h_ranges], dim=-1)
    # generate points for wz and level 2
    h_ranges = torch.linspace(0.5, tpv_h-0.5, num_points_in_pillar[0]) / tpv_h
    h_ranges = h_ranges.unsqueeze(0).expand(tpv_w*tpv_z, -1)
    z_ranges = torch.linspace(0.5, tpv_z-0.5, tpv_z) / tpv_z
    z_ranges = z_ranges.reshape(1, -1, 1).expand(tpv_w, -1, num_points_in_pillar[0]).flatten(0, 1)
    wz_zh = torch.stack([h_ranges, z_ranges], dim=-1)
    # generate points for wz and level 3
    w_ranges = torch.linspace(0.5, tpv_w-0.5, tpv_w) / tpv_w
    w_ranges = w_ranges.reshape(-1, 1, 1).expand(-1, tpv_z, num_points_in_pillar[0]).flatten(0, 1)
    z_ranges = torch.linspace(0.5, tpv_z-0.5, tpv_z) / tpv_z
    z_ranges = z_ranges.reshape(1, -1, 1).expand(tpv_w, -1, num_points_in_pillar[0]).flatten(0, 1)
    wz_wz = torch.stack([z_ranges, w_ranges], dim=-1)

    reference_points = torch.cat([
        torch.stack([hw_hw, hw_zh, hw_wz], dim=1),
        torch.stack([zh_hw, zh_zh, zh_wz], dim=1),
        torch.stack([wz_hw, wz_zh, wz_wz], dim=1)
    ], dim=0) # hw+zh+wz, 3, #p, 2
    
    return reference_points


def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cpu', dtype=torch.float):
    """Get the reference points used in image cross-attention and single plane self-attention.
    Args:
        H, W: spatial shape of tpv.
        Z: hight of pillar.
        D: sample D points uniformly from each pillar.
        device (obj:`device`): The device where
            reference_points should be.
    Returns:
        Tensor: reference points used in decoder, has \
            shape (bs, num_keys, num_levels, 2).
    """

    # reference points in 3D space, used in image cross-attention
    if dim == '3d':
        zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                            device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, -1).expand(num_points_in_pillar, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, -1, 1).expand(num_points_in_pillar, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
        return ref_3d

    # reference points on 2D tpv plane, used in self attention in tpvformer04 
    # which is an older version. Now we use get_cross_view_ref_points instead.
    elif dim == '2d':
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(
                0.5, W - 0.5, W, dtype=dtype, device=device))
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
        return ref_2d
    
# This function must use fp32!!!
@torch.cuda.amp.autocast(enabled=False)
def point_sampling(reference_points, img_metas):
    reference_points = reference_points.float()

    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    if isinstance(lidar2img[0], (np.ndarray, list)):
        lidar2img = np.asarray(lidar2img)
        if len(lidar2img.shape) == 5:
            B, F, N, _, _ = lidar2img.shape
            lidar2img = lidar2img.reshape(B*F, N, 4, 4)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
    else:
        lidar2img = torch.stack(lidar2img, dim=0)

    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1)

    reference_points = reference_points.permute(1, 0, 2, 3)
    D, B, num_query = reference_points.size()[:3]
    num_cam = lidar2img.size(1)

    reference_points = reference_points.view(
        D, B, 1, num_query, 4, 1)

    lidar2img = lidar2img.view(
        1, B, num_cam, 1, 4, 4)

    reference_points_cam = torch.matmul(
        lidar2img.to(torch.float32),
        reference_points.to(torch.float32)).squeeze(-1)
    
    eps = 1e-5

    tpv_mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam[..., 0:2] = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
    if img_metas[0].get('img_aug_matrix') is not None:
        img_aug_matrix = []
        for img_meta in img_metas:
            img_aug_matrix.append(np.stack(img_meta['img_aug_matrix'], axis=0))
        img_aug_matrix = np.stack(img_aug_matrix, axis=0)
        if len(img_aug_matrix.shape) == 5:
            B, F, N, _, _ = img_aug_matrix.shape
            img_aug_matrix = img_aug_matrix.reshape(B*F, N, 4, 4)
        img_aug_matrix = torch.from_numpy(img_aug_matrix).cuda()[None, :, :, None]
        reference_points_cam = torch.matmul(img_aug_matrix, reference_points_cam[..., None]).squeeze(-1)
    
    reference_points_cam = reference_points_cam[..., :2]
    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0] # D, B, N, Q, 2

    tpv_mask = (tpv_mask & (reference_points_cam[..., 1:2] > 0.0)
                & (reference_points_cam[..., 1:2] < 1.0)
                & (reference_points_cam[..., 0:1] < 1.0)
                & (reference_points_cam[..., 0:1] > 0.0))

    tpv_mask = torch.nan_to_num(tpv_mask)

    reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4) # N, B, Q, D, 2
    tpv_mask = tpv_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

    return reference_points_cam, tpv_mask

