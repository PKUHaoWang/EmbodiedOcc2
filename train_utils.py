import numba as nb
import numpy as np
import torch
import math

def restore_depth_from_LID(indices, depth_min, depth_max, num_bins):
    """
    Restores the continuous depth map from LID bin indices, handling out-of-bound indices
    Args:
        indices [torch.Tensor(H, W)]: Depth bin indices from LID discretization
        depth_min [float]: Minimum depth value
        depth_max [float]: Maximum depth value
        num_bins [int]: Number of depth bins
    Returns:
        depth_map [torch.Tensor(H, W)]: Restored continuous depth map
    """
    # Ensure indices are within valid range [0, num_bins - 1]
    indices = torch.clamp(indices, 0, num_bins - 1)
    
    # Compute bin size for LID
    bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
    
    # Restore depth map from indices
    depth_map = depth_min + (bin_size / 8) * (4 * (indices + 0.5) ** 2 - 1)
    
    return depth_map


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

def downsample_label(label, voxel_size=(60, 60, 36), downscale=4):
    """Downsample the labeled data."""
    if downscale == 1:
        return label
    ds = downscale
    small_size = (
        voxel_size[0] // ds,
        voxel_size[1] // ds,
        voxel_size[2] // ds,
    )
    label_downscale = torch.zeros(small_size, dtype=torch.uint8)
    empty_t = 0.95 * ds * ds * ds
    s01 = small_size[0] * small_size[1]
    label_i = torch.zeros((ds, ds, ds), dtype=torch.int32)

    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = i // s01
        y = (i - z * s01) // small_size[0]
        x = i - z * s01 - y * small_size[0]

        label_i[:, :, :] = label[
            x * ds : (x + 1) * ds, y * ds : (y + 1) * ds, z * ds : (z + 1) * ds
        ]
        label_bin = label_i.flatten()

        zero_count_12 = (label_bin == 12).sum().item() # empty
        zero_count_0 = (label_bin == 0).sum().item() # unknown

        zero_count = zero_count_12 + zero_count_0
        if zero_count > empty_t:
            label_downscale[x, y, z] = 12 if zero_count_12 > zero_count_0 else 0
        else:
            label_i_s = label_bin[(label_bin > 0) & (label_bin < 12)]
            if label_i_s.numel() > 0:
                label_downscale[x, y, z] = torch.argmax(torch.bincount(label_i_s)).item()
    return label_downscale

def compute_CP_mega_matrix(target, is_binary=False):
    """
    Computes the CP mega matrix for voxel relations.
    """
    label = target.reshape(-1)
    N = label.shape[0]
    super_voxel_size = [i // 2 for i in target.shape]
    matrix_size = (2 if is_binary else 4, N, super_voxel_size[0] * super_voxel_size[1] * super_voxel_size[2])
    matrix = torch.zeros(matrix_size, dtype=torch.uint8)

    for xx in range(super_voxel_size[0]):
        for yy in range(super_voxel_size[1]):
            for zz in range(super_voxel_size[2]):
                col_idx = xx * (super_voxel_size[1] * super_voxel_size[2]) + yy * super_voxel_size[2] + zz
                label_col_megas = torch.tensor([
                    target[xx * 2,     yy * 2,     zz * 2],
                    target[xx * 2 + 1, yy * 2,     zz * 2],
                    target[xx * 2,     yy * 2 + 1, zz * 2],
                    target[xx * 2,     yy * 2,     zz * 2 + 1],
                    target[xx * 2 + 1, yy * 2 + 1, zz * 2],
                    target[xx * 2 + 1, yy * 2,     zz * 2 + 1],
                    target[xx * 2,     yy * 2 + 1, zz * 2 + 1],
                    target[xx * 2 + 1, yy * 2 + 1, zz * 2 + 1],
                ])
                label_col_megas = label_col_megas[label_col_megas != 0]
                     
                for label_col_mega in label_col_megas:
                    label_col = torch.full((N,), label_col_mega)
                    if not is_binary:
                        matrix[0, (label != 0) & (label_col == label) & (label_col != 12), col_idx] = 1
                        matrix[1, (label != 0) & (label_col != label) & (label_col != 12) & (label != 12), col_idx] = 1
                        matrix[2, (label != 0) & (label == label_col) & (label_col == 12), col_idx] = 1
                        matrix[3, (label != 0) & (label != label_col) & ((label == 12) | (label_col == 12)), col_idx] = 1
                    else:
                        matrix[0, (label != 0) & (label_col != label), col_idx] = 1
                        matrix[1, (label != 0) & (label_col == label), col_idx] = 1
    return matrix
    