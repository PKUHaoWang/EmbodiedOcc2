import torch
from typing_extensions import Literal


class GridMeterMapping:

    def __init__(
        self,
        w_size=200,
        h_size=200,
        z_size=16,
        pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    ) -> None:
        self.h_size = h_size
        self.w_size = w_size
        self.z_size = z_size
        self.w_range = [pc_range[0], pc_range[3]]
        self.h_range = [pc_range[1], pc_range[4]]
        self.z_range = [pc_range[2], pc_range[5]]
    
    def grid2meter(self, grid):
        # grid: [..., (w, h, z)]
        w, h = grid[..., 0], grid[..., 1]
        if grid.shape[-1] == 3:
            z = grid[..., 2]
        else:
            z = None
        
        # deal with w
        x = (w / self.w_size) * (self.w_range[1] - self.w_range[0]) + self.w_range[0]

        # deal with h
        y = (h / self.h_size) * (self.h_range[1] - self.h_range[0]) + self.h_range[0]

        # deal with z
        if z is not None:
            z = (z / self.z_size) * (self.z_range[1] - self.z_range[0]) + self.z_range[0]
            return torch.stack([x, y, z], dim=-1)
        else:
            return torch.stack([x, y], dim=-1)
        