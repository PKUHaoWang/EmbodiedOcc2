import torch, torch.nn as nn, torch.nn.functional
from mmengine.registry import MODELS
from mmengine.model import BaseModule
from utils.chamfer_distance import ChamferDistance


@MODELS.register_module()
class BEVRenderHead(BaseModule):
    def __init__(
        self, bev_w, bev_h, bev_z, nbr_classes=1, in_dims=64, hidden_dims=128, 
        out_dims=None, cls_dims=None,
    ):
        super().__init__()
        self.bev_w = bev_w
        self.bev_h = bev_h
        self.bev_z = bev_z

        out_dims = in_dims if out_dims is None else out_dims
        cls_dims = out_dims if cls_dims is None else cls_dims
        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims)
        )

        self.classifier = nn.Linear(cls_dims, nbr_classes)
        self.classes = nbr_classes

        self.ce_loss_func = nn.CrossEntropyLoss()
        self.chamfer_dist = ChamferDistance()
    
    def forward(self, bev_feat, metas, points, label, output_dict, test_mode=False):
        points_split = [meta['points_split'] for meta in metas]
        
        B, F, W, H, C = bev_feat.shape
        assert W == self.bev_w and H == self.bev_h

        bev_feat = self.decoder(bev_feat)
        voxel_feat = bev_feat.reshape(B, F, self.bev_w, self.bev_h, self.bev_z, -1)

        feats, targets = [], []
        for i in range(B):
            for j in range(F):
                start, end = points_split[i][j], points_split[i][j+1]
                sample_loc = points[i:i+1, start:end, :, [2, 1, 0]].unsqueeze(0)             # B, N, D, 3   -->   1, 1, N, D, 3
                sample_feat = voxel_feat[i:i+1, j].permute(0, 4, 1, 2, 3)                    # B, F, W, H, Z, C   -->   1, W, H, Z, C
                feats.append(torch.nn.functional.grid_sample(sample_feat,
                        sample_loc).permute(0, 2, 3, 4, 1).squeeze(0).squeeze(0))            # N, D, C
            targets.append(label[i, :, 4])                                                   # B, N, 5 (dir + bin + label)
        feats = torch.cat(feats, dim=0)
        targets = torch.cat(targets, dim=0)
        logits = self.classifier(feats).squeeze(-1)

        output_dict['ce_input'] = logits
        output_dict['ce_label'] = targets
        
        # render points & cal chamfer distance
        if test_mode:
            pred = torch.argmax(logits, dim=-1).float()
            directions, depth_bin, target = label[..., :3], label[..., 3:4], label[..., 4:]   # B, N, 1
            chamfer_dist = []
            points_sum = 0
            for i in range(B):
                for j in range(F):
                    start, end = points_split[i][j], points_split[i][j+1]
                    vec = directions[i:i+1, start:end] * depth_bin[i:i+1, start:end]
                    cur_pred_pts = vec * pred[None, start+points_sum:end+points_sum, None]
                    cur_target_pts = vec * target[i, start:end]
                    dist1, dist2 = self.chamfer_dist(cur_pred_pts, cur_target_pts)
                    chamfer_dist.append((torch.mean(dist1) + torch.mean(dist2)).item())
                points_sum += points_split[i][-1]
            output_dict['chamfer_dist'] = chamfer_dist

        return output_dict