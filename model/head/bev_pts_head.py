import torch, torch.nn as nn, torch.nn.functional
from mmengine.registry import MODELS
from mmengine.model import BaseModule
from torch.utils.checkpoint import checkpoint as cp
from einops import rearrange


@MODELS.register_module()
class BEVPtsHead(BaseModule):
    def __init__(
        self, bev_w, bev_h, bev_z, nbr_classes=2, in_dims=64, hidden_dims=128, 
        out_dims=None, cls_dims=None
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
    
    def forward(self, bev_feat, points, label, output_dict, **kwargs):
        B, F, W, H, D = label.shape
        label = label.reshape(B*F, W, H, D)
        assert bev_feat.shape[0] == B and bev_feat.shape[1] == F

        bev_feat = self.decoder(bev_feat)
        voxel_feat = bev_feat.reshape(B*F, self.bev_w, self.bev_h, self.bev_z, -1).permute(0, 4, 1, 2, 3)
        voxel_feat = torch.nn.functional.interpolate(voxel_feat, size=[W, H, D], mode='trilinear', align_corners=True).contiguous()
        logits = self.classifier(voxel_feat.permute(0, 2, 3, 4, 1)).squeeze(-1)
        output_dict['ce_input'] = logits
        output_dict['ce_label'] = label

        if points is not None:
            with torch.cuda.amp.autocast(False):
                voxel_feat = voxel_feat.float()
                assert B == points.shape[0] and F == points.shape[1]
                points = points.flatten(0, 1)
                occ_feat = torch.nn.functional.grid_sample(voxel_feat, points[..., [2, 1, 0]], padding_mode="border", align_corners=True)
                occ_predict = self.classifier(occ_feat.permute(0, 2, 3, 4, 1)).squeeze(-1)
                output_dict['occ3d_predict'] = occ_predict
        
        return output_dict


class UnetBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(out_c)
        self.norm2 = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()
        if in_c == out_c:
            self.shortcut = nn.Identity()
        if in_c != out_c:
            self.shortcut = nn.Conv2d(in_c, out_c, 1, 1, 0)
    def forward(self, input):
        output = self.conv1(input)
        output = self.norm1(output)
        output = self.act(output)
        output = self.conv2(output)
        output = self.norm2(output)
        output = output + self.shortcut(input)
        output = self.act(output)
        return output


@MODELS.register_module()
class VoxPtsHead(BaseModule):
    def __init__(
        self,
        cls_dims=19,
        in_dims=128,
        in_dims_3d=24,
        num_level=3,
        num_level_3d=2,
        with_cp=True,
    ):
        super(VoxPtsHead, self).__init__()

        self.with_cp = with_cp
        self.num_level = num_level
        
        # bev encoder
        self.bev_convs = nn.ModuleList()
        for i in range(self.num_level):
            self.bev_convs.append(nn.Sequential(UnetBlock(in_dims, in_dims), UnetBlock(in_dims, in_dims)))
        self.bev_downsample = nn.ModuleList()
        for i in range(self.num_level):
            downsample = nn.Conv2d(in_dims, in_dims, kernel_size=2, stride=2, padding=0) if i < self.num_level-1 else nn.Identity()
            self.bev_downsample.append(downsample)

        # voxel encoder
        self.occ_convs = nn.ModuleList()
        for i in range(num_level_3d):
            occ_conv = nn.Sequential(
                nn.Conv3d(in_dims_3d, in_dims_3d, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(in_dims_3d),
                nn.ReLU(inplace=True))
            self.occ_convs.append(occ_conv)

        self.occ_pred_conv = nn.Sequential(
                nn.Conv3d(in_dims_3d, in_dims_3d*2, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(in_dims_3d*2),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_dims_3d*2, cls_dims, kernel_size=1, stride=1, padding=0))
    
    def forward(self, bev_feat, points, label, output_dict, **kwargs):
        B, F, W, H, D = label.shape
        label = label.reshape(B*F, W, H, D)
        assert bev_feat.shape[0] == B and bev_feat.shape[1] == F
        bev_feat = rearrange(bev_feat, 'b f h w c -> (b f) c h w').contiguous()

        bev_feats = []
        x = bev_feat
        for conv, down in zip(self.bev_convs, self.bev_downsample):
            if self.with_cp and self.training:
                x = cp(conv, x)
            else:
                x = conv(x)
            bev_feats.append(x)
            if self.with_cp and self.training:
                x = cp(down, x)
            else:
                x = down(x)
                
        
        for i in range(len(bev_feats)):
            bev_feats[i] = torch.nn.functional.interpolate(bev_feats[i], size=[W, H], mode='bilinear')
        x = torch.cat(bev_feats, dim=1)

        x = rearrange(x, 'bf c h w -> bf h w c').reshape(B*F, W, H, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        for conv in self.occ_convs:
            if self.with_cp and self.training:
                x = cp(conv, x)
            else:
                x = conv(x)

        if self.with_cp and self.training:
            logits = cp(self.occ_pred_conv, x)
        else:
            logits = self.occ_pred_conv(x)
        
        logits[torch.isnan(logits)] = 0
        logits[torch.isinf(logits)] = 0
        assert torch.isnan(logits).sum().item() == 0

        output_dict['occ_pred'] = logits
        output_dict['occ_target'] = label

        if points is not None:
            with torch.cuda.amp.autocast(False):
                x = x.float()
                assert B == points.shape[0] and F == points.shape[1]
                points = points.flatten(0, 1)
                occ_feat = torch.nn.functional.grid_sample(x, points[..., [2, 1, 0]], padding_mode="border", align_corners=True)
                if self.with_cp and self.training:
                    occ_predict = cp(self.occ_pred_conv, occ_feat)
                else:
                    occ_predict = self.occ_pred_conv(occ_feat)
                output_dict['occ3d_predict'] = occ_predict

        return output_dict