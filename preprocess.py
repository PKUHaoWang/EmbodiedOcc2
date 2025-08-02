import torch
import numpy as np
from torch import nn
from surface_normal_uncertainty.models.NNET import NNET
import surface_normal_uncertainty.utils.utils as utils
from torchvision import transforms
from PIL import Image
import os
import glob
from tqdm import tqdm
import cv2
from torchvision.transforms import Compose
from dataset.transform_ import Resize, NormalizeImage, PrepareForNet
import argparse
import sys

# 添加深度模型的路径
cwd = os.getcwd()
sys.path.append(cwd + '/Depth-Anything-V2/metric_depth')
from depth_anything_v2.dpt import DepthAnythingV2


class NormalKappaPrior(nn.Module):
    """法线和Kappa预测模型"""
    def __init__(self, device):
        super().__init__()
        checkpoint = 'checkpoints/scannet.pt'
        print('Loading normal/kappa checkpoint... {}'.format(checkpoint))
        self.device = device
        self.model = NNET(nnet_config).to(self.device)
        self.model = utils.load_checkpoint(checkpoint, self.model)
        self.model.eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.input_width = 640
        self.input_height = 480
    
    def preprocess(self, img_path):
        """预处理图像"""
        img = Image.open(img_path).convert("RGB").resize(size=(self.input_width, self.input_height), resample=Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = self.normalize(img)

        img_name = img_path.split('/')[-1]
        img_name = img_name.split('.png')[0] if '.png' in img_name else img_name.split('.jpg')[0]

        sample = {'img': img.unsqueeze(0), 'img_name': img_name}
        return sample
    
    def forward(self, rgb_path):
        """前向传播"""
        data_dict = self.preprocess(rgb_path)
        with torch.no_grad():
            img = data_dict['img'].to(self.device)
            norm_out_list, _, _ = self.model(img)
            norm_out = norm_out_list[-1]
            # 获取法线和kappa
            pred_norm = norm_out[:, :3, :, :]
            pred_kappa = norm_out[:, 3:, :, :]
            return pred_norm, pred_kappa


class DepthPrior(nn.Module):
    """深度预测模型"""
    def __init__(self, device):
        super().__init__()
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.device = device
        self.depthanything = DepthAnythingV2(**{**model_configs['vitb'], 'max_depth': 20}).to(self.device)
        checkpoint = torch.load('checkpoints/finetune_scannet_depthanythingv2.pth', map_location='cpu')['model']
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                new_key = k[len('module.'):]
            else:
                new_key = k
            new_state_dict[new_key] = v
        self.depthanything.load_state_dict(new_state_dict)
        self.depthanything.eval()
        
    def forward(self, rgb_path):
        """前向传播"""
        transform = Compose([
            Resize(
                width=480,
                height=480,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        img_depthbranch = cv2.imread(rgb_path)
        img_depthbranch = cv2.resize(img_depthbranch, (640, 480), interpolation=cv2.INTER_NEAREST)
        img_depthbranch = cv2.cvtColor(img_depthbranch, cv2.COLOR_BGR2RGB) / 255.0
        sample = transform({'image': img_depthbranch})
        img_depthbranch = torch.from_numpy(sample['image']).unsqueeze(0).to(self.device)
        depth_pred = self.depthanything.infer_image(img_depthbranch, 480, 640, 480)
        return depth_pred


# 配置类
class nnet_config:
    architecture = 'BN'
    pretrained = 'scannet'
    sampling_ratio = 0.4
    importance_ratio = 0.7
    input_height = 480
    input_width = 640


def process_scene(scene_folder, models, output_dirs, features_to_generate):
    """处理单个场景"""
    scene_name = os.path.basename(scene_folder)
    print(f"\nProcessing scene: {scene_name}")
    
    # 创建对应的输出目录
    scene_output_dirs = {}
    for feature_type, output_dir in output_dirs.items():
        if features_to_generate in [feature_type, 'all']:
            scene_output_dirs[feature_type] = os.path.join(output_dir, scene_name)
            os.makedirs(scene_output_dirs[feature_type], exist_ok=True)
    
    # 获取所有jpg文件
    jpg_files = glob.glob(os.path.join(scene_folder, '*.jpg'))
    jpg_files.sort()
    
    # 处理每张图像
    for jpg_path in tqdm(jpg_files, desc=f"Processing {scene_name}"):
        base_name = os.path.basename(jpg_path).replace('.jpg', '')
        
        # 检查是否需要处理
        skip_processing = True
        for feature_type in scene_output_dirs.keys():
            output_path = os.path.join(scene_output_dirs[feature_type], f"{base_name}.npy")
            if not os.path.exists(output_path):
                skip_processing = False
                break
        
        if skip_processing:
            continue
        
        try:
            # 处理法线和kappa
            if 'normal_kappa' in models:
                pred_norm, pred_kappa = models['normal_kappa'](jpg_path)
                
                if features_to_generate in ['normal', 'all']:
                    output_path = os.path.join(scene_output_dirs['normal'], f"{base_name}.npy")
                    pred_norm_np = pred_norm.cpu().numpy()
                    np.save(output_path, pred_norm_np)
                
                if features_to_generate in ['kappa', 'all']:
                    output_path = os.path.join(scene_output_dirs['kappa'], f"{base_name}.npy")
                    pred_kappa_np = pred_kappa.cpu().numpy()
                    np.save(output_path, pred_kappa_np)
            
            # 处理深度
            if 'depth' in models:
                pred_depth = models['depth'](jpg_path)
                
                if features_to_generate in ['depth', 'all']:
                    output_path = os.path.join(scene_output_dirs['depth'], f"{base_name}.npy")
                    pred_depth_np = pred_depth.detach().cpu().numpy()
                    np.save(output_path, pred_depth_np)
                    
        except Exception as e:
            print(f"\nError processing {jpg_path}: {str(e)}")
            continue


def main():
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='all', 
                       choices=['depth', 'normal', 'kappa', 'all'],
                       help='要生成的特征类型: depth, normal, kappa, 或 all')
    args = parser.parse_args()
    features_to_generate = args.features

    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化模型
    models = {}
    if features_to_generate in ['normal', 'kappa', 'all']:
        print("Initializing normal/kappa model...")
        models['normal_kappa'] = NormalKappaPrior(device).to(device).eval()

    if features_to_generate in ['depth', 'all']:
        print("Initializing depth model...")
        models['depth'] = DepthPrior(device).to(device).eval()

    # 定义输入和输出目录
    input_base_dir = './data/occscannet/posed_images'
    output_dirs = {
        'depth': './data/occscannet/depthanything',
        'normal': './data/occscannet/normals',
        'kappa': './data/occscannet/kappas'
    }

    # 创建输出目录
    for feature_type, output_dir in output_dirs.items():
        if features_to_generate in [feature_type, 'all']:
            os.makedirs(output_dir, exist_ok=True)

    # 获取所有场景文件夹
    scene_folders = glob.glob(os.path.join(input_base_dir, 'scene*'))
    scene_folders.sort()

    # 处理每个场景文件夹
    for scene_folder in scene_folders:
        process_scene(scene_folder, models, output_dirs, features_to_generate)

    print("\nProcessing completed!")


if __name__ == '__main__':
    main()