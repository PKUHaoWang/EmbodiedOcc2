#include <torch/extension.h>
#include "local_aggregate.h"
#include <tuple>
#include <vector>
#include <algorithm>
#include <iostream>

// 简化版的GetInfluentialGaussians实现，避免使用复杂的C++容器和智能指针
// 并过滤掉尺度过大的高斯点
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor>
GetInfluentialGaussiansCUDA(
    const torch::Tensor& pts,
    const torch::Tensor& points_int,
    const torch::Tensor& means3D,
    const torch::Tensor& means3D_int,
    const torch::Tensor& opacity,
    const torch::Tensor& radii,
    const torch::Tensor& cov3D,
    const int H, int W, int D)
{
    // 获取数据尺寸
    const int N = pts.size(0);
    const int P = means3D.size(0);
    
    std::cout << "GetInfluentialGaussiansCUDA: N=" << N << ", P=" << P << std::endl;
    
    // 创建输出张量 - 直接分配足够大的空间
    // 每个点最多关联50个高斯点 (增加了最大关联数量)
    const int max_pairs = N * 50;
    auto voxel_indices = torch::zeros({max_pairs}, torch::dtype(torch::kInt32).device(pts.device()));
    auto gaussian_indices = torch::zeros({max_pairs}, torch::dtype(torch::kInt32).device(pts.device()));
    auto influence_values = torch::zeros({max_pairs}, torch::dtype(torch::kFloat32).device(pts.device()));
    
    // 获取CPU数据
    auto pts_cpu = pts.cpu();
    auto points_int_cpu = points_int.cpu();
    auto means3D_cpu = means3D.cpu();
    auto opacity_cpu = opacity.cpu();
    auto cov3D_cpu = cov3D.cpu();
    auto radii_cpu = radii.cpu();
    
    const float* pts_data = pts_cpu.data_ptr<float>();
    const int* points_int_data = points_int_cpu.data_ptr<int>();
    const float* means3D_data = means3D_cpu.data_ptr<float>();
    const float* opacity_data = opacity_cpu.data_ptr<float>();
    const float* cov3D_data = cov3D_cpu.data_ptr<float>();
    const int* radii_data = radii_cpu.data_ptr<int>();
    
    // 用于填充输出张量的索引
    int pair_count = 0;
    
    // 处理前10个体素时进行详细分析
    const int MAX_VERBOSE_VOXELS = 10;
    int verbose_count = 0;
    
    // 标记可能是"空"高斯点的高斯点
    // 检查哪些高斯点具有异常大的尺度/半径
    std::vector<bool> is_empty_gaussian(P, false);
    int empty_count = 0;
    
    // 查找最大半径值及其索引
    int max_radius = 0;
    int max_radius_idx = -1;
    for (int p = 0; p < P; p++) {
        if (radii_data[p] > max_radius) {
            max_radius = radii_data[p];
            max_radius_idx = p;
        }
    }
    
    // 确定异常值阈值 - 提高阈值以只过滤掉极端异常的空点
    const int EMPTY_RADIUS_THRESHOLD = 5000;  // 提高了阈值
    
    // 标记半径异常大的高斯点
    for (int p = 0; p < P; p++) {
        if (radii_data[p] > EMPTY_RADIUS_THRESHOLD) {
            is_empty_gaussian[p] = true;
            empty_count++;
            std::cout << "标记可能是空点的高斯点 #" << p << "，半径=" << radii_data[p] << std::endl;
        }
    }
    
    std::cout << "检测到 " << empty_count << " 个可能是空点的高斯点" << std::endl;
    std::cout << "最大半径值: " << max_radius << "，索引: " << max_radius_idx << std::endl;
    
    // 遍历所有体素
    for (int n = 0; n < N && pair_count < max_pairs; n++) {
        // 每处理100000个体素打印一次进度
        if (n % 100000 == 0) {
            std::cout << "Processing voxel " << n << "/" << N << std::endl;
        }
        
        // 计算体素索引
        int voxel_x = points_int_data[n * 3];
        int voxel_y = points_int_data[n * 3 + 1];
        int voxel_z = points_int_data[n * 3 + 2];
        int voxel_idx = voxel_x * W * D + voxel_y * D + voxel_z;
        
        // 体素中心坐标
        float voxel_center[3] = {
            pts_data[n * 3],
            pts_data[n * 3 + 1],
            pts_data[n * 3 + 2]
        };
        
        // 临时存储这个体素的影响值
        std::vector<std::pair<int, float>> influences;
        influences.reserve(P);
        
        // 遍历所有高斯点（跳过空点）
        for (int p = 0; p < P; p++) {
            // 跳过被标记为空点的高斯点
            if (is_empty_gaussian[p]) {
                continue;
            }
            
            // 计算距离向量
            float dx = means3D_data[p * 3] - voxel_center[0];
            float dy = means3D_data[p * 3 + 1] - voxel_center[1];
            float dz = means3D_data[p * 3 + 2] - voxel_center[2];
            
            // 计算欧氏距离平方，提前过滤掉明显太远的点 (增加了距离阈值)
            float dist_sq = dx*dx + dy*dy + dz*dz;
            if (dist_sq > 400.0f) {  // 从100增加到400，增大了搜索范围
                continue;
            }
            
            // 计算马氏距离
            float power = cov3D_data[p * 6] * dx * dx + 
                         cov3D_data[p * 6 + 1] * dy * dy + 
                         cov3D_data[p * 6 + 2] * dz * dz;
            power = -0.5f * power - (cov3D_data[p * 6 + 3] * dx * dy + 
                                   cov3D_data[p * 6 + 4] * dy * dz + 
                                   cov3D_data[p * 6 + 5] * dx * dz);
            
            // 应用高斯函数并结合不透明度
            float influence = opacity_data[p] * exp(power);
            
            // 只添加有影响的高斯点 (降低了阈值)
            if (influence > 1e-6) {  // 从1e-5降低到1e-6
                influences.push_back(std::make_pair(p, influence));
            }
        }
        
        // 对影响值排序
        std::sort(influences.begin(), influences.end(), 
                 [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                     return a.second > b.second;
                 });
        
        // 是否为详细分析的体素
        bool is_verbose = (verbose_count < MAX_VERBOSE_VOXELS && !influences.empty());
        if (is_verbose) {
            verbose_count++;
            
            std::cout << "========= 体素 " << voxel_idx << " (" << voxel_x << "," << voxel_y << "," << voxel_z << ") 详细影响分析（去除空点后）=========" << std::endl;
            std::cout << "体素中心坐标: (" << voxel_center[0] << "," << voxel_center[1] << "," << voxel_center[2] << ")" << std::endl;
            std::cout << "找到 " << influences.size() << " 个有影响的高斯点" << std::endl;
            
            if (!influences.empty()) {
                float total_influence = 0.0f;
                for (const auto& infl : influences) {
                    total_influence += infl.second;
                }
                
                std::cout << "总影响值: " << total_influence << std::endl;
                std::cout << "前10个高斯点的影响:" << std::endl;
                for (size_t i = 0; i < std::min(size_t(10), influences.size()); i++) {
                    const auto& infl = influences[i];
                    int gs_idx = infl.first;
                    std::cout << "高斯点 " << gs_idx << ": 影响值=" << infl.second 
                              << ", 坐标=(" << means3D_data[gs_idx*3] << "," << means3D_data[gs_idx*3+1] << "," << means3D_data[gs_idx*3+2] << ")"
                              << ", 不透明度=" << opacity_data[gs_idx]
                              << ", 贡献百分比=" << (infl.second / total_influence * 100.0f) << "%" 
                              << std::endl;
                }
            }
        }
        
        // 限制每个体素的高斯点数量 (增加了保留数量)
        int count = std::min(50, static_cast<int>(influences.size()));
        
        // 填充输出张量
        for (int i = 0; i < count && pair_count < max_pairs; i++) {
            voxel_indices[pair_count] = voxel_idx;
            gaussian_indices[pair_count] = influences[i].first;
            influence_values[pair_count] = influences[i].second;
            pair_count++;
        }
    }
    
    std::cout << "Total voxel-gaussian pairs: " << pair_count << std::endl;
    
    if (pair_count == 0) {
        std::cout << "警告: 没有找到有效的高斯点-体素对!" << std::endl;
        
        // 创建空的输出张量
        auto empty_voxel_indices = torch::zeros({0}, torch::dtype(torch::kInt32).device(pts.device()));
        auto empty_gaussian_indices = torch::zeros({0}, torch::dtype(torch::kInt32).device(pts.device()));
        auto empty_influence_values = torch::zeros({0}, torch::dtype(torch::kFloat32).device(pts.device()));
        
        return std::make_tuple(0, empty_voxel_indices, empty_gaussian_indices, empty_influence_values);
    }
    
    // 裁剪输出张量到实际大小
    auto final_voxel_indices = voxel_indices.slice(0, 0, pair_count);
    auto final_gaussian_indices = gaussian_indices.slice(0, 0, pair_count);
    auto final_influence_values = influence_values.slice(0, 0, pair_count);
    
    return std::make_tuple(pair_count, final_voxel_indices, final_gaussian_indices, final_influence_values);
} 