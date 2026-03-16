//
// Created by zhiyu on 25-5-18.
//
#pragma once
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

class CFRMapper_depthwise : public torch::nn::Module {
public:

    CFRMapper_depthwise(int hidden_dim) {
        // 深度可分离卷积网络 - 替代原来的三个MLP
        // 输入形状: [3, H, W]，输出形状: [3, H, W]
        int channels = 3;
        // 定义三通道独立的深度可分离卷积模块
        conv = register_module("conv", torch::nn::Sequential(
                // 1. 1x1深度卷积（每个通道隔离）
                torch::nn::Conv2d(
                        torch::nn::Conv2dOptions(3, 3, 3)
                                .groups(3)
                                .padding(1) // 关键：3个分组，每个通道独立处理
                                .bias(false)
                ),
                torch::nn::LeakyReLU(),

                // 2. 1x1分组卷积（通道内扩展，不跨组融合）
                torch::nn::Conv2d(
                        torch::nn::Conv2dOptions(3, 3 * hidden_dim, 1)
                                .groups(3)
                                .bias(true)
                ),
                torch::nn::LeakyReLU(),

                // 3. 1x1分组卷积（还原通道数，保持独立）
                torch::nn::Conv2d(
                        torch::nn::Conv2dOptions(3 * hidden_dim, 3, 1)
                                .groups(3)  // 最终分组输出，确保通道无融合
                                .bias(true)
                ),
                torch::nn::Sigmoid()
        ));

        // 优化器
        optimizer_cfr_ = std::make_shared<torch::optim::Adam>(
                parameters(),
                torch::optim::AdamOptions(0.01)
        );

        this->to(torch::kCUDA);
    }


//    CFRMapper_depthwise(int hidden_dim) {
//        // 深度可分离卷积网络 - 替代原来的三个MLP
//        // 输入形状: [3, H, W]，输出形状: [3, H, W]
//        int channels = 3;
//        conv = register_module("conv", torch::nn::Sequential(
//                torch::nn::Conv2d(
//                        torch::nn::Conv2dOptions(channels, channels, 3)
//                                .groups(channels)  // 深度卷积关键参数
//                                .padding(1)        // 保持空间尺寸
//                                .bias(true)
//                ),
//                torch::nn::ReLU(),
//
//                // 逐点卷积 - 1x1卷积融合信息
//                torch::nn::Conv2d(
//                        torch::nn::Conv2dOptions(channels, hidden_dim, 1)
//                                .bias(true)
//                ),
//                torch::nn::ReLU(),
//
//                // 输出层 - 映射回3个通道并使用Sigmoid
//                torch::nn::Conv2d(
//                        torch::nn::Conv2dOptions(hidden_dim, channels, 1)
//                                .bias(true)
//                ),
//                torch::nn::Sigmoid()
//        ));
//
//        // 优化器
//        optimizer_cfr_ = std::make_shared<torch::optim::Adam>(
//                parameters(),
//                torch::optim::AdamOptions(0.005)
//        );
//
//        this->to(torch::kCUDA);
//    }

    // 前向传播
    torch::Tensor forward(torch::Tensor &x, torch::Tensor &exposure_t, torch::Tensor &log_wb_gains);

    torch::Tensor unit_exp_loss();

    void save_CFR(const std::string& path, float a, float b, int N);

    std::shared_ptr<torch::optim::Adam> optimizer_cfr_;

private:
    torch::nn::Sequential conv{nullptr};
    torch::Tensor unit_gt = 0.73 * torch::ones({3, 1}, torch::kFloat32).to(torch::kCUDA);
    torch::Tensor unit_input = torch::zeros({1, 3, 3, 3}, torch::kFloat32).to(torch::kCUDA);
};





/// we 2D-mapper the log radiance (ln e)+ log exposure time (ln t) to color
/// input is the ln e (3, H, W) tensor
/// output is the color (3, H, W) tensor
/// gray image using the same mlps
class CFRMapper_mlp: public torch::nn::Module{

public:
    CFRMapper_mlp(int W){



//        wb_r = register_parameter("wb_r", torch::ones({1}, torch::kFloat32));
//        wb_g = register_parameter("wb_g", torch::ones({1}, torch::kFloat32));
//        wb_b = register_parameter("wb_b", torch::ones({1}, torch::kFloat32));


        // 定义 MLP: [3, W, W, 3], 中间层 ReLU, 输出层 Sigmoid
        mlp_r = register_module("mlp_r", torch::nn::Sequential(
                torch::nn::Linear(1, W),
                torch::nn::ReLU(),
                torch::nn::Linear(W, 1),
                torch::nn::Sigmoid()
        ));

        mlp_g = register_module("mlp_g", torch::nn::Sequential(
                torch::nn::Linear(1, W),
                torch::nn::ReLU(),
                torch::nn::Linear(W, 1),
                torch::nn::Sigmoid()
        ));

        mlp_b = register_module("mlp_b", torch::nn::Sequential(
                torch::nn::Linear(1, W),
                torch::nn::ReLU(),
                torch::nn::Linear(W, 1),
                torch::nn::Sigmoid()
        ));

        // 直接使用当前模块的所有参数，无需手动收集
        optimizer_cfr_ = std::make_shared<torch::optim::Adam>(
                parameters(),  // 优化当前模块的所有参数（包括子模块）
                torch::optim::AdamOptions(0.005)  // 学习率设为 0.001
        );

        this->to(torch::kCUDA); // 强制所有参数和缓冲区移动到CUDA

    }
    torch::Tensor forward( torch::Tensor& x, torch::Tensor& expsure_t);

    torch::Tensor unit_exp_loss();

    void save_CFR(const std::string& path, float a, float b, int N);

    std::shared_ptr<torch::optim::Adam> optimizer_cfr_;


private:
    torch::Tensor wb_r, wb_g, wb_b;
    torch::nn::Sequential mlp_r{nullptr}, mlp_g{nullptr}, mlp_b{nullptr};
    torch::Tensor unit_gt = 0.73 * torch::ones({3, 1}, torch::kFloat32).to(torch::kCUDA);    // 0.73-> 0.7
    torch::Tensor unit_input = torch::zeros({1}, torch::kFloat32).to(torch::kCUDA);
};
