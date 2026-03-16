//
// Created by zhiyu on 25-5-18.
//
#include "include/response_mapper.h"

torch::Tensor CFRMapper_depthwise::forward(torch::Tensor &x, torch::Tensor &exposure_t, torch::Tensor &log_wb_gains) {
    // 计算对数曝光组合 (与MLP版本保持一致)
    auto lne = x + torch::log(exposure_t) + log_wb_gains.unsqueeze(1).unsqueeze(2);  // 形状: [3, H, W]

    // 深度可分离卷积需要批次维度，增加一个维度
    auto input = lne.unsqueeze(0);  // 形状变为 [1, 3, H, W]

    // 通过卷积网络处理，直接得到输出
    auto output = conv->forward(input);  // 形状: [1, 3, H, W]

    // 移除批次维度，恢复为 [3, H, W] (与MLP版本输出形状一致)
    return output.squeeze(0);
}

torch::Tensor CFRMapper_depthwise::unit_exp_loss() {
    // 直接使用预定义的unit_input（已适配[1,3,1,1]），无需再调整维度
    auto unit_output = conv->forward(this->unit_input);

    // 仅需两步压缩维度，直接匹配unit_gt的[3,1]
    auto unit_output_flat = unit_output
            .squeeze(0)  // [1,3,1,1] → [3,1,1]
            .squeeze(-1); // [3,1,1] → [3,1]

    return torch::mse_loss(unit_output_flat, this->unit_gt);
}

torch::Tensor CFRMapper_mlp::forward(torch::Tensor &x, torch::Tensor &expsure_t) {
    // 计算对数曝光组合 (保留梯度)

    const int H = x.size(1);
    const int W = x.size(2);
    const int WH = H * W;

    auto lne = x + torch::log(expsure_t);  // 形状: [3, H, W]

    // 恢复形状 - 使用预存的H和W，避免重复查询尺寸
    auto out_r = mlp_r->forward(lne[0].view({WH, 1})).view({H, W});
    auto out_g = mlp_g->forward(lne[1].view({WH, 1})).view({H, W});
    auto out_b = mlp_b->forward(lne[2].view({WH, 1})).view({H, W});

    // 合并通道 - 直接使用已计算的张量构建结果
    return torch::stack({out_r, out_g, out_b}, 0);  // [3, H, W]
}





torch::Tensor CFRMapper_mlp::unit_exp_loss(){

    auto unit_output_r = mlp_r->forward(this->unit_input);
    auto unit_output_g = mlp_g->forward(this->unit_input);
    auto unit_output_b = mlp_b->forward(this->unit_input);

    auto unit_output = torch::stack({unit_output_r, unit_output_g, unit_output_b}, 0);  // [3, 1]

    return torch::mse_loss(unit_output, this->unit_gt);
}

void CFRMapper_mlp::save_CFR(const std::string& path, float a, float b, int N) {
    // 验证范围有效性
    if (a >= b) {
        std::cerr << "错误: 范围参数 a 必须小于 b (当前 a=" << a << ", b=" << b << ")" << std::endl;
        return;
    }

    // 设置设备为CUDA
    torch::Device device(torch::kCUDA);

    // 生成[a, b]之间的均匀分布有序序列
    torch::Tensor lnx = torch::linspace(a, b, N, device);

    // repeat成[1000, 1]形状
    lnx = lnx.view({N, 1});
    auto t = torch::tensor({1.0}, device);

    auto color_r = mlp_r ->forward(lnx);
    auto color_g = mlp_g ->forward(lnx);
    auto color_b = mlp_b ->forward(lnx);

    lnx = lnx.squeeze().to(torch::kCPU).reshape(-1);
    color_r = color_r.squeeze().to(torch::kCPU).reshape(-1);
    color_g = color_g.squeeze().to(torch::kCPU).reshape(-1);
    color_b = color_b.squeeze().to(torch::kCPU).reshape(-1);

    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "错误: 无法打开文件 " << path << std::endl;
        return;
    }
    file << "lnx color_r color_g color_b" << std::endl;

    // 写入数据行
    for (int i = 0; i < N; ++i) {
        file << lnx[i].item<float>() << " "
             << color_r[i].item<float>() << " "
             << color_g[i].item<float>() << " "
             << color_b[i].item<float>() << std::endl;
    }

    // 关闭文件
    file.close();
    std::cout << "成功将张量保存到文件: " << path << " (范围: [" << a << ", " << b << "])" << std::endl;
}
