/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Photo-SLAM.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <torch/torch.h>

const double DEG_TO_RAD = M_PI / 180.0;

namespace general_utils {

    inline torch::Tensor inverse_sigmoid(const torch::Tensor &x) {
        return torch::log(x / (1 - x));
    }

    inline torch::Tensor build_rotation(torch::Tensor &r) {
        auto r0 = r.index({torch::indexing::Slice(), 0});
        auto r1 = r.index({torch::indexing::Slice(), 1});
        auto r2 = r.index({torch::indexing::Slice(), 2});
        auto r3 = r.index({torch::indexing::Slice(), 3});
        auto norm = torch::sqrt(r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3);

        auto q = r / norm.unsqueeze(/*dim=*/1);
        r = q.index({torch::indexing::Slice(), 0});
        auto x = q.index({torch::indexing::Slice(), 1});
        auto y = q.index({torch::indexing::Slice(), 2});
        auto z = q.index({torch::indexing::Slice(), 3});

        auto R = torch::zeros({q.size(0), 3, 3}, torch::TensorOptions().device(torch::kCUDA));
        R.select(1, 0).select(1, 0).copy_(1 - 2 * (y * y + z * z));
        R.select(1, 0).select(1, 1).copy_(2 * (x * y - r * z));
        R.select(1, 0).select(1, 2).copy_(2 * (x * z + r * y));
        R.select(1, 1).select(1, 0).copy_(2 * (x * y + r * z));
        R.select(1, 1).select(1, 1).copy_(1 - 2 * (x * x + z * z));
        R.select(1, 1).select(1, 2).copy_(2 * (y * z - r * x));
        R.select(1, 2).select(1, 0).copy_(2 * (x * z - r * y));
        R.select(1, 2).select(1, 1).copy_(2 * (y * z + r * x));
        R.select(1, 2).select(1, 2).copy_(1 - 2 * (x * x + y * y));
        return R;
    }

    inline Sophus::SE3d perturbSE3Impl(const Eigen::Matrix4d &T, double d, double t, std::mt19937 &rand_gen) {
        // 将Eigen矩阵转换为Sophus的SE3类型
        Sophus::SE3d se3_T(T);

        // 1. 处理旋转扰动：保持旋转轴不变，角度在[-d, d]度范围内随机扰动
        const Eigen::Quaterniond &original_quat = se3_T.unit_quaternion();
        Eigen::AngleAxisd original_aa(original_quat);

        // 生成角度扰动 [-d, d]度
        std::uniform_real_distribution<double> angle_dist(-d, d);
        double delta_angle_rad = angle_dist(rand_gen) * DEG_TO_RAD;

        // 保持原旋转轴，叠加角度扰动
        Eigen::AngleAxisd perturbed_aa(original_aa.angle() + delta_angle_rad,
                                       original_aa.axis());
        // 将AngleAxisd转换为四元数
        Eigen::Quaterniond perturbed_quat(perturbed_aa);

        // 2. 处理平移扰动：x,y,z方向分别在[-t, t]范围内随机扰动
        std::uniform_real_distribution<double> trans_dist(-t, t);
        Eigen::Vector3d delta_trans(
                trans_dist(rand_gen),
                trans_dist(rand_gen),
                trans_dist(rand_gen)
        );

        // 3. 构建扰动后的SE3（使用四元数作为旋转参数）
        Sophus::SE3d se3_perturbed(perturbed_quat, se3_T.translation() + delta_trans);

        return se3_perturbed;
    }

    inline Sophus::SE3d perturbSE3(const Eigen::Matrix4d &T, double d, double t) {
        // 静态随机数生成器，第一次调用时初始化，之后保持不变
        static std::mt19937 rand_gen(std::random_device{}());

        return perturbSE3Impl(T, d, t, rand_gen);
    }

}
