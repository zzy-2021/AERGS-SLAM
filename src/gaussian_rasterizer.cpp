/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * This file is Derivative Works of Gaussian Splatting,
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023,
 * as part of Photo-SLAM.
 */

#include "include/gaussian_rasterizer.h"

torch::Tensor
GaussianRasterizer::markVisibleGaussians(
        torch::Tensor &positions) {
    // Mark visible points (based on frustum culling for camera) with a boolean
    torch::NoGradGuard no_grad;
    auto raster_settings = this->raster_settings_;
    return markVisible(positions, raster_settings.viewmatrix_, raster_settings.projmatrix_);
}

/// 第3层
torch::autograd::tensor_list
GaussianRasterizerFunction::forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor means3D,
        torch::Tensor means2D,
        torch::Tensor sh,
        torch::Tensor colors_precomp,
        torch::Tensor opacities,
        torch::Tensor scales,
        torch::Tensor rotations,
        torch::Tensor cov3Ds_precomp,
        torch::Tensor theta,
        torch::Tensor rho,
        GaussianRasterizationSettings raster_settings) {

    auto rasterization_result = RasterizeGaussiansCUDA(
            raster_settings.bg_,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier_,
            cov3Ds_precomp,
            raster_settings.viewmatrix_,
            raster_settings.projmatrix_,
            raster_settings.projmatrix_raw_,
            raster_settings.tanfovx_,
            raster_settings.tanfovy_,
            raster_settings.image_height_,
            raster_settings.image_width_,
            sh,
            raster_settings.sh_degree_,
            raster_settings.campos_,
            raster_settings.prefiltered_,
            false
    );

    auto num_rendered = std::get<0>(rasterization_result);
    auto color = std::get<1>(rasterization_result);
    auto radii = std::get<2>(rasterization_result);
    auto geomBuffer = std::get<3>(rasterization_result);
    auto binningBuffer = std::get<4>(rasterization_result);
    auto imgBuffer = std::get<5>(rasterization_result);

    auto depth = std::get<6>(rasterization_result);
    auto opacity = std::get<7>(rasterization_result);
    auto n_touched = std::get<8>(rasterization_result);

    // Keep relevant tensors for backward
    ctx->saved_data["num_rendered"] = num_rendered;
    ctx->saved_data["scale_modifier"] = raster_settings.scale_modifier_;
    ctx->saved_data["tanfovx"] = raster_settings.tanfovx_;
    ctx->saved_data["tanfovy"] = raster_settings.tanfovy_;
    ctx->saved_data["sh_degree"] = raster_settings.sh_degree_;

    ctx->save_for_backward({raster_settings.bg_,
                            raster_settings.viewmatrix_,
                            raster_settings.projmatrix_,
                            raster_settings.projmatrix_raw_,
                            raster_settings.campos_,
                            colors_precomp,
                            means3D,
                            scales,
                            rotations,
                            cov3Ds_precomp,
                            radii,
                            sh,
                            geomBuffer,
                            binningBuffer,
                            imgBuffer});

    return {color, radii, depth, opacity, n_touched};
}

torch::autograd::tensor_list
GaussianRasterizerFunction::backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs) {
    // Restore necessary values from context
    auto num_rendered = ctx->saved_data["num_rendered"].toInt();
    auto scale_modifier = static_cast<float>(ctx->saved_data["scale_modifier"].toDouble());
    auto tanfovx = static_cast<float>(ctx->saved_data["tanfovx"].toDouble());
    auto tanfovy = static_cast<float>(ctx->saved_data["tanfovy"].toDouble());
    auto sh_degree = ctx->saved_data["sh_degree"].toInt();

    auto saved = ctx->get_saved_variables();

    auto bg = saved[0];
    auto viewmatrix = saved[1];
    auto projmatrix = saved[2];
    auto projmatrix_raw = saved[3];
    auto campos = saved[4];
    auto colors_precomp = saved[5];
    auto means3D = saved[6];
    auto scales = saved[7];
    auto rotations = saved[8];
    auto cov3Ds_precomp = saved[9];
    auto radii = saved[10];
    auto sh = saved[11];
    auto geomBuffer = saved[12];
    auto binningBuffer = saved[13];
    auto imgBuffer = saved[14];

    // Compute gradients for relevant tensors by invoking backward method
    auto grad_out_color = grad_outputs[0];
    auto grad_out_depth = grad_outputs[2];

    auto rasterization_backward_result = RasterizeGaussiansBackwardCUDA(
            bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            scale_modifier,
            cov3Ds_precomp,
            viewmatrix,
            projmatrix,
            projmatrix_raw,
            tanfovx,
            tanfovy,
            grad_out_color,
            grad_out_depth,
            sh,
            sh_degree,
            campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            false
    );


    auto grad_tau = std::get<8>(rasterization_backward_result); /*dL_dtau*/
    // 先进行维度重塑（-1表示自动计算维度）
    auto reshaped_grad_tau = grad_tau.view({-1, 6});
    // 沿dim=0求和（压缩维度）
    grad_tau = torch::sum(reshaped_grad_tau, /*dim=*/0);

    // 切片操作（取前3个元素）
    auto grad_rho = grad_tau.index({torch::indexing::Slice(0, 3)})
            .view({1, -1})
            .contiguous();  // 确保内存连续性

    auto grad_theta = grad_tau.index({torch::indexing::Slice(3, torch::indexing::None)})
            .view({1, -1})
            .contiguous();

    return {
            std::get<3>(rasterization_backward_result)/*dL_dmeans3D*/,
            std::get<0>(rasterization_backward_result)/*dL_dmeans2D*/,
            std::get<5>(rasterization_backward_result)/*dL_dsh*/,
            std::get<1>(rasterization_backward_result)/*dL_dcolors*/,
            std::get<2>(rasterization_backward_result)/*dL_dopacity*/,
            std::get<6>(rasterization_backward_result)/*dL_dscales*/,
            std::get<7>(rasterization_backward_result)/*dL_drotations*/,
            std::get<4>(rasterization_backward_result)/*dL_dcov3D*/,
            grad_theta,
            grad_rho,
            torch::Tensor()
    };
}

/// 第1层
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianRasterizer::forward(
        torch::Tensor means3D,
        torch::Tensor means2D,
        torch::Tensor opacities,
        bool has_shs,
        bool has_colors_precomp,
        bool has_scales,
        bool has_rotations,
        bool has_cov3D_precomp,
        torch::Tensor shs,
        torch::Tensor colors_precomp,
        torch::Tensor scales,
        torch::Tensor rotations,
        torch::Tensor cov3D_precomp,
        torch::Tensor theta,
        torch::Tensor rho) {
    auto raster_settings = this->raster_settings_;

    if ((!has_shs/*shs is None*/ && !has_colors_precomp/*colors_precomp is None*/)
        || (has_shs/*shs is not None*/ && has_colors_precomp/*colors_precomp is not None*/))
        throw std::runtime_error("Please provide excatly one of either SHs or precomputed colors!");

    if (((!has_scales/*scales is None*/ || !has_rotations/*rotations is None*/) &&
         !has_cov3D_precomp/*cov3D_precomp is None*/)
        || ((has_scales/*scales is not None*/ || has_rotations/*rotations is not None*/) &&
            has_cov3D_precomp/*cov3D_precomp is not None*/))
        throw std::runtime_error(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!");

    torch::TensorOptions options;
    if (!has_shs)
        shs = torch::tensor({}, options.device(torch::kCUDA));
    if (!has_colors_precomp)
        colors_precomp = torch::tensor({}, options.device(torch::kCUDA));
    if (!has_scales)
        scales = torch::tensor({}, options.device(torch::kCUDA));
    if (!has_rotations)
        rotations = torch::tensor({}, options.device(torch::kCUDA));
    if (!has_cov3D_precomp)
        cov3D_precomp = torch::tensor({}, options.device(torch::kCUDA));

    auto result = rasterizeGaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            theta,
            rho,
            raster_settings
    );

    return std::make_tuple(result[0]/*color*/, result[1]/*radii*/, result[2]/*depth*/, result[3]/*opacity*/, result[4]/*n_touched*/);   ///返回值要改
}
