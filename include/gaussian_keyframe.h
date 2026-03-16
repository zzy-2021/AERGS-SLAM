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

#include <memory>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>

#include "AirSLAM/3rdparty/Sophus/sophus/se3.hpp"

#include "types.h"
#include "gaussian_view.h"
#include "point2d.h"
#include "general_utils.h"
#include "graphics_utils.h"
#include "tensor_utils.h"

class GaussianKeyframe
{
public:
    GaussianKeyframe() {}

    GaussianKeyframe(std::size_t fid, int creation_iter = 0)
        : fid_(fid), creation_iter_(creation_iter) {

        //  把exposure_t_tensor_注册为可学习的参数
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        this->exposure_t_ = torch::tensor({1.0}, options.requires_grad(true));
        this->Tensor_vec_exposure_t_ = {this->exposure_t_};


        // 把exposure_a, b注册为可学习参数
        this->exp_a = torch::tensor({0.0}, options.requires_grad(true));
        this->exp_b = torch::tensor({0.0}, options.requires_grad(true));
        this->Tensor_vec_affine_ = {this->exp_a, this->exp_b};


//        this->wbalance_param_ = 0.33 * torch::zeros({3}, options.requires_grad(true));
//        this->Tensor_vec_wbalance_param_ = {this->wbalance_param_};

        torch::optim::AdamOptions adam_options;
        adam_options.set_lr(0.0);
        adam_options.eps() = 1e-15;
        this->optimizer_exposure_t_.reset(new torch::optim::Adam(this->Tensor_vec_exposure_t_, adam_options));
        this->optimizer_exposure_t_->param_groups()[0].options().set_lr(0.02);  // 这个要高一点  EuRoC 0.02

        this->optimizer_affine_.reset(new torch::optim::Adam(this->Tensor_vec_affine_, adam_options));
        this->optimizer_affine_->param_groups()[0].options().set_lr(0.001);  // 这个要高一点  EuRoC 0.02

//        this->optimizer_exposure_t_->add_param_group(this->Tensor_vec_wbalance_param_);
//        this->optimizer_exposure_t_->param_groups()[1].options().set_lr(0.01);


        // camera pose
        this->cam_rot_delta_ = torch::zeros({3},options.requires_grad(true));
        this->cam_trans_delta_ = torch::zeros({3},options.requires_grad(true));
        this->Tensor_vec_cam_rot_delta_ = {this->cam_rot_delta_};
        this->Tensor_vec_cam_trans_delta_ = {this->cam_trans_delta_};

        this->optimizer_pose_.reset(new torch::optim::Adam(Tensor_vec_cam_rot_delta_, adam_options));
        this->optimizer_pose_->param_groups()[0].options().set_lr(0.003);
        this->optimizer_pose_->add_param_group(Tensor_vec_cam_trans_delta_);
        this->optimizer_pose_->param_groups()[1].options().set_lr(0.001);
    }

    void setPose(
        const double qw,
        const double qx,
        const double qy,
        const double qz,
        const double tx,
        const double ty,
        const double tz);
    
    void setPose(
        const Eigen::Quaterniond& q,
        const Eigen::Vector3d& t);


    void setPose( const Eigen::Matrix4d& Tcw);

    Sophus::SE3d getPose();
    Sophus::SE3f getPosef();

    void setCameraParams(const GaussianView& camera);

    void setPoints2D(const std::vector<Eigen::Vector2d>& points2D);
    void setPoint3DIdxForPoint2D(
        const point2D_idx_t point2D_idx,
        const point3D_id_t point3D_id);

    void computeTransformTensors();

    Eigen::Matrix4f getWorld2View2(
        const Eigen::Vector3f& trans = {0.0f, 0.0f, 0.0f},
        float scale = 1.0f);

    torch::Tensor getProjectionMatrix(
        float znear,
        float zfar,
        float fovX,
        float fovY,
        torch::DeviceType device_type = torch::kCUDA);

    int getCurrentGausPyramidLevel();

    void updatePoseFromRender();

public:
    std::size_t fid_;
    double time_stamp_;
    int creation_iter_;
    int remaining_times_of_use_ = 0;

    int alived_time_ = 0;

    bool set_camera_ = false;
    torch::Tensor wbalance_param_;
    std::vector<torch::Tensor> Tensor_vec_wbalance_param_;

    torch::Tensor exposure_t_;
    std::shared_ptr<torch::optim::Adam> optimizer_exposure_t_;
    std::vector<torch::Tensor> Tensor_vec_exposure_t_;

    torch::Tensor exp_a, exp_b;
    std::shared_ptr<torch::optim::Adam> optimizer_affine_;
    std::vector<torch::Tensor> Tensor_vec_affine_;


    torch::Tensor cam_rot_delta_;
    torch::Tensor cam_trans_delta_;
    std::vector<torch::Tensor> Tensor_vec_cam_rot_delta_;
    std::vector<torch::Tensor> Tensor_vec_cam_trans_delta_;
    std::shared_ptr<torch::optim::Adam> optimizer_pose_;

    camera_id_t camera_id_;
    int camera_model_id_ = 0;

    std::string img_filename_;
    cv::Mat img_undist_, img_auxiliary_undist_;
    torch::Tensor original_image_; ///< image
    int image_width_;              ///< image
    int image_height_;             ///< image

    int num_gaus_pyramid_sub_levels_;
    std::vector<int> gaus_pyramid_times_of_use_;
    std::vector<std::size_t> gaus_pyramid_width_;            ///< gaus_pyramid image
    std::vector<std::size_t> gaus_pyramid_height_;           ///< gaus_pyramid image
    std::vector<torch::Tensor> gaus_pyramid_original_image_; ///< gaus_pyramid image
    // Tensor gt_alpha_mask_;

    std::vector<float> intr_; ///< intrinsics

    float FoVx_; ///< intrinsics
    float FoVy_; ///< intrinsics

    bool set_pose_ = false;
    bool set_projection_matrix_ = false;

    Eigen::Quaterniond R_quaternion_;  ///< extrinsics
    Eigen::Vector3d t_;                ///< extrinsics
    Sophus::SE3d Tcw_;                 ///< extrinsics
    Eigen::Matrix4f Twcf_;

    torch::Tensor R_tensor_; ///< extrinsics
    torch::Tensor t_tensor_; ///< extrinsics

    float zfar_ = 100.0f;
    float znear_ = 0.01f;

    Eigen::Vector3f trans_ = {0.0f, 0.0f, 0.0f};
    float scale_ = 1.0f;

    torch::Tensor world_view_transform_;    ///< transform tensors
    torch::Tensor projection_matrix_;       ///< transform tensors
    torch::Tensor full_proj_transform_;     ///< transform tensors
    torch::Tensor camera_center_;           ///< transform tensors

    std::vector<Point2D> points2D_;
    std::vector<float> kps_pixel_;
    std::vector<float> kps_point_local_;

    bool done_inactive_geo_densify_ = false;

    torch::Tensor edges_loss_norm_; /// timing gs 用
};
