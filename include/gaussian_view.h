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

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/cudawarping.hpp>

#include "types.h"
#include "tensor_utils.h"

class GaussianView
{
public:
    GaussianView() {}

    GaussianView (
        camera_id_t camera_id,
        std::size_t width,
        std::size_t height,
        std::vector<double> params,
        int model_id = 0)
        : camera_id_(camera_id),
          width_(width),
          height_(height),
          params_(params),
          model_id_(model_id){}

    enum CameraModelType{
        INVALID = 0,
        PINHOLE = 1};

public:
    inline void setModelId(const CameraModelType model_id)
    {
        model_id_ = model_id;
        switch (model_id_)
        {
        case PINHOLE: // Pinhole
            params_.resize(4);
            break;

        default:
            break;
        }
    }


public:
    camera_id_t camera_id_ = 0U;

    int model_id_ = 0;

    std::size_t width_ = 0UL;
    std::size_t height_ = 0UL;

    int num_gaus_pyramid_sub_levels_ = 0;
    std::vector<std::size_t> gaus_pyramid_width_;
    std::vector<std::size_t> gaus_pyramid_height_;

    std::vector<double> params_;
};
