//
// Created by zhiyu on 25-6-26.
//
#pragma once

#include <torch/torch.h>
#include "cuda_lanczos/lanczos.h"

class GaussianResolutionScheduler
{
public:
    GaussianResolutionScheduler(int reso_sample_num)
    {
        reso_sample_num_ = reso_sample_num;


    };

    inline torch::Tensor lanczos_resample(torch::Tensor input, float scale_factor){
        // Input shape default follows (H, W, C)
        auto input_h = input.size(0);
        auto input_w = input.size(1);

        // 计算输出尺寸
        int output_h = static_cast<int>(input_h / scale_factor);
        int output_w = static_cast<int>(input_w / scale_factor);

        return LanczosResampling(input, output_h, output_w, 2);
    };

//    inline float get_scale_from_alived_time(int alived_time){
//        if (alived_time>0)
//            return 1.0;
//
//        float scale = -0.062* alived_time + 8;
//
//        return scale;
//    };


    // EUROC 效果比较好的
    inline float get_scale_from_alived_time(int alived_time){
        if (alived_time>100) // euroc  100
            return 1.5;      // euroc  1.5

        float scale = -0.065* alived_time + 8;

        return scale;
    };

    // zed 效果比较好的
//    inline float get_scale_from_alived_time(int alived_time){
//        if (alived_time>50) // euroc  100
//            return 1.0;      // euroc  1.5
//
//        float scale = -(3.0/50.0)* alived_time + 4;
//
//        return scale;
//    };



public:
    int reso_sample_num_;

};

