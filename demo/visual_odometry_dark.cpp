#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <ros/ros.h>
#include <thread>

#include "read_configs.h"
#include "dataset.h"
#include "map_builder.h"
#include "include/gaussian_mapper.h"
//#include "viewer/imgui_viewer.h"

cv::RNG rng(119);


cv::Mat randomAdjustSaturation(cv::Mat& bgrImg, float Scale)
{
    // 1. 输入合法性校验
    if (bgrImg.empty()) {
        throw std::invalid_argument("输入图像为空！");
    }
    if (bgrImg.channels() != 3) {
        throw std::invalid_argument("输入必须是3通道BGR图像！当前通道数：" + std::to_string(bgrImg.channels()));
    }
    if (bgrImg.type() != CV_8UC3) {
        throw std::invalid_argument("输入必须是8位无符号3通道图像（CV_8UC3）！");
    }


    // 3. BGR转HSV（HSV空间中S通道单独控制饱和度）
    cv::Mat hsvImg;
    cv::cvtColor(bgrImg, hsvImg, cv::COLOR_BGR2HSV);

    // 4. 遍历像素调整饱和度（S通道），超出范围自动clip
    int height = hsvImg.rows;
    int width = hsvImg.cols;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            cv::Vec3b& hsv = hsvImg.at<cv::Vec3b>(y, x);
            // HSV通道说明：
            // hsv[0] = H（色相，0-179），hsv[1] = S（饱和度，0-255），hsv[2] = V（明度，0-255）
            uchar originalS = hsv[1];
            // 调整饱和度：乘以随机系数，然后clip到0-255
            float newS = originalS * Scale;
            hsv[1] = static_cast<uchar>(std::clamp(newS, 0.0f, 255.0f)); // clip核心操作
        }
    }

    // 5. HSV转回BGR
    cv::Mat resultImg;
    cv::cvtColor(hsvImg, resultImg, cv::COLOR_HSV2BGR);

    return resultImg;
}

cv::Mat grayToColorWithCoeff(const cv::Mat& grayImg, float coeffB = 0.8f, float coeffG = 0.7f, float coeffR = 0.9f)
{
    // 1. 输入合法性校验
    if (grayImg.empty()) {
        throw std::invalid_argument("输入的灰度图像为空！");
    }
    if (grayImg.channels() != 1) {
        throw std::invalid_argument("输入必须是单通道灰度图！当前通道数：" + std::to_string(grayImg.channels()));
    }
    if (grayImg.type() != CV_8UC1) {
        throw std::invalid_argument("输入必须是8位无符号灰度图（CV_8UC1）！");
    }

    // 2. 初始化3通道BGR图像（先复制灰度值到所有通道）
    cv::Mat colorImg(grayImg.size(), CV_8UC3);
    for (int y = 0; y < grayImg.rows; y++) {
        for (int x = 0; x < grayImg.cols; x++) {
            // 获取当前像素的灰度值
            uchar grayVal = grayImg.at<uchar>(y, x);

            // 核心规则：每个通道 = 灰度值 × 固定系数，且截断在0-255
            uchar bVal = static_cast<uchar>(std::clamp(grayVal * coeffB, 0.0f, 255.0f));
            uchar gVal = static_cast<uchar>(std::clamp(grayVal * coeffG, 0.0f, 255.0f));
            uchar rVal = static_cast<uchar>(std::clamp(grayVal * coeffR, 0.0f, 255.0f));

            // 赋值给BGR通道（OpenCV通道顺序：B→G→R）
            colorImg.at<cv::Vec3b>(y, x) = cv::Vec3b(bVal, gVal, rVal);
        }
    }

    return colorImg;
}



bool readTUMTrajectory(const std::string &file_path,
                       std::vector<std::string> &timestamps,
                       std::vector<Eigen::Matrix4d> &poses) {
    // 初始化容器
    timestamps.clear();
    poses.clear();
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "文件打开失败: " << file_path << std::endl;
        return false;
    }

    std::string line;
    // 逐行解析
    for (int line_idx = 0; std::getline(file, line); line_idx++) {
        // 跳过空行/注释行
        if (line.empty() || line[0] == '#') continue;

        // 解析8列数据（时间戳+3平移+4四元数）
        std::istringstream iss(line);

        // 第一列是整数时间戳，直接作为字符串读取
        std::string ts_str;
        if (!(iss >> ts_str)) {
            std::cerr << "第" << line_idx + 1 << "行格式错误：无法读取时间戳" << std::endl;
            return false;
        }

        // 读取剩下的7个数值（3平移+4四元数）
        double data[7];
        for (int i = 0; i < 7; i++) {
            if (!(iss >> data[i])) {
                std::cerr << "第" << line_idx + 1 << "行格式错误：在第" << i + 2 << "列" << std::endl;
                return false;
            }
        }

        // 1. 时间戳直接使用读取的字符串
        timestamps.push_back(ts_str);

        // 2. 构造4x4位姿矩阵
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        // 平移向量 (tx, ty, tz) - 对应数据中的前3个数值
        pose.block<3, 1>(0, 3) = Eigen::Vector3d(data[0], data[1], data[2]);
        // 四元数转旋转矩阵（qx,qy,qz,qw → 归一化）- 对应数据中的后4个数值
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        pose.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
        poses.push_back(pose);
    }
    file.close();

    // 基础校验
    if (timestamps.size() != poses.size()) {
        std::cerr << "时间戳与位姿数量不匹配" << std::endl;
        return false;
    }
    std::cout << "成功读取 " << timestamps.size() << " 帧数据" << std::endl;
    return true;
}


/**
 * 从曝光因子文件加载时间戳与对应因子到map
 * @param file_path MH01_factor.txt等文件的路径
 * @param name_factors 存储结果的map，key为时间戳字符串，value为曝光因子
 * @return 加载成功返回true，否则返回false
 */
bool loadExposureFactors(const std::string &file_path_str, std::map<std::string, float> &name_factors) {
    // 清空输出map，避免残留数据影响
    name_factors.clear();

    // 将string转换为filesystem::path用于文件操作
    std::filesystem::path file_path(file_path_str);

    // 检查文件是否存在
    if (!std::filesystem::exists(file_path)) {
        throw std::invalid_argument("文件不存在: " + file_path_str);
    }

    // 检查是否是常规文件（不是目录）
    if (!std::filesystem::is_regular_file(file_path)) {
        throw std::invalid_argument("路径不是文件: " + file_path_str);
    }

    // 尝试打开文件
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + file_path_str + "，可能是权限问题");
    }

    std::string line;
    unsigned int line_num = 0;

    // 逐行读取文件
    while (std::getline(file, line)) {
        line_num++;

        // 跳过空行
        if (line.empty()) {
            continue;
        }

        // 处理可能包含注释的行（#后面的内容忽略）
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
            // 如果处理后变成空行，则跳过
            if (line.empty()) {
                continue;
            }
        }

        // 分割行内容（时间戳和曝光因子）
        std::istringstream iss(line);
        std::string timestamp;
        float factor;

        // 提取时间戳和因子，检查格式是否正确
        if (!(iss >> timestamp >> factor)) {
            // 抛出包含详细信息的异常
            throw std::runtime_error("文件 " + file_path_str +
                                     " 第 " + std::to_string(line_num) +
                                     " 行格式错误: " + line);
        }

        // 检查是否有多余内容
        std::string extra;
        if (iss >> extra) {
            throw std::runtime_error("文件 " + file_path_str +
                                     " 第 " + std::to_string(line_num) +
                                     " 行包含多余内容: " + extra);
        }

        // 检查曝光因子是否在合理范围（根据你的数据调整为0.3~2.0）
        if (factor < 0.3f - 1e-6f || factor > 2.0f + 1e-6f) {
            throw std::runtime_error("文件 " + file_path_str +
                                     " 第 " + std::to_string(line_num) +
                                     " 行曝光因子超出合理范围: " + std::to_string(factor));
        }

        // 检查是否有重复的时间戳
        if (name_factors.find(timestamp) != name_factors.end()) {
            throw std::runtime_error("文件 " + file_path_str +
                                     " 第 " + std::to_string(line_num) +
                                     " 行存在重复时间戳: " + timestamp);
        }

        // 插入到map中
        name_factors[timestamp] = factor;
    }

    // 检查文件读取过程是否出现错误
    if (file.bad()) {
        throw std::runtime_error("读取文件 " + file_path_str + " 时发生错误");
    }

    // 检查是否读取到数据
    if (name_factors.empty()) {
        throw std::runtime_error("文件 " + file_path_str + " 中没有有效数据");
    }

    return true;
}


void adjustGainWithNoise(cv::Mat &image, float exposure_gain, float base_noise_std = 0.01) {

    // 1. 检查输入有效性
    if (image.empty()) {
        throw std::invalid_argument("输入图像为空，无法进行曝光调整");
    }

    image.convertTo(image, CV_32F, 1.0 / 255);

    image = image * exposure_gain;

    float noise_std = base_noise_std * pow(exposure_gain, 2);
    cv::Mat noise(image.size(), CV_32F);

    rng.fill(noise, cv::RNG::NORMAL, 0.0, noise_std);

    image = image + noise;
    cv::threshold(image, image, 1.0, 1.0, cv::THRESH_TRUNC);  // 超过1的截断为1
    cv::threshold(image, image, 0.0, 0.0, cv::THRESH_TOZERO);    // 低于0的置为0
    image = image * 255;
    cv::Mat result;
    image.convertTo(image, CV_8U);

}

// 调整单张图像的曝光（原地修改输入图像）
void adjustExposure(cv::Mat &image, float exposureFactor) {
    // 1. 检查输入有效性
    if (image.empty()) {
        throw std::invalid_argument("输入图像为空，无法进行曝光调整");
    }
    if (exposureFactor <= 0) {
        throw std::invalid_argument("曝光因子必须为正数，当前值: " + std::to_string(exposureFactor));
    }

    // 2. 处理单通道灰度图
    if (image.channels() == 1) {
        // 转换为浮点型避免计算精度丢失，调整后裁剪到0-255范围
        image.convertTo(image, CV_32F);
        image *= exposureFactor;
        cv::threshold(image, image, 255.0, 255.0, cv::THRESH_TRUNC);  // 超过255的截断为255
        cv::threshold(image, image, 0.0, 0.0, cv::THRESH_TOZERO);    // 低于0的置为0
        image.convertTo(image, CV_8U);  // 转回8位无符号整型
    }
        // 3. 处理3通道RGB图（通过HSV空间调整亮度通道，避免色彩失真）
    else if (image.channels() == 3) {
        cv::Mat hsv;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);  // BGR转HSV（OpenCV默认存储格式为BGR）

        // 拆分HSV通道（H:色调, S:饱和度, V:亮度）
        std::vector<cv::Mat> hsv_channels;
        cv::split(hsv, hsv_channels);

        // 仅调整V通道（亮度），逻辑同灰度图
        hsv_channels[2].convertTo(hsv_channels[2], CV_32F);
        hsv_channels[2] *= exposureFactor;
        cv::threshold(hsv_channels[2], hsv_channels[2], 255.0, 255.0, cv::THRESH_TRUNC);
        cv::threshold(hsv_channels[2], hsv_channels[2], 0.0, 0.0, cv::THRESH_TOZERO);
        hsv_channels[2].convertTo(hsv_channels[2], CV_8U);

        // 合并通道并转回BGR格式
        cv::merge(hsv_channels, hsv);
        cv::cvtColor(hsv, image, cv::COLOR_HSV2BGR);
    }
        // 4. 不支持的通道类型
    else {
        throw std::invalid_argument("不支持的图像通道数: " + std::to_string(image.channels()) +
                                    "，仅支持单通道灰度图和3通道RGB图");
    }
}

int mode = 1; // do not change

int main(int argc, char **argv) {

    // 打印参数个数（argc 本身包含程序名作为第一个参数）
    std::cout << "参数总数: " << argc << std::endl;

    // 循环打印每个参数
    for (int i = 0; i < argc; ++i) {
        std::cout << "参数 " << i << ": " << argv[i] << std::endl;
    }

    ros::init(argc, argv, "AERGS_SLAM");

    std::string config_path, mr_config_path, gs_config_path, model_dir, voc_path, exposure_factor_path;
    ros::param::get("~config_path", config_path);
    ros::param::get("~mr_config_path", mr_config_path);
    ros::param::get("~model_dir", model_dir);
    ros::param::get("~voc_path", voc_path);
    ros::param::get("~gaussian_config_path", gs_config_path);
    ros::param::get("~exposure_file_path", exposure_factor_path);

    VisualOdometryConfigs configs(config_path, model_dir);
    MapRefinementConfigs mr_config(mr_config_path, model_dir);

    std::cout << "config done" << std::endl;

    ros::param::get("~dataroot", configs.dataroot);
    ros::param::get("~camera_config_path", configs.camera_config_path);
    ros::param::get("~saving_dir", configs.saving_dir);


    std::string cache_path_gt = configs.saving_dir + "/cache/gt";
    std::string cache_path_render = configs.saving_dir + "/cache/render";
    std::filesystem::create_directories(cache_path_gt);
    std::filesystem::create_directories(cache_path_render);

    if (!std::filesystem::exists(configs.saving_dir)) {
        std::filesystem::create_directories(configs.saving_dir);
    }

    std::map<std::string, float> name_factors;
    loadExposureFactors(exposure_factor_path, name_factors);

    ros::NodeHandle nh;
    std::shared_ptr<MapBuilder> pSLAM = std::make_shared<MapBuilder>(configs, mr_config, voc_path, nh);

    std::cout << "map_builder done" << std::endl;
    Dataset dataset(configs.dataroot, pSLAM->UseIMU());
    size_t dataset_length = dataset.GetDatasetLength();
    std::cout << "dataset done" << std::endl;

    // Device
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }

    std::filesystem::path result_dir = std::filesystem::path(configs.saving_dir);
    std::filesystem::path gaussian_config_file_path = std::filesystem::path(gs_config_path);
    std::shared_ptr<GaussianMapper> pGausMapper = std::make_shared<GaussianMapper>(pSLAM, gaussian_config_file_path,
                                                                                   result_dir, 0, device_type);
    std::thread training_thd(&GaussianMapper::run, pGausMapper.get());

    double sum_time = 0;
    int image_num = 0;
    for (size_t i = 0; i < dataset_length && ros::ok(); ++i) {
        cv::Mat image_left, image_right;
        double timestamp;
        std::string name;
        ImuDataList batch_imu_data;
        if (!dataset.GetData(i, image_left, image_right, batch_imu_data, timestamp, name)) continue;

        float ex_factor = name_factors[name];

        if (mode == 1)
        {
            adjustExposure(image_left, ex_factor);
            adjustExposure(image_right, ex_factor);
        } else if (mode == 2)
        {
            adjustGainWithNoise(image_left, ex_factor);
            adjustGainWithNoise(image_right, ex_factor);
        } else
        {
            image_left = grayToColorWithCoeff(image_left);
            image_right = grayToColorWithCoeff(image_right);

            image_left = randomAdjustSaturation(image_left, ex_factor);
            image_right = randomAdjustSaturation(image_right, ex_factor);
        }

        cv::imshow("image_left", image_left);
        cv::waitKey(2);

        InputDataPtr data = std::shared_ptr<InputData>(new InputData());
        data->index = i;
        data->time = timestamp;
        data->image_left = image_left;
        data->image_right = image_right;
        data->batch_imu_data = batch_imu_data;
        data->name = name;

        auto before_infer = std::chrono::high_resolution_clock::now();
        pSLAM->AddInput(data);
        auto after_infer = std::chrono::high_resolution_clock::now();
        auto cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(after_infer - before_infer).count();
        sum_time += (double) cost_time;
        image_num++;
    }
    std::cout << "Average FPS = " << image_num / (sum_time / 1000.0) << std::endl;

    std::cout << "Starting to refining..." << std::endl;
    pSLAM->Refining();

    std::cout << "Waiting to stop..." << std::endl;
    pSLAM->Stop();
    while (!pSLAM->IsStopped()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << "Map building has been stopped" << std::endl;
    ros::shutdown();
    ros::waitForShutdown();

    if (training_thd.joinable()) {
        training_thd.join();
    }

    // 在此处保存pose
    std::string trajectory_keyframe_path = ConcatenateFolderAndFileName(configs.saving_dir,
                                                                        "KeyFrameTrajectory_TUM.txt");
    pSLAM->SaveTrajectory(trajectory_keyframe_path);

    std::string trajectory_allframe_path = ConcatenateFolderAndFileName(configs.saving_dir, "CameraTrajectory_TUM.txt");
    pSLAM->SaveAllFrameTrajectory(trajectory_allframe_path);



    ////   evaluating code, exclude training data ////
    std::vector<std::string> ts_list, ts_list_kf;
    std::vector<Eigen::Matrix4d> pose_list, _;

    readTUMTrajectory(trajectory_keyframe_path, ts_list_kf, _);
    if (readTUMTrajectory(trajectory_allframe_path, ts_list, pose_list)) {

        float avg_pnsr = 0;
        float avg_ssim = 0;

        std::vector<std::pair<float, float>> frame_results;
        namespace fs = std::filesystem;

        fs::path saving_dir(configs.saving_dir);
        if (!fs::exists(saving_dir)) fs::create_directories(saving_dir);

        std::string frame_results_path = (saving_dir / "frame_metric.txt").string();
        std::string final_stats_path = (saving_dir / "final_metric.txt").string();
        std::string exposure_results = (saving_dir / "exposure_metric.txt").string();

        std::ofstream frame_file(frame_results_path);
        std::ofstream exposure_file(exposure_results);

        if (!frame_file.is_open()) {
            std::cerr << "无法打开帧结果文件: " << frame_results_path << std::endl;
            return -1;
        }
        frame_file << "frame_index\tpsnr(dB)\tssim" << std::endl;
        exposure_file << "frame_index\testimate_exposure" << std::endl;

        // 显示
        cv::namedWindow("Image vs Render", cv::WINDOW_NORMAL);
        cv::resizeWindow("Image vs Render", 1280, 480); // 设置合适的窗口大小

        float test_image_num = 0;
        for (int i = 0; i < ts_list.size(); i++) {
            std::string ts = ts_list[i];
            if (std::find(ts_list_kf.begin(), ts_list_kf.end(), ts) != ts_list_kf.end())
                continue;

            Eigen::Matrix4d Twc = pose_list[i];

            // 读取图像
            cv::Mat image_left, image, render, combined;
            float psnr, ssim, exposure_t, ex_factor;

            image_left = cv::imread(configs.dataroot + "/cam0/data/" + ts + ".png", cv::IMREAD_UNCHANGED);
            pSLAM->GetCameraPtr()->UndistortImage(image_left, image_left);

            ex_factor = name_factors[ts];

            if (mode == 1)
            {
                adjustExposure(image_left, ex_factor);
            } else if (mode == 2)
            {
                adjustGainWithNoise(image_left, ex_factor);
            } else
            {
                image_left = grayToColorWithCoeff(image_left);
                image_left = randomAdjustSaturation(image_left, ex_factor);
            }

            if (image_left.channels() == 1)
                cv::cvtColor(image_left, image_left, cv::COLOR_GRAY2RGB);

            cv::imwrite(cache_path_gt + "/" + ts + ".png", image_left);
            image_left.convertTo(image, CV_32F, 1.0 / 255.0);

            // evaluate
            pGausMapper->evaluateImage(psnr, ssim, exposure_t, Twc, image, render);
            cv::imwrite(cache_path_render + "/" + ts + ".png", render);

            cv::hconcat(image_left, render, combined);
            cv::putText(combined,
                        "Frame: " + std::to_string(i) + "  PSNR: " + std::to_string(psnr),
                        cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX,
                        1.0,
                        cv::Scalar(0, 255, 0),
                        2);
            cv::imshow("Image vs Render", combined);
            cv::waitKey(1); // 等待1毫秒，允许窗口刷新

            // 记录当前帧结果
            avg_pnsr += psnr;
            avg_ssim += ssim;
            frame_results.emplace_back(psnr, ssim);

            // 写入TXT文件，使用制表符分隔
            frame_file << ts << "\t"
                       << std::fixed << std::setprecision(4) << psnr << "\t"
                       << std::fixed << std::setprecision(4) << ssim << std::endl;


            exposure_file << ts << "\t"
                          << std::fixed << std::setprecision(4) << exposure_t << std::endl;


            test_image_num += 1;
            // 显示进度条
            int progress = static_cast<int>((i + 1) * 100.0 / ts_list.size());
            std::cout << "\r[";
            int pos = 100 * (i + 1) / ts_list.size();
            for (int j = 0; j < 100; ++j) {
                if (j < pos) std::cout << "=";
                else if (j == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << progress << "% (" << (i + 1) << "/" << ts_list.size() << ")"
                      << std::flush;
        }

        std::cout << std::endl;
        frame_file.close();

        avg_pnsr /= test_image_num;
        avg_ssim /= test_image_num;

        // 打印最终结果
        std::cout << "最终平均PSNR: " << std::fixed << std::setprecision(4) << avg_pnsr << " dB" << std::endl;
        std::cout << "最终平均SSIM: " << std::fixed << std::setprecision(4) << avg_ssim << std::endl;

        // 保存最终统计结果（TXT格式）
        std::ofstream final_file(final_stats_path);
        if (final_file.is_open()) {
            final_file << "评估统计结果" << std::endl;
            final_file << "====================" << std::endl;
            final_file << "测试集总帧数: " << (int) test_image_num << std::endl;
            final_file << "平均PSNR: " << std::fixed << std::setprecision(4) << avg_pnsr << " dB" << std::endl;
            final_file << "平均SSIM: " << std::fixed << std::setprecision(4) << avg_ssim << std::endl;
            final_file << "====================" << std::endl;
            final_file.close();
        } else {
            std::cerr << "无法打开最终统计文件: " << final_stats_path << std::endl;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
