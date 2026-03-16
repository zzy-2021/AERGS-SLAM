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
#include<geometry_msgs/Vector3Stamped.h>

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

class ExposureGrabber {
public:
    ExposureGrabber() {};

    void GrabExposure(const geometry_msgs::Vector3StampedConstPtr &exposure_msg);

    std::vector<std::tuple<std::string, float>> GetExposure();

    std::map<std::string, float> exposureBuf;
    std::mutex mBufMutex;
};

class ImageGrabber {
public:
    ImageGrabber(std::shared_ptr<GaussianMapper> pGausMapper, std::shared_ptr<MapUser> pSLAM , ExposureGrabber *pExposureGb, std::string results_path)
            : mpGausMapper(pGausMapper),mpSLAM(pSLAM), mpExposureGrabber(pExposureGb), mstrResults_path(results_path) {}

    void GrabImageLeft(const sensor_msgs::ImageConstPtr &msg);

    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);

    void Process();

    std::queue<sensor_msgs::ImageConstPtr> imgLeftBuf;
    std::mutex mBufMutexLeft;
    std::string mstrResults_path;

    std::shared_ptr<GaussianMapper> mpGausMapper;
    std::shared_ptr<MapUser> mpSLAM;
    ExposureGrabber *mpExposureGrabber;
    // 新增：退出标志
    std::atomic<bool> mbStop{false};
    // 新增：最后一次收到数据的时间
    std::chrono::steady_clock::time_point mLastDataTime;
    // 超时阈值（例如5秒，可根据实际情况调整）
    const double mTimeout = 5.0;

};


std::string ros_time_to_string(const ros::Time &time) {
    std::stringstream ss;
    // 输出秒数（无格式限制）
    ss << time.sec;
    // 输出纳秒数，固定9位，不足补前导零（确保nsec占9位）
    ss << std::setw(9) << std::setfill('0') << time.nsec;
    return ss.str();
}

void ExposureGrabber::GrabExposure(const geometry_msgs::Vector3StampedConstPtr &exposure_msg) {
    mBufMutex.lock();

    std::string timestamp = ros_time_to_string(exposure_msg->header.stamp);  // 时间戳
    auto exposure_rate = (exposure_msg->vector.x) * 0.1;
    auto gain = exposure_msg->vector.y;

    //float fps = 30.0 | 19.97; 15.0 | 19.97;  60.0 | 10.84072;  100.0 | 10.106624;
    float real_exposure_time, weight_exposure;
    if (exposure_rate > 0.001)
        real_exposure_time = exposure_rate * (19.97 - 0.17072);
    else
        real_exposure_time = 0.17072;

    if (gain > 0)
        weight_exposure = real_exposure_time * (gain / 100.0);
    else
        weight_exposure = real_exposure_time;
    exposureBuf[timestamp] = weight_exposure;

    mBufMutex.unlock();

}


void ImageGrabber::GrabImageLeft(const sensor_msgs::ImageConstPtr &img_msg) {
    mBufMutexLeft.lock();
//    if (!imgLeftBuf.empty())
//        imgLeftBuf.pop();
    imgLeftBuf.push(img_msg);
    mBufMutexLeft.unlock();
}


cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg) {
    //ROS_INFO("Original image encoding: %s", img_msg->encoding.c_str());

    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    if (cv_ptr->image.type() == CV_8UC1 || cv_ptr->image.type() == CV_8UC3) {
//
//        cv::cvtColor(cv_ptr->image, cv_ptr->image, CV_BGR2RGB);
        return cv_ptr->image.clone();

    } else {
        std::cout << "Error type" << std::endl;
        return cv_ptr->image.clone();
    }
}

void ImageGrabber::Process() {

    std::vector <std::string> ts_list_kf, ts_list;
    std::vector <Eigen::Matrix4d> pose_list,  _;

    std::string trajectory_keyframe_path = ConcatenateFolderAndFileName(mstrResults_path, "KeyFrameTrajectory_TUM.txt");
    std::string trajectory_allframe_path = ConcatenateFolderAndFileName(mstrResults_path, "CameraTrajectory_TUM.txt");

    std::string frame_results_path = mstrResults_path + "/frame_metric.txt";
    std::string final_stats_path = mstrResults_path + "/final_metric.txt";
    std::string exposure_results = mstrResults_path + "/exposure_metric.txt";


    std::ofstream frame_file(frame_results_path);
    std::ofstream exposure_file(exposure_results);

    if (!frame_file.is_open()) {
        std::cerr << "无法打开帧结果文件: " << frame_results_path << std::endl;
        return;
    }
    frame_file << "frame_index\tpsnr(dB)\tssim" << std::endl;
    exposure_file << "frame_index\tgt_exposure\testimate_exposure" << std::endl;
    // 显示
    cv::namedWindow("Image vs Render", cv::WINDOW_NORMAL);
    cv::resizeWindow("Image vs Render", 1280, 480); // 设置合适的窗口大小

    readTUMTrajectory(trajectory_keyframe_path, ts_list_kf, _);
    readTUMTrajectory(trajectory_allframe_path, ts_list, pose_list);

    int idx = 0;
    // 新增：标记是否已收到过有效数据（左图、右图、IMU都非空才算）
    bool hasReceivedValidData = false;
    // 初始化最后数据时间
    mLastDataTime = std::chrono::steady_clock::now();

    float avg_pnsr = 0;
    float avg_ssim = 0;
    std::vector<std::pair<float, float>> frame_results;

    while (!mbStop && ros::ok()) {
        cv::Mat image_left, image;
        std::string str_tImLeft;

        if (!imgLeftBuf.empty() && !mpExposureGrabber->exposureBuf.empty()) {
            // 收到有效数据，更新标志和最后数据时间
            hasReceivedValidData = true;
            mLastDataTime = std::chrono::steady_clock::now();

            str_tImLeft = ros_time_to_string(imgLeftBuf.front()->header.stamp);

            if (std::find(ts_list.begin(), ts_list.end(), str_tImLeft) == ts_list.end())
            {
                imgLeftBuf.pop();
                continue;  // 这个帧结果里面没有, 跳过
            }


            this->mBufMutexLeft.lock();
            image_left = GetImage(imgLeftBuf.front());
            imgLeftBuf.pop();
            mpSLAM->GetCamera()->UndistortImage(image_left, image_left);
            this->mBufMutexLeft.unlock();

            float psnr, ssim, exposure_t, gt_exposure_t;
            cv::Mat render, combined;
            Eigen::Matrix4d Tcw = pose_list[std::find(ts_list.begin(), ts_list.end(), str_tImLeft) - ts_list.begin()];

            image_left.convertTo(image, CV_32F, 1.0 / 255.0);

            mpGausMapper->evaluateImage(psnr, ssim, exposure_t, Tcw, image, render);

            gt_exposure_t = mpExposureGrabber->exposureBuf[str_tImLeft];

            cv::cvtColor(render, render, CV_BGR2RGB);

            cv::hconcat(image_left, render, combined);
            cv::putText(combined,
                        "Frame: " + std::to_string(idx) + "  PSNR: " + std::to_string(psnr),
                        cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX,
                        1.0,
                        cv::Scalar(0, 255, 0),
                        2);
            cv::imshow("Image vs Render", combined);
            cv::waitKey(1); // 等待1毫秒，允许窗口刷新

            if (std::find(ts_list_kf.begin(), ts_list_kf.end(), str_tImLeft) != ts_list_kf.end())
                continue;  // 是关键帧, 不记录结果，跳过

            // 记录当前帧结果
            avg_pnsr += psnr;
            avg_ssim += ssim;
            frame_results.emplace_back(psnr, ssim);

            cv::imwrite(mstrResults_path+"/cache/gt/"+ str_tImLeft+".png", image_left);
            cv::imwrite(mstrResults_path+"/cache/render/"+ str_tImLeft+".png", render);

            // 写入TXT文件，使用制表符分隔
            frame_file << str_tImLeft << "\t"
                       << std::fixed << std::setprecision(4) << psnr << "\t"
                       << std::fixed << std::setprecision(4) << ssim << std::endl;

            exposure_file << str_tImLeft << "\t"
                       << std::fixed << std::setprecision(4) << gt_exposure_t << "\t"
                       << std::fixed << std::setprecision(4) << exposure_t << std::endl;

            std::chrono::milliseconds tSleep(1);
            std::this_thread::sleep_for(tSleep);
            idx++;

        } else {
            // 仅当“已收到过有效数据”时，才检查超时（避免没数据时直接退出）
            if (hasReceivedValidData ) {
                auto now = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed = now - mLastDataTime;

                // 超时（如5秒），认为数据断流（bag播完），退出线程
                if (elapsed.count() > mTimeout) {
                    ROS_INFO("No new data for %.1fs after receiving valid data, exiting SyncWithImu thread.", mTimeout);
                    mbStop = true;
                    break;
                }
            }
            // 没收到过有效数据时，只休眠等待，不做超时判断
            std::chrono::milliseconds tSleep(10);
            std::this_thread::sleep_for(tSleep);
        }
    }

    avg_pnsr /= idx;
    avg_ssim /= idx;

    // 打印最终结果
    std::cout << "最终平均PSNR: " << std::fixed << std::setprecision(4) << avg_pnsr << " dB" << std::endl;
    std::cout << "最终平均SSIM: " << std::fixed << std::setprecision(4) << avg_ssim << std::endl;

    // 保存最终统计结果（TXT格式）
    std::ofstream final_file(final_stats_path);
    if (final_file.is_open()) {
        final_file << "评估统计结果" << std::endl;
        final_file << "====================" << std::endl;
        final_file << "测试集总帧数: " << (int)idx << std::endl;
        final_file << "平均PSNR: " << std::fixed << std::setprecision(4) << avg_pnsr << " dB" << std::endl;
        final_file << "平均SSIM: " << std::fixed << std::setprecision(4) << avg_ssim << std::endl;
        final_file << "====================" << std::endl;
        final_file.close();
    } else {
        std::cerr << "无法打开最终统计文件: " << final_stats_path << std::endl;
    }

    ROS_INFO("SyncWithImu thread exited.");
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

bool saveExposureDataToFile(
        const std::vector<std::tuple<std::string, float>> &exposureBuf,
        const std::string &filePath
) {
    // 打开文件（若文件不存在则创建，若存在则覆盖）
    std::ofstream outFile(filePath);
    if (!outFile.is_open()) {  // 检查文件是否成功打开
        // 可根据需要添加错误日志，这里简化处理
        return false;
    }

    // 遍历vector，按顺序写入每行数据
    for (const auto &data: exposureBuf) {
        // 提取tuple中的时间戳（第一个元素）和曝光时间（第二个元素）
        const std::string &timestamp = std::get<0>(data);  // 时间戳字符串
        float exposureTime = std::get<1>(data);            // 曝光时间（float）

        // 写入格式：时间戳 + 空格 + 曝光时间（保留6位小数，可根据需求调整）
        outFile << timestamp
                << " "
                << std::fixed << std::setprecision(6)  // 控制浮点数精度
                << exposureTime
                << "\n";  // 换行
    }

    // 关闭文件（ofstream析构时会自动关闭，显式关闭更稳妥）
    outFile.close();

    return true;  // 保存成功
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

int main(int argc, char **argv) {

    // 打印参数个数（argc 本身包含程序名作为第一个参数）
    std::cout << "参数总数: " << argc << std::endl;

    // 循环打印每个参数
    for (int i = 0; i < argc; ++i) {
        std::cout << "参数 " << i << ": " << argv[i] << std::endl;
    }

    ros::init(argc, argv, "ZED_EVALUATE");
    ros::NodeHandle nh;

    std::string reloc_config_path, model_dir, gs_config_path, saving_dir;
    ros::param::get("~model_dir", model_dir);
    ros::param::get("~gs_config_path", gs_config_path);
    ros::param::get("~reloc_config_path", reloc_config_path);
    ros::param::get("~saving_dir", saving_dir);

    RelocalizationConfigs reloc_configs(reloc_config_path, model_dir);
    ros::param::get("~camera_config_path", reloc_configs.camera_config_path);
    std::cout << "config done" << std::endl;


    std::shared_ptr<MapUser> pMapUser = std::make_shared<MapUser>(reloc_configs, nh);
    std::shared_ptr<GaussianMapper> pGausMapper = std::make_shared<GaussianMapper>(pMapUser, gs_config_path,
                                                                                   saving_dir, 0, torch::kCUDA);

    ExposureGrabber exgb;
    ImageGrabber igb(pGausMapper, pMapUser,  &exgb, saving_dir);

    ros::Subscriber sub_img_left = nh.subscribe("/zed2i/zed_node/left/image_rect_color", 100,
                                                &ImageGrabber::GrabImageLeft, &igb);
    ros::Subscriber sub_exposure = nh.subscribe("/zed2i/zed_node/exposure_gain_with_timestamp", 100,
                                                &ExposureGrabber::GrabExposure, &exgb);


    std::thread sync_thread(&ImageGrabber::Process, &igb);

//    ros::spin();
    ros::Rate rate(100);  // 循环频率（不影响回调处理，仅控制循环速度）
    while (!igb.mbStop) {   // 当节点存活时（未收到退出信号）
        ros::spinOnce();  // 处理一次消息队列中的回调
        rate.sleep();     // 按设定频率休眠，避免CPU占用过高
    }

    return 0;
}
