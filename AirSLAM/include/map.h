#ifndef MAP_H_
#define MAP_H_

#include <opencv2/highgui/highgui.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>

#include "read_configs.h"
#include "camera.h"
#include "mappoint.h"
#include "mapline.h"
#include "frame.h"
#include "g2o_optimization/types.h"
#include "ros_publisher.h"
#include "bow/database.h"
#include "sophus/se3.hpp"

class MappingOperation {
public:
    enum OprType {
        LocalMappingBA = 1,
        LoopClosingBA = 2,
        MapRefinement = 3
    };

private:
    MappingOperation(
            const MappingOperation &opr,
            const std::lock_guard<std::mutex> &,
            const std::lock_guard<std::mutex> &)
            : mvAssociatedKeyFrames(std::move(opr.mvAssociatedKeyFrames)),
              mvAssociatedMapPoints(std::move(opr.mvAssociatedMapPoints)),
              meOperationType(opr.meOperationType),
              mfScale(opr.mfScale),
              mT(opr.mT) {}

public:
    MappingOperation(
            OprType type,
            const float scale = 1.0f,
            const Sophus::SE3f T = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()),
            const std::size_t nKFs = 0UL,
            const std::size_t nMPs = 0UL)
            : meOperationType(type),
              mfScale(scale),
              mT(T) {
        mvAssociatedKeyFrames.reserve(nKFs);
        int length = nMPs * 3;
        std::get<0>(mvAssociatedMapPoints).reserve(length);
        std::get<1>(mvAssociatedMapPoints).reserve(length);
    }

    MappingOperation(const MappingOperation &opr)
            : MappingOperation(
            opr,
            std::lock_guard<std::mutex>(opr.mMutexKeyFrames),
            std::lock_guard<std::mutex>(opr.mMutexMapPoints)) {}

public:
    void reserveKeyFrames(const std::size_t nKFs) {
        mvAssociatedKeyFrames.reserve(nKFs);
    }

    void addKeyFrame(FramePtr pKF, bool isLoopClosureKF = false) // 把关键帧放入mvAssociatedKeyFrames
    {
        std::unique_lock<std::mutex> lock(mMutexKeyFrames);
        mvAssociatedKeyFrames.emplace_back(
                std::make_tuple(
                        pKF->GetFrameId(),
                        pKF->GetPose(),
                        pKF->imgRGB.clone(),
                        isLoopClosureKF));
    }

    std::vector<std::tuple<
            unsigned long,
            Eigen::Matrix4d,
            cv::Mat,
            bool>> &associatedKeyFrames() { return mvAssociatedKeyFrames; }

    void reserveMapPoints(const std::size_t nMPs) {
        int length = nMPs * 3;
        std::get<0>(mvAssociatedMapPoints).reserve(length);
        std::get<1>(mvAssociatedMapPoints).reserve(length);
    }

    void addMapPoint(MappointPtr pMP)  // 地图点的内容
    {
        std::unique_lock<std::mutex> lock(mMutexMapPoints);
        auto pt = pMP->GetPosition();
        std::get<0>(mvAssociatedMapPoints).emplace_back(pt.x());
        std::get<0>(mvAssociatedMapPoints).emplace_back(pt.y());
        std::get<0>(mvAssociatedMapPoints).emplace_back(pt.z());
        auto color = pMP->GetColor();

//        std::cout << "特征点的颜色值为: "
//                  << "R: " << color[0] << ", "
//                  << "G: " << color[1] << ", "
//                  << "B: " << color[2] << std::endl;

//        std::get<1>(mvAssociatedMapPoints).emplace_back(0.5);
//        std::get<1>(mvAssociatedMapPoints).emplace_back(0.5);
//        std::get<1>(mvAssociatedMapPoints).emplace_back(0.5);
        std::get<1>(mvAssociatedMapPoints).emplace_back(color.x());
        std::get<1>(mvAssociatedMapPoints).emplace_back(color.y());
        std::get<1>(mvAssociatedMapPoints).emplace_back(color.z());

    }

    std::tuple<std::vector<float/*pos*/>, std::vector<float/*color*/>> &
    associatedMapPoints() { return mvAssociatedMapPoints; }

public:
    // Type
    OprType meOperationType;

    // Data
    float mfScale; ///<  ScaleRefinement: global; LoopClosingBA: only for visible; LocalMappingBA: meaningless
    Sophus::SE3f mT;

protected:
    // Data
    std::tuple<std::vector<float/*pos*/>,
            std::vector<float/*color*/>> mvAssociatedMapPoints;

    std::vector<std::tuple<
            unsigned long/*Id*/,
            Eigen::Matrix4d/*pose*/,
            cv::Mat/*image*/,
            bool/*isLoopClosure*/>> mvAssociatedKeyFrames;

    // Mutex
    mutable std::mutex mMutexMapPoints;
    mutable std::mutex mMutexKeyFrames;
};

class MapRefiner;

class Map {
public:
    Map();

    Map(OptimizationConfig &backend_optimization_config, CameraPtr camera, RosPublisherPtr ros_publisher);

    void InsertKeyframe(FramePtr frame);

    void InsertMappoint(MappointPtr mappoint);

    void InsertMapline(MaplinePtr mapline);

    bool UppdateMapline(MaplinePtr mapline);

    void UpdateMaplineEndpoints(MaplinePtr mapline);

    void CheckAndDeleteMappoint(MappointPtr mpt);

    void CheckAndDeleteMapline(MaplinePtr mpl);

    void DeleteKeyframe(FramePtr frame);

    CameraPtr GetCameraPtr();

    FramePtr GetFramePtr(int frame_id);

    MappointPtr GetMappointPtr(int mappoint_id);

    MaplinePtr GetMaplinePtr(int mapline_id);

    bool TriangulateMappoint(MappointPtr mappoint);

    bool TriangulateMaplineByMappoints(MaplinePtr mapline);

    bool UpdateMappointDescriptor(MappointPtr mappoint);

    void LocalMapOptimization(FramePtr new_frame, MappingOperation &opr);

    std::pair<FramePtr, FramePtr> MakeFramePair(FramePtr frame0, FramePtr frame1);

    void RemoveOutliers(const std::vector<std::pair<FramePtr, MappointPtr>> &outliers);

    void RemoveLineOutliers(const std::vector<std::pair<FramePtr, MaplinePtr>> &line_outliers);

    int UpdateFrameTrackIds(int track_id);

    int UpdateFrameLineTrackIds(int line_track_id);

    void SearchByProjection(FramePtr frame, std::vector<MappointPtr> &mappoints,
                            int thr, std::vector<std::pair<int, MappointPtr>> &good_projections);

    void SaveKeyframeTrajectory(std::string save_root);

    void SaveAllframeTrajectory(std::string save_root);

    void SaveKeyframeTrajectory_Gaussian(std::string save_root, std::map<size_t, Eigen::Matrix4f> kfposes);

    void SaveAllframeTrajectory_Gaussian(std::string save_root, std::map<size_t, Eigen::Matrix4f> kfposes);

    bool InitializeIMU(FramePtr frame);

    void SetRwg(const Eigen::Matrix3d &Rwg);

    Eigen::Matrix3d GetRwg();

    void SetIMUInit(bool imu_init);

    bool IMUInit();

    void SaveMap(const std::string &map_root);

    void SetRosPublisher(RosPublisherPtr ros_publisher);

    void Publish(double time, bool clear_old_message = false);


    // for offline optimization
    std::map<int, MappointPtr> &GetAllMappoints();

    std::map<int, MaplinePtr> &GetAllMaplines();

    std::map<int, FramePtr> &GetAllKeyframes();

    int RemoveInValidMappoints();

    int RemoveInValidMaplines();

    void UpdateCovisibilityGraph();

    void UpdateFrameCovisibility(FramePtr frame);

    void GetConnectedFrames(FramePtr frame, std::map<FramePtr, int> &covi_frames);

    // visualization
    double MapScale();

    // debug
    void CheckMap();

    int GetNumKeyframes();

public:
    // tmp parameters
    std::vector<std::pair<FramePtr, int>> to_update_track_id;
    std::vector<std::pair<FramePtr, int>> to_update_line_track_id;

    // timestamp, reference frame id, relative pose to ref_frame
    std::vector<std::tuple<std::string, int, Eigen::Matrix4d>> normal_frame_pose;

    // for imu
    FramePtr last_keyframe;

    double imu_init_time;
    FramePtr imu_init_frame;
    int imu_init_stage;

    std::mutex mMutexMapUpdate;  // SLAM在更新的时候，高斯不能读取



    void pushMappingOperation(MappingOperation opr);
    MappingOperation getAndPopMappingOperation();
    bool hasMappingOperation();
    void clearMappingOperation();


private:
    friend class MapRefiner;

    friend class MapUser;

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & _camera;
        ar & _mappoints;
        ar & _maplines;
        ar & _keyframes;
        // ar & _keyframe_ids;
        ar & _imu_init;
        ar & boost::serialization::make_array(_Rwg.data(), _Rwg.size());

        ar & _covisibile_frames;
        ar & _database;
        ar & _junction_database;
        ar & _junction_voc;
    }

private:
    OptimizationConfig _backend_optimization_config;
    CameraPtr _camera;
    std::map<int, MappointPtr> _mappoints;
    std::map<int, MaplinePtr> _maplines;
    std::map<int, FramePtr> _keyframes;
    std::vector<int> _keyframe_ids;
    RosPublisherPtr _ros_publisher;

    // for imu
    bool _imu_init;
    Eigen::Matrix3d _Rwg;

    // for loop detection adn relocalization
    std::map<FramePtr, std::map<FramePtr, int>> _covisibile_frames;
    DatabasePtr _database;

    DatabasePtr _junction_database;
    SuperpointVocabularyPtr _junction_voc;

    std::queue<MappingOperation> mqMappingOperations;  // 一个队列，保持着核心的地图操作
    std::mutex mMutexMappingOperations;



};

typedef std::shared_ptr<Map> MapPtr;

#endif // MAP_H_