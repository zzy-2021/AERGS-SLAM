cmake_minimum_required(VERSION 3.5)
project(air_slam)

# 设置 CMake 策略以使用 <PackageName>_ROOT 变量
cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_BUILD_TYPE "debug")
set(CMAKE_BUILD_TYPE "release")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
add_definitions(-w)

add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/tensorrtbuffer)
add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/DBoW2)


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

set(PYTHON_EXECUTABLE "/usr/bin/python3")
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  image_transport
  nav_msgs
  roscpp
  rospy
  std_msgs
  sensor_msgs
  tf
)


set(cv_bridge_DIR "${PROJECT_SOURCE_DIR}/3rd_party/vision_opencv-noetic/cv_bridge/build/devel/share/cv_bridge/cmake")
find_package(cv_bridge REQUIRED)

# 打印找到的 cv_bridge 信息（用于调试）
message(STATUS "cv_bridge_DIR: ${cv_bridge_DIR}")
message(STATUS "catkin_LIBRARIES: ${catkin_LIBRARIES}")

find_package(OpenCV 4 REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)


# 指定 yaml-cpp 头文件目录和库文件路径
set(YAML_CPP_INCLUDE_DIR /usr/local/include)
set(YAML_CPP_LIBRARY /usr/local/lib/libyaml-cpp.a)

find_package(Boost REQUIRED)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

find_package(G2O REQUIRED)
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
message(STATUS "G2O_INCLUDE_DIR: ${G2O_INCLUDE_DIR}")
message(STATUS "G2O_LIBRARIES: ${G2O_LIBRARIES}")

set(TENSORRT_INCLUDE_DIRS /home/zhiyu/Software/TensorRT/TRT/TRT/include)  # 根据实际路径修改
set(TENSORRT_LIBRARIES /home/zhiyu/Software/TensorRT/TRT/TRT/lib/libnvinfer.so)  # 根据实际路径修改

find_package(Gflags REQUIRED)
find_package(Glog REQUIRED)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES ${PROJECT_NAME}_lib
 CATKIN_DEPENDS geometry_msgs image_transport nav_msgs roscpp rospy std_msgs
)


include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${TENSORRT_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIR}
  ${GFLAGS_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/3rdparty/Sophus

)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)


add_library(${PROJECT_NAME}_lib SHARED
  src/g2o_optimization/vertex_line3d.cc
  src/g2o_optimization/vertex_vi_pose.cc
  src/g2o_optimization/vertex_imu.cc
  src/g2o_optimization/edge_project_point.cc
  src/g2o_optimization/edge_project_line.cc
  src/g2o_optimization/edge_imu.cc
  src/g2o_optimization/edge_relative_pose.cc
  src/g2o_optimization/g2o_optimization.cc
  src/bow/FSuperpoint.cc
  src/bow/database.cc
  src/super_point.cpp
  src/feature_detector.cc
  src/super_glue.cpp
  src/light_glue.cpp
  src/plnet.cpp
  src/utils.cc
  src/camera.cc
  src/imu.cc
  src/dataset.cc
  src/frame.cc
  src/point_matcher.cc
  src/mappoint.cc
  src/mapline.cc
  src/line_processor.cc
  src/ros_publisher.cc
  src/map.cc
  src/map_builder.cc
  src/map_refiner.cc
  src/map_user.cc
  src/timer.cc
  src/debug.cc
)

target_link_libraries(${PROJECT_NAME}_lib
  ${TENSORRT_LIBRARIES}
  nvonnxparser
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${Boost_LIBRARIES}
  ${G2O_LIBRARIES}
  ${GFLAGS_LIBRARIES} 
  ${GLOG_LIBRARIES}
  ${G2O_OPENGL_HELPER_LIBRARY}
  ${YAML_CPP_LIBRARY}
  tensorrtbuffer
  DBoW2
  -lboost_serialization
  ${PROJECT_SOURCE_DIR}/3rdparty/vision_opencv-noetic/cv_bridge/build/devel/lib/libcv_bridge.so
        )

#add_executable(visual_odometry demo/visual_odometry.cpp)
#target_link_libraries(visual_odometry ${PROJECT_NAME}_lib ${catkin_LIBRARIES} ${YAML_CPP_LIBRARY})
#
#add_executable(map_refinement demo/map_refinement.cpp)
#target_link_libraries(map_refinement ${PROJECT_NAME}_lib ${catkin_LIBRARIES})
#
#add_executable(relocalization demo/relocalization.cpp)
#target_link_libraries(relocalization ${PROJECT_NAME}_lib ${catkin_LIBRARIES})
#
#add_executable(test_feature demo/test_feature.cpp)
#target_link_libraries(test_feature ${PROJECT_NAME}_lib ${catkin_LIBRARIES})
