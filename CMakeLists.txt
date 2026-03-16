cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
set(CMAKE_CUDA_ARCHITECTURES "89-real")
add_definitions(-DTCNN_MIN_GPU_ARCH=89)
project(aergs-slam LANGUAGES CXX CUDA)
set(PYTHON_EXECUTABLE /usr/bin/python3)

find_package(catkin REQUIRED COMPONENTS
        geometry_msgs
        image_transport
        nav_msgs
        roscpp
        rospy
        std_msgs
        sensor_msgs
        geometry_msgs
        tf
        )

set(cv_bridge_DIR "${PROJECT_SOURCE_DIR}/third_party/vision_opencv-noetic/cv_bridge/build/devel/share/cv_bridge/cmake")
find_package(cv_bridge REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    #set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

## gcc 优化选项

## cuda 优化选项

set(CAFFE2_USE_CUDNN on)

set(Torch_DIR "${PROJECT_SOURCE_DIR}/third_party/libtorch-2.0.0+cu118/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(OpenCV 4 REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

find_package(jsoncpp REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Python3 REQUIRED)

set(TENSORRT_INCLUDE_DIRS /home/zhiyu/Software/TensorRT/TRT/TRT/include)  # 根据实际路径修改
set(TENSORRT_LIBRARIES /home/zhiyu/Software/TensorRT/TRT/TRT/lib/libnvinfer.so)  # 根据实际路径修改


set(AIR_SLAM_SOURCE_DIR "${PROJECT_SOURCE_DIR}/AirSLAM")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${AIR_SLAM_SOURCE_DIR}
    ${AIR_SLAM_SOURCE_DIR}/include
    ${AIR_SLAM_SOURCE_DIR}/3rdparty/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${catkin_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

##################################################################################
##  Build the gaussian splatting component libraries to ${PROJECT_SOURCE_DIR}/lib
##################################################################################
add_library(simple_knn SHARED
    third_party/simple-knn/simple_knn.cu
    third_party/simple-knn/simple_knn.h
    third_party/simple-knn/spatial.cu
    third_party/simple-knn/spatial.h)
target_link_libraries(simple_knn "${TORCH_LIBRARIES}")

add_library(cuda_lanczos SHARED
        cuda_lanczos/lanczos.cu
        cuda_lanczos/lanczos.h)
target_link_libraries(cuda_lanczos "${TORCH_LIBRARIES}")

add_library(cuda_rasterizer SHARED
    include/operate_points.h
    src/operate_points.cu
    include/rasterize_points.h
    src/rasterize_points.cu
    include/stereo_vision.h
    src/stereo_vision.cu
    cuda_rasterizer/auxiliary.h
    cuda_rasterizer/backward.cu
    cuda_rasterizer/backward.h
    cuda_rasterizer/config.h
    cuda_rasterizer/forward.cu
    cuda_rasterizer/forward.h
    cuda_rasterizer/operate_points.h
    cuda_rasterizer/rasterizer.h
    cuda_rasterizer/rasterizer_impl.cu
    cuda_rasterizer/rasterizer_impl.h
    cuda_rasterizer/stereo_vision.h)
set_target_properties(cuda_rasterizer PROPERTIES CUDA_ARCHITECTURES "75;86")

target_include_directories(cuda_rasterizer PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuda_rasterizer
    glm::glm
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen)

##################################################################################
##  Build the gaussian mapper library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(gaussian_mapper SHARED
    third_party/tinyply/tinyply.h
    third_party/tinyply/tinyply.cpp
    include/gaussian_keyframe.h
    include/gaussian_model.h
    include/gaussian_parameters.h
    include/gaussian_rasterizer.h
    include/gaussian_renderer.h
    include/gaussian_scene.h
    include/gaussian_trainer.h
    include/gaussian_mapper.h
    include/general_utils.h
    include/graphics_utils.h
    include/loss_utils.h
    include/sh_utils.h
    include/tensor_utils.h
    include/gaussian_view.h
    include/point_cloud.h
    include/point2d.h
    include/point3d.h
    include/types.h
    include/ssim.h
    include/gaussian_reso_scheduler.h
    include/response_mapper.h
    src/gaussian_keyframe.cpp
    src/gaussian_model.cpp
    src/gaussian_parameters.cpp
    src/gaussian_rasterizer.cpp
    src/gaussian_renderer.cpp
    src/gaussian_scene.cpp
    src/gaussian_trainer.cpp
    src/ssim.cu
    src/response_mapper.cpp
    src/gaussian_mapper.cpp)

target_include_directories(gaussian_mapper PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(gaussian_mapper
    ${AIR_SLAM_SOURCE_DIR}/lib/libair_slam_lib.so
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen
    simple_knn
    cuda_rasterizer
    cuda_lanczos)

##################################################################################
##  Build the mapping examples to ${PROJECT_SOURCE_DIR}/bin
##################################################################################

add_executable(visual_odometry_dark demo/visual_odometry_dark.cpp)
target_link_libraries(visual_odometry_dark
        gaussian_mapper
        ${OpenCV_LIBRARIES}
        ${AIR_SLAM_SOURCE_DIR}/lib/libair_slam_lib.so
        ${catkin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/third_party/vision_opencv-noetic/cv_bridge/build/devel/lib/libcv_bridge.so
        )


add_executable(zed_ros demo/zed_ros.cpp)
target_link_libraries(zed_ros
        gaussian_mapper
        ${OpenCV_LIBRARIES}
        ${AIR_SLAM_SOURCE_DIR}/lib/libair_slam_lib.so
        ${catkin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/third_party/vision_opencv-noetic/cv_bridge/build/devel/lib/libcv_bridge.so
        )

add_executable(zed_evaluate evaluate/zed_data_evaluate/zed_evaluate_psnr_ssim.cpp)
target_link_libraries(zed_evaluate
        gaussian_mapper
        ${OpenCV_LIBRARIES}
        ${AIR_SLAM_SOURCE_DIR}/lib/libair_slam_lib.so
        ${catkin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/third_party/vision_opencv-noetic/cv_bridge/build/devel/lib/libcv_bridge.so
        )









