cmake_minimum_required(VERSION 3.0)
project(BEVFusion_Backbone)

set(CMAKE_PREFIX_PATH "../libtorch")  # 替换为你的LibTorch路径
find_package(Torch REQUIRED)

add_executable(lidar_backbone sparse_conv.h main.cpp lidar_backbone.h)
target_link_libraries(lidar_backbone "${TORCH_LIBRARIES}")
set_property(TARGET lidar_backbone PROPERTY CXX_STANDARD 17)