cmake_minimum_required(VERSION 3.0)
project(BEVFusion_Backbone)

set(CMAKE_PREFIX_PATH "../libtorch")  # 替换为你的LibTorch路径
find_package(Torch REQUIRED)

add_executable(camera_backbone camera_backbone.cpp)
target_link_libraries(camera_backbone "${TORCH_LIBRARIES}")
set_property(TARGET camera_backbone PROPERTY CXX_STANDARD 17)