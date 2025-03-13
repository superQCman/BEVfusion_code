cmake_minimum_required(VERSION 3.0)
project(BEVFusion_Backbone)

set(CMAKE_PREFIX_PATH "../../libtorch")  # 替换为你的LibTorch路径
find_package(Torch REQUIRED)

add_library(camera_backbone SHARED camera_backbone.cpp)
target_link_libraries(camera_backbone "${TORCH_LIBRARIES}")
set_property(TARGET camera_backbone PROPERTY CXX_STANDARD 17)

set_target_properties(camera_backbone PROPERTIES
    VERSION 1.0.0
    SOVERSION 1)