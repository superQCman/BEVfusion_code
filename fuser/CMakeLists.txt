cmake_minimum_required(VERSION 3.0)
project(BEVFusion_Fuser)

set(CMAKE_PREFIX_PATH "/home/ting/SourceCode/libtorch")  # 替换为你的LibTorch路径
find_package(Torch REQUIRED)

# 将可执行文件改为动态库
add_library(fuser SHARED fuser.cpp)
target_link_libraries(fuser "${TORCH_LIBRARIES}")
set_property(TARGET fuser PROPERTY CXX_STANDARD 17)

# 设置动态库的版本信息（可选）
set_target_properties(fuser PROPERTIES
    VERSION 1.0.0
    SOVERSION 1)