cmake_minimum_required(VERSION 3.10)
project(camera_vtransform)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

add_library(camera_vtransform SHARED camera_vtransform.cpp) 
target_compile_options(camera_vtransform PRIVATE -g -O0)
message(STATUS "GDB调试已启用，使用以下命令调试：")