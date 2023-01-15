cmake_minimum_required(VERSION 3.18) 

project(reduce LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set up cuda envs
find_package(CUDA REQUIRED)
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

set(COMMON_HEADER_DIRS ${CUDA_PATH}/include)
set(COMMON_LIB_DIRS ${CUDA_PATH}/lib64)

message(STATUS "include headers: {COMMON_HEADER_DIRS}")
include_directories(${COMMON_HEADER_DIRS})
link_directories(${COMMON_LIB_DIRS})

add_executable(reduce1 reduce1.cu)
add_executable(reduce2 reduce2.cu)
add_executable(reduce3 reduce3.cu)
add_executable(reduce4 reduce4.cu)


