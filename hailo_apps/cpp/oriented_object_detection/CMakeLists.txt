cmake_minimum_required(VERSION 3.16)
project(oriented_obj_det)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compilation options for warnings, debugging, and optimization
# Enable AVX2, FMA, and OpenMP for SIMD and parallelization optimizations
set(COMPILE_OPTIONS -Wall -Wextra -O3 -fconcepts -Wno-ignored-qualifiers -Wno-extra -Wno-stringop-truncation -Wno-reorder -mavx2 -mfma -fopenmp)

# Find necessary packages
find_package(Threads)
find_package(HailoRT REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "Found OpenCV: " ${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES
    ./*.cpp
    ./utils/*.cpp
)
list(APPEND SOURCES ../common/toolbox.cpp ../common/hailo_infer.cpp)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/utils ${CMAKE_CURRENT_SOURCE_DIR}/../common)

# Define the executable and link libraries
link_libraries(stdc++fs)
add_executable(${PROJECT_NAME} ${SOURCES})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${PROJECT_NAME} Threads::Threads HailoRT::libhailort)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
