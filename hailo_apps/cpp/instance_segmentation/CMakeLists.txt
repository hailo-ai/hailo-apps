cmake_minimum_required(VERSION 3.11)
project(instance_segmentation CXX)

set(CMAKE_CXX_STANDARD 20)
set(COMPILE_OPTIONS_CPP
    -O3 -DNDEBUG -Wall
    -Wno-reorder -Wno-ignored-qualifiers -Wno-unused-local-typedefs
    -Wno-parentheses -Wno-array-bounds -Wno-unused-value
    -Wno-conversion -Wno-unused-parameter -Wno-missing-field-initializers
    -Wno-sign-compare -Wno-uninitialized
)

# Find required packages
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(HailoRT REQUIRED)

# External dependencies
include(ExternalProject)
ExternalProject_Add(xtl-test
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/external
)
ExternalProject_Add(xtensor-test
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/external -Dxtl_DIR=${CMAKE_BINARY_DIR}/external/share/cmake/xtl/
)

# Create executable
add_executable(${PROJECT_NAME}
    instance_segmentation.cpp
    instance_seg_postprocess.cpp
    ../common/toolbox.cpp
    ../common/hailo_infer.cpp
)

target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS_CPP})

# Dependencies and linking
add_dependencies(${PROJECT_NAME} xtl-test xtensor-test)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_BINARY_DIR}/external/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    HailoRT::libhailort
    Threads::Threads
    ${OpenCV_LIBS}
)
