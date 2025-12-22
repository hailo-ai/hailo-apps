cmake_minimum_required(VERSION 3.10.0)
project(mono_depth_estimation)

# Set C++ standard to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads)
find_package(OpenCV REQUIRED)
find_package(HailoRT REQUIRED)

set(COMPILE_OPTIONS_CPP -Wall -Werror -O3)

set(SOURCES
    ${PROJECT_NAME}.cpp
    ../common/toolbox.cpp
    ../common/hailo_infer.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../common)

include_directories(/usr/include ./)

target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS_CPP})
target_link_libraries(${PROJECT_NAME} HailoRT::libhailort Threads::Threads)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

