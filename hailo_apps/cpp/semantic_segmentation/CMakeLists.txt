cmake_minimum_required(VERSION 3.10.0)
project(semantic_segmentation)

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

target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS_CPP})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} HailoRT::libhailort Threads::Threads)