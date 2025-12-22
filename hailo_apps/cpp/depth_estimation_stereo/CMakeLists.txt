cmake_minimum_required(VERSION 3.11)
project(stereo_depth_estimation)

set(CMAKE_CXX_STANDARD 20)

set(COMPILE_OPTIONS -Werror -Wall -Wextra -Wconversion -O3 -std=c++17 -Wno-reorder -Wno-ignored-qualifiers -Wno-extra -Wno-unused-parameter -Wno-error=conversion -Wno-conversion)

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads)

find_package(HailoRT REQUIRED)

find_package( OpenCV REQUIRED)
message(STATUS "Found OpenCV: " ${OpenCV_INCLUDE_DIRS})

set(SOURCES
    ${PROJECT_NAME}.cpp
    ../common/toolbox.cpp
    ../common/hailo_infer.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS} -fconcepts)
target_link_libraries(${PROJECT_NAME} HailoRT::libhailort ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS})

