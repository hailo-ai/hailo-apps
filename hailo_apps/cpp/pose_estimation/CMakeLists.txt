cmake_minimum_required(VERSION 3.11)
project(pose_estimation)

set(CMAKE_CXX_STANDARD 20)

set(COMPILE_OPTIONS -Werror -Wall -Wextra -Wconversion -O3 -Wno-reorder -Wno-ignored-qualifiers -Wno-extra -Wno-unused-local-typedefs -Wno-conversion -Wno-parentheses -Wno-unused-but-set-variable -Wno-array-bounds -Wno-unused-value)

set(BASE_DIR /path/to/yolov8/example/folder)

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

file(GLOB SOURCES
    ./*.cpp
)
list(APPEND SOURCES ../common/toolbox.cpp ../common/hailo_infer.cpp)
include(ExternalProject)
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

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


include_directories(${EXTERNAL_INSTALL_LOCATION}/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

link_libraries(stdc++fs)
add_executable(${PROJECT_NAME} ${SOURCES})
add_dependencies(${PROJECT_NAME} xtl-test xtensor-test)
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../common)
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS} -fconcepts)
target_link_libraries(${PROJECT_NAME} HailoRT::libhailort ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS})