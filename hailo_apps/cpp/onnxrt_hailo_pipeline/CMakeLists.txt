cmake_minimum_required(VERSION 3.12)
project(onnxrt_hailo_pipeline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# ---- Find xtensor (header-only) ----
# Try system paths first; if you bundled xtensor to third_party/xtensor/include, it will also be picked up.
find_path(XTENSOR_INCLUDE_DIR xtensor/xarray.hpp
    PATH_SUFFIXES include
    PATHS /usr/include /usr/local/include ${CMAKE_SOURCE_DIR}/third_party/xtensor/include)
if (NOT XTENSOR_INCLUDE_DIR)
    message(FATAL_ERROR "xtensor headers not found. Install libxtensor-dev or place headers under third_party/xtensor/include")
endif()

# ---- ONNX Runtime under /opt ----
file(GLOB ALL_SUBDIRS "/opt/*")
set(ONNXRUNTIME_DIR "")
foreach(DIR ${ALL_SUBDIRS})
    if (IS_DIRECTORY ${DIR} AND "${DIR}" MATCHES ".*onnxruntime.*")
        set(ONNXRUNTIME_DIR ${DIR})
        break()
    endif()
endforeach()
if (ONNXRUNTIME_DIR)
    message(STATUS "Found directory: ${ONNXRUNTIME_DIR}")
    find_library(ONNXRUNTIME onnxruntime ${ONNXRUNTIME_DIR}/lib)
    set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)
else()
    message(FATAL_ERROR "No directory containing 'onnxruntime' was found in /opt/")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${XTENSOR_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/../common
)

add_executable(onnxrt_hailo_pipeline
    onnxrt_hailo_pipeline.cpp
    onnx_decode.cpp
    instance_seg_postprocess.cpp
    ../common/toolbox.cpp
    ../common/hailo_infer.cpp
)

target_link_libraries(onnxrt_hailo_pipeline
    ${OpenCV_LIBS}
    Threads::Threads
    ${ONNXRUNTIME}
    hailort
)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(onnxrt_hailo_pipeline PRIVATE -O3 -DNDEBUG)
else()
  target_compile_options(onnxrt_hailo_pipeline PRIVATE -O0 -g)
endif()
