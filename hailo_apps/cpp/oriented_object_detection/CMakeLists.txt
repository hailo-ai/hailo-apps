cmake_minimum_required(VERSION 3.16)
project(oriented_obj_det)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Base compilation options (portable)
set(COMPILE_OPTIONS
    -Wall -Wextra -O3 -fconcepts
    -Wno-ignored-qualifiers -Wno-extra -Wno-stringop-truncation -Wno-reorder
    -fopenmp
)

# Add AVX2/FMA only if compiler supports them (x86)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" HAS_AVX2)
check_cxx_compiler_flag("-mfma"  HAS_FMA)

if (HAS_AVX2)
    list(APPEND COMPILE_OPTIONS -mavx2)
endif()

if (HAS_FMA)
    list(APPEND COMPILE_OPTIONS -mfma)
endif()

find_package(Threads)
find_package(HailoRT REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS}")

file(GLOB SOURCES
    ./*.cpp
    ./utils/*.cpp
)
list(APPEND SOURCES ../common/toolbox.cpp ../common/hailo_infer.cpp)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

add_executable(${PROJECT_NAME} ${SOURCES})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${PROJECT_NAME} Threads::Threads HailoRT::libhailort ${OpenCV_LIBS})