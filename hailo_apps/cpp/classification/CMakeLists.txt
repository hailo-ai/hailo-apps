cmake_minimum_required(VERSION 3.10.0)
project(classifier)

set(COMPILE_OPTIONS -Wall -std=gnu++2a -Werror -O3)

find_package(HailoRT REQUIRED)
find_package(Threads)
find_package(OpenCV REQUIRED)

message(STATUS "Found OpenCV: " ${OpenCV_INCLUDE_DIRS})

set(SOURCES
    ${PROJECT_NAME}.cpp
    ../common/toolbox.cpp
    ../common/hailo_infer.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../common)

include_directories( ${OpenCV_INCLUDE_DIRS} )
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTIONS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} HailoRT::libhailort Threads::Threads)
