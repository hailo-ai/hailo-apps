cmake_minimum_required(VERSION 3.12)
project(onnxrt_hailo_pipeline LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../external")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.0")
        message(FATAL_ERROR
            "GCC 11+ required. Found version ${CMAKE_CXX_COMPILER_VERSION} at ${CMAKE_CXX_COMPILER}\n"
            "Fix:\n"
            "  • Install GCC 11+: sudo apt install gcc-11 g++-11\n"
            "  • Reconfigure: cmake -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11 <src>\n"
            "  • Delete your build directory and rerun cmake.")
    endif()
endif()

find_package(Threads REQUIRED)
# Prefer system-installed HailoRT over a separate /usr/local installation.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT HailoRT_DIR)
    find_package(HailoRT CONFIG QUIET PATHS /usr NO_DEFAULT_PATH)
endif()
find_package(HailoRT CONFIG REQUIRED)
message(STATUS "HailoRT configuration: ${HailoRT_DIR}")
find_package(OpenCV REQUIRED)

message(STATUS "Found OpenCV includes: ${OpenCV_INCLUDE_DIRS}")

find_package(yaml-cpp QUIET)

if(yaml-cpp_FOUND)
    message(STATUS "yaml-cpp: using system ${yaml-cpp_VERSION}")
else()
    message(STATUS "yaml-cpp: building from submodule")
    if(NOT EXISTS "${THIRD_PARTY_DIR}/yaml-cpp/CMakeLists.txt")
        message(FATAL_ERROR
            "yaml-cpp submodule missing.\n"
            "Run:  git submodule update --init --recursive\n"
            "Or:   sudo apt-get install libyaml-cpp-dev")
    endif()
    set(YAML_CPP_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
    add_subdirectory("${THIRD_PARTY_DIR}/yaml-cpp" "${CMAKE_BINARY_DIR}/_deps/yaml-cpp")
endif()

if(TARGET yaml-cpp::yaml-cpp)
    set(YAMLCPP_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
    set(YAMLCPP_TARGET yaml-cpp)
else()
    message(FATAL_ERROR "yaml-cpp target was not created.")
endif()

find_package(CURL QUIET)

if(CURL_FOUND)
    message(STATUS "curl: using system ${CURL_VERSION_STRING}")
else()
    message(STATUS "curl: building HTTP-only from submodule")
    if(NOT EXISTS "${THIRD_PARTY_DIR}/curl/CMakeLists.txt")
        message(FATAL_ERROR
            "curl submodule missing.\n"
            "Run:  git submodule update --init --recursive\n"
            "Or:   sudo apt-get install libcurl4-openssl-dev")
    endif()
    set(BUILD_CURL_EXE      OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
    set(CURL_DISABLE_TESTS  ON  CACHE BOOL "" FORCE)
    set(CURL_USE_LIBPSL     OFF CACHE BOOL "" FORCE)
    set(CURL_BROTLI         OFF CACHE BOOL "" FORCE)
    set(CURL_ZSTD           OFF CACHE BOOL "" FORCE)
    set(USE_NGHTTP2         OFF CACHE BOOL "" FORCE)
    set(CURL_USE_LIBSSH2    OFF CACHE BOOL "" FORCE)
    set(CURL_USE_GSSAPI     OFF CACHE BOOL "" FORCE)
    set(CURL_DISABLE_DICT   ON  CACHE BOOL "" FORCE)
    set(CURL_DISABLE_FILE   ON  CACHE BOOL "" FORCE)
    set(CURL_DISABLE_FTP    ON  CACHE BOOL "" FORCE)
    set(CURL_DISABLE_GOPHER ON  CACHE BOOL "" FORCE)
    set(CURL_DISABLE_IMAP   ON  CACHE BOOL "" FORCE)
    set(CURL_DISABLE_LDAP   ON  CACHE BOOL "" FORCE)
    set(CURL_DISABLE_LDAPS  ON  CACHE BOOL "" FORCE)
    set(CURL_DISABLE_MQTT   ON  CACHE BOOL "" FORCE)
    set(CURL_DISABLE_POP3   ON  CACHE BOOL "" FORCE)
    set(CURL_DISABLE_RTSP   ON  CACHE BOOL "" FORCE)
    set(CURL_DISABLE_SMB    ON  CACHE BOOL "" FORCE)
    set(CURL_DISABLE_SMTP   ON  CACHE BOOL "" FORCE)
    set(CURL_DISABLE_TELNET ON  CACHE BOOL "" FORCE)
    set(CURL_DISABLE_TFTP   ON  CACHE BOOL "" FORCE)
    if(WIN32)
        set(CURL_USE_SCHANNEL ON  CACHE BOOL "" FORCE)
        set(CURL_USE_OPENSSL  OFF CACHE BOOL "" FORCE)
    endif()
    add_subdirectory("${THIRD_PARTY_DIR}/curl" "${CMAKE_BINARY_DIR}/_deps/curl")
    if(TARGET libcurl AND NOT TARGET CURL::libcurl)
        add_library(CURL::libcurl ALIAS libcurl)
    endif()
endif()

if(NOT TARGET CURL::libcurl)
    message(FATAL_ERROR "libcurl target missing. Check curl submodule or system install.")
endif()

set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime root directory (contains include/ and lib/)")

if (NOT ONNXRUNTIME_DIR)
    if(DEFINED ENV{ONNXRUNTIME_DIR})
        set(ONNXRUNTIME_DIR "$ENV{ONNXRUNTIME_DIR}")
    endif()
endif()

if (NOT ONNXRUNTIME_DIR)
    if (WIN32)
        message(FATAL_ERROR
            "ONNXRUNTIME_DIR is not set on Windows.\n"
            "Fix:\n"
            "  - Set -DONNXRUNTIME_DIR=C:/path/to/onnxruntime\n"
            "  - Or set env var ONNXRUNTIME_DIR\n"
            "Expected layout:\n"
            "  <ONNXRUNTIME_DIR>/include\n"
            "  <ONNXRUNTIME_DIR>/lib\n")
    else()
        file(GLOB ALL_SUBDIRS "/opt/*")
        foreach(DIR ${ALL_SUBDIRS})
            if (IS_DIRECTORY "${DIR}" AND "${DIR}" MATCHES ".*onnxruntime.*")
                set(ONNXRUNTIME_DIR "${DIR}")
                break()
            endif()
        endforeach()
    endif()
endif()

if (ONNXRUNTIME_DIR)
    message(STATUS "Using ONNX Runtime directory: ${ONNXRUNTIME_DIR}")

    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS "${ONNXRUNTIME_DIR}/include"
              "${ONNXRUNTIME_DIR}/include/onnxruntime/core/session"
        REQUIRED
    )

    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        HINTS "${ONNXRUNTIME_DIR}/lib"
        REQUIRED
    )
else()
    message(FATAL_ERROR "Failed to locate ONNX Runtime. Set ONNXRUNTIME_DIR.")
endif()

include(ExternalProject)

set(EXTERNAL_PREFIX ${CMAKE_BINARY_DIR}/external)

ExternalProject_Add(xtl-test
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl
    GIT_TAG 0.7.7
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PREFIX}
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    INSTALL_DIR ${EXTERNAL_PREFIX}
)

ExternalProject_Add(xtensor-test
    DEPENDS xtl-test
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
    GIT_TAG 0.24.6
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PREFIX}
        -Dxtl_DIR=${EXTERNAL_PREFIX}/share/cmake/xtl
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    INSTALL_DIR ${EXTERNAL_PREFIX}
)

set(XTENSOR_INCLUDE_DIR ${EXTERNAL_PREFIX}/include)

add_executable(${PROJECT_NAME}
    onnxrt_hailo_pipeline.cpp
    onnx_decode.cpp
    instance_seg_postprocess.cpp
    ../common/toolbox.cpp
    ../common/hailo_infer.cpp
    ../common/resources_manager.cpp
)

add_dependencies(${PROJECT_NAME} xtl-test xtensor-test)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${XTENSOR_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_BINARY_DIR}/external/include
)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        /w /O2
        /EHsc
        /permissive-
        /Zc:__cplusplus
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        -O3 -DNDEBUG -w

    )
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
    Threads::Threads
    HailoRT::libhailort
    ${OpenCV_LIBS}
    ${YAMLCPP_TARGET}
    CURL::libcurl
    ${ONNXRUNTIME_LIB}
)

set(HAILO_APPS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")  # onnxrt_hailo_pipeline -> cpp -> hailo_apps
set(RESOURCES_YAML_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../config/resources_config.yaml")

if(NOT EXISTS "${RESOURCES_YAML_SRC}")
    message(FATAL_ERROR "resources_config.yaml not found at: ${RESOURCES_YAML_SRC}")
endif()

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${PROJECT_NAME}>/config"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${RESOURCES_YAML_SRC}"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/config/resources_config.yaml"
    COMMENT "Copy resources_config.yaml next to the executable"
)
