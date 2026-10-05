cmake_minimum_required(VERSION 3.20)
project(instance_segmentation LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../external")

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

if(HAILO_USE_SYSTEM_DEPS)
    find_package(xtl CONFIG REQUIRED)
    find_package(xtensor CONFIG REQUIRED)
    set(USE_SYSTEM_XTENSOR ON)
else()
    message(STATUS "Fetching xtl/xtensor (build tree only)")
    include(FetchContent)

    FetchContent_Declare(
        xtl
        GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
        GIT_TAG        0.8.0
        PATCH_COMMAND  sed -i "s/cmake_minimum_required(VERSION 3.29)/cmake_minimum_required(VERSION 3.16)/" CMakeLists.txt
    )
    FetchContent_Declare(
        xtensor
        GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
        GIT_TAG        0.26.0
        PATCH_COMMAND  sed -i "s/cmake_minimum_required(VERSION 3.29)/cmake_minimum_required(VERSION 3.16)/" CMakeLists.txt
    )
    FetchContent_MakeAvailable(xtl xtensor)
    set(USE_SYSTEM_XTENSOR OFF)
endif()

add_executable(${PROJECT_NAME}
    instance_segmentation.cpp
    instance_seg_postprocess.cpp
    ../common/toolbox.cpp
    ../common/hailo_infer.cpp
    ../common/resources_manager.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

if(USE_SYSTEM_XTENSOR)
    target_link_libraries(${PROJECT_NAME} PRIVATE xtl xtensor)
else()
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${xtl_SOURCE_DIR}/include
        ${xtensor_SOURCE_DIR}/include
    )
endif()

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
)

set(HAILO_APPS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")  # instance_segmentation -> cpp -> hailo_apps
set(RESOURCES_YAML_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../config/resources_config.yaml")

if(NOT EXISTS "${RESOURCES_YAML_SRC}")
    message(FATAL_ERROR "resources_config.yaml not found at: ${RESOURCES_YAML_SRC}")
endif()

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${PROJECT_NAME}>/config"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${RESOURCES_YAML_SRC}"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/config/resources_config.yaml"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/visualization_config.yaml"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/config/visualization_config.yaml"
    COMMENT "Copy config files next to the executable"
)
