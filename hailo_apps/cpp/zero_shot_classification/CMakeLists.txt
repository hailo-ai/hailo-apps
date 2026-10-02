cmake_minimum_required(VERSION 3.16)
project(zero_shot_classification LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
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

# ZLIB (required by cnpy for .npz support)
include(FetchContent)

find_package(ZLIB QUIET)

if(NOT ZLIB_FOUND)
    message(STATUS "ZLIB not found - fetching zlib via FetchContent")

    FetchContent_Declare(
        zlib
        URL https://zlib.net/zlib-1.3.1.tar.gz
    )

    # Keep it static to avoid runtime DLL hassles
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(zlib)

    if(TARGET zlibstatic)
        add_library(ZLIB::ZLIB ALIAS zlibstatic)
    elseif(TARGET zlib)
        add_library(ZLIB::ZLIB ALIAS zlib)
    else()
        message(FATAL_ERROR "Fetched zlib but no target was created (expected zlibstatic or zlib).")
    endif()
endif()

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

set(SOURCES
    ./tokenizer/tokenizer.cpp
    ./tokenizer/cnpy.cpp
    ./clip_example.cpp
    ../common/toolbox.cpp
    ../common/hailo_infer.cpp
    ../common/resources_manager.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        /w
        /O2
        /EHsc
        /permissive-
        /Zc:__cplusplus
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        -O3
        -w

    )
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
    Threads::Threads
    HailoRT::libhailort
    ${OpenCV_LIBS}
    ${YAMLCPP_TARGET}
    CURL::libcurl
    ZLIB::ZLIB
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/config

    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/../../config/resources_config.yaml
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/config/resources_config.yaml
)
