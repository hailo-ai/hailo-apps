#include "toolbox.hpp"
#include "hailo_infer.hpp"
#include "resources_manager.hpp"
#include <chrono>
#include <thread>
#include <filesystem>
#include <fstream>
#include <string>
#include <stdexcept>
#include <regex>
#include <sstream>
#include <unordered_set>
#include <yaml-cpp/yaml.h>
#include <vector>
#include <cstdio>
#include <array>
#include <algorithm>

#include <vector>
#include <string>

#ifdef _WIN32
#include <opencv2/videoio.hpp>
#else
#include <filesystem>
#include <cstdio>
#include <array>
#endif

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#define popen  _popen
#define pclose _pclose
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

namespace hailo_utils {
namespace fs = std::filesystem;

const std::unordered_map<std::string, std::pair<int,int>> RESOLUTION_MAP = {
    {"sd",  {640, 480}},
    {"hd",  {1280, 720}},
    {"fhd", {1920, 1080}}
};

fs::path get_executable_dir()
{
#ifdef _WIN32
    char buf[MAX_PATH];
    DWORD len = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (len == 0 || len == MAX_PATH)
        throw std::runtime_error("GetModuleFileNameA failed");
    return fs::path(std::string(buf, len)).parent_path();
#else
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0)
        throw std::runtime_error("readlink(/proc/self/exe) failed");
    buf[len] = '\0';
    return fs::path(buf).parent_path();
#endif
}

VisualizationParams load_visualization_params(const std::string &path)
{
    YAML::Node root;

    fs::path resolved = get_executable_dir() / "config" / fs::path(path).filename();

    try {
        root = YAML::LoadFile(resolved.string());
    } catch (const std::exception &e) {
        throw std::runtime_error(
            "Failed to load visualization config file: " + resolved.string() +
            " | " + e.what());
    }

    if (!root["visualization_params"]) {
        throw std::runtime_error(
            "Missing 'visualization_params' section in config file");
    }

    const YAML::Node vp = root["visualization_params"];

    VisualizationParams params;

    // Required Fields
    if (!vp["score_thresh"]) {
        throw std::runtime_error(
            "Missing visualization_params.score_thresh");
    }

    if (!vp["max_boxes_to_draw"]) {
        throw std::runtime_error(
            "Missing visualization_params.max_boxes_to_draw");
    }

    params.score_thresh = vp["score_thresh"].as<float>();
    params.max_boxes_to_draw = vp["max_boxes_to_draw"].as<int>();

    // Optional Fields (Instance Segmentation)
    if (vp["mask_thresh"]) {
        params.mask_thresh = vp["mask_thresh"].as<float>();
    }

    if (vp["mask_alpha"]) {
        params.mask_alpha = vp["mask_alpha"].as<float>();
    }

    return params;
}


void validate_visualization_params(const VisualizationParams &vis, AppVisMode mode)
{
    // basic required fields
    if (vis.max_boxes_to_draw < 0) {
        throw std::runtime_error("visualization_params.max_boxes_to_draw must be >= 0");
    }
    if (vis.score_thresh < 0.0f || vis.score_thresh > 1.0f) {
        throw std::runtime_error("visualization_params.score_thresh must be in [0,1]");
    }

    // per-app requirements
    if (mode == AppVisMode::instance_seg) {
        if (!vis.mask_thresh || !vis.mask_alpha) {
            throw std::runtime_error(
                "Instance segmentation requires visualization_params.mask_thresh and mask_alpha");
        }
        if (*vis.mask_thresh < 0.0f || *vis.mask_thresh > 1.0f) {
            throw std::runtime_error("visualization_params.mask_thresh must be in [0,1]");
        }
        if (*vis.mask_alpha < 0.0f || *vis.mask_alpha > 1.0f) {
            throw std::runtime_error("visualization_params.mask_alpha must be in [0,1]");
        }
    }
}

static bool has_gstreamer_element(const std::string &element_name);

static std::string make_csi_gst_pipeline(const std::string &device, int w, int h)
{
    // On Astrial/IMX8: use hardware G2D scaler/converter if available,
    // fall back to software.
    if (has_gstreamer_element("imxvideoconvert_g2d")) {
        return "v4l2src device=" + device +
               " ! imxvideoconvert_g2d ! video/x-raw,format=BGRx"
               ",width=" + std::to_string(w) + ",height=" + std::to_string(h) +
               " ! videoconvert ! video/x-raw,format=BGR"
               " ! appsink max-buffers=1 drop=true sync=false";
    }
    return "v4l2src device=" + device +
           " ! videoscale ! videoconvert"
           " ! video/x-raw,format=BGR,width=" + std::to_string(w) +
           ",height=" + std::to_string(h) +
           " ! appsink max-buffers=1 drop=true sync=false";
}

static std::string make_rpi_gst_pipeline(int w, int h, int fps)
{
    return "libcamerasrc ! "
           "video/x-raw,format=NV12,width=" + std::to_string(w) +
           ",height=" + std::to_string(h) +
           ",framerate=" + std::to_string(fps) + "/1 ! "
           "videoconvert ! "
           "video/x-raw,format=BGR ! "
           "appsink max-buffers=1 drop=true sync=false";
}


static bool is_raspberry_pi()
{
#ifdef _WIN32
    return false;
#else
    const std::string RPI_POSSIBLE_NAME = "Raspberry Pi";

    std::ifstream f("/proc/device-tree/model");
    if (!f.is_open()) {
        return false;
    }

    std::string model;
    std::getline(f, model);

    return model.find(RPI_POSSIBLE_NAME) != std::string::npos;
#endif
}

static bool is_yocto()
{
#ifdef _WIN32
    return false;
#else
    // Known identifiers for Yocto/OpenEmbedded-based distros.
    // Checked case-insensitively against the ID, ID_LIKE, and CPE_NAME fields.
    static const std::vector<std::string> YOCTO_MARKERS = {
        "poky", "yocto", "openembedded", "fsl-imx", "buildroot", "openwrt"
    };

    std::ifstream f("/etc/os-release");
    if (!f.is_open()) return false;

    std::string line;
    while (std::getline(f, line)) {
        // Only inspect relevant keys
        const bool relevant =
            line.rfind("ID=", 0)       == 0 ||
            line.rfind("ID_LIKE=", 0)  == 0 ||
            line.rfind("CPE_NAME=", 0) == 0;
        if (!relevant) continue;

        std::string lower = line;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        for (const auto &marker : YOCTO_MARKERS) {
            if (lower.find(marker) != std::string::npos)
                return true;
        }
    }
    return false;
#endif
}


hailo_status wait_and_check_threads(
    std::future<hailo_status> &f1, const std::string &name1,
    std::future<hailo_status> &f2, const std::string &name2,
    std::future<hailo_status> &f3, const std::string &name3,
    std::future<hailo_status> *f4, const std::string &name4)
{
    auto get_or_report = [](std::future<hailo_status> &f, const std::string &name) -> hailo_status {
        try {
            auto st = f.get();
            if (HAILO_SUCCESS != st) {
                std::cerr << name << " failed with status " << st << std::endl;
            }
            return st;
        } catch (const std::exception &e) {
            std::cerr << name << " threw exception: " << e.what() << std::endl;
            return HAILO_INTERNAL_FAILURE;
        } catch (...) {
            std::cerr << name << " threw unknown exception" << std::endl;
            return HAILO_INTERNAL_FAILURE;
        }
    };

    hailo_status status = get_or_report(f1, name1);
    if (HAILO_SUCCESS != status) return status;

    status = get_or_report(f2, name2);
    if (HAILO_SUCCESS != status) return status;

    status = get_or_report(f3, name3);
    if (HAILO_SUCCESS != status) return status;

    if (f4) {
        status = get_or_report(*f4, name4.empty() ? "thread4" : name4);
        if (HAILO_SUCCESS != status) return status;
    }

    return HAILO_SUCCESS;
}

bool is_image_file(const std::string& path) {
    static const std::vector<std::string> image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"};
    std::string extension = fs::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    return std::find(image_extensions.begin(), image_extensions.end(), extension) != image_extensions.end();
}

bool is_video_file(const std::string& path) {
    static const std::vector<std::string> video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"};
    std::string extension = fs::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    return std::find(video_extensions.begin(), video_extensions.end(), extension) != video_extensions.end();
}

bool is_directory_of_images(const std::string& path, size_t &entry_count, size_t batch_size) {
    entry_count = 0;
    if (fs::exists(path) && fs::is_directory(path)) {
        bool has_images = false;
        for (const auto& entry : fs::directory_iterator(path)) {
            if (fs::is_regular_file(entry)) {
                entry_count++;
                if (!is_image_file(entry.path().string())) {
                    // Found a non-image file
                    return false;
                }
                has_images = true; 
            }
        }
        if (entry_count % batch_size != 0) {
            throw std::invalid_argument("Directory contains " + std::to_string(entry_count) + " images, which is not divisible by batch size " + std::to_string(batch_size));
        }
        return has_images; 
    }
    return false;
}


std::string parse_output_resolution_arg(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--output-resolution" || arg == "-or") {

            if (i + 1 >= argc) {
                std::cerr << "-E- --output-resolution requires a value.\n";
                std::exit(1);
            }

            std::string first = argv[i + 1];

            // Convert presets directly to WxH
            if (first == "sd")  return "640x480";
            if (first == "hd")  return "1280x720";
            if (first == "fhd") return "1920x1080";

            // Custom width & height
            if (i + 2 < argc) {
                std::string second = argv[i + 2];

                auto is_digits = [](const std::string &s) {
                    return !s.empty() &&
                           std::all_of(s.begin(), s.end(),
                           [](unsigned char c){ return std::isdigit(c); });
                };

                if (is_digits(first) && is_digits(second)) {
                    return first + "x" + second;  // "1920x1080"
                }
            }

            std::cerr << "-E- Invalid --output-resolution value.\n"
                      << "    Allowed: sd | hd | fhd | <width> <height>\n";
            std::exit(1);
        }
    }

    return "";  // No resolution provided
}


bool is_image(const std::string& path) {
    return fs::exists(path) && fs::is_regular_file(path) && is_image_file(path);
}

bool is_video(const std::string& path) {
    return fs::exists(path) && fs::is_regular_file(path) && is_video_file(path);
}

std::string get_hef_name(const std::string &path)
{
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}


std::string getCmdOption(int argc, char *argv[], const std::string &option)
{
    std::string cmd;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (0 == arg.find(option, 0)) {
            std::size_t found = arg.find("=", 0) + 1;
            cmd = arg.substr(found);
            return cmd;
        }
    }
    return cmd;
}

bool has_flag(int argc, char *argv[], const std::string &flag) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == flag) {
            return true;
        }
    }
    return false;
}
std::string getCmdOptionWithShortFlag(int argc, char *argv[], const std::string &longOption, const std::string &shortOption) {
    std::string cmd;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == longOption || arg == shortOption) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                cmd = argv[i + 1];
                return cmd;
            }
        }
    }
    return cmd;
}

CommandLineArgs parse_command_line_arguments(int argc, char** argv) {

    std::string batch_str = getCmdOptionWithShortFlag(argc, argv, "--batch-size", "-b");
    std::string fps_str = getCmdOptionWithShortFlag(argc, argv, "--framerate", "-f");

    // Convert to proper types with defaults
    size_t batch_size = batch_str.empty() ? static_cast<size_t>(1) : static_cast<size_t>(std::stoul(batch_str));
    double framerate = fps_str.empty() ? 30.0 : std::stod(fps_str);
    std::string out_res_str = parse_output_resolution_arg(argc, argv);

    return {
        getCmdOptionWithShortFlag(argc, argv, "--net",   "-n"),
        getCmdOptionWithShortFlag(argc, argv, "--input", "-i"),
        getCmdOptionWithShortFlag(argc, argv, "--output-dir", "-o"),
        getCmdOptionWithShortFlag(argc, argv, "--camera-resolution", "-cr"),
        out_res_str,
        has_flag(argc, argv, "-s") || has_flag(argc, argv, "--save-stream-output"),
        has_flag(argc, argv, "--no-display"),
        batch_size,
        framerate,
    };
}

void post_parse_args(const std::string &app, CommandLineArgs &args, int argc, char **argv)
{
    try {
        hailo_apps::ResourcesManager rm;
        if (has_flag(argc, argv, "--list-models")) {
            rm.print_models(app);
            std::exit(0);
        }

        if (has_flag(argc, argv, "--list-inputs")) {
            rm.print_inputs(app);
            std::exit(0);
        }

        args.net = rm.resolve_net_arg(app, args.net);
        args.input = rm.resolve_input_arg(app, args.input);
    }
    catch (const std::exception &e) {
            std::cerr << "ResourcesManager ERROR: " << e.what() << std::endl;
            std::exit(1);
    }
}

std::string get_model_meta_value(const std::string &app,
                                  const std::string &model_name,
                                  const std::string &key)
{
    try {
        hailo_apps::ResourcesManager rm;
        return rm.get_model_meta_value(app, model_name, key);
    } catch (const std::exception &e) {
        std::cerr << "get_model_meta_value error: "
                  << e.what() << std::endl;
        return "N/A";
    }
    catch (...) {
        std::cerr << "get_model_meta_value unknown error."
                  << std::endl;
        return "N/A";
    }
}

#ifndef _WIN32
static std::string run_command(const std::string &cmd)
{
    std::array<char, 256> buffer{};
    std::string output;

    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return "";
    }

    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        output += buffer.data();
    }

    pclose(pipe);
    return output;
}


static std::vector<CameraDevice> get_usb_video_devices_linux()
{
    namespace fs = std::filesystem;

    std::cout << "Scanning /dev for video devices..." << std::endl;

    std::vector<CameraDevice> usb_video_devices;
    std::vector<std::string> video_devices;

    for (const auto &entry : fs::directory_iterator("/dev")) {
        const std::string device_name = entry.path().filename().string();

        if (device_name.rfind("video", 0) == 0) {
            video_devices.push_back("/dev/" + device_name);
        }
    }

    for (const auto &device : video_devices) {
        try {

            const std::string cmd =
                "udevadm info --query=all --name=" + device + " 2>/dev/null";
            const std::string output = run_command(cmd);

            if (output.find("ID_BUS=usb") != std::string::npos &&
                output.find(":capture:") != std::string::npos) {
                std::cout << "USB camera detected: " << device << std::endl;
                usb_video_devices.push_back(device);
            }
        } catch (const std::exception &e) {
            std::cerr << "Error checking device " << device << ": " << e.what() << std::endl;
        }
    }

    std::cout << "USB video devices found on Linux:" << std::endl;
    for (const auto &device : usb_video_devices) {
        std::cout << "  " << std::get<std::string>(device) << std::endl;
    }

    return usb_video_devices;
}

static std::string find_csi_video_device_linux()
{
    std::vector<std::string> candidates;
    for (const auto &entry : fs::directory_iterator("/dev")) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("video", 0) == 0)
            candidates.push_back("/dev/" + name);
    }
    std::sort(candidates.begin(), candidates.end());

    for (const auto &device : candidates) {
        const std::string info = run_command(
            "udevadm info --query=all --name=" + device + " 2>/dev/null");

        if (info.find("ID_BUS=usb") != std::string::npos)
            continue;

        // Check capture capability: prefer udev property, fall back to v4l2-ctl
        if (info.find("ID_V4L_CAPABILITIES") != std::string::npos) {
            if (info.find(":capture:") == std::string::npos) continue;
        } else {
            const std::string cap = run_command(
                "v4l2-ctl --device=" + device + " --info 2>/dev/null");
            if (cap.find("Video Capture") == std::string::npos) continue;
        }

        std::cout << "CSI camera detected: " << device << "\n";
        return device;
    }
    return "";
}
#endif

#ifdef _WIN32
static std::vector<CameraDevice> get_usb_video_devices_windows()
{
    std::vector<CameraDevice> usb_video_devices;

    try {
        for (int index = 0; index < 10; ++index) {
            cv::VideoCapture cap(index, cv::CAP_DSHOW);

            if (cap.isOpened()) {
                std::cout << "USB camera detected: index=" << index << std::endl;
                usb_video_devices.push_back(index);
                cap.release();
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "Failed to enumerate Windows video devices: " << e.what() << std::endl;
        return {};
    }

    std::cout << "USB video devices found on Windows:" << std::endl;
    for (const auto &device : usb_video_devices) {
        std::cout << "  " << std::get<int>(device) << std::endl;
    }

    return usb_video_devices;
}
#endif

std::vector<CameraDevice> get_usb_video_devices()
{
#ifdef _WIN32
    std::cout << "Detecting USB video devices on Windows..." << std::endl;
    return get_usb_video_devices_windows();
#else
    std::cout << "Detecting USB video devices on Linux..." << std::endl;
    return get_usb_video_devices_linux();
#endif
}

static bool is_digits_only(const std::string &s)
{
    return !s.empty() && std::all_of(s.begin(), s.end(),
                                    [](unsigned char c){ return std::isdigit(c); });
}


InputType determine_input_type(const std::string& input_path,
                               cv::VideoCapture &capture,
                               double &org_height,
                               double &org_width,
                               size_t &frame_count,
                               size_t batch_size,
                               const std::string &camera_resolution)
{
    InputType input_type;

    // --------------------------------------------
    // 1) Images / directory / video
    // --------------------------------------------
    if (is_directory_of_images(input_path, frame_count, batch_size)) {
        input_type.is_directory = true;

    } else if (is_image(input_path)) {
        input_type.is_image = true;
        frame_count = 1;

    } else if (is_video(input_path)) {
        input_type.is_video = true;
        capture = open_video_capture(input_path, capture, org_height, org_width, frame_count,
                                     false /*is_camera*/, "");
    // --------------------------------------------
    // 2) Camera inputs
    // --------------------------------------------

    //1. Windows camera index: "0", "1", ...
    } else if (is_digits_only(input_path)) {
        input_type.is_camera = true;
    #ifdef _WIN32
        std::cout << "Using USB camera: index " << input_path << "\n";
        capture = open_video_capture(input_path, capture, org_height, org_width, frame_count,
                                    true /*is_camera*/, camera_resolution);
    #else
        // On Linux this is not a valid camera spec; keep it strict
        throw std::runtime_error("Numeric camera index is supported only on Windows. Use /dev/videoX or 'usb'.");
    #endif

    //2. Linux explicit device: /dev/videoX
    } else if (input_path.rfind("/dev/video", 0) == 0) {
        input_type.is_camera = true;
    #ifdef _WIN32
        throw std::runtime_error("'/dev/videoX' is supported only on Linux. On Windows use 'usb' or a camera index.");
    #else
        std::cout << "Using USB camera: " << input_path << "\n";
        capture = open_video_capture(input_path, capture, org_height, org_width, frame_count,
                                    true /*is_camera*/, camera_resolution);
    #endif

    //3. "usb" shortcut: auto-select default camera
    } else if (input_path == "usb") {
        input_type.is_camera = true;

        const auto usb_devices = get_usb_video_devices();
        if (usb_devices.empty()) {
            throw std::runtime_error("No USB camera detected");
        }

    #ifdef _WIN32
        const int camera_index = std::get<int>(usb_devices.front());
        std::cout << "Using USB camera: index " << camera_index << "\n";
        capture = open_video_capture(std::to_string(camera_index), capture, org_height, org_width,
                                     frame_count, true /*is_camera*/, camera_resolution);
    #else
        const std::string video_device = std::get<std::string>(usb_devices.front());
        std::cout << "Using USB camera: " << video_device << "\n";
        capture = open_video_capture(video_device, capture, org_height, org_width, frame_count,
                                     true /*is_camera*/, camera_resolution);
    #endif

    //4. RPI camera shortcut
    } else if (input_path == "rpi") {
        if (!is_raspberry_pi()) {
            throw std::runtime_error(
                "'rpi' camera input is supported only on Raspberry Pi devices.");
        }
        input_type.is_camera = true;
        std::cout << "Using RPI camera\n";
        capture = open_video_capture(input_path, capture, org_height, org_width, frame_count,
                                     true /*is_camera*/, camera_resolution);

    //5. CSI camera shortcut (Astrial/IMX8 and other platform CSI cameras)
    } else if (input_path == "csi") {
#ifdef _WIN32
        throw std::runtime_error("'csi' camera input is not supported on Windows.");
#else
        if (!is_yocto()) {
            throw std::runtime_error(
                "'csi' camera input is only supported on Yocto-based embedded systems (e.g. Astrial/IMX8).\n"
                "Use '-i usb' for a USB camera or '-i rpi' on Raspberry Pi.");
        }
        const std::string csi_device = find_csi_video_device_linux();
        if (csi_device.empty()) {
            std::cerr << "-W- No CSI camera detected.\n"
                      << "-W- On Astrial/IMX8, initialize the ISP first:\n"
                      << "-W-   cd /opt/imx8-isp/bin && ./run.sh -lm -c dual_imx219_1080p60 &\n";
            throw std::runtime_error("No CSI camera detected");
        }
        input_type.is_camera = true;
        std::cout << "Using CSI camera: " << csi_device << "\n";
        capture = open_video_capture("csi:" + csi_device, capture, org_height, org_width,
                                     frame_count, true /*is_camera*/, camera_resolution);
#endif

    } else {
        throw std::runtime_error("Unsupported input type: " + input_path);
    }

    return input_type;
}


void show_progress_helper(size_t current, size_t total)
{
    if (total == 0) {
        std::cout << "\rProcessed: " << (current + 1) << " frames" << std::flush;
        return;
    }
    int progress = static_cast<int>((static_cast<float>(current + 1) / static_cast<float>(total)) * 100);
    int bar_width = 50; 
    int pos = static_cast<int>(bar_width * (current + 1) / total);

    std::cout << "\rProgress: [";
    for (int j = 0; j < bar_width; ++j) {
        if (j < pos) std::cout << "=";
        else if (j == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << progress
              << "% (" << std::setw(3) << (current + 1) << "/" << total << ")" << std::flush;
}

void show_progress(hailo_utils::InputType &input_type, int progress, size_t frame_count) {
    if (input_type.is_video || input_type.is_directory) {
        show_progress_helper(progress, frame_count);
    }
}

void print_inference_statistics(std::chrono::duration<double> inference_time,
    const std::string &hef_file,
    double frame_count,
    std::chrono::duration<double> total_time)
{
    std::cout << BOLDGREEN << "\n-I-----------------------------------------------" << std::endl;
    std::cout << "-I- Inference & Postprocess                        " << std::endl;
    std::cout << "-I- Average FPS:  " << frame_count / (inference_time.count()) << std::endl;
    std::cout << "-I- Total time:   " << inference_time.count() << " sec" << std::endl;
    std::cout << "-I- Latency:      "
    << 1.0 / (frame_count / (inference_time.count()) / 1000) << " ms" << std::endl;
    std::cout << "-I-----------------------------------------------" << std::endl;
    std::cout << BOLDBLUE << "\n-I- Application finished successfully" << RESET << std::endl;
    std::cout << BOLDBLUE << "-I- Total application run time: " << (double)total_time.count() << " sec" << RESET << std::endl;
}

void print_net_banner(const std::string &detection_model_name,
    const std::vector<hailort::InferModel::InferStream> &inputs,
    const std::vector<hailort::InferModel::InferStream> &outputs)
{
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I-  Network Name                               " << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I   " << detection_model_name << std::endl << RESET;
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    for (auto &input : inputs) {
        auto shape = input.shape();
        std::cout << MAGENTA << "-I-  Input: " << input.name()
        << ", Shape: (" << shape.height << ", " << shape.width << ", " << shape.features << ")"
        << std::endl << RESET;
    }
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------" << std::endl << RESET;
    for (auto &output : outputs) {
        auto shape = output.shape();
        std::cout << MAGENTA << "-I-  Output: " << output.name()
        << ", Shape: (" << shape.height << ", " << shape.width << ", " << shape.features << ")"
        << std::endl << RESET;
    }
    std::cout << BOLDMAGENTA << "-I-----------------------------------------------\n" << std::endl << RESET;
}

void init_video_writer(const std::string &output_path, cv::VideoWriter &video, double framerate, int org_width, int org_height) {
    cv::Size sz(org_width, org_height);
    video.open(output_path, cv::VideoWriter::fourcc('m','p','4','v'), framerate, sz);
    if (!video.isOpened()) {
        std::string avi_path = output_path.substr(0, output_path.rfind('.')) + ".avi";
        video.open(avi_path, cv::VideoWriter::fourcc('M','J','P','G'), framerate, sz);
        if (!video.isOpened()) {
            std::cerr << "-W- VideoWriter failed to open (tried mp4v + MJPG). Output will not be saved.\n";
        }
    }
}

static bool has_gstreamer_element(const std::string &element_name)
{
    const std::string cmd =
        "gst-inspect-1.0 " + element_name + " > /dev/null 2>&1";

    const int ret = std::system(cmd.c_str());
    return ret == 0;
}

cv::VideoCapture open_video_capture(const std::string &input_path,
    cv::VideoCapture &capture,
    double &org_height,
    double &org_width,
    size_t &frame_count,
    bool is_camera,
    const std::string &camera_resolution)
    {
    const bool is_rpi_input = (input_path == "rpi");
    const bool is_csi       = (input_path.rfind("csi:", 0) == 0);
    const std::string device = is_csi ? input_path.substr(4) : input_path;
    // Validate platform
    if (is_rpi_input && !is_raspberry_pi()) {
        throw std::runtime_error(
            "You requested '-i rpi', but this system is not a Raspberry Pi.\n"
            "Use '-i usb' or a video file instead.");
    }

    // Default camera resolution
    int width  = 640;
    int height = 480;
    int fps    = 30;

    // Apply user camera resolution
    if (is_camera && !camera_resolution.empty()) {
        auto it = RESOLUTION_MAP.find(camera_resolution);
        if (it == RESOLUTION_MAP.end()) {
            std::cerr << "-W- Unknown camera resolution \"" << camera_resolution
                      << "\". Supported values: sd, hd, fhd\n";
        } else {
            width  = it->second.first;
            height = it->second.second;
        }
    }

    if (is_rpi_input) {
        if (!has_gstreamer_element("libcamerasrc")) {
            throw std::runtime_error(
                "Missing required GStreamer element 'libcamerasrc'.\n"
                "Install it with:\n"
                "    sudo apt install gstreamer1.0-libcamera"
            );
        }
        org_width  = width  = 800;
        org_height = height = 600;
        std::string pipeline = make_rpi_gst_pipeline(width, height, fps);
        capture.open(pipeline, cv::CAP_GSTREAMER);
    } else if (is_csi) {
        std::string pipeline = make_csi_gst_pipeline(device, width, height);
        capture.open(pipeline, cv::CAP_GSTREAMER);
    } else {
    #ifdef _WIN32
        if (is_camera) {
            // Windows: open camera by index (e.g. "0"), not by a device path
            int idx = 0;
            try { idx = std::stoi(input_path); } catch (...) { idx = 0; }

            // Prefer DirectShow; if it fails on some machines you can try MSMF
            capture.open(idx, cv::CAP_DSHOW);
        } else {
            // Video/image file path
            capture.open(input_path, cv::CAP_ANY);
        }
    #else
        if (!is_camera && has_gstreamer_element("v4l2h264dec") &&
            has_gstreamer_element("imxvideoconvert_g2d")) {
            // On NXP i.MX8: VPU decode + G2D GPU scale/convert.
            // Key fix: decodebin's internal multiqueue defaults to max-size-bytes=1MB,
            // which only fits ~1 decoded NV12 frame (614KB for 640x640).
            // We add an explicit queue after the decoder (max-size-bytes=0 = unlimited,
            // max-size-buffers=15) so the VPU can stay 15 frames ahead of consumption,
            // matching the throughput seen with the USB camera path.
            // Output BGRx (4 channels) directly - no software videoconvert needed.
            // preprocess_callback handles BGRA2RGB in one pass.
            // Strategy 1: explicit qtdemux+h264parse+v4l2h264dec with large pre/post-decode queues
            // G2D converts format only (NV12→BGRx); preprocess_callback handles resize to model size.
            // This preserves original video resolution in org_frames for proper display output.
            std::string pipeline =
                "filesrc location=" + input_path +
                " ! qtdemux"
                " ! queue max-size-buffers=200 max-size-bytes=0 max-size-time=0"
                " ! h264parse"
                " ! v4l2h264dec"
                " ! queue max-size-buffers=15 max-size-bytes=0 max-size-time=0"
                " ! imxvideoconvert_g2d ! video/x-raw,format=BGRx"
                " ! appsink name=appsink0 sync=false async=false";
            capture.open(pipeline, cv::CAP_GSTREAMER);

            if (!capture.isOpened()) {
                // Strategy 2: decodebin with explicit queue to work around the 1MB buffer limit
                std::string pipeline2 =
                    "filesrc location=" + input_path +
                    " ! decodebin"
                    " ! queue max-size-buffers=15 max-size-bytes=0 max-size-time=0"
                    " ! imxvideoconvert_g2d ! video/x-raw,format=BGRx"
                    " ! appsink name=appsink0 sync=false async=false";
                capture.open(pipeline2, cv::CAP_GSTREAMER);
            }

            if (!capture.isOpened()) {
                std::cerr << "-W- HW G2D pipeline failed, falling back to software decode\n";
                capture.open(input_path, cv::CAP_ANY);
            }
        } else {
            // Linux: camera or platform without HW decoder
            capture.open(input_path, cv::CAP_ANY);
        }
    #endif

        if (is_camera) { // apply camera settings
            // Request MJPG to reduce USB bandwidth vs raw YUYV; industrial/CSI cameras may
            // not support it, so we check whether the driver actually accepted the fourcc
            // and fall back silently rather than aborting.
            double mjpg = cv::VideoWriter::fourcc('M','J','P','G');
            capture.set(cv::CAP_PROP_FOURCC, mjpg);
            if (capture.get(cv::CAP_PROP_FOURCC) != mjpg) {
                std::cerr << "-W- Camera did not accept MJPG fourcc, using default format\n";
            }
            capture.set(cv::CAP_PROP_FRAME_WIDTH,  width);
            capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
            capture.set(cv::CAP_PROP_FPS, fps);
        }
    }

    if (!capture.isOpened()) {
        throw std::runtime_error("Unable to open input file / camera");
    }

    if (!is_rpi_input) {
        org_width  = capture.get(cv::CAP_PROP_FRAME_WIDTH);
        org_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        // GStreamer pipelines without explicit size caps return 0 — probe with one frame
        if ((org_width == 0 || org_height == 0) && !is_camera) {
            cv::Mat probe;
            if (capture.read(probe) && !probe.empty()) {
                org_width  = probe.cols;
                org_height = probe.rows;
            }
        }
    }

    frame_count = is_camera ? 0 : static_cast<size_t>(capture.get(cv::CAP_PROP_FRAME_COUNT));
    // GStreamer custom pipelines may return 0 — fall back to querying the file directly
    if (!is_camera && frame_count == 0) {
        cv::VideoCapture tmp(input_path, cv::CAP_ANY);
        if (tmp.isOpened()) {
            frame_count = static_cast<size_t>(tmp.get(cv::CAP_PROP_FRAME_COUNT));
            tmp.release();
        }
    }
    return capture;
}


void preprocess_video_frames(cv::VideoCapture &capture,
                             uint32_t &width, uint32_t &height,
                             size_t &batch_size, double &framerate,
                             std::shared_ptr<BoundedTSQueue<
                                 std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue,
                             PreprocessCallback preprocess_callback)
{
    std::vector<cv::Mat> org_frames;
    std::vector<cv::Mat> preprocessed_frames;

    const bool limit_fps = (framerate > 0.0);
    using clock = std::chrono::steady_clock;

    clock::duration frame_interval{};
    clock::time_point next_frame_time{};

    if (limit_fps) {
        // Convert from duration<double> to the clock's native duration type
        frame_interval = std::chrono::duration_cast<clock::duration>(
            std::chrono::duration<double>(1.0 / framerate));
        next_frame_time = clock::now() + frame_interval;
    }

    while (true) {
        if (limit_fps) {
            auto now = clock::now();
            if (now < next_frame_time) {
                // sleep_until avoids manual subtraction and type headaches
                std::this_thread::sleep_until(next_frame_time);
            }
        }

        cv::Mat org_frame;
        capture >> org_frame;
        if (org_frame.empty()) {
            preprocessed_batch_queue->stop();
            break;
        }

        org_frames.push_back(org_frame);

        if (org_frames.size() == batch_size) {
            preprocessed_frames.clear();
            preprocess_callback(org_frames, preprocessed_frames, width, height);
            preprocessed_batch_queue->push(std::make_pair(org_frames, preprocessed_frames));
            if (preprocessed_batch_queue->is_stopped()) break;
            org_frames.clear();
        }

        if (limit_fps) {
            next_frame_time += frame_interval;
        }
    }
}


void preprocess_image_frames(const std::string &input_path,
                          uint32_t &width, uint32_t &height, size_t &batch_size,
                          std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue,
                          PreprocessCallback preprocess_callback) {
    cv::Mat org_frame = cv::imread(input_path);
    std::vector<cv::Mat> org_frames = {org_frame}; 
    std::vector<cv::Mat> preprocessed_frames;
    preprocess_callback(org_frames, preprocessed_frames, width, height);
    preprocessed_batch_queue->push(std::make_pair(org_frames, preprocessed_frames));
    
    preprocessed_batch_queue->stop();
}

void preprocess_directory_of_images(const std::string &input_path,
                                uint32_t &width, uint32_t &height, size_t &batch_size,
                                std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue,
                                PreprocessCallback preprocess_callback) {
    std::vector<cv::Mat> org_frames;
    std::vector<cv::Mat> preprocessed_frames;
    
    for (const auto &entry : fs::directory_iterator(input_path)) {
        if (is_image_file(entry.path().string())) {
            cv::Mat org_frame = cv::imread(entry.path().string());
            if (!org_frame.empty()) {
                org_frames.push_back(org_frame);
                
                if (org_frames.size() == batch_size) {
                    preprocessed_frames.clear();
                    preprocess_callback(org_frames, preprocessed_frames, width, height);
                    preprocessed_batch_queue->push(std::make_pair(org_frames, preprocessed_frames));
                    org_frames.clear();
                }
            }
        }
    }    
    preprocessed_batch_queue->stop();
}


/**
 * @brief Pad (bottom/right) and crop (top-left) an image to the target size.
 *
 * If the input image is smaller, it is padded with black pixels on the
 * bottom and right sides. If it is larger, it is cropped from the top-left
 * corner to exactly match the target size. No resizing is performed, so
 * geometry is preserved.
 *
 * @param img        Input image (cv::Mat).
 * @param target_h   Target height in pixels.
 * @param target_w   Target width in pixels.
 * @return cv::Mat   Image of shape (target_h, target_w).
 */
static inline cv::Mat pad_crop_to_target(const cv::Mat &img,
                                         int target_h,
                                         int target_w)
{
    // Calculate padding needed
    const int pad_h = std::max(target_h - img.rows, 0);
    const int pad_w = std::max(target_w - img.cols, 0);

    cv::Mat padded;
    if (pad_h > 0 || pad_w > 0) {
        cv::copyMakeBorder(img,
                           padded,
                           /*top*/ 0,
                           /*bottom*/ pad_h,
                           /*left*/ 0,
                           /*right*/ pad_w,
                           cv::BORDER_CONSTANT,
                           cv::Scalar(0, 0, 0));
    } else {
        padded = img;  // no padding needed
    }

    // Crop (top-left anchored) if larger than target
    const cv::Rect roi(0,
                       0,
                       std::min(target_w, padded.cols),
                       std::min(target_h, padded.rows));

    return padded(roi).clone();
}

cv::Mat resize_with_letterbox(const cv::Mat &src, int target_w, int target_h);

void preprocess_frames(const std::vector<cv::Mat>& org_frames,
                         std::vector<cv::Mat>& preprocessed_frames,
                         uint32_t target_width, uint32_t target_height)
{
    preprocessed_frames.clear();
    preprocessed_frames.reserve(org_frames.size());

    for (const auto &src_bgr : org_frames) {
        // Skip invalid frames but keep vector alignment (optional: push empty)
        if (src_bgr.empty()) {
            preprocessed_frames.emplace_back();
            continue;
        }
        cv::Mat rgb;
        // 1) Convert to RGB
        switch (src_bgr.channels()) {
            case 3:  cv::cvtColor(src_bgr, rgb, cv::COLOR_BGR2RGB);  break;
            case 4:  cv::cvtColor(src_bgr, rgb, cv::COLOR_BGRA2RGB); break;
            case 1:  cv::cvtColor(src_bgr, rgb, cv::COLOR_GRAY2RGB); break;
            default: {
                // Fallback: force 3 channels
                std::vector<cv::Mat> ch(3, src_bgr);
                cv::merge(ch, rgb);
                cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
            } break;
        }

        // 2) Letterbox-scale to target: preserve full frame, pad with black to fill
        cv::Mat fitted = resize_with_letterbox(rgb, static_cast<int>(target_width), static_cast<int>(target_height));
        
        // 3) Ensure contiguous buffer
        if (!fitted.isContinuous()) {
            fitted = fitted.clone();
        }
        // 4) Push to output vector
        preprocessed_frames.push_back(std::move(fitted));
    }
}

cv::Mat resize_with_letterbox(const cv::Mat &src, int target_w, int target_h)
{
    if (src.empty()) {
        return src;
    }

    int src_w = src.cols;
    int src_h = src.rows;

    double scale = std::min(
        static_cast<double>(target_w) / src_w,
        static_cast<double>(target_h) / src_h
    );

    int new_w = static_cast<int>(src_w * scale);
    int new_h = static_cast<int>(src_h * scale);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    cv::Mat canvas(target_h, target_w, src.type(), cv::Scalar(0, 0, 0));

    int x = (target_w - new_w) / 2;
    int y = (target_h - new_h) / 2;

    resized.copyTo(canvas(cv::Rect(x, y, new_w, new_h)));
    return canvas;
}


hailo_status run_post_process(
    InputType &input_type,
    double &org_height,
    double &org_width,
    size_t &frame_count,
    cv::VideoCapture &capture,
    double &framerate,
    size_t &batch_size,
    const bool &save_stream_output,
    const bool &no_display,
    const std::string &output_dir,
    const std::string &output_resolution,
    std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue,
    ModelInputQueuesMap preprocess_queues,
    PostprocessCallback postprocess_callback)
{
    size_t i = 0;
    cv::VideoWriter video;
    InferenceResult output_item;

    const bool is_stream      = (input_type.is_video || input_type.is_camera);
    const bool is_image_like  = (input_type.is_image || input_type.is_directory);
    const char *kWindowName   = "Processed Video";

    // Ensure output directory exists if needed
    if (save_stream_output || is_image_like) {
        if (!output_dir.empty()) {
            try {
                fs::create_directories(output_dir);
            } catch (const std::exception &e) {
                std::cerr << "-E- Failed to create output dir " << output_dir
                          << ": " << e.what() << "\n";
                return HAILO_INVALID_ARGUMENT;
            }
        }
    }

    bool have_output_res = false;
    int out_w = 0;
    int out_h = 0;

    if (!output_resolution.empty()) {
        auto pos = output_resolution.find('x');
        out_w = std::stoi(output_resolution.substr(0, pos));
        out_h = std::stoi(output_resolution.substr(pos + 1));

        have_output_res = true;
        std::cout << "-I- Using output resolution: "  << out_w << "x" << out_h << "\n";
    }

    std::string video_path;
    if (save_stream_output && is_stream) {
        video_path = output_dir.empty()
            ? "processed_video.mp4"
            : (output_dir + "/processed_video.mp4");
        std::cout << "-I- Saving processed video to: " << video_path << "\n";
    }

    auto handle_stream_frame = [&](cv::Mat &frame) -> bool {

        if (have_output_res && !frame.empty()) {
            frame = resize_with_letterbox(frame, out_w, out_h);
        }

        if (!no_display) {
            cv::imshow(kWindowName, frame);

            const int key = cv::waitKey(1);
            if (key == 'q' || key == 27) {
                std::cout << "\nUser requested stop.\n";
                for (auto &[name, q] : preprocess_queues) q->stop();
                return false;
            }
        }

        if (save_stream_output) {
            if (!video.isOpened() && !frame.empty()) {
                int writer_w = have_output_res ? out_w : frame.cols;
                int writer_h = have_output_res ? out_h : frame.rows;
                try {
                    init_video_writer(video_path, video, framerate, writer_w, writer_h);
                } catch (const std::exception &e) {
                    std::cerr << "-W- VideoWriter init failed: " << e.what() << "\n";
                }
            }
            if (video.isOpened()) {
                cv::Mat write_frame;
                if (frame.channels() == 4)
                    cv::cvtColor(frame, write_frame, cv::COLOR_BGRA2BGR);
                else
                    write_frame = frame;
                video.write(write_frame);
            }
        }

        return true;
    };

    // Consume results until queue is closed or user quits.
    while (results_queue->pop(output_item)) {

        if (input_type.is_camera)
            frame_count++;

        show_progress(input_type, i, frame_count);
        cv::Mat &frame = output_item.org_frame;
        if (postprocess_callback && !output_item.output_data_and_infos.empty()) {
            postprocess_callback(frame, output_item.output_data_and_infos);
        }

        if (is_stream) {
            if (!handle_stream_frame(frame)) break;
        } else if (is_image_like) {
            std::string img_path =
                output_dir.empty()
                    ? ("processed_image_" + std::to_string(i) + ".jpg")
                    : (output_dir + "/processed_image_" + std::to_string(i) + ".jpg");

            cv::Mat frame_to_save = frame;
            if (have_output_res && !frame.empty()) {
                frame_to_save = resize_with_letterbox(frame, out_w, out_h);
            }

            cv::imwrite(img_path, frame_to_save);
            if (i + 1 == frame_count) {
                break;
            }
        }
        i++;
    }

    if (input_type.is_video)
        frame_count = i;

    release_resources(capture, video, input_type, no_display, nullptr, results_queue);
    return HAILO_SUCCESS;
}


hailo_status run_inference_async(HailoInfer& model,
                            std::chrono::duration<double>& inference_time,
                            ModelInputQueuesMap &named_input_queues,
                            std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue) {
    
    const auto start_time = std::chrono::high_resolution_clock::now();
    const size_t outputs_per_binding = model.get_infer_model()->get_output_names().size();
    if (named_input_queues.empty()) return HAILO_INVALID_ARGUMENT;

    bool jobs_submitted = false;

    while (true) {
        //build InputMap and capture originals in one pass
        InputMap inputs_map;
        std::vector<cv::Mat> org_frames;
        bool have_org = false;

        for (const auto &[input_name, queue] : named_input_queues) {
            std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>> pack;
            if (!queue->pop(pack)) goto done;

            if (!have_org) {
                org_frames = std::move(pack.first);
                have_org = true;
            }
            inputs_map.emplace(input_name, std::move(pack.second));
        }

        model.infer(
            inputs_map,
            [org_frames = std::move(org_frames), results_queue, outputs_per_binding]
            (const hailort::AsyncInferCompletionInfo &,
             const std::vector<std::pair<uint8_t*, hailo_vstream_info_t>> &flat_outputs,
             const std::vector<std::shared_ptr<uint8_t>> &flat_guards)
            {
                const size_t batch_size = org_frames.size();
                for (size_t i = 0; i < batch_size; ++i) {
                    InferenceResult out;
                    out.org_frame = org_frames[i];

                    const size_t start = i * outputs_per_binding;
                    const size_t end   = start + outputs_per_binding;
                    out.output_data_and_infos.insert(out.output_data_and_infos.end(),
                                                     flat_outputs.begin() + start,
                                                     flat_outputs.begin() + end);
                    out.output_guards.insert(out.output_guards.end(),
                                             flat_guards.begin() + start,
                                             flat_guards.begin() + end);

                    results_queue->push(std::move(out));
                }
            }
        );

        jobs_submitted = true;
    }

done:
    if (jobs_submitted) model.wait_for_last_job();
    results_queue->stop();
    inference_time = std::chrono::high_resolution_clock::now() - start_time;
    return HAILO_SUCCESS;
}

hailo_status run_preprocess(const std::string& input_path,
        const std::string& hef_path,
        HailoInfer &model, 
        InputType &input_type,
        cv::VideoCapture &capture,
        size_t &batch_size,
        double &framerate,
        std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue,
        PreprocessCallback preprocess_callback) 
{

    auto model_input_shape = model.get_infer_model()->hef().get_input_vstream_infos().release()[0].shape;
    print_net_banner(get_hef_name(hef_path), std::ref(model.get_inputs()), std::ref(model.get_outputs()));

    if (input_type.is_image) {
        preprocess_image_frames(input_path, model_input_shape.width, model_input_shape.height, batch_size, preprocessed_batch_queue, preprocess_callback);
    }
    else if (input_type.is_directory) {
        preprocess_directory_of_images(input_path, model_input_shape.width, model_input_shape.height, batch_size, preprocessed_batch_queue, preprocess_callback);
    }
    else{
        preprocess_video_frames(capture, model_input_shape.width, model_input_shape.height, batch_size, framerate, preprocessed_batch_queue, preprocess_callback);
    } 
    return HAILO_SUCCESS;
}

void release_resources(cv::VideoCapture &capture, cv::VideoWriter &video, InputType &input_type,
                      bool no_display,
                      std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue,
                      std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue) {

    if (input_type.is_video || input_type.is_camera) {
        video.release();

        if (!no_display) {
            cv::destroyAllWindows();
        }
    }

    if (preprocessed_batch_queue) {
        preprocessed_batch_queue->stop();
    }

    if (results_queue) {
        results_queue->stop();
    }
}
}