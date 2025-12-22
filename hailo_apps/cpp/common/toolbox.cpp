#include "toolbox.hpp"
#include "hailo_infer.hpp"
#include <chrono>
#include <thread>


namespace hailo_utils {

namespace fs = std::filesystem;
const std::unordered_map<std::string, std::pair<int,int>> RESOLUTION_MAP = {
    {"sd",  {640, 480}},
    {"hd",  {1280, 720}},
    {"fhd", {1920, 1080}}
};

static fs::path executable_dir()
{
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    buf[len] = '\0';
    return fs::path(buf).parent_path();
}

static fs::path find_scripts(const fs::path &start)
{
    fs::path dir = start;
    for (int i = 0; i < 10; i++) {
        fs::path config_dir = dir / "config";
        if (fs::exists(config_dir / "get_hef.sh") && fs::exists(config_dir / "get_input.sh")) {
            return config_dir;
        }

        if (!dir.has_parent_path()) break;
        dir = dir.parent_path();
    }
    throw std::runtime_error("Could not find get_hef.sh / get_input.sh folder.");
}

const fs::path GET_HEF_BASH_SCRIPT_PATH = find_scripts(executable_dir()) / "get_hef.sh";
const fs::path GET_INPUT_BASH_SCRIPT_PATH = find_scripts(executable_dir()) / "get_input.sh";


hailo_status check_status(const hailo_status &status, const std::string &message) {
    if (HAILO_SUCCESS != status) {
        std::cerr << message << " with status " << status << std::endl;
        return status;
    }
    return HAILO_SUCCESS;
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
        batch_size,
        framerate,
    };
}

void post_parse_args(const std::string &app, CommandLineArgs &args, int argc, char **argv)
{
    if (has_flag(argc, argv, "--list-nets")) {
        list_networks(app);
        std::exit(0);
    }

    if (has_flag(argc, argv, "--list-inputs")) {
        list_inputs(app);
        std::exit(0);
    }

    args.net = resolve_net_arg(app, args.net);
    args.input = resolve_input_arg(app, args.input);
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
    if (is_directory_of_images(input_path, frame_count, batch_size)) {
        input_type.is_directory = true;

    } else if (is_image(input_path)) {
        input_type.is_image = true;
        frame_count = 1; // Single image

    } else if (is_video(input_path)) {
        std::cout << "-I- Detected video file input: " << input_path << "\n";
        input_type.is_video = true;
        capture = open_video_capture(input_path, std::ref(capture), std::ref(org_height), std::ref(org_width), std::ref(frame_count), false);
    } else if (input_path == "camera") { // default to /dev/video0
        input_type.is_camera = true;
        capture = open_video_capture("/dev/video0", std::ref(capture), std::ref(org_height), std::ref(org_width), std::ref(frame_count), true, std::ref(camera_resolution));
    } else if (input_path.rfind("/dev/video", 0) == 0) { // user gave explicit device like /dev/video1
        input_type.is_camera = true;
        capture = open_video_capture(input_path, std::ref(capture), std::ref(org_height), std::ref(org_width), std::ref(frame_count), true, std::ref(camera_resolution));
    } else {
        throw std::runtime_error("Unsupported input type: " + input_path);
    }

    return input_type;
}

struct ProcessResult {
    int exit_code = -1;
    std::string stdout_str;
    std::string stderr_str;
};

// run a bash script and capture output (stdout + stderr)
static ProcessResult run_bash_helper(const fs::path &script_path,
                                     const std::vector<std::string> &args)
{
    ProcessResult result;

    if (!fs::exists(script_path)) {
        std::ostringstream oss;
        oss << "File not found: " << script_path;
        result.exit_code = 1;
        result.stderr_str = oss.str();
        std::cerr << oss.str() << std::endl;
        return result;
    }

    std::ostringstream cmd;
    cmd << script_path.string();
    for (const auto &a : args) {
        cmd << " " << a;   // simple join; args are simple flags/words
    }
    cmd << " 2>&1";       // redirect stderr to stdout

    FILE *pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        result.exit_code = 1;
        result.stderr_str = "Failed to run helper script";
        std::cerr << result.stderr_str << std::endl;
        return result;
    }

    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result.stdout_str += buffer;
    }
    int status = pclose(pipe);
    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    } else {
        result.exit_code = 1;
    }

    // For simplicity, treat everything as stderr on error
    if (result.exit_code != 0) {
        result.stderr_str = result.stdout_str;
    }

    return result;
}

static ProcessResult run_get_hef_command(const std::vector<std::string> &args)
{
    auto result = run_bash_helper(GET_HEF_BASH_SCRIPT_PATH, args);
    if (result.exit_code != 0) {
        const std::string &stderr_str = result.stderr_str;

        if (stderr_str.find("No device detected") != std::string::npos) {
            std::cerr
                << "\nNo Hailo device was detected.\n"
                << "This application uses the connected device to choose the correct HEF "
                << "(e.g., hailo8 vs hailo10h).\n"
                << "Please plug in a Hailo device and run the app again.\n"
                << "If you want to download a model without hardware, run get_hef.sh directly "
                << "and pass --hw-arch explicitly (e.g., hailo8).\n";
        } else if (!stderr_str.empty()) {
            std::cerr << stderr_str << std::endl;
        }
        std::exit(result.exit_code);
    }
    return result;
}

static ProcessResult run_get_input_command(const std::vector<std::string> &args)
{
    auto result = run_bash_helper(GET_INPUT_BASH_SCRIPT_PATH, args);
    if (result.exit_code != 0) {
        if (!result.stdout_str.empty())
            std::cerr << result.stdout_str << std::endl;
        if (!result.stderr_str.empty())
            std::cerr << result.stderr_str << std::endl;
        std::exit(result.exit_code);
    }
    return result;
}

// --- list_networks / list_inputs ---

void list_networks(const std::string &app)
{
    std::vector<std::string> cmd_args = {"list", "--app", app};
    std::cout << "Fetching networks list... please wait\n";
    auto result = run_get_hef_command(cmd_args);

    std::string out = result.stdout_str;
    if (!out.empty()) {
        std::string footer =
            "\n\033[33mPick any network name from the list above and pass it with --net "
            "(without extension).\n"
            "Example:  --net <name>\033[0m\n";
        std::cout << "\n" << out << footer << std::endl;
    }
}

void list_inputs(const std::string &app)
{
    std::cout << "Listing predefined inputs for app '" << app << "'...\n";
    auto result = run_get_input_command({"list", "--app", app});

    std::string out = result.stdout_str;
    if (!out.empty()) {
        std::string footer =
            "\n\n\033[33mPick any name from the list above and pass it with -i/--input "
            "(without extension).\n"
            "Example:  -i <name>\033[0m\n";
        std::cout << "\n" << out << footer << std::endl;
    }
}

// --- HEF helpers (verify arch + get hef) ---
static void verify_hef_arch(const std::string &app, const fs::path &hef_path)
{
    (void)app; // kept for symmetry with Python; not needed by the script itself
    std::vector<std::string> args = {
        "verify-arch",
        "--hef", hef_path.string()
    };

    auto result = run_get_hef_command(args);
    if (result.exit_code != 0) {
        if (!result.stderr_str.empty())
            std::cerr << result.stderr_str << std::endl;
        std::exit(result.exit_code);
    }
}

static std::string get_hef(const std::string &app,
                           const std::string &net,
                           const std::string &dest_dir = "hefs")
{
    fs::path dest(dest_dir);
    fs::create_directories(dest);

    std::vector<std::string> args = {
        "get",
        "--app", app,
        "--net", net,
        "--dest", dest.string()
    };

    auto result = run_get_hef_command(args);
    std::string stdout_str = result.stdout_str;
    if (stdout_str.empty()) {
        std::cerr << "get_hef.sh returned empty stdout; cannot determine downloaded path.\n";
        std::exit(1);
    }

    // Last line is the path
    std::istringstream iss(stdout_str);
    std::string line, last_line;
    while (std::getline(iss, line)) {
        if (!line.empty()) last_line = line;
    }

    fs::path hef_path = fs::path(last_line).lexically_normal();
    hef_path = fs::absolute(hef_path);

    std::cout << "\033[32mDownload complete: " << hef_path << "\033[0m\n";
    return hef_path.string();
}

// --- resolve_net_arg ---
std::string resolve_net_arg(const std::string &app,
                            const std::string &net_arg_in,
                            const std::string &dest_dir)
{
    if (net_arg_in.empty()) {
        std::cerr << "No --net was provided.\n";
        list_networks(app);
        std::exit(1);
    }

    fs::path dest(dest_dir);
    fs::create_directories(dest);

    fs::path candidate(net_arg_in);

    // 1) Existing path case
    if (fs::exists(candidate)) {
        if (fs::is_regular_file(candidate) && candidate.extension() == ".hef") {
            fs::path hef_path = fs::absolute(candidate);
            std::cout << "Using local HEF file: " << hef_path << std::endl;
            // verify_hef_arch(app, hef_path);
            return hef_path.string();
        } else {
            std::cerr
                << "Path '" << net_arg_in << "' exists but is not a .hef file.\n"
                << "Please provide either:\n"
                << "  • A valid .hef file\n"
                << "  • OR a network name (without extension)\n"
                << "\033[33mTo see all available network names, run:  --list-nets\033[0m\n";
            std::exit(1);
        }
    }

    // 2) Non-existing .hef
    if (candidate.extension() == ".hef") {
        std::cerr << "HEF file not found: " << net_arg_in << "\n"
                  << "\033[33mTo see all available network names, run:  --list-nets\033[0m\n";
        std::exit(1);
    }

    // 3) Treat as model name
    std::string net_name = net_arg_in;
    fs::path existing_hef = dest / (net_name + ".hef");

    std::cout << "You passed a model name: '" << net_name
              << "'. Searching for this model in the supported networks... please wait.\n";

    if (fs::exists(existing_hef)) {
        std::string answer;
        std::cout
            << "A HEF file already exists for network '" << net_name << "': " << existing_hef << "\n"
            << "Do you want to reuse this file instead of downloading it again? [Y/n]: ";
        if (!std::getline(std::cin, answer)) {
            answer = "y"; // non-interactive -> reuse
        }

        auto to_lower = [](std::string s) {
            std::transform(s.begin(), s.end(), s.begin(),
                           [](unsigned char c){ return std::tolower(c); });
            return s;
        };

        if (answer.empty() || to_lower(answer) == "y" || to_lower(answer) == "yes") {
            fs::path hef_path = fs::absolute(existing_hef);
            std::cout << "Reusing existing HEF: " << hef_path << std::endl;
            verify_hef_arch(app, hef_path);
            return hef_path.string();
        }

        std::string answer2;
        std::cout << "Do you want to re-download and replace '" << existing_hef << "'? [Y/n]: ";
        if (!std::getline(std::cin, answer2)) {
            answer2 = "n";
        }
        std::string a2 = to_lower(answer2);

        if (a2.empty() || a2 == "y" || a2 == "yes") {
            std::cout << "Re-downloading network '" << net_name
                      << "' and replacing existing HEF...\n";
            std::string hef_path_str = get_hef(app, net_name, dest_dir);
            fs::path hef_path = fs::absolute(hef_path_str);
            verify_hef_arch(app, hef_path);
            return hef_path.string();
        } else {
            std::cerr
                << "Aborting: existing HEF was neither reused nor replaced. "
                << "Please provide a different --net or remove the file manually.\n";
            std::exit(1);
        }
    }

    // 4) No existing HEF -> download
    std::cout << "Downloading model name " << net_name << ", please wait...\n";
    std::string hef_path_str = get_hef(app, net_name, dest_dir);
    fs::path hef_path = fs::absolute(hef_path_str);
    verify_hef_arch(app, hef_path);
    return hef_path.string();
}

// --- resolve_input_arg ---

static std::string download_input(const std::string &app,
                                  const std::string &input_id,
                                  const std::string &target_dir = "inputs")
{
    fs::path target(target_dir);
    fs::create_directories(target);

    std::cout << "Downloading input '" << input_id
              << "' for app '" << app << "' from resources...\n";

    auto result = run_get_input_command({
        "get",
        "--app", app,
        "--target-dir", target.string(),
        "--i", input_id
    });

    std::string stdout_str = result.stdout_str;
    if (stdout_str.empty()) {
        std::cerr << "get_input.sh returned empty stdout; cannot determine downloaded path.\n";
        std::exit(1);
    }

    std::istringstream iss(stdout_str);
    std::string line, last_line;
    while (std::getline(iss, line)) {
        if (!line.empty()) last_line = line;
    }

    fs::path downloaded_path = fs::absolute(last_line);
    std::cout << "\033[32mDownload complete: " << downloaded_path << "\033[0m\n";
    return downloaded_path.string();
}

std::string resolve_input_arg(const std::string &app,
                              const std::string &input_arg_in)
{
    // No input -> offer default resource
    if (input_arg_in.empty()) {
        std::string answer;
        std::cout
            << "No --input was provided for app '" << app << "'. "
            << "Do you want to download and use the default input from resources? [Y/n]: ";
        if (!std::getline(std::cin, answer)) {
            std::cerr << "No input provided and cannot prompt interactively. "
                      << "Please specify -i/--input explicitly.\n";
            std::exit(1);
        }

        auto to_lower = [](std::string s) {
            std::transform(s.begin(), s.end(), s.begin(),
                           [](unsigned char c){ return std::tolower(c); });
            return s;
        };
        std::string a = to_lower(answer);

        if (a.empty() || a == "y" || a == "yes") {
            return download_input(app, "default", "inputs");
        }

        std::cerr
            << "No input provided. Please run again with -i/--input or accept the default resource.\n";
        std::exit(1);
    }

    // Camera is passed as-is
    if (input_arg_in == "camera") {
        return input_arg_in;
    }

    fs::path candidate(input_arg_in);

    if (fs::exists(candidate)) {
        return fs::absolute(candidate).string();
    }

    // Has extension but doesn't exist -> error + list
    if (!candidate.extension().empty()) {
        std::cerr << "Input file not found: " << input_arg_in << "\n";
        std::cout << "Available predefined inputs for this app:\n";
        list_inputs(app);
        std::exit(1);
    }

    // No extension and doesn't exist -> treat as logical ID in resources
    std::cout
        << "Input '" << input_arg_in
        << "' does not exist as a local file or directory. "
        << "Assuming this is a resource ID and downloading from inputs.json...\n";

    return download_input(app, input_arg_in, "inputs");
}

// Safely query a metadata value from networks.json via get_hef.sh get_key_value.
// - If the script succeeds → return the raw first token exactly as printed.
// - If anything goes wrong → return "N/A".
std::string get_network_meta_value(const std::string &app,
                                   const std::string &model_name,
                                   const std::string &key,
                                   const std::string &sub_key)
{
    std::vector<std::string> args = {
        "get_key_value",
        "--app",  app,
        "--name", model_name,
        "--key",  key
    };
    if (!sub_key.empty()) {
        args.push_back("--sub_key");
        args.push_back(sub_key);
    }

    ProcessResult result = run_bash_helper(GET_HEF_BASH_SCRIPT_PATH, args);

    // Any error or empty output → "N/A"
    if (result.exit_code != 0 || result.stdout_str.empty()) {
        return "N/A";
    }

    // Return the FIRST token exactly as-is
    std::istringstream iss(result.stdout_str);
    std::string token;
    if (!(iss >> token)) {
        return "N/A";
    }
    return token;
}

void show_progress_helper(size_t current, size_t total)
{
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
    video.open(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), framerate, cv::Size(org_width, org_height));
    if (!video.isOpened()) {
        throw std::runtime_error("Error when writing video");
    }
}

cv::VideoCapture open_video_capture(const std::string &input_path,
    cv::VideoCapture &capture,
    double &org_height,
    double &org_width,
    size_t &frame_count,
    bool is_camera,
    const std::string &camera_resolution) 
    {

    capture.open(input_path, cv::CAP_ANY); 
    if (!capture.isOpened()) {
        throw std::runtime_error("Unable to read input file");
    }

    // If this is a camera and user requested a resolution, apply it
    if (is_camera && !camera_resolution.empty()) {
        auto it = RESOLUTION_MAP.find(camera_resolution);
        if (it == RESOLUTION_MAP.end()) {
            std::cerr << "-W- Unknown camera resolution \"" << camera_resolution
                      << "\". Supported values are: sd, hd, fhd.\n";
        } else {
            int width  = it->second.first;
            int height = it->second.second;
            capture.set(cv::CAP_PROP_FRAME_WIDTH,  width);
            capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        }
    }
    // Query back the actual resolution 
    org_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    org_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    
    if (is_camera) {
        frame_count = 0;   // cameras have no known frame count
    } else {
        frame_count = static_cast<size_t>(capture.get(cv::CAP_PROP_FRAME_COUNT));
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

        // 2) Geometry-preserving fit: pad (bottom/right) then crop (top-left) to target
        cv::Mat fitted = pad_crop_to_target(rgb, static_cast<int>(target_height), static_cast<int>(target_width));
        
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
    const std::string &output_dir,
    const std::string &output_resolution,
    std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue,
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

    // Only init writer if we’re actually saving a stream
    if (save_stream_output && is_stream) {
        std::string video_path =
            output_dir.empty()
                ? "processed_video.mp4"
                : (output_dir + "/processed_video.mp4");

        int base_w = static_cast<int>(org_width);
        int base_h = static_cast<int>(org_height);

        int writer_w = have_output_res ? out_w : base_w;
        int writer_h = have_output_res ? out_h : base_h;

        init_video_writer(video_path, video, framerate, writer_w, writer_h);
        std::cout << "-I- Saving processed video to: " << video_path << "\n";
    }

    auto handle_stream_frame = [&](cv::Mat &frame) -> bool {

        if (have_output_res && !frame.empty()) {
            frame = resize_with_letterbox(frame, out_w, out_h);
        }
        
        cv::imshow(kWindowName, frame);
        if (save_stream_output) {
            video.write(frame);
        }

        const int key = cv::waitKey(1);
        if (key == 'q' || key == 27) { // 'q' or ESC
            std::cout << "\n\033[31mUser requested stop.\033[0m\n";
            return false;
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
                break; // stop after writing the last image
            }
        }
        i++;
    }

    release_resources(capture, video, input_type, nullptr, results_queue);
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
                      std::shared_ptr<BoundedTSQueue<std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>>>> preprocessed_batch_queue,
                      std::shared_ptr<BoundedTSQueue<InferenceResult>> results_queue) {
    if (input_type.is_video) {
        video.release();
    }
    if (input_type.is_camera) {
        capture.release();
        cv::destroyAllWindows();
    }
    if (preprocessed_batch_queue) {
        preprocessed_batch_queue->stop();
    }
    if (results_queue) {
        results_queue->stop();
    }
}
}
