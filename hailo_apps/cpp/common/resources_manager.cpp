#include "resources_manager.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <yaml-cpp/yaml.h>

// libcurl
#include <curl/curl.h>

#ifdef _WIN32
  #include <Windows.h>
#else
  #include <unistd.h>
#endif

namespace hailo_apps {
namespace fs = std::filesystem;

namespace {

// ------------------------------
// utilities
// ------------------------------

static std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

static std::string getenv_str(const char *k)
{
    const char *v = std::getenv(k);
    return v ? std::string(v) : std::string();
}

static std::string stem_no_ext(const std::string &name)
{
    fs::path p(name);
    std::string s = p.stem().string();
    return s.empty() ? name : s;
}

// ------------------------------
// Executable dir (for default YAML location)
// ------------------------------

static fs::path get_executable_dir()
{
#ifdef _WIN32
    char buf[MAX_PATH];
    DWORD len = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (len == 0 || len == MAX_PATH) {
        throw std::runtime_error("GetModuleFileNameA failed");
    }
    return fs::path(std::string(buf, len)).parent_path();
#else
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) {
        throw std::runtime_error("readlink(/proc/self/exe) failed");
    }
    buf[len] = '\0';
    return fs::path(buf).parent_path();
#endif
}

static fs::path default_resources_yaml_path()
{
    // 1) explicit override
    if (const char *env = std::getenv("RESOURCES_YAML")) {
        return fs::path(env);
    }
    // 2) default next to exe
    return get_executable_dir() / "config" / "resources_config.yaml";
}

static YAML::Node load_yaml_resource_file(const fs::path &p)
{
    if (!fs::exists(p)) {
        throw std::runtime_error("resources YAML not found: " + p.string());
    }
    return YAML::LoadFile(p.string());
}

static bool tag_contains(const YAML::Node &tags, const std::string &app_key)
{
    if (!tags || !tags.IsSequence()) return false;
    for (const auto &t : tags) {
        if (t.IsScalar() && t.as<std::string>() == app_key) return true;
    }
    return false;
}

static std::string get_string_any(const YAML::Node &m, std::initializer_list<const char*> keys)
{
    for (auto k : keys) {
        if (m[k] && m[k].IsScalar()) return m[k].as<std::string>();
    }
    return {};
}

// ------------------------------
// HTTP download (libcurl)
// ------------------------------

struct HeadInfo {
    long status = 0;
    curl_off_t size = -1;
};

static HeadInfo head_request(const std::string &url)
{
    HeadInfo info;

    CURL *curl = curl_easy_init();
    if (!curl) return info;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);        // HEAD request
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "hailo-apps-resources/1.0");

    curl_easy_perform(curl);

    long http_code = 0;
    if (curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code) == CURLE_OK) {
        info.status = http_code;
    }

    curl_off_t cl = -1;
    if (curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &cl) == CURLE_OK) {
        if (cl >= 0) info.size = cl;
    }

    curl_easy_cleanup(curl);
    return info;
}

struct ProgressCtx {
    curl_off_t last_printed = -1;
    bool done = false;
};

static int xferinfo_cb(void *p, curl_off_t dltotal, curl_off_t dlnow,
                       curl_off_t /*ultotal*/, curl_off_t /*ulnow*/)
{
    auto *ctx = static_cast<ProgressCtx*>(p);

    // ✅ if we already printed "done", never print again (fixes repeated 100% lines)
    if (ctx->done) return 0;

    // ✅ don't spam same value
    if (dlnow == ctx->last_printed) return 0;
    ctx->last_printed = dlnow;

    const int width = 40;
    const double total_mb = (dltotal > 0) ? (double)dltotal / (1024.0 * 1024.0) : 0.0;
    const double now_mb   = (dlnow  > 0) ? (double)dlnow  / (1024.0 * 1024.0) : 0.0;
    const double frac     = (dltotal > 0) ? (double)dlnow / (double)dltotal : 0.0;

    int filled = (int)(frac * width);
    if (filled > width) filled = width;

    std::cout << "\r[";
    for (int i = 0; i < width; i++) std::cout << (i < filled ? "=" : (i == filled ? "-" : " "));
    if (dltotal > 0) {
        std::cout << "] " << (int)(frac * 100.0) << "% (" << now_mb << "/" << total_mb << " MB)";
    } else {
        std::cout << "] " << now_mb << " MB";
    }
    std::cout << std::flush;

    // ✅ print newline ONCE
    if (dltotal > 0 && dlnow >= dltotal) {
        ctx->done = true;
        std::cout << "\n";
    }

    return 0;
}


static size_t write_file_cb(void *ptr, size_t size, size_t nmemb, void *stream)
{
    return std::fwrite(ptr, size, nmemb, static_cast<FILE*>(stream));
}

/**
 * @brief Download a URL into target_dir. If save_as is empty, uses filename from URL.
 * @return Absolute path to downloaded file.
 */
 static fs::path download_http(const std::string &url,
    const fs::path &target_dir,
    const std::string &save_as = {})
{
fs::create_directories(target_dir);

std::string filename = save_as;
if (filename.empty()) {
const auto pos = url.find_last_of('/');
if (pos == std::string::npos || pos + 1 >= url.size()) {
throw std::runtime_error("URL has no filename: " + url);
}
filename = url.substr(pos + 1);
if (filename.empty()) {
throw std::runtime_error("URL has empty filename: " + url);
}
}

const fs::path dest = target_dir / filename;

// Optional: show HEAD info (status/size) before GET
const auto head = head_request(url);
if (head.status != 0) {
std::cout << "  URL valid: " << url << "\n";
std::cout << "  Status: " << head.status << "\n";
if (head.size > 0) std::cout << "  Size: " << (long long)head.size << " bytes\n";
if (head.status >= 400) {
throw std::runtime_error("Download blocked (HTTP " + std::to_string(head.status) + "): " + url);
}
}

CURL *curl = curl_easy_init();
if (!curl) throw std::runtime_error("curl_easy_init failed");

#ifdef _WIN32
FILE *fp = nullptr;
if (fopen_s(&fp, dest.string().c_str(), "wb") != 0 || !fp) {
curl_easy_cleanup(curl);
throw std::runtime_error("Failed to open: " + dest.string());
}
#else
FILE *fp = std::fopen(dest.string().c_str(), "wb");
if (!fp) {
curl_easy_cleanup(curl);
throw std::runtime_error("Failed to open: " + dest.string());
}
#endif

curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
curl_easy_setopt(curl, CURLOPT_USERAGENT, "hailo-apps-resources/1.0");

// Write callback
curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_file_cb);
curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);

// Make HTTP 4xx/5xx fail the transfer (very important!)
curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

// Progress bar
ProgressCtx progress;
curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, xferinfo_cb);
curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progress);

const CURLcode res = curl_easy_perform(curl);

long http_code = 0;
curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

std::fclose(fp);
curl_easy_cleanup(curl);

// If curl failed (includes many HTTP errors due to FAILONERROR)
if (res != CURLE_OK) {
// remove bad/partial file
std::error_code ec;
fs::remove(dest, ec);
throw std::runtime_error(std::string("Download failed: ") + curl_easy_strerror(res) +
       " (HTTP " + std::to_string(http_code) + ") url=" + url);
}

// Extra safety: if server returned 4xx/5xx but curl still said OK
if (http_code >= 400) {
std::error_code ec;
fs::remove(dest, ec);
throw std::runtime_error("Download failed (HTTP " + std::to_string(http_code) + "): " + url);
}

return fs::absolute(dest);
}

// ------------------------------
// YAML extraction helpers (inputs)
// ------------------------------

static std::vector<ResourcesManager::ResourceEntry>
collect_resources_by_tag(const YAML::Node &root,
                         const std::string &kind,
                         const std::string &app_key)
{
    std::vector<ResourcesManager::ResourceEntry> out;
    const YAML::Node sec = root[kind];
    if (!sec || !sec.IsSequence()) return out;

    for (const auto &item : sec) {
        if (!item || !item.IsMap()) continue;

        const YAML::Node tags = item["tag"] ? item["tag"] : item["tags"];
        if (!tag_contains(tags, app_key)) continue;

        ResourcesManager::ResourceEntry e;
        e.kind = kind;
        e.name = get_string_any(item, {"name"});
        e.description = get_string_any(item, {"description", "desc"});
        e.url = get_string_any(item, {"url", "path"});
        e.source = get_string_any(item, {"source"});

        if (!e.name.empty()) out.push_back(std::move(e));
    }
    return out;
}

/**
 * @brief Build default CSData-style URL for shared resources when YAML doesn't include explicit URL.
 * NOTE: This mirrors your previous behavior. Adjust base if your buckets move.
 */
static std::string build_csdata_resource_url(const ResourcesManager::ResourceEntry &e)
{
    // These folder names reflect your earlier layout in code comments:
    // images -> resources/images
    // videos -> resources/video
    // json   -> resources/json
    // npy    -> resources/npy
    const std::string base = "https://hailo-csdata.s3.amazonaws.com/resources/";

    std::string folder = e.kind;
    if (folder == "videos") folder = "video";
    if (folder == "images") folder = "images";
    if (folder == "json")   folder = "json";
    if (folder == "npy")    folder = "npy";

    return base + folder + "/" + e.name;
}

// ------------------------------
// HailoRT / device arch detection (hailortcli)
// ------------------------------

static std::string run_cmd_capture(const std::string &cmd)
{
    std::string out;

#ifdef _WIN32
    FILE *pipe = _popen(cmd.c_str(), "r");
#else
    FILE *pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe) return out;

    char buf[1024];
    while (std::fgets(buf, sizeof(buf), pipe) != nullptr) {
        out += buf;
    }

#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return out;
}

static std::string detect_device_arch()
{
    // Same behavior as your original code:
    // try windows redirect first, then posix redirect
    std::string out = run_cmd_capture("hailortcli fw-control identify 2>nul");
    if (out.empty()) out = run_cmd_capture("hailortcli fw-control identify 2>/dev/null");

    std::regex re(R"(Device Architecture\s*:\s*([A-Za-z0-9]+))");
    std::smatch m;
    if (!std::regex_search(out, m, re)) return "";

    std::string arch = to_lower(m[1].str());

    // Normalize: original behavior mapped hailo15h -> hailo10h
    if (arch == "hailo15h") arch = "hailo10h";
    return arch;
}

static std::string hailort_version()
{
    std::string out = run_cmd_capture("hailortcli -v 2>nul");
    if (out.empty()) out = run_cmd_capture("hailortcli -v 2>/dev/null");

    // best-effort: first X.Y or X.Y.Z
    std::regex re(R"(([0-9]+(\.[0-9]+){1,2}))");
    std::smatch m;
    if (!std::regex_search(out, m, re)) return "";
    return m[1].str();
}

static std::string validate_arch(const std::string &a)
{
    const std::string lc = to_lower(a);
    if (lc == "hailo8" || lc == "hailo8l" || lc == "hailo10h") return lc;
    throw std::runtime_error("Invalid architecture '" + a + "'. Supported: hailo8, hailo8l, hailo10h");
}

static std::string detect_hef_arch(const std::filesystem::path &hef_path)
{
    auto run = [&](const std::string &cmd) {
#ifdef _WIN32
        return run_cmd_capture(cmd + " 2>nul");
#else
        return run_cmd_capture(cmd + " 2>/dev/null");
#endif
    };

    const std::string p = hef_path.string();
    std::string out = run("hailortcli parse-hef \"" + p + "\"");
    if (out.empty()) return "";

    std::smatch m;

    // Match either:
    // 1) "HEF Compatible for: HAILO15H, HAILO10H"
    // 2) "Architecture HEF was compiled for: HAILO8"
    {
        std::regex re(R"(HEF\s+Compatible\s+for\s*:\s*([^\r\n]+))", std::regex::icase);
        if (!std::regex_search(out, m, re)) {
            re = std::regex(R"(Architecture\s+HEF\s+was\s+compiled\s+for\s*:\s*([^\r\n]+))", std::regex::icase);
            if (!std::regex_search(out, m, re)) return "";
        }
    }

    std::string list = to_lower(m[1].str());

    if (list.find("hailo10h") != std::string::npos) return "hailo10h";
    if (list.find("hailo15h") != std::string::npos) return "hailo10h";
    if (list.find("hailo8l")  != std::string::npos) return "hailo8l";
    if (list.find("hailo8")   != std::string::npos) return "hailo8";

    std::regex tok(R"(([A-Za-z0-9]+))");
    if (std::regex_search(list, m, tok)) {
        std::string arch = to_lower(m[1].str());
        if (arch == "hailo15h") arch = "hailo10h";
        return arch;
    }
    return "";
}

static bool is_hef_compatible_with_device(const std::filesystem::path &hef_path,
                                           const std::string &device_arch)
{
    const std::string hef_arch = detect_hef_arch(hef_path);


    // ❌ if arch detection failed → ERROR immediately
    if (hef_arch.empty()) {
        throw std::runtime_error(
            "ERROR: Failed to detect HEF architecture for: " +
            std::filesystem::absolute(hef_path).string());
    }

    // ❌ mismatch → ERROR
    if (hef_arch != device_arch) {
        throw std::runtime_error(
            "ERROR: HEF architecture mismatch.\n"
            "  HEF:    " + std::filesystem::absolute(hef_path).string() + "\n"
            "  HEF arch: " + hef_arch + "\n"
            "  Device:   " + device_arch);
    }

    return true;
}

/**
 * @brief Model Zoo compatibility mapping (copied from your original logic).
 * Update this map if you add new HailoRT versions.
 */
static std::string modelzoo_version_for(const std::string &hw_arch, const std::string &hailort_ver)
{
    // hailo10h: 5.x -> v5.x (kept in sync with config.yaml's model_zoo_mapping comment:
    // HailoRT 5.1.x -> v5.1.0, 5.2.x -> v5.2.0, 5.3.x -> v5.3.0, 5.4.x -> v5.4.0)
    static const std::unordered_map<std::string,std::string> compat_10h = {
        {"5.0.1","v5.0.0"},
        {"5.0.0","v5.0.0"},
        {"5.1.0","v5.1.0"},
        {"5.1.1","v5.1.0"},
        {"5.1.2","v5.1.0"},
        {"5.2.0","v5.2.0"},
        {"5.3.0","v5.3.0"},
        {"5.4.0","v5.4.0"},
    };

    // hailo8/8l: 4.x -> v2.xx (kept in sync with config.yaml's model_zoo_mapping comment:
    // HailoRT 4.23.x -> v2.18.0, HailoRT 4.24.x -> v2.19.0)
    static const std::unordered_map<std::string,std::string> compat_8 = {
        {"4.23.0","v2.18.0"},
        {"4.22.0","v2.16.0"},
        {"4.21.0","v2.15.0"},
        {"4.20.0","v2.14.0"},
        {"4.24.0","v2.19.0"},
    };

    if (hw_arch == "hailo10h") {
        auto it = compat_10h.find(hailort_ver);
        if (it == compat_10h.end()) {
            throw std::runtime_error("HailoRT " + hailort_ver + " is not compatible with " + hw_arch + ". Use HailoRT 5.x.x.");
        }
        return it->second;
    }

    if (hw_arch == "hailo8" || hw_arch == "hailo8l") {
        auto it = compat_8.find(hailort_ver);
        if (it == compat_8.end()) {
            throw std::runtime_error("HailoRT " + hailort_ver + " is not compatible with " + hw_arch + ". Use HailoRT 4.x.x.");
        }
        return it->second;
    }

    throw std::runtime_error("Unknown hw-arch " + hw_arch);
}

// ------------------------------
// YAML extraction helpers (models)
// ------------------------------

static std::vector<ResourcesManager::ModelEntry>
collect_models(const YAML::Node &root,
               const std::string &app_key,
               const std::string &hw_arch,
               const std::string &bucket) // "default" or "extra"
{
    std::vector<ResourcesManager::ModelEntry> out;

    if (!root[app_key] || !root[app_key]["models"] || !root[app_key]["models"][hw_arch]) {
        return out;
    }

    const YAML::Node sec = root[app_key]["models"][hw_arch][bucket];
    if (!sec || !sec.IsSequence()) return out;

    for (const auto &item : sec) {
        if (!item || !item.IsMap()) continue;

        ResourcesManager::ModelEntry m;
        m.name   = get_string_any(item, {"name"});
        m.source = get_string_any(item, {"source"});
        m.url    = get_string_any(item, {"url"});

        if (!m.name.empty()) out.push_back(std::move(m));
    }

    return out;
}

static ResourcesManager::ModelEntry
find_model_entry(const YAML::Node &root,
                 const std::string &app_key,
                 const std::string &hw_arch,
                 const std::string &model_name)
{
    const auto defaults = collect_models(root, app_key, hw_arch, "default");
    const auto extras   = collect_models(root, app_key, hw_arch, "extra");

    auto match = [&](const ResourcesManager::ModelEntry &m){
        return (m.name == model_name) ||
               (stem_no_ext(m.name) == stem_no_ext(model_name));
    };

    for (const auto &m : defaults) if (match(m)) return m;
    for (const auto &m : extras)   if (match(m)) return m;

    throw std::runtime_error("Model '" + model_name + "' not found under app=" + app_key + " arch=" + hw_arch);
}

static bool model_supported_on_arch(const YAML::Node &root,
                                    const std::string &app_key,
                                    const std::string &hw_arch,
                                    const std::string &model_name)
{
    try {
        (void)find_model_entry(root, app_key, hw_arch, model_name);
        return true;
    } catch (...) {
        return false;
    }
}

/**
 * @brief Build HEF URL given source + versions.
 * This preserves your previous behavior:
 *  - explicit url in YAML wins
 *  - otherwise build from source + mz_ver + hw_arch + model name
 */
static std::string build_hef_url(const std::string &source,
                                 const std::string &mz_ver,
                                 const std::string &hw_arch,
                                 const std::string &model_name)
{
    // NOTE: These are placeholders matching common patterns in Hailo repos.
    // If your real endpoints differ, update here only (single source of truth).
    const std::string name = (fs::path(model_name).extension() == ".hef")
                               ? fs::path(model_name).filename().string()
                               : (model_name + ".hef");

    if (source == "mz") {
        // Example style: https://hailo-model-zoo.s3.amazonaws.com/<mz_ver>/<arch>/<name>
        return "https://hailo-model-zoo.s3.amazonaws.com/ModelZoo/Compiled/" + mz_ver + "/" + hw_arch + "/" + name;
    }

    if (source == "s3") {
        // s3 uses short arch names (h8/h8l/h10h) with no version component
        std::string short_arch = hw_arch;
        if      (hw_arch == "hailo8")  short_arch = "h8";
        else if (hw_arch == "hailo8l") short_arch = "h8l";
        else if (hw_arch == "hailo10h") short_arch = "h10h";
        return "https://hailo-csdata.s3.amazonaws.com/resources/hefs/" + short_arch + "/" + name;
    }

    if (source == "gen-ai-mz") {
        return "https://hailo-genai-model-zoo.s3.amazonaws.com/" + mz_ver + "/" + hw_arch + "/" + name;
    }

    throw std::runtime_error("Unknown model source '" + source + "'");
}

// ------------------------------
// Input URL resolution logic
// ------------------------------

static std::string resolve_input_url_from_yaml(const YAML::Node &root,
                                               const std::string &app,
                                               const std::string &input_name)
{
    // Collect YAML resources tagged for this application
    const auto images = collect_resources_by_tag(root, "images", app);
    const auto videos = collect_resources_by_tag(root, "videos", app);

     // Search images first
    for (const auto &e : images) {
        if (e.name == input_name) {
            if (!e.url.empty()) return e.url;
            return build_csdata_resource_url(e);
        }
    }

    // Then search videos
    for (const auto &e : videos) {
        if (e.name == input_name) {
            if (!e.url.empty()) return e.url;
            return build_csdata_resource_url(e);
        }
    }

    throw std::runtime_error(
        "Input '" + input_name + "' not found in YAML for app=" + app);
}

static std::string download_input_yaml(const YAML::Node &root,
                                       const std::string &app,
                                       const std::string &input_name,
                                       const fs::path &target_dir)
{
    fs::create_directories(target_dir);

    // 1) If already exists locally → reuse
    const fs::path local = target_dir / fs::path(input_name).filename();
    if (fs::exists(local) && fs::is_regular_file(local)) {
        std::cout << "Found input in resources: "
                  << fs::absolute(local) << "\n";
        return fs::absolute(local).string();
    }

    // 2) Resolve URL from YAML (throws if not found)
    const std::string url = resolve_input_url_from_yaml(root, app, input_name);

    // 3) Preserve filename if user passed an extension (e.g., bus.mp4, zidane.jpg)
    std::string save_as;
    if (!input_name.empty() && fs::path(input_name).has_extension()) {
        save_as = fs::path(input_name).filename().string();
    }

    // 4) Neutral + accurate log (doesn't promise YAML exists elsewhere)
    std::cout << "\nWWARNING: Input '" << input_name << "' is not downloaded.\n"
              << "   Downloading input for " << app << "...\n"
              << "   This may take a while depending on your internet connection.\n\n";

    // 5) Download
    const fs::path out = download_http(url, target_dir, save_as);
    std::cout << "\033[32mDownload complete: " << out << "\n";
    return out.string();
}


// ------------------------------
// HEF resolution / download logic
// ------------------------------

static std::string download_hef_yaml(const YAML::Node &root,
                                    const std::string &app,
                                    const std::string &net,
                                    const fs::path &dest_dir)
{
    std::string hw_arch = detect_device_arch();
    if (hw_arch.empty()) {
        std::cerr
            << "\nNo Hailo device was detected.\n"
            << "This application uses the connected device to choose the correct HEF "
            << "(e.g., hailo8 vs hailo10h).\n"
            << "Please plug in a Hailo device and run the app again.\n";
        throw std::runtime_error("No Hailo device detected");
    }
    hw_arch = validate_arch(hw_arch);

    if (!model_supported_on_arch(root, app, hw_arch, net)) {
        throw std::runtime_error("Net '" + net + "' does not support hw-arch=" + hw_arch + " (or not found).");
    }

    const auto m = find_model_entry(root, app, hw_arch, net);

    fs::create_directories(dest_dir);

    // Only resolve the installed HailoRT -> Model Zoo version mapping when it's
    // actually needed to build the download URL. Explicit "url" entries and the
    // version-less "s3" source don't depend on it, so they must not fail just
    // because the installed HailoRT version isn't in the (hardcoded) compat map.
    std::string url;
    if (!m.url.empty()) {
        url = m.url; // explicit URL in YAML
    } else if (m.source == "s3") {
        url = build_hef_url(m.source, /*mz_ver=*/{}, hw_arch, m.name);
    } else {
        const std::string hv = hailort_version();
        if (hv.empty()) {
            throw std::runtime_error("Cannot parse HailoRT version. Is hailortcli installed?");
        }
        const std::string mz_ver = modelzoo_version_for(hw_arch, hv);
        url = build_hef_url(m.source, mz_ver, hw_arch, m.name);
    }

    // Save name: <net>.hef (if user gave net without .hef)
    const std::string save_as =
        (fs::path(net).extension() == ".hef")
            ? fs::path(net).filename().string()
            : (net + ".hef");

    const fs::path out = download_http(url, dest_dir, save_as);
    std::cout << "\033[32mDownload complete: " << out << "\033[0m\n";
    return out.string();
}

} // anonymous namespace




// ============================================================================
// ResourcesManager implementation
// ============================================================================

ResourcesManager::ResourcesManager(std::optional<fs::path> yaml_path)
    : m_yaml_path(yaml_path.has_value() ? *yaml_path : default_resources_yaml_path())
{
    // Validate early so failures are obvious.
    (void)load_yaml_resource_file(m_yaml_path);
}

static std::filesystem::path default_resources_root()
{
#ifdef _WIN32
    return std::filesystem::path("C:\\usr\\local\\hailo\\resources");
#else
    return std::filesystem::path("/usr/local/hailo/resources");
#endif
}

std::filesystem::path ResourcesManager::resources_root() const
{
    const std::string env = getenv_str("HAILO_RESOURCES_DIR");
    if (!env.empty()) return std::filesystem::path(env);
    return default_resources_root();
}

std::filesystem::path ResourcesManager::models_dir_for_arch(const std::string &hw_arch) const
{
    return resources_root() / "models" / hw_arch;
}

std::filesystem::path ResourcesManager::inputs_dir_for_kind(const std::string &kind) const
{
    return resources_root() / kind;
}


std::vector<ResourcesManager::ResourceEntry>
ResourcesManager::list_images(const std::string &app) const
{
    const YAML::Node root = load_yaml_resource_file(m_yaml_path);
    return collect_resources_by_tag(root, "images", app);
}

std::vector<ResourcesManager::ResourceEntry>
ResourcesManager::list_videos(const std::string &app) const
{
    const YAML::Node root = load_yaml_resource_file(m_yaml_path);
    return collect_resources_by_tag(root, "videos", app);
}

void ResourcesManager::print_inputs(const std::string &app, std::ostream &os) const
{
    const YAML::Node root = load_yaml_resource_file(m_yaml_path);
    const auto images = collect_resources_by_tag(root, "images", app);
    const auto videos = collect_resources_by_tag(root, "videos", app);

    os << "\n============================================================\n";
    os << "Available inputs for: " << app << "\n";
    os << "============================================================\n\n";

    os << "[Images]\n";
    if (images.empty()) {
        os << "   - none\n";
    } else {
        for (const auto &e : images) {
            os << "   - " << e.name;
            if (!e.description.empty()) {
                os << " - " << e.description;
            }
            os << "\n";
        }
    }

    os << "\n[Videos]\n";
    if (videos.empty()) {
        os << "   - none\n";
    } else {
        for (const auto &e : videos) {
            os << "   - " << e.name;
            if (!e.description.empty()) {
                os << " - " << e.description;
            }
            os << "\n";
        }
    }

    os << "\n============================================================\n";
}

std::string ResourcesManager::resolve_input_arg(const std::string &app,
                                               const std::string &input_arg,
                                               const fs::path &target_dir) const
{
    namespace fs = std::filesystem;
    // ------------------------------------------------
    // (1) Camera inputs (pass-through):
    //  - "usb" / "rpi"
    //  - Windows: "0", "1", ...
    //  - Linux: "/dev/videoX"
    if (input_arg == "usb" || input_arg == "rpi" || input_arg == "csi" ||
        (!input_arg.empty() && std::all_of(input_arg.begin(), input_arg.end(), ::isdigit)) ||
        (input_arg.rfind("/dev/video", 0) == 0))
    {
        return input_arg;
    }

    // ------------------------------------------------
    // (2) If user provided an explicit PATH (absolute/relative),
    // ------------------------------------------------
    if (!input_arg.empty()) {
        fs::path candidate(input_arg);

        // If it looks like a path (has parent directory part), treat as strict path.
        if (candidate.has_parent_path()) {
            if (!fs::exists(candidate)) {
                throw std::runtime_error("Input path does not exist: " + candidate.string());
            }
            if (!fs::is_regular_file(candidate) && !fs::is_directory(candidate)) {
                throw std::runtime_error("Input path is not a file or directory: " + candidate.string());
            }
            return fs::absolute(candidate).string();
        }

        // If it's a filename-only and exists in current working directory -> use it.
        fs::path cwd_candidate = fs::current_path() / candidate;
        if (fs::exists(cwd_candidate) && (fs::is_regular_file(cwd_candidate) || fs::is_directory(cwd_candidate))) {
            return fs::absolute(cwd_candidate).string();
        }
    }

    // Load YAML once
    const YAML::Node root = load_yaml_resource_file(m_yaml_path);

    // ------------------------------------------------
    // (3) Empty input -> auto-select default from YAML:
    //     prefer first image, else first video.
    // ------------------------------------------------
    if (input_arg.empty()) {
        std::cerr << "Using default bundled input resource for "
                  << app << "\n";

        const auto images = collect_resources_by_tag(root, "images", app);
        const auto videos = collect_resources_by_tag(root, "videos", app);

        if (!images.empty()) {
            const fs::path dir = inputs_dir_for_kind("images");
            return download_input_yaml(root, app, images.front().name, dir);
        }

        if (!videos.empty()) {
            const fs::path dir = inputs_dir_for_kind("videos");
            return download_input_yaml(root, app, videos.front().name, dir);
        }

        throw std::runtime_error("No inputs defined in YAML for app=" + app);
    }

    // ------------------------------------------------
    // (4) Non-empty input that was NOT an explicit path and
    //     was NOT found locally -> treat as YAML resource name.
    //     Determine kind (image/video) to pick the right download dir.
    // ------------------------------------------------
    const auto images = collect_resources_by_tag(root, "images", app);
    for (const auto &e : images) {
        if (e.name == input_arg) {
            return download_input_yaml(root, app, input_arg, inputs_dir_for_kind("images"));
        }
    }
    const auto videos = collect_resources_by_tag(root, "videos", app);
    for (const auto &e : videos) {
        if (e.name == input_arg) {
            return download_input_yaml(root, app, input_arg, inputs_dir_for_kind("videos"));
        }
    }
    return download_input_yaml(root, app, input_arg, target_dir);
}

void ResourcesManager::print_models(const std::string &app, std::ostream &os) const
{
    const YAML::Node root = load_yaml_resource_file(m_yaml_path);

    std::string hw_arch = detect_device_arch();
    if (hw_arch.empty()) {
        os << "No Hailo device detected.\n";
        return;
    }
    hw_arch = validate_arch(hw_arch);

    const auto defaults = collect_models(root, app, hw_arch, "default");
    const auto extras   = collect_models(root, app, hw_arch, "extra");

    os << "\n============================================================\n";
    os << "Available models for: " << app << " (" << hw_arch << ") [standalone]\n";
    os << "============================================================\n\n";

    os << "[Default Models]\n";
    if (defaults.empty()) {
        os << "   - none\n";
    } else {
        for (const auto &m : defaults) {
            os << "   - " << stem_no_ext(m.name) << "\n";
        }
    }

    os << "\n[Extra Models]\n";
    if (extras.empty()) {
        os << "   - none\n";
    } else {
        for (const auto &m : extras) {
            os << "   - " << stem_no_ext(m.name) << "\n";
        }
    }

    os << "\n============================================================\n";
    os << "Total: " << defaults.size() << " default, " << extras.size() << " extra\n\n";
    os << "Usage: --hef-path <model_name>\n";
    os << "       Model will be auto-downloaded if not found locally.\n";
}

std::string ResourcesManager::resolve_net_arg(const std::string &app,
                                             const std::string &net_arg,
                                             const fs::path &) const
{
    namespace fs = std::filesystem;

    // ------------------------------------------------
    // Detect connected device arch once
    // ------------------------------------------------
    std::string hw_arch = detect_device_arch();
    if (hw_arch.empty()) {
        throw std::runtime_error("No Hailo device detected");
    }
    hw_arch = validate_arch(hw_arch);

    // ------------------------------------------------
    // Resolve requested model name (default if empty)
    // ------------------------------------------------
    std::string resolved = net_arg;
    if (resolved.empty()) {
        const YAML::Node root = load_yaml_resource_file(m_yaml_path);

        const auto defaults = collect_models(root, app, hw_arch, "default");
        if (defaults.empty()) {
            throw std::runtime_error("No default model defined in resources_config.yaml for app: " + app);
        }

        resolved = stem_no_ext(defaults.front().name);
        std::cout << "Using default model: " << resolved << "\n";
    }

    fs::path arg_path(resolved);
    const bool user_has_hef_ext = (to_lower(arg_path.extension().string()) == ".hef");

    // always normalize to filename we want to locate
    const std::string hef_filename = user_has_hef_ext ? arg_path.filename().string() : (resolved + ".hef");


    // ------------------------------------------------
    // (1) If user provided a path (absolute/relative) that exists, prefer it
    //     but only if it's a .hef and compatible
    // ------------------------------------------------
    if (!resolved.empty() && fs::exists(arg_path)) {
        if (!fs::is_regular_file(arg_path)) {
            throw std::runtime_error("Path exists but is not a file: " + fs::absolute(arg_path).string());
        }
        if (!user_has_hef_ext) {
            throw std::runtime_error("Path exists but is not a .hef file: " + fs::absolute(arg_path).string());
        }

        fs::path abs_p = fs::absolute(arg_path);

        if (is_hef_compatible_with_device(abs_p, hw_arch)) {
            std::cout << "Using HEF from path: " << abs_p << "\n";
            return abs_p.string();
        }
    }


    // ------------------------------------------------
    // (2) If user passed filename only "yolov8m.hef":
    // ------------------------------------------------
    const bool strict_user_path = arg_path.has_parent_path();

    if (user_has_hef_ext && !strict_user_path) {
        fs::path cwd_candidate = fs::current_path() / hef_filename;
        if (fs::exists(cwd_candidate) && fs::is_regular_file(cwd_candidate)) {
            fs::path abs_p = fs::absolute(cwd_candidate);

            if (is_hef_compatible_with_device(abs_p, hw_arch)) {
                std::cout << "Found HEF in current dir: " << abs_p << "\n";
                return abs_p.string();
            }
        }
    }

    // ------------------------------------------------
    // (3) Check resources dir: <resources_root>/models/<arch>/<hef>
    // ------------------------------------------------
    fs::path resources_dir = models_dir_for_arch(hw_arch); // resources_root()/models/hw_arch
    fs::create_directories(resources_dir);

    fs::path resources_candidate = resources_dir / hef_filename;
    if (fs::exists(resources_candidate) && fs::is_regular_file(resources_candidate)) {
        fs::path abs_p = fs::absolute(resources_candidate);

        // resources should match arch anyway
        if (is_hef_compatible_with_device(abs_p, hw_arch)) {
            std::cout << "Found HEF in resources: " << abs_p << "\n";
            return abs_p.string();
        }
    }

    // ------------------------------------------------
    // (4) Not found -> download into resources_dir
    // If user passed ".hef", use stem name for YAML lookup
    // ------------------------------------------------
    const std::string model_name_for_yaml = user_has_hef_ext ? stem_no_ext(hef_filename) : resolved;

    std::cout << "WARNING: Model '" << model_name_for_yaml << "' is not downloaded.\n"
              << "   Downloading model for " << app << "/" << hw_arch << "...\n"
              << "   This may take a while depending on your internet connection.\n\n";

    const YAML::Node root = load_yaml_resource_file(m_yaml_path);
    return download_hef_yaml(root, app, model_name_for_yaml, resources_dir);
}

std::string ResourcesManager::get_model_meta_value(const std::string &app,
                                                     const std::string &model_name,
                                                     const std::string &key) const
{
    try {
        const YAML::Node root = load_yaml_resource_file(m_yaml_path);

        std::string arch = detect_device_arch();
        if (arch.empty()) {
            std::cerr << "Warning: failed to detect device arch\n";
            return "N/A";
        }
        arch = validate_arch(arch);
        const YAML::Node models = root[app]["models"][arch];
        auto find_in_group = [&](const YAML::Node &group) -> std::string {
            if (!group || !group.IsSequence()) return "N/A";

            for (const auto &m : group) {
                if (!m["name"] || !m["name"].IsScalar()) continue;
                if (m["name"].as<std::string>() != model_name) continue;

                const YAML::Node meta = m["meta_data"];
                if (!meta || !meta.IsMap()) return "N/A";

                const YAML::Node val = meta[key];
                if (!val || !val.IsScalar()) return "N/A";

                return val.as<std::string>();
            }
            return "N/A";
        };

        std::string v = find_in_group(models["default"]);
        if (v != "N/A") return v;

        v = find_in_group(models["extra"]);
        if (v != "N/A") return v;

        std::cerr << "Warning: metadata key '" << key << "' not found for model '"
                  << model_name << "' (app='" << app << "', arch='" << arch << "')\n";
        return "N/A";
    }
    catch (const std::exception &e) {
        std::cerr << "Warning: get_model_meta_value failed: " << e.what() << "\n";
        return "N/A";
    }
    catch (...) {
        std::cerr << "Warning: get_model_meta_value failed (unknown error)\n";
        return "N/A";
    }
}

} // namespace hailo_apps