$ErrorActionPreference = "Stop"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$CLEAN = $false
$ARGS_LIST = @()

$DEPS_INSTALL_DIR = Join-Path $SCRIPT_DIR "deps"
$YAML_CPP_SUBMODULE = Join-Path $SCRIPT_DIR "external\yaml-cpp"
$CURL_SUBMODULE = Join-Path $SCRIPT_DIR "external\curl"

$GIT_USR_BIN = "C:\Program Files\Git\usr\bin"
if (Test-Path $GIT_USR_BIN) {
    $env:PATH = "$GIT_USR_BIN;$env:PATH"
}

function Check-LastCommand {
    param([string]$Message)

    if ($LASTEXITCODE -ne 0) {
        throw $Message
    }
}

function Require-Tool {
    param(
        [string]$ToolName,
        [string]$InstallHint
    )

    $tool = Get-Command $ToolName -ErrorAction SilentlyContinue
    if (-not $tool) {
        throw "$ToolName was not found in PATH. $InstallHint"
    }

    Write-Host "-I- Found ${ToolName}: $($tool.Source)"
}

foreach ($arg in $args) {
    if ($arg -eq "--rebuild") {
        $CLEAN = $true
    } else {
        $ARGS_LIST += ($arg.TrimEnd('\', '/'))
    }
}

if ($ARGS_LIST -contains "-h" -or $ARGS_LIST -contains "--help") {
    Write-Host "Usage: .\build.ps1 [--rebuild] [app1 app2 ...]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  --rebuild   Remove build directories before building"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\build.ps1"
    Write-Host "  .\build.ps1 object_detection"
    Write-Host "  .\build.ps1 object_detection pose_estimation"
    Write-Host "  .\build.ps1 --rebuild object_detection"
    Write-Host "  .\build.ps1 --rebuild"
    Write-Host ""
    Write-Host "Windows requirements:"
    Write-Host "  Git for Windows must be installed because some dependencies require sed."
    Write-Host ""
    Write-Host "For ONNX app:"
    Write-Host '  $env:ONNXRUNTIME_DIR="C:\path\to\onnxruntime"'
    Write-Host "  .\build.ps1 onnxrt_hailo_pipeline"
    exit 0
}

Require-Tool "cmake" "Install CMake and make sure it is available in PATH."
Require-Tool "sed" "Install Git for Windows, then reopen PowerShell."

if (!(Test-Path (Join-Path $YAML_CPP_SUBMODULE "CMakeLists.txt"))) {
    throw "yaml-cpp submodule is missing. Run: git submodule update --init --recursive"
}

if (!(Test-Path (Join-Path $CURL_SUBMODULE "CMakeLists.txt"))) {
    throw "curl submodule is missing. Run: git submodule update --init --recursive"
}

if (!(Test-Path (Join-Path $DEPS_INSTALL_DIR "lib\yaml-cpp.lib"))) {
    Write-Host "=========================================="
    Write-Host " Building shared dependency: yaml-cpp"
    Write-Host "=========================================="

    cmake -S $YAML_CPP_SUBMODULE `
          -B "$YAML_CPP_SUBMODULE\build" `
          -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL_DIR" `
          -DYAML_CPP_BUILD_TESTS=OFF `
          -DYAML_CPP_BUILD_TOOLS=OFF `
          -DYAML_CPP_BUILD_CONTRIB=OFF

    Check-LastCommand "Failed to configure yaml-cpp"

    cmake --build "$YAML_CPP_SUBMODULE\build" --config Release
    Check-LastCommand "Failed to build yaml-cpp"

    cmake --install "$YAML_CPP_SUBMODULE\build" --config Release
    Check-LastCommand "Failed to install yaml-cpp"
}

if (!(Test-Path (Join-Path $DEPS_INSTALL_DIR "lib\libcurl.lib"))) {
    Write-Host "=========================================="
    Write-Host " Building shared dependency: curl"
    Write-Host "=========================================="

    cmake -S $CURL_SUBMODULE `
          -B "$CURL_SUBMODULE\build" `
          -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL_DIR" `
          -DBUILD_CURL_EXE=OFF `
          -DBUILD_SHARED_LIBS=OFF `
          -DCURL_DISABLE_TESTS=ON `
          -DCURL_USE_LIBPSL=OFF `
          -DCURL_USE_SCHANNEL=ON `
          -DCURL_USE_OPENSSL=OFF

    Check-LastCommand "Failed to configure curl"

    cmake --build "$CURL_SUBMODULE\build" --config Release
    Check-LastCommand "Failed to build curl"

    cmake --install "$CURL_SUBMODULE\build" --config Release
    Check-LastCommand "Failed to install curl"
}

$CMAKE_PREFIX_PATH_FOR_APPS = $DEPS_INSTALL_DIR

$ALL_APPS = @(
    "classification",
    "depth_estimation_mono",
    "depth_estimation_stereo",
    "instance_segmentation",
    "object_detection",
    "oriented_object_detection",
    "pose_estimation",
    "semantic_segmentation",
    "zero_shot_classification",
    "onnxrt_hailo_pipeline"
)

if ($ARGS_LIST.Count -gt 0) {
    $APPS = $ARGS_LIST
} else {
    $APPS = $ALL_APPS
}

$FAILED = @()

foreach ($APP in $APPS) {
    Write-Host ""
    Write-Host "=========================================="
    Write-Host " Building: $APP"
    Write-Host "=========================================="

    $APP_DIR = Join-Path $SCRIPT_DIR $APP
    $BUILD_DIR = Join-Path $APP_DIR "build"

    if (!(Test-Path $APP_DIR)) {
        Write-Host "-E- ${APP}: directory does not exist"
        $FAILED += $APP
        continue
    }

    if ($CLEAN -and (Test-Path $BUILD_DIR)) {
        Remove-Item $BUILD_DIR -Recurse -Force
        Write-Host "-I- ${APP}: build dir cleaned"
    }

    Push-Location $APP_DIR

    try {
        if ($APP -eq "onnxrt_hailo_pipeline") {
            if (-not $env:ONNXRUNTIME_DIR) {
                throw "ONNXRUNTIME_DIR is not set. Example: `$env:ONNXRUNTIME_DIR='C:\path\to\onnxruntime'"
            }

            if (!(Test-Path $env:ONNXRUNTIME_DIR)) {
                throw "ONNXRUNTIME_DIR path does not exist: $env:ONNXRUNTIME_DIR"
            }

            cmake -S . -B build `
                -DCMAKE_FIND_PACKAGE_RESOLVE_SYMLINKS=True `
                -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH_FOR_APPS" `
                -DONNXRUNTIME_DIR="$env:ONNXRUNTIME_DIR"

            Check-LastCommand "CMake configure failed for ${APP}"

            cmake --build build --config Release
            Check-LastCommand "Build failed for ${APP}"

            Copy-Item "$env:ONNXRUNTIME_DIR\lib\*.dll" ".\build\Release\" -Force
            Write-Host "-I- ${APP}: copied ONNX Runtime DLLs"
        }
        else {
            cmake -S . -B build `
                -DCMAKE_FIND_PACKAGE_RESOLVE_SYMLINKS=True `
                -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH_FOR_APPS"

            Check-LastCommand "CMake configure failed for ${APP}"

            cmake --build build --config Release
            Check-LastCommand "Build failed for ${APP}"
        }

        Write-Host "-I- ${APP}: OK"
    }
    catch {
        Write-Host "-E- ${APP}: FAILED"
        Write-Host $_.Exception.Message
        $FAILED += $APP
    }
    finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "=========================================="
Write-Host " Build Summary"
Write-Host "=========================================="

foreach ($APP in $APPS) {
    if ($FAILED -contains $APP) {
        Write-Host "  X  $APP"
    } else {
        Write-Host "  OK $APP"
    }
}

Write-Host "=========================================="

if ($FAILED.Count -gt 0) {
    exit 1
}