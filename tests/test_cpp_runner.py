"""
C++ App Test Runner

Tests for Hailo C++ standalone applications.

Test suites:
  test_cpp_build[app]   - verify build.sh <app> exits 0 (no device needed)
  test_cpp_image[app]   - run <app> with a still image, expect clean exit
  test_cpp_video[app]   - run <app> with a video file for VIDEO_RUN_TIME seconds
  test_cpp_camera[app]  - run <app> with USB camera

Special handling:
  depth_estimation_stereo  : uses --left / --right instead of --input
  onnxrt_hailo_pipeline    : requires yolov8m_seg.hef (auto-resolved) + yolov8m-seg_post.onnx
  zero_shot_classification : different CLI (-te= -ie= -i= -p=); needs text_projection.bin
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT  = Path(__file__).resolve().parents[1]
CPP_DIR    = REPO_ROOT / "hailo_apps" / "cpp"

if sys.platform == "win32":
    RES_DIR = Path(r"C:\usr\local\hailo\resources")
else:
    RES_DIR = REPO_ROOT / "resources"

IMAGES_DIR = RES_DIR / "images"
VIDEOS_DIR = RES_DIR / "videos"
MODELS_DIR = RES_DIR / "models"

S3_BASE = "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources"

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

BUILD_TIMEOUT   = 300   # seconds — per-app CMake build
IMAGE_TIMEOUT   = 90    # seconds — image test (app exits naturally after one frame)
VIDEO_RUN_TIME  = 20    # seconds — video run duration before SIGTERM
CAMERA_RUN_TIME = 15    # seconds — camera run duration before SIGTERM
TERM_TIMEOUT    = 10    # seconds — wait for SIGTERM before SIGKILL
MIN_STABLE_RUN  = 3.0   # seconds — minimum elapsed time to count as stable

# ---------------------------------------------------------------------------
# App table
# ---------------------------------------------------------------------------

@dataclass
class _AppCfg:
    name: str
    binary: str                             # path relative to CPP_DIR
    input_mode: str                         # "standard" | "stereo" | "zero_shot"
    supports_camera: bool = True
    extra: Dict[str, List[str]] = field(default_factory=dict)  # per input_type static extra args
    run_cwd: Optional[str] = None           # relative to CPP_DIR; None → CPP_DIR
    skip_image:  Optional[str] = None
    skip_video:  Optional[str] = None
    skip_camera: Optional[str] = None


_APPS: List[_AppCfg] = [
    _AppCfg(
        name="classification",
        binary="classification/build/classifier",
        input_mode="standard",
    ),
    _AppCfg(
        name="depth_estimation_mono",
        binary="depth_estimation_mono/build/mono_depth_estimation",
        input_mode="standard",
    ),
    _AppCfg(
        name="depth_estimation_stereo",
        binary="depth_estimation_stereo/build/stereo_depth_estimation",
        input_mode="stereo",
        supports_camera=False,
        skip_video="depth_estimation_stereo supports image input only",
        skip_camera="Stereo depth requires two independent camera devices",
    ),
    _AppCfg(
        name="instance_segmentation",
        binary="instance_segmentation/build/instance_segmentation",
        input_mode="standard",
    ),
    _AppCfg(
        name="object_detection",
        binary="object_detection/build/object_detection",
        input_mode="standard",
    ),
    # onnxrt extra args are built dynamically in _build_cmd (arch-aware HEF lookup)
    _AppCfg(
        name="onnxrt_hailo_pipeline",
        binary="onnxrt_hailo_pipeline/build/onnxrt_hailo_pipeline",
        input_mode="standard",
    ),
    _AppCfg(
        name="oriented_object_detection",
        binary="oriented_object_detection/build/oriented_obj_det",
        input_mode="standard",
    ),
    _AppCfg(
        name="pose_estimation",
        binary="pose_estimation/build/pose_estimation",
        input_mode="standard",
    ),
    _AppCfg(
        name="semantic_segmentation",
        binary="semantic_segmentation/build/semantic_segmentation",
        input_mode="standard",
    ),
    _AppCfg(
        name="zero_shot_classification",
        binary="zero_shot_classification/build/zero_shot_classification",
        input_mode="zero_shot",
        supports_camera=False,
        run_cwd="zero_shot_classification",
        skip_image="zero_shot_classification: build test only",
        skip_video="zero_shot_classification: build test only",
        skip_camera="zero_shot_classification: build test only",
    ),
]

_APP_MAP  = {c.name: c for c in _APPS}
_ALL_NAMES = [c.name for c in _APPS]

# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

class _Redirect308Handler(urllib.request.HTTPRedirectHandler):
    """urllib does not handle 308 by default; delegate it to the 307 handler."""
    def http_error_308(self, req, fp, code, msg, headers):
        return self.http_error_307(req, fp, code, msg, headers)

_http_opener = urllib.request.build_opener(_Redirect308Handler())


def _download_if_missing(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", url, dest)
    req = urllib.request.Request(url, headers={"User-Agent": "hailo-cpp-tests/1.0"})
    # timeout=30 is per socket operation (connect + each read chunk).
    # This prevents a stalled server from blocking the test suite indefinitely.
    try:
        with _http_opener.open(req, timeout=30) as resp:
            with open(dest, "wb") as fh:
                while chunk := resp.read(1 << 16):
                    fh.write(chunk)
    except Exception:
        if dest.exists():
            dest.unlink()
        raise

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_arch() -> Optional[str]:
    try:
        from hailo_apps.python.core.common.installation_utils import detect_hailo_arch
        arch = detect_hailo_arch()
        print(f"Detected arch: {arch}")
        return arch
    except Exception as e:
        print(f"_detect_arch failed: {type(e).__name__}: {e}")
        return None


def _binary(cfg: _AppCfg) -> Path:
    if sys.platform == "win32":
        parent, name = cfg.binary.rsplit("/", 1)
        return CPP_DIR / parent / "Release" / (name + ".exe")
    return CPP_DIR / cfg.binary




def _run_cwd(cfg: _AppCfg) -> Path:
    return CPP_DIR / cfg.run_cwd if cfg.run_cwd else CPP_DIR


def _zero_shot_ready(arch: str) -> bool:
    zs_dir = CPP_DIR / "zero_shot_classification"
    return (
        (zs_dir / "text_projection.bin").exists()
        and (MODELS_DIR / arch / "clip_text_encoder_vit_l_14_laion2B.hef").exists()
        and (MODELS_DIR / arch / "clip_vit_l_14_laion2B_image_encoder.hef").exists()
    )


def _onnxrt_onnx_path() -> Optional[Path]:
    """Return path to yolov8m-seg_post.onnx (lives next to the binary), or None if missing."""
    p = CPP_DIR / "onnxrt_hailo_pipeline" / "yolov8m-seg_post.onnx"
    return p if p.exists() else None


def _build_cmd(cfg: _AppCfg, input_type: str, arch: str, output_dir: Optional[Path]) -> List[str]:
    """Return argv for running cfg with input_type.

    Calls pytest.skip() when required resources are absent.
    """
    binary = str(_binary(cfg))

    # ---------------------------------------------------------------- zero_shot
    if cfg.input_mode == "zero_shot":
        if not _zero_shot_ready(arch):
            pytest.skip(
                f"zero_shot_classification: missing resources. "
                f"Run hailo_apps/cpp/zero_shot_classification/download_resources.sh"
            )
        te = str(MODELS_DIR / arch / "clip_text_encoder_vit_l_14_laion2B.hef")
        ie = str(MODELS_DIR / arch / "clip_vit_l_14_laion2B_image_encoder.hef")
        base = [binary, f"-te={te}", f"-ie={ie}", "-p=dog,cat"]
        if input_type == "image":
            return base + [f"-i={IMAGES_DIR / 'bus.jpg'}", "-n=1"]
        if input_type == "video":
            return base + [f"-i={VIDEOS_DIR / 'example.mp4'}"]
        pytest.skip(cfg.skip_camera or "Camera not supported for zero_shot test")

    # ---------------------------------------------------------------- stereo
    if cfg.input_mode == "stereo":
        cmd = [binary]
        if output_dir:
            cmd += ["--output-dir", str(output_dir)]
        if input_type == "image":
            stereo_dir = CPP_DIR / "depth_estimation_stereo"
            cmd += ["--left",  str(stereo_dir / "left.jpg"),
                    "--right", str(stereo_dir / "right.jpg")]
        elif input_type == "video":
            cmd += ["--left",  str(VIDEOS_DIR / "example.mp4"),
                    "--right", str(VIDEOS_DIR / "example_640.mp4")]
        else:
            pytest.skip(cfg.skip_camera or "Stereo camera requires two camera devices")
        return cmd

    # ---------------------------------------------------------------- standard
    cmd = [binary]
    if output_dir:
        cmd += ["--output-dir", str(output_dir)]

    if input_type == "image":
        cmd += ["--input", str(IMAGES_DIR / "bus.jpg")]
    elif input_type == "video":
        cmd += ["--input", str(VIDEOS_DIR / "example.mp4")]
    else:
        cmd += ["--input", "usb"]

    # onnxrt: HEF is auto-resolved by ResourcesManager (yolov8m_seg); only --onnx is needed
    if cfg.name == "onnxrt_hailo_pipeline":
        onnx = _onnxrt_onnx_path()
        if onnx is None:
            pytest.skip(
                "onnxrt_hailo_pipeline: yolov8m-seg_post.onnx not found. "
                "Run hailo_apps/cpp/onnxrt_hailo_pipeline/download_resources.sh"
            )
        cmd += ["--onnx", str(onnx)]
    else:
        cmd += cfg.extra.get(input_type, [])

    return cmd


# ---------------------------------------------------------------------------
# Process runner
# ---------------------------------------------------------------------------

class _Result:
    __slots__ = ("returncode", "stdout", "stderr", "elapsed", "early_exit")

    def __init__(self, returncode, stdout, stderr, elapsed=0.0, early_exit=False):
        self.returncode = returncode
        self.stdout     = stdout
        self.stderr     = stderr
        self.elapsed    = elapsed
        self.early_exit = early_exit


def _cpp_env() -> Dict[str, str]:
    """Keep display forwarding while excluding Python OpenCV's bundled Qt paths."""
    env = os.environ.copy()
    for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        if "cv2/qt/" in env.get(key, "").replace("\\", "/"):
            env.pop(key)
    return env


def _run_timed(cmd: List[str], cwd: Path, run_time: int) -> _Result:
    """Run cmd for up to run_time seconds, then send SIGTERM."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(cwd),
        env=_cpp_env(),
    )

    start      = time.monotonic()
    early_exit = False

    while (time.monotonic() - start) < run_time:
        time.sleep(0.25)
        if proc.poll() is not None:
            early_exit = True
            break

    elapsed = time.monotonic() - start

    if not early_exit:
        proc.terminate()  # SIGTERM on Linux/macOS, TerminateProcess on Windows

    try:
        stdout, stderr = proc.communicate(timeout=TERM_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()

    return _Result(proc.returncode, stdout, stderr, elapsed, early_exit)


# ---------------------------------------------------------------------------
# Logging + assertion
# ---------------------------------------------------------------------------

_LOG_DIR = REPO_ROOT / "tests" / "tests_logs" / "cpp"

_FATAL = [
    "segmentation fault", "core dumped",
    "hailo_status_failure", "failed to open",
    "no such file or directory", "cannot open",
    "abort()", "terminate called",
]


def _log_path(app: str, tag: str) -> Path:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_DIR / f"{app}_{tag}.log"


def _write_log(path: Path, cmd: List[str], result: _Result) -> None:
    with open(path, "wb") as fh:
        fh.write(("cmd: " + " ".join(cmd) + "\n\n").encode())
        fh.write(b"--- stdout ---\n" + result.stdout)
        fh.write(b"\n--- stderr ---\n" + result.stderr)


def _assert_clean(result: _Result, app: str, tag: str, log: Path,
                  min_runtime: float = 0.0) -> None:
    stderr = result.stderr.decode(errors="replace").lower()

    # early_exit=True  → process quit by itself; rc must be 0
    # early_exit=False → harness killed it (terminate/SIGTERM); any rc is expected
    bad_exit = result.early_exit and result.returncode != 0

    fatal = next((kw for kw in _FATAL if kw in stderr), None)
    too_short = min_runtime > 0 and result.elapsed < min_runtime

    assert not bad_exit, (
        f"[{app}][{tag}] exit {result.returncode} "
        f"(early={result.early_exit}, elapsed={result.elapsed:.1f}s) — "
        f"log: {log}\n"
        + result.stderr.decode(errors="replace")[-2000:]
    )
    assert fatal is None, (
        f"[{app}][{tag}] fatal keyword '{fatal}' in stderr — log: {log}\n"
        + result.stderr.decode(errors="replace")[-2000:]
    )
    if too_short:
        pytest.fail(
            f"[{app}][{tag}] exited after {result.elapsed:.1f}s "
            f"(expected ≥ {min_runtime}s) — log: {log}"
        )


# ---------------------------------------------------------------------------
# Session fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def _arch():
    return _detect_arch()


@pytest.fixture(scope="session")
def _resources(_arch):
    """Download all inputs needed by C++ functional tests."""
    if not _arch:
        pytest.skip("No Hailo device detected — skipping C++ functional tests")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    _download_if_missing(f"{S3_BASE}/images/bus.jpg", IMAGES_DIR / "bus.jpg")
    _download_if_missing(f"{S3_BASE}/video/example.mp4",     VIDEOS_DIR / "example.mp4")
    _download_if_missing(f"{S3_BASE}/video/example_640.mp4", VIDEOS_DIR / "example_640.mp4")

    # onnxrt: yolov8m-seg_post.onnx must live next to the binary
    _download_if_missing(
        f"{S3_BASE}/onnxs/yolov8m-seg_post.onnx",
        CPP_DIR / "onnxrt_hailo_pipeline" / "yolov8m-seg_post.onnx",
    )

    # zero_shot extra: text_projection.bin must live next to the binary
    zs_dir = CPP_DIR / "zero_shot_classification"
    _download_if_missing(
        f"{S3_BASE}/external+bin+files/text_projection.bin",
        zs_dir / "text_projection.bin",
    )


# ---------------------------------------------------------------------------
# 1. Build tests — no device required
# ---------------------------------------------------------------------------

@pytest.mark.cpp
@pytest.mark.parametrize("app_name", _ALL_NAMES)
def test_cpp_build(app_name):
    """Build <app> and assert exit code 0."""
    log = _log_path(app_name, "build")

    if sys.platform == "win32":
        display_cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", ".\\build.ps1", app_name]
    else:
        display_cmd = ["bash", "build.sh", app_name]

    p = subprocess.run(
        display_cmd,
        cwd=str(CPP_DIR),
        capture_output=True,
        timeout=BUILD_TIMEOUT,
    )
    result = _Result(p.returncode, p.stdout, p.stderr)

    _write_log(log, display_cmd, result)
    assert result.returncode == 0, (
        f"Build failed for {app_name}. Log: {log}\n"
        + result.stdout.decode(errors="replace")[-3000:]
        + "\n"
        + result.stderr.decode(errors="replace")[-500:]
    )


# ---------------------------------------------------------------------------
# 2. Image tests — one frame, app exits naturally
# ---------------------------------------------------------------------------

@pytest.mark.cpp
@pytest.mark.requires_device
@pytest.mark.parametrize("app_name", _ALL_NAMES)
def test_cpp_image(app_name, _resources, _arch, tmp_path):
    """Run <app> with a still image; expect clean exit within IMAGE_TIMEOUT."""
    cfg = _APP_MAP[app_name]
    if cfg.skip_image:
        pytest.skip(cfg.skip_image)
    if not _binary(cfg).exists():
        pytest.skip(f"Binary not found for {app_name} — run build tests first")

    output_dir = tmp_path / "output"
    cmd = _build_cmd(cfg, "image", _arch, output_dir)
    log = _log_path(app_name, f"image_{_arch}")

    logger.info("[%s] image cmd: %s", app_name, " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            cwd=str(_run_cwd(cfg)),
            env=_cpp_env(),
            timeout=IMAGE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"[{app_name}] image test timed out after {IMAGE_TIMEOUT}s")

    result = _Result(proc.returncode, proc.stdout, proc.stderr)
    _write_log(log, cmd, result)
    _assert_clean(result, app_name, f"image[{_arch}]", log)


# ---------------------------------------------------------------------------
# 3. Video tests — run for VIDEO_RUN_TIME seconds then SIGTERM
# ---------------------------------------------------------------------------

@pytest.mark.cpp
@pytest.mark.requires_device
@pytest.mark.parametrize("app_name", _ALL_NAMES)
def test_cpp_video(app_name, _resources, _arch):
    """Run <app> with a video file; expect stable run without early crash."""
    cfg = _APP_MAP[app_name]
    if cfg.skip_video:
        pytest.skip(cfg.skip_video)
    if not _binary(cfg).exists():
        pytest.skip(f"Binary not found for {app_name} — run build tests first")

    cmd = _build_cmd(cfg, "video", _arch, None)
    log = _log_path(app_name, f"video_{_arch}")

    logger.info("[%s] video cmd: %s", app_name, " ".join(cmd))

    result = _run_timed(cmd, _run_cwd(cfg), VIDEO_RUN_TIME)
    _write_log(log, cmd, result)
    _assert_clean(result, app_name, f"video[{_arch}]", log, min_runtime=MIN_STABLE_RUN)


# ---------------------------------------------------------------------------
# 4. Camera tests
# ---------------------------------------------------------------------------

@pytest.mark.cpp
@pytest.mark.requires_device
@pytest.mark.parametrize("app_name", [c.name for c in _APPS if c.supports_camera])
def test_cpp_camera(app_name, _resources, _arch):

    cfg = _APP_MAP[app_name]
    if cfg.skip_camera:
        pytest.skip(cfg.skip_camera)
    if not _binary(cfg).exists():
        pytest.skip(f"Binary not found for {app_name} — run build tests first")

    cmd = _build_cmd(cfg, "camera", _arch, None)
    log = _log_path(app_name, f"camera_{_arch}")

    logger.info("[%s] camera cmd: %s", app_name, " ".join(cmd))

    result = _run_timed(cmd, _run_cwd(cfg), CAMERA_RUN_TIME)
    _write_log(log, cmd, result)
    _assert_clean(result, app_name, f"camera[{_arch}]", log, min_runtime=MIN_STABLE_RUN)
