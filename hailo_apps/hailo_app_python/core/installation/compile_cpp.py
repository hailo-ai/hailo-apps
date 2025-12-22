import subprocess
import sys
from pathlib import Path

from hailo_apps.hailo_app_python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)


def compile_postprocess():
    hailo_logger.debug("Entering compile_postprocess()")

    # 1) locate hailo-apps package
    here = Path(__file__).resolve()
    pkg_root = here.parents[3]
    hailo_logger.debug(f"Resolved package root: {pkg_root}")

    # 2) point at the correct folder structure: hailo_app_python/core/cpp_postprocess
    pp_dir = pkg_root / "hailo_app_python" / "core" / "cpp_postprocess"
    hailo_logger.debug(f"Postprocess directory: {pp_dir}")

    if not (pp_dir / "meson.build").exists():
        hailo_logger.error(f"meson.build not found in {pp_dir}")
        sys.exit(1)

    # 3) call the script from there
    script = pp_dir / "compile_postprocess.sh"
    hailo_logger.debug(f"Running compile script: {script}")

    ret = subprocess.run(["bash", str(script), "release"], cwd=str(pp_dir), check=False)
    if ret.returncode:
        hailo_logger.error(f"C++ postprocess build failed (exit {ret.returncode})")
        sys.exit(ret.returncode)
    hailo_logger.info("C++ postprocess compiled successfully.")


def main():
    hailo_logger.debug("Entering main() in compile_cpp.py")
    try:
        compile_postprocess()
    except Exception as e:
        hailo_logger.exception(f"Failed to compile C++ postprocess module: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
