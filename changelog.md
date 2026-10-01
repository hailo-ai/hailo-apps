## 26.09.0

### Fixed
- Fixed super-resolution post-processing: model output is now kept at its native upscaled resolution (2x for ESRGAN, 4x for ESPCN) and passed through model-specific post-processing (float→uint8 conversion, color handling) before building the side-by-side comparison image
- Corrected installed-package detection logic in `install.sh` and `scripts/check_installed_packages.sh`

### Improved (Installation)
- `install.sh` now self-elevates to root via `sudo -E`, preserving the caller's environment (including an already-active virtual environment such as the one pre-activated inside the Hailo AI Software Suite Docker) — running `./install.sh` no longer requires prefixing with `sudo`
- Added detection of an existing PyHailoRT virtual environment and of an already-installed TAPPAS core, avoiding redundant reinstallation
- Added support for installing inside the Hailo AI Software Suite Docker and the HailoRT Docker container, with corresponding documentation in the prerequisites and installation guides
- Wheel-supplied PyHailoRT/PyTAPPAS versions (`--pyhailort` / `--pytappas`) are now correctly extracted from the wheel filename and take precedence over stale/absent detected versions

### Documentation
- Reworked and condensed the [installation guide](./doc/user_guide/installation.md) and [running applications guide](./doc/user_guide/running_applications.md)
- Added a new [prerequisites guide](./doc/user_guide/prerequisites.md) covering platform/docker requirements
- Updated README and various app/developer guide READMEs

## 26.03.1

### Fixed
- Improved Raspberry Pi compatibility by supporting both legacy and new HailoRT package names
- Fixed H10 default Model Zoo version auto-resolution and added compatibility support for HailoRT 5.3.0
- Refactored USB/Raspberry Pi camera detection with earlier validation when required camera components are unavailable
- Strengthened installation prerequisite checks and version compatibility validation across supported architectures
- Removed deprecated resource scripts `get_inputs.sh` and `get_hef.sh`