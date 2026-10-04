#!/usr/bin/env python3
"""
Resource Download Manager for Hailo Apps Infrastructure.

This module provides a robust, optimized system for downloading ML models,
videos, images, and configuration files for Hailo applications.

Features:
- Parallel downloads with configurable workers
- Retry mechanism with exponential backoff
- File size validation (replaces corrupted/partial files)
- Atomic file operations (temp file + move)
- Progress tracking with visual progress bar
- Dry-run mode for previewing downloads
- Force-redownload capability
"""

import argparse
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import yaml

from hailo_apps.python.core.common.hailo_logger import get_logger

hailo_logger = get_logger(__name__)

from hailo_apps.config.config_manager import get_images_for_app, get_videos_for_app, _load_yaml as load_config

from hailo_apps.python.core.common.core import load_environment
from hailo_apps.python.core.common.defines import (
    DEFAULT_RESOURCES_CONFIG_PATH,
    HAILO8_ARCH,
    HAILO8L_ARCH,
    HAILO10H_ARCH,
    HAILO_FILE_EXTENSION,
    HAILORT_VERSION_KEY,
    MODEL_ZOO_URL,
    MODEL_ZOO_VERSION_DEFAULT,
    MODEL_ZOO_VERSION_KEY,
    RESOURCES_JSON_DIR_NAME,
    RESOURCES_NPY_DIR_NAME,
    RESOURCES_MODELS_DIR_NAME,
    RESOURCES_ROOT_PATH_DEFAULT,
    RESOURCES_VIDEOS_DIR_NAME,
    S3_RESOURCES_BASE_URL,
    VALID_H8_MODEL_ZOO_VERSION,
    VALID_H10_MODEL_ZOO_VERSION,
    RESOURCE_TYPE_MODEL,
    RESOURCE_TYPE_IMAGE,
    RESOURCE_TYPE_VIDEO,
    RESOURCE_TYPE_ONNX,
    RESOURCE_TYPES,
)

from hailo_apps.python.core.common.installation_utils import detect_hailo_arch, auto_detect_hailort_version


# =============================================================================
# Configuration
# =============================================================================

# User-Agent string to avoid 403 errors from Cloudflare-protected servers
# Some servers (like dev-public.hailo.ai) block requests with Python's default User-Agent
DEFAULT_USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

@dataclass
class DownloadConfig:
    """Configuration for the resource downloader."""
    max_retries: int = 3
    retry_delay: float = 1.0  # Base delay in seconds (exponential backoff)
    timeout: int = 300  # Download timeout in seconds
    parallel_workers: int = 4
    show_progress: bool = True
    dry_run: bool = False
    force_redownload: bool = False
    include_gen_ai: bool = False  # Whether to include gen-ai models


@dataclass
class DownloadTask:
    """Represents a single download task."""
    url: str
    dest_path: Path
    resource_type: str  # 'model', 'video', 'image', 'json'
    name: str
    expected_size: Optional[int] = None
    
    def __hash__(self):
        return hash((self.url, str(self.dest_path)))
    
    def __eq__(self, other):
        if not isinstance(other, DownloadTask):
            return False
        return self.url == other.url and self.dest_path == other.dest_path


@dataclass
class DownloadResult:
    """Result of a download operation."""
    task: DownloadTask
    success: bool
    message: str
    skipped: bool = False
    file_size: int = 0


# =============================================================================
# Utility Functions
# =============================================================================

def is_none_value(value) -> bool:
    """Check if a value represents None (handles YAML None parsing)."""
    if value is None:
        return True
    if isinstance(value, str) and value.lower() == "none":
        return True
    return False


def is_valid_model_entry(entry) -> bool:
    """Check if a model entry is valid (not None, has valid name)."""
    if is_none_value(entry):
        return False
    if isinstance(entry, dict):
        name = entry.get("name")
        return not is_none_value(name) and bool(name)
    if isinstance(entry, str):
        return not is_none_value(entry) and bool(entry)
    return False


def test_url(url: str) -> bool:
    """Test if a URL is reachable and valid."""
    try:
        request = urllib.request.Request(
            url, 
            method='HEAD',
            headers={'User-Agent': DEFAULT_USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            print(f"✓ URL valid: {url}")
            print(f"  Status: {response.status}")
            print(f"  Size: {response.headers.get('Content-Length', 'unknown')} bytes")
            return True
    except urllib.error.HTTPError as e:
        print(f"✗ HTTP Error {e.code}: {url}")
        return False
    except urllib.error.URLError as e:
        print(f"✗ URL Error: {e.reason} - {url}")
        return False


def map_arch_to_config_key(hailo_arch: str) -> str:
    """Map Hailo architecture to config key (H8 or H10)."""
    if hailo_arch in (HAILO8_ARCH, HAILO8L_ARCH):
        return "H8"
    elif hailo_arch == HAILO10H_ARCH:
        return "H10"
    else:
        hailo_logger.warning(f"Unknown architecture {hailo_arch}, defaulting to H8")
        return "H8"


def map_arch_to_s3_path(hailo_arch: str) -> str:
    """Map Hailo architecture to S3 path architecture."""
    arch_map = {
        HAILO8_ARCH: "h8",
        HAILO8L_ARCH: "h8l",
        HAILO10H_ARCH: "h10",
    }
    if hailo_arch not in arch_map:
        hailo_logger.warning(f"Unknown architecture {hailo_arch}, defaulting to h8")
        return "h8"
    return arch_map[hailo_arch]


def get_model_zoo_version_for_arch(hailo_arch: str) -> tuple[str, str]:
    """Get Model Zoo version and download architecture for a given Hailo architecture.

    For H10: Derives from HailoRT version (5.1.x -> v5.1.0, 5.2.x -> v5.2.0, etc.)
    For H8/H8L: Derives from HailoRT version (4.24.x -> v2.19.0, 4.23.x -> v2.18.0)
    """
    download_arch = hailo_arch

    # First check if explicitly set via environment
    model_zoo_version = os.getenv(MODEL_ZOO_VERSION_KEY)

    if model_zoo_version is None:
        hailort_version = os.getenv(
            HAILORT_VERSION_KEY,
            auto_detect_hailort_version()
        )

        # Auto-select default model zoo version based on device architecture
        if hailo_arch == HAILO10H_ARCH:
            if not hailort_version:
                raise RuntimeError(
                    "Failed to determine HailoRT version for Hailo-10H. "
                    f"Please set {MODEL_ZOO_VERSION_KEY} manually."
                )
            # Keep 5.1 pinned to v5.1.0 for backward compatibility
            if hailort_version.startswith("5.1"):
                model_zoo_version = "v5.1.0"
            else:
                # For newer versions, use the exact HailoRT version
                model_zoo_version = f"v{hailort_version}"
        else:
            # H8/H8L: Derive from HailoRT version
            if hailort_version and hailort_version.startswith("4.24"):
                model_zoo_version = "v2.19.0"
            elif hailort_version and hailort_version.startswith("4.23"):
                model_zoo_version = "v2.18.0"
            else:
                # Fallback to newest
                model_zoo_version = VALID_H8_MODEL_ZOO_VERSION[0]

    if hailo_arch == HAILO10H_ARCH and model_zoo_version.startswith("v5.1."):
        model_zoo_version = "v5.1.0"

    # Validate the version; fall back to the config default if invalid
    if hailo_arch == HAILO10H_ARCH and model_zoo_version not in VALID_H10_MODEL_ZOO_VERSION:
        model_zoo_version = VALID_H10_MODEL_ZOO_VERSION[0]
    if hailo_arch in (HAILO8_ARCH, HAILO8L_ARCH) and model_zoo_version not in VALID_H8_MODEL_ZOO_VERSION:
        model_zoo_version = VALID_H8_MODEL_ZOO_VERSION[0]

    return model_zoo_version, download_arch


def get_remote_file_size(url: str, timeout: int = 30) -> Optional[int]:
    """Get the size of a remote file without downloading it."""
    try:
        request = urllib.request.Request(
            url, 
            method='HEAD',
            headers={'User-Agent': DEFAULT_USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_length = response.headers.get('Content-Length')
            if content_length:
                return int(content_length)
    except Exception as e:
        hailo_logger.debug(f"Could not get remote file size for {url}: {e}")
    return None


def is_gen_ai_source(source: str) -> bool:
    """Check if the source is a gen-ai model source."""
    return source == "gen-ai-mz"

def _ensure_hef_filename(name: str) -> str:
    """Return a .hef filename for a model name."""
    if name.endswith(HAILO_FILE_EXTENSION):
        return name
    return f"{name}{HAILO_FILE_EXTENSION}"

# =============================================================================
# Progress Display
# =============================================================================

class ProgressTracker:
    """Tracks and displays download progress."""
    
    def __init__(self, show_progress: bool = True):
        self.show_progress = show_progress
        self._last_percent = -1
    
    def update(self, block_num: int, block_size: int, total_size: int):
        """Callback function to show download progress."""
        if not self.show_progress:
            return
        
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            if percent == self._last_percent:
                return
            self._last_percent = percent
            
            bar_length = 40
            filled = int(bar_length * downloaded // total_size)
            bar = '=' * filled + '-' * (bar_length - filled)
            size_mb = total_size / (1024 * 1024)
            downloaded_mb = downloaded / (1024 * 1024)
            print(f"\r[{bar}] {percent}% ({downloaded_mb:.2f}/{size_mb:.2f} MB)", end='', flush=True)
        else:
            downloaded_mb = downloaded / (1024 * 1024)
            if downloaded_mb < 0.01:
                downloaded_kb = downloaded / 1024
                print(f"\rDownloading... {downloaded_kb:.2f} KB", end='', flush=True)
            else:
                print(f"\rDownloading... {downloaded_mb:.2f} MB", end='', flush=True)
    
    def finish(self):
        """Print newline after progress bar."""
        if self.show_progress:
            print()
            self._last_percent = -1


# =============================================================================
# Resource Downloader Class
# =============================================================================

class ResourceDownloader:
    """
    Robust resource downloader with retry, parallel downloads, and validation.
    """
    
    def __init__(
        self,
        config: dict,
        hailo_arch: str,
        resource_root: Path,
        download_config: Optional[DownloadConfig] = None
    ):
        self.config = config
        self.hailo_arch = hailo_arch
        self.resource_root = Path(resource_root)
        self.download_config = download_config or DownloadConfig()
        
        # Setup model zoo parameters
        self.model_zoo_version, self.download_arch = get_model_zoo_version_for_arch(hailo_arch)
        
        # Track download tasks
        self._tasks: set[DownloadTask] = set()
        self._results: list[DownloadResult] = []
    
    # -------------------------------------------------------------------------
    # File Download Core
    # -------------------------------------------------------------------------
    
    def _should_download(self, dest_path: Path, expected_size: Optional[int] = None) -> tuple[bool, str]:
        """
        Determine if a file should be downloaded.
        
        Returns:
            Tuple of (should_download, reason)
        """
        if self.download_config.force_redownload:
            return True, "Force redownload requested"
        
        if not dest_path.exists():
            return True, "File does not exist"
        
        # Check file size if we have expected size
        if expected_size is not None:
            local_size = dest_path.stat().st_size
            if local_size != expected_size:
                hailo_logger.info(
                    f"File size mismatch for {dest_path.name}: "
                    f"local={local_size}, remote={expected_size}. Will re-download."
                )
                return True, f"Size mismatch (local: {local_size}, remote: {expected_size})"
        
        # Check if file is empty (likely corrupted/partial)
        if dest_path.stat().st_size == 0:
            return True, "File is empty (likely corrupted)"
        
        return False, "File already exists and appears valid"
    
    def _download_file_with_retry(self, task: DownloadTask) -> DownloadResult:
        """
        Download a file with retry mechanism and atomic operations.
        """
        url = task.url
        dest_path = task.dest_path
        
        # Check remote file size for validation
        remote_size = task.expected_size
        if remote_size is None:
            remote_size = get_remote_file_size(url, timeout=30)
        
        # Check if download is needed
        should_download, reason = self._should_download(dest_path, remote_size)
        
        if not should_download:
            hailo_logger.info(f"Skipping {dest_path.name}: {reason}")
            return DownloadResult(
                task=task,
                success=True,
                message=reason,
                skipped=True,
                file_size=dest_path.stat().st_size if dest_path.exists() else 0
            )
        
        # Dry run mode
        if self.download_config.dry_run:
            hailo_logger.info(f"[DRY RUN] Would download: {url} → {dest_path}")
            return DownloadResult(
                task=task,
                success=True,
                message="Dry run - would download",
                skipped=True,
                file_size=0
            )
        
        # Ensure destination directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove existing file if re-downloading
        if dest_path.exists():
            try:
                dest_path.unlink()
                hailo_logger.debug(f"Removed existing file: {dest_path}")
            except Exception as e:
                hailo_logger.warning(f"Could not remove existing file {dest_path}: {e}")
        
        # Download with retries
        last_error = None
        for attempt in range(self.download_config.max_retries):
            temp_path = None
            try:
                # Create temp file in same directory for atomic move
                fd, temp_path = tempfile.mkstemp(
                    dir=dest_path.parent,
                    prefix=f".{dest_path.name}.",
                    suffix=".tmp"
                )
                os.close(fd)
                temp_path = Path(temp_path)
                
                # Download to temp file
                progress = ProgressTracker(self.download_config.show_progress)
                hailo_logger.info(f"Downloading: {url}")
                
                # Use urlopen with custom User-Agent to avoid 403 from Cloudflare-protected servers
                request = urllib.request.Request(
                    url,
                    headers={'User-Agent': DEFAULT_USER_AGENT}
                )
                
                with urllib.request.urlopen(request, timeout=self.download_config.timeout) as response:
                    total_size = int(response.headers.get('Content-Length', 0))
                    block_size = 8192
                    downloaded = 0
                    
                    with open(temp_path, 'wb') as out_file:
                        while True:
                            block = response.read(block_size)
                            if not block:
                                break
                            out_file.write(block)
                            downloaded += len(block)
                            if self.download_config.show_progress:
                                progress.update(downloaded // block_size, block_size, total_size)
                
                progress.finish()
                
                # Verify download size
                downloaded_size = temp_path.stat().st_size
                if remote_size is not None and downloaded_size != remote_size:
                    raise ValueError(
                        f"Downloaded file size ({downloaded_size}) doesn't match "
                        f"expected size ({remote_size})"
                    )
                
                if downloaded_size == 0:
                    raise ValueError("Downloaded file is empty")
                
                # Atomic move to final destination
                temp_path.rename(dest_path)
                
                hailo_logger.info(f"Downloaded to {dest_path}")
                return DownloadResult(
                    task=task,
                    success=True,
                    message="Download successful",
                    skipped=False,
                    file_size=downloaded_size
                )
                
            except Exception as e:
                last_error = e
                hailo_logger.warning(
                    f"Download attempt {attempt + 1}/{self.download_config.max_retries} "
                    f"failed for {url}: {e}"
                )
                
                # Cleanup temp file
                if temp_path and Path(temp_path).exists():
                    try:
                        Path(temp_path).unlink()
                    except Exception:
                        pass
                
                # Fall back only for an unavailable MZ artifact, not network errors.
                if (
                    isinstance(e, urllib.error.HTTPError) and e.code in (403, 404)
                    and task.name == "stereonet" and self.hailo_arch == HAILO10H_ARCH
                    and self.model_zoo_version.startswith("v5.1.")
                    and url == f"{MODEL_ZOO_URL}/{self.model_zoo_version}/{self.download_arch}/stereonet{HAILO_FILE_EXTENSION}"
                ):
                    latest = max(VALID_H10_MODEL_ZOO_VERSION,
                                 key=lambda version: tuple(map(int, version.lstrip("v").split("."))))
                    fallback_url = f"{MODEL_ZOO_URL}/{latest}/{self.download_arch}/stereonet{HAILO_FILE_EXTENSION}"
                    if fallback_url != url:
                        hailo_logger.warning(
                            f"StereoNet unavailable in {self.model_zoo_version}; trying latest Model Zoo {latest}"
                        )
                        return self._download_file_with_retry(
                            replace(task, url=fallback_url, expected_size=None)
                        )

                # Exponential backoff
                if attempt < self.download_config.max_retries - 1:
                    delay = self.download_config.retry_delay * (2 ** attempt)
                    hailo_logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
        
        # All retries failed
        error_msg = f"Failed to download after {self.download_config.max_retries} attempts: {last_error}"
        hailo_logger.error(error_msg)
        return DownloadResult(
            task=task,
            success=False,
            message=error_msg,
            skipped=False,
            file_size=0
        )
    
    # -------------------------------------------------------------------------
    # Task Building
    # -------------------------------------------------------------------------
    
    def _build_model_url(self, model_entry: dict, source: str) -> Optional[str]:
        """Build download URL for a model based on its source."""
        name = model_entry.get("name")
        
        # Check for explicit URL first (works for any source including gen-ai-mz)
        if "url" in model_entry:
            return model_entry["url"]
        
        # Build URL based on source
        if source == "s3":
            s3_arch = map_arch_to_s3_path(self.hailo_arch)
            url = f"{S3_RESOURCES_BASE_URL}/hefs/{s3_arch}/{name}{HAILO_FILE_EXTENSION}"
            if test_url(url=url):
                return url
            return f"{S3_RESOURCES_BASE_URL}/hefs/{s3_arch}/{name}{HAILO_FILE_EXTENSION}"
        elif source == "mz":
            url = f"{MODEL_ZOO_URL}/{self.model_zoo_version}/{self.download_arch}/{name}{HAILO_FILE_EXTENSION}"
            test_url(url=url)  # Print URL validation info
            return url
        elif source == "gen-ai-mz":
            # Gen-AI models default to metadata-driven URL construction:
            #   {gen_ai_base}/{version}/blob/{model}.hef
            gen_ai_base = (
                self.config.get("metadata", {})
                .get("s3_endpoints", {})
                .get("gen_ai_mz", "https://dev-public.hailo.ai")
            )

            url = f"{gen_ai_base}/{self.model_zoo_version}/blob/{_ensure_hef_filename(name)}"
            test_url(url=url)  # Print URL validation info
            return url
        else:
            hailo_logger.warning(f"Unknown source '{source}' for model '{name}'")
            return None
    
    def _add_model_task(self, model_entry, is_gen_ai_allowed: bool = False):
        """Add a model download task from a model entry."""
        if not is_valid_model_entry(model_entry):
            return
        
        if isinstance(model_entry, str):
            # Legacy string format (assumed Model Zoo)
            name = model_entry
            source = "mz"
            model_entry = {"name": name, "source": source}
        else:
            name = model_entry.get("name")
            source = model_entry.get("source", "mz")
        
        # Skip gen-ai models unless explicitly allowed
        if is_gen_ai_source(source) and not is_gen_ai_allowed:
            hailo_logger.debug(f"Skipping gen-ai model: {name}")
            return
        
        url = self._build_model_url(model_entry, source)
        if not url:
            return
        
        dest = (
            self.resource_root
            / RESOURCES_MODELS_DIR_NAME
            / self.hailo_arch
            / f"{name}{HAILO_FILE_EXTENSION}"
        )
        
        task = DownloadTask(
            url=url,
            dest_path=dest,
            resource_type="model",
            name=name
        )
        self._tasks.add(task)

    def _add_onnx_task(self, onnx_name: str):
        """Add an ONNX sidecar download task by filename."""
        s3_arch = map_arch_to_s3_path(self.hailo_arch)
        url = f"{S3_RESOURCES_BASE_URL}/hefs/{s3_arch}/{onnx_name}"
        dest = self.resource_root / RESOURCES_MODELS_DIR_NAME / self.hailo_arch / onnx_name
        task = DownloadTask(
            url=url,
            dest_path=dest,
            resource_type="onnx",
            name=onnx_name,
        )
        self._tasks.add(task)
    
    def _add_video_task(self, video_entry):
        """Add a video download task from a video entry."""
        if is_none_value(video_entry):
            return
        
        if isinstance(video_entry, dict):
            video_name = video_entry.get("name")
            source = video_entry.get("source")
            video_url = video_entry.get("url")
            
            if not video_name:
                hailo_logger.warning(f"Video entry missing name: {video_entry}")
                return
            
            dest = self.resource_root / RESOURCES_VIDEOS_DIR_NAME / video_name
            
            if source == "s3":
                url = video_url or f"{S3_RESOURCES_BASE_URL}/video/{video_name}"
            elif video_url:
                url = video_url
            else:
                hailo_logger.warning(f"Video '{video_name}' missing URL and source is not 's3'")
                return
        elif isinstance(video_entry, str) and video_entry.startswith(("http://", "https://")):
            url = video_entry
            video_name = Path(video_entry).name
            dest = self.resource_root / RESOURCES_VIDEOS_DIR_NAME / video_name
        else:
            return
        
        task = DownloadTask(
            url=url,
            dest_path=dest,
            resource_type="video",
            name=video_name
        )
        self._tasks.add(task)
    
    def _add_image_task(self, image_entry):
        """Add an image download task from an image entry."""
        if is_none_value(image_entry):
            return
        
        if isinstance(image_entry, dict):
            image_name = image_entry.get("name")
            source = image_entry.get("source")
            image_url = image_entry.get("url")
            
            if not image_name:
                hailo_logger.warning(f"Image entry missing name: {image_entry}")
                return
            
            dest = self.resource_root / "images" / image_name
            
            if source == "s3":
                url = image_url or f"{S3_RESOURCES_BASE_URL}/images/{image_name}"
            elif image_url:
                url = image_url
            else:
                hailo_logger.warning(f"Image '{image_name}' missing URL and source is not 's3'")
                return
        elif isinstance(image_entry, str) and image_entry.startswith(("http://", "https://")):
            url = image_entry
            image_name = Path(image_entry).name
            dest = self.resource_root / "images" / image_name
        else:
            return
        
        task = DownloadTask(
            url=url,
            dest_path=dest,
            resource_type="image",
            name=image_name
        )
        self._tasks.add(task)
    
    def _add_json_task(self, json_entry):
        """Add a JSON download task from a JSON entry."""
        if is_none_value(json_entry):
            return
        
        if isinstance(json_entry, dict):
            json_name = json_entry.get("name")
            source = json_entry.get("source")
            json_url = json_entry.get("url")
            
            if not json_name:
                hailo_logger.warning(f"JSON entry missing name: {json_entry}")
                return
            
            dest = self.resource_root / RESOURCES_JSON_DIR_NAME / json_name
            
            if source == "s3":
                url = json_url or f"{S3_RESOURCES_BASE_URL}/configs/{json_name}"
            elif json_url:
                url = json_url
            else:
                hailo_logger.warning(f"JSON '{json_name}' missing URL and source is not 's3'")
                return
        elif isinstance(json_entry, str) and json_entry.startswith(("http://", "https://")):
            url = json_entry
            json_name = Path(json_entry).name
            dest = self.resource_root / RESOURCES_JSON_DIR_NAME / json_name
        else:
            return
        
        task = DownloadTask(
            url=url,
            dest_path=dest,
            resource_type="json",
            name=json_name
        )
        self._tasks.add(task)
    
    def _add_npy_task(self, npy_entry):
        """Add a NPY download task from a NPY entry."""
        if is_none_value(npy_entry):
            return

        if isinstance(npy_entry, dict):
            npy_name = npy_entry.get("name")
            source = npy_entry.get("source")
            npy_url = npy_entry.get("url")

            if not npy_name:
                hailo_logger.warning(f"NPY entry missing name: {npy_entry}")
                return

            dest = self.resource_root / RESOURCES_NPY_DIR_NAME / npy_name

            if source == "s3":
                url = npy_url or f"{S3_RESOURCES_BASE_URL}/npy/{npy_name}"
            elif npy_url:
                url = npy_url
            else:
                hailo_logger.warning(f"NPY '{npy_name}' missing URL and source is not 's3'")
                return
        elif isinstance(npy_entry, str) and npy_entry.startswith(("http://", "https://")):
            url = npy_entry
            npy_name = Path(npy_entry).name
            dest = self.resource_root / RESOURCES_NPY_DIR_NAME / npy_name
        else:
            return

        task = DownloadTask(
            url=url,
            dest_path=dest,
            resource_type="npy",
            name=npy_name
        )
        self._tasks.add(task)
    
    def _add_npy_task(self, npy_entry):
        """Add a NPY download task from a NPY entry."""
        if is_none_value(npy_entry):
            return

        if isinstance(npy_entry, dict):
            npy_name = npy_entry.get("name")
            source = npy_entry.get("source")
            npy_url = npy_entry.get("url")

            if not npy_name:
                hailo_logger.warning(f"NPY entry missing name: {npy_entry}")
                return

            dest = self.resource_root / RESOURCES_NPY_DIR_NAME / npy_name

            if source == "s3":
                url = npy_url or f"{S3_RESOURCES_BASE_URL}/npy/{npy_name}"
            elif npy_url:
                url = npy_url
            else:
                hailo_logger.warning(f"NPY '{npy_name}' missing URL and source is not 's3'")
                return
        elif isinstance(npy_entry, str) and npy_entry.startswith(("http://", "https://")):
            url = npy_entry
            npy_name = Path(npy_entry).name
            dest = self.resource_root / RESOURCES_NPY_DIR_NAME / npy_name
        else:
            return

        task = DownloadTask(
            url=url,
            dest_path=dest,
            resource_type="npy",
            name=npy_name
        )
        self._tasks.add(task)
    # -------------------------------------------------------------------------
    # High-Level Collection Methods
    # -------------------------------------------------------------------------
    
    def collect_all_videos(self):
        """Collect all video download tasks from config."""
        if "videos" in self.config:
            for video_entry in self.config["videos"]:
                self._add_video_task(video_entry)
    
    def collect_all_images(self):
        """Collect all image download tasks from config."""
        if "images" in self.config:
            for image_entry in self.config["images"]:
                self._add_image_task(image_entry)

    def collect_specific_image_for_app(self, app_name: str, image_name: str):
        """Collect a specific image by name, restricted to a specific app."""
        images_config = get_images_for_app(app_name=app_name)

        for image_entry in images_config:
            if isinstance(image_entry, dict):
                name = image_entry.get("name")
            elif isinstance(image_entry, str):
                name = Path(image_entry).name
            else:
                continue

            if name == image_name:
                self._add_image_task(image_entry)
                hailo_logger.info(f"Collected image '{image_name}' for app '{app_name}'")
                return

        hailo_logger.warning(
            f"Image '{image_name}' not found for app '{app_name}'"
        )


    def collect_specific_video_for_app(self, app_name: str, video_name: str):
        """Collect a specific video by name, restricted to a specific app."""
        videos_config = get_videos_for_app(app_name=app_name)

        for video_entry in videos_config:
            if isinstance(video_entry, dict):
                name = video_entry.get("name")
            elif isinstance(video_entry, str):
                name = Path(video_entry).name
            else:
                continue

            if name == video_name:
                self._add_video_task(video_entry)
                hailo_logger.info(f"Collected video '{video_name}' for app '{app_name}'")
                return

        hailo_logger.warning(
            f"Video '{video_name}' not found for app '{app_name}'"
        )

    def collect_all_json_files(self):
        """Collect all JSON download tasks from top-level json section."""
        if "json" in self.config:
            for json_entry in self.config["json"]:
                self._add_json_task(json_entry)
    def collect_all_npy_files(self):
        """Collect all NPY download tasks from top-level npy section."""
        if "npy" in self.config:
            for npy_entry in self.config["npy"]:
                self._add_npy_task(npy_entry)

    def collect_npy_by_tag(self, tag: str):
        """Collect NPY download tasks filtered by tag."""
        if "npy" not in self.config:
            return
        for npy_entry in self.config["npy"]:
            if isinstance(npy_entry, dict) and tag in npy_entry.get("tag", []):
                self._add_npy_task(npy_entry)

    def collect_models_for_app(
        self,
        app_name: str,
        include_extra: bool = False,
        is_gen_ai_allowed: bool = False
    ):
        """Collect model download tasks for a specific app."""
        app_config = self.config.get(app_name)
        if not isinstance(app_config, dict) or "models" not in app_config:
            hailo_logger.warning(f"App '{app_name}' not found or has no models")
            return
        
        models_config = app_config["models"]
        if self.hailo_arch not in models_config:
            hailo_logger.warning(f"App '{app_name}' has no models for {self.hailo_arch}")
            return
        
        arch_models = models_config[self.hailo_arch]
        models_found = False
        
        # Collect default model(s)
        if "default" in arch_models:
            default_model = arch_models["default"]
            if is_none_value(default_model):
                hailo_logger.warning(
                    f"⚠️  App '{app_name}' has no models available for {self.hailo_arch}. "
                    f"This app may only support other architectures (e.g., hailo10h for gen-ai apps)."
                )
            elif isinstance(default_model, list):
                for model_entry in default_model:
                    if is_valid_model_entry(model_entry):
                        self._add_model_task(model_entry, is_gen_ai_allowed)
                        models_found = True
            else:
                if is_valid_model_entry(default_model):
                    self._add_model_task(default_model, is_gen_ai_allowed)
                    models_found = True
        
        # Collect extra models if requested
        if include_extra and "extra" in arch_models:
            for model_entry in arch_models["extra"]:
                if is_valid_model_entry(model_entry):
                    self._add_model_task(model_entry, is_gen_ai_allowed)
                    models_found = True
        
        if models_found:
            hailo_logger.info(f"Collected models for app '{app_name}' ({self.hailo_arch})")
    
    def collect_all_default_models(
        self,
        include_extra: bool = False,
        exclude_gen_ai_apps: bool = True
    ):
        """Collect default (and optionally extra) models for all apps."""
        for app_name, app_config in self.config.items():
            if not isinstance(app_config, dict) or "models" not in app_config:
                continue
            
            # Check if this is a gen-ai app
            is_gen_ai_app = self._is_gen_ai_app(app_config)
            
            if exclude_gen_ai_apps and is_gen_ai_app:
                hailo_logger.debug(f"Skipping gen-ai app: {app_name}")
                continue
            
            self.collect_models_for_app(
                app_name,
                include_extra=include_extra,
                is_gen_ai_allowed=not exclude_gen_ai_apps
            )

    def collect_specific_model_for_app(self, app_name: str, model_name: str):
        """Collect a specific model by name, restricted to a specific app."""
        app_config = self.config.get(app_name)

        if not isinstance(app_config, dict) or "models" not in app_config:
            hailo_logger.warning(f"App '{app_name}' not found or has no models section")
            return

        models_config = app_config["models"]
        if self.hailo_arch not in models_config:
            hailo_logger.warning(
                f"Architecture '{self.hailo_arch}' not found for app '{app_name}'"
            )
            return

        arch_models = models_config[self.hailo_arch]

        # Check default model
        if "default" in arch_models:
            default_model = arch_models["default"]
            if self._find_and_add_model_by_name(default_model, model_name):
                return

        # Check extra models
        if "extra" in arch_models:
            for model_entry in arch_models["extra"]:
                if self._find_and_add_model_by_name(model_entry, model_name):
                    return

        hailo_logger.warning(
            f"Model '{model_name}' not found for app '{app_name}' "
            f"and architecture '{self.hailo_arch}'"
        )

    def collect_specific_onnx_for_app(self, app_name: str, onnx_name: str):
        """Collect a specific ONNX sidecar artifact for a specific app."""
        self._add_onnx_task(onnx_name)

    def collect_specific_model(self, model_name: str):
        """Collect a specific model by name."""
        for app_name, app_config in self.config.items():
            if not isinstance(app_config, dict) or "models" not in app_config:
                continue
            
            models_config = app_config["models"]
            if self.hailo_arch not in models_config:
                continue
            
            arch_models = models_config[self.hailo_arch]
            
            # Check default model
            if "default" in arch_models:
                default_model = arch_models["default"]
                if self._find_and_add_model_by_name(default_model, model_name):
                    return
            
            # Check extra models
            if "extra" in arch_models:
                for model_entry in arch_models["extra"]:
                    if self._find_and_add_model_by_name(model_entry, model_name):
                        return
        
        hailo_logger.warning(f"Model '{model_name}' not found for architecture {self.hailo_arch}")
    
    def _find_and_add_model_by_name(self, model_entry, target_name: str) -> bool:
        """Find and add a model if it matches the target name. Returns True if found."""
        if is_none_value(model_entry):
            return False
        
        if isinstance(model_entry, list):
            for entry in model_entry:
                if self._find_and_add_model_by_name(entry, target_name):
                    return True
            return False
        
        if isinstance(model_entry, dict):
            if model_entry.get("name") == target_name:
                self._add_model_task(model_entry, is_gen_ai_allowed=True)
                return True
        elif isinstance(model_entry, str) and model_entry == target_name:
            self._add_model_task(model_entry, is_gen_ai_allowed=True)
            return True
        
        return False
    
    def collect_group_resources(self, group_name: str):
        """Collect all resources for a specific group/app."""
        if group_name not in self.config:
            hailo_logger.error(f"Group '{group_name}' not found in config")
            available = [k for k in self.config.keys() if isinstance(self.config.get(k), dict)]
            hailo_logger.info(f"Available groups: {', '.join(available)}")
            return
        
        group_config = self.config[group_name]
        if not isinstance(group_config, dict):
            hailo_logger.error(f"Group '{group_name}' config is not a dictionary")
            return
        
        # Check if this is a gen-ai app (allow gen-ai models for explicit group downloads)
        is_gen_ai_app = self._is_gen_ai_app(group_config)
        
        # Collect models for this group
        self.collect_models_for_app(
            group_name,
            include_extra=True,
            is_gen_ai_allowed=is_gen_ai_app
        )
        
        # Collect videos, images, and JSON files (shared across all apps)
        self.collect_all_videos()
        self.collect_all_images()
        self.collect_all_json_files()
        self.collect_all_npy_files()
    
    def _is_gen_ai_app(self, app_config: dict) -> bool:
        """Check if an app is a gen-ai app."""
        if not isinstance(app_config, dict) or "models" not in app_config:
            return False
        
        models_config = app_config.get("models", {})
        for arch_models in models_config.values():
            if not isinstance(arch_models, dict):
                continue
            
            # Check default model
            if "default" in arch_models:
                if self._has_gen_ai_model(arch_models["default"]):
                    return True
            
            # Check extra models
            if "extra" in arch_models:
                for model_entry in arch_models["extra"]:
                    if self._has_gen_ai_model(model_entry):
                        return True
        
        return False
    
    def _has_gen_ai_model(self, model_entry) -> bool:
        """Check if a model entry is a gen-ai model."""
        if is_none_value(model_entry):
            return False
        
        if isinstance(model_entry, dict):
            return model_entry.get("source") == "gen-ai-mz"
        elif isinstance(model_entry, list):
            return any(
                isinstance(e, dict) and e.get("source") == "gen-ai-mz"
                for e in model_entry
            )
        return False
    
    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------
    
    def execute(self, parallel: bool = True) -> list[DownloadResult]:
        """
        Execute all collected download tasks.
        
        Args:
            parallel: If True, download files in parallel
        
        Returns:
            List of DownloadResult objects
        """
        if not self._tasks:
            hailo_logger.info("No download tasks to execute")
            return []
        
        hailo_logger.info(f"Executing {len(self._tasks)} download tasks...")
        
        if self.download_config.dry_run:
            hailo_logger.info("=== DRY RUN MODE ===")
        
        tasks_list = list(self._tasks)
        
        if parallel and len(tasks_list) > 1 and not self.download_config.dry_run:
            self._results = self._execute_parallel(tasks_list)
        else:
            self._results = self._execute_sequential(tasks_list)
        
        # Summary
        successful = sum(1 for r in self._results if r.success and not r.skipped)
        skipped = sum(1 for r in self._results if r.skipped)
        failed = sum(1 for r in self._results if not r.success)
        
        hailo_logger.info(
            f"Download summary: {successful} downloaded, {skipped} skipped, {failed} failed"
        )
        
        if failed > 0:
            hailo_logger.warning("Some downloads failed:")
            for result in self._results:
                if not result.success:
                    hailo_logger.warning(f"  - {result.task.name}: {result.message}")
            failures = "; ".join(
                f"{result.task.name}: {result.message}"
                for result in self._results if not result.success
            )
            raise RuntimeError(f"{failed} resource download(s) failed: {failures}")

        return self._results
    
    def _execute_sequential(self, tasks: list[DownloadTask]) -> list[DownloadResult]:
        """Execute tasks sequentially."""
        results = []
        for i, task in enumerate(tasks, 1):
            hailo_logger.info(f"[{i}/{len(tasks)}] Processing {task.name}...")
            result = self._download_file_with_retry(task)
            results.append(result)
        return results
    
    def _execute_parallel(self, tasks: list[DownloadTask]) -> list[DownloadResult]:
        """Execute tasks in parallel using thread pool."""
        results = []
        max_workers = min(self.download_config.parallel_workers, len(tasks))
        
        # Disable progress for parallel downloads (would be messy)
        original_show_progress = self.download_config.show_progress
        self.download_config.show_progress = False
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(self._download_file_with_retry, task): task
                    for task in tasks
                }
                
                completed = 0
                for future in as_completed(future_to_task):
                    completed += 1
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        results.append(result)
                        status = "✓" if result.success else "✗"
                        hailo_logger.info(
                            f"[{completed}/{len(tasks)}] {status} {task.name}"
                        )
                    except Exception as e:
                        hailo_logger.error(f"Unexpected error for {task.name}: {e}")
                        results.append(DownloadResult(
                            task=task,
                            success=False,
                            message=str(e),
                            skipped=False
                        ))
        finally:
            self.download_config.show_progress = original_show_progress
        
        return results
    
    def clear_tasks(self):
        """Clear all collected tasks."""
        self._tasks.clear()
        self._results.clear()
    
    @property
    def tasks(self) -> set[DownloadTask]:
        """Get current download tasks."""
        return self._tasks
    
    @property
    def results(self) -> list[DownloadResult]:
        """Get download results."""
        return self._results


# =============================================================================
# Legacy API Compatibility
# =============================================================================

def download_file(url: str, dest_path: Path, show_progress: bool = True):
    """Legacy function: Download a file from URL to destination path."""
    config = DownloadConfig(show_progress=show_progress)
    task = DownloadTask(
        url=url,
        dest_path=Path(dest_path),
        resource_type="unknown",
        name=Path(dest_path).name
    )
    
    # Create a minimal downloader just for this download
    downloader = ResourceDownloader(
        config={},
        hailo_arch=HAILO8_ARCH,
        resource_root=Path(dest_path).parent,
        download_config=config
    )
    result = downloader._download_file_with_retry(task)
    
    if not result.success:
        raise RuntimeError(result.message)


def download_group_resources(
    group_name: str,
    resource_config_path: str | None = None,
    arch: str | None = None
):
    """Legacy function: Download resources for a specific group/app."""
    cfg_path = Path(resource_config_path or DEFAULT_RESOURCES_CONFIG_PATH)
    if not cfg_path.is_file():
        hailo_logger.error(f"Config file not found at {cfg_path}")
        return
    
    config = load_config(cfg_path)
    
    hailo_arch = arch or detect_hailo_arch()
    if not hailo_arch:
        hailo_logger.error("Could not detect Hailo architecture.")
        print(
            "\n❌ ERROR: Could not detect Hailo device architecture.\n"
            "   Please ensure:\n"
            "   - A Hailo device is connected\n"
            "   - The HailoRT driver is installed and loaded\n"
            "   - You have permissions to access the device\n"
            "\n   Alternatively, specify the architecture manually with --arch (e.g., --arch hailo8)\n",
            file=sys.stderr
        )
        sys.exit(1)
    
    hailo_logger.info(f"Using Hailo architecture: {hailo_arch}")
    
    resource_root = Path(RESOURCES_ROOT_PATH_DEFAULT)
    
    downloader = ResourceDownloader(
        config=config,
        hailo_arch=hailo_arch,
        resource_root=resource_root
    )
    
    downloader.collect_group_resources(group_name)
    downloader.execute(parallel=True)


def _create_downloader(
    resource_config_path: str | None,
    arch: str | None,
    dry_run: bool,
    force: bool,
    include_gen_ai: bool,
):
    """
    Create and initialize a ResourceDownloader.

    Args:
        resource_config_path: Path to the resources config file. If None, use default.
        arch: Hailo architecture override ("hailo8", "hailo8l", "hailo10h"). Auto-detected if None.
        dry_run: If True, do not download files, only log planned actions.
        force: If True, force re-download even if files already exist.
        include_gen_ai: If True, allow downloading gen-ai models/resources.

    Returns:
        ResourceDownloader instance on success, None if configuration loading fails.
    """

    # ------------------------------------------------------------
    # Resolve and validate resources configuration path
    # ------------------------------------------------------------
    cfg_path = Path(resource_config_path or DEFAULT_RESOURCES_CONFIG_PATH)
    if not cfg_path.is_file():
        hailo_logger.error(f"Config file not found at {cfg_path}")
        return None

    # Load resources configuration
    config = load_config(cfg_path)
    hailo_logger.info(f"Using resource config from: {cfg_path}")

    # ------------------------------------------------------------
    # Detect or validate Hailo architecture
    # ------------------------------------------------------------
    hailo_arch = arch or detect_hailo_arch()
    if not hailo_arch:
        hailo_logger.error("Could not detect Hailo architecture.")
        print(
            "\n❌ ERROR: Could not detect Hailo device architecture.\n"
            "   Please ensure:\n"
            "   - A Hailo device is connected\n"
            "   - The HailoRT driver is installed and loaded\n"
            "   - You have permissions to access the device\n"
            "\n   Alternatively, specify the architecture manually with --arch (e.g., --arch hailo8)\n",
            file=sys.stderr
        )
        sys.exit(1)

    hailo_logger.info(f"Using Hailo architecture: {hailo_arch}")

    # ------------------------------------------------------------
    # Build download configuration
    # ------------------------------------------------------------
    download_config = DownloadConfig(
        dry_run=dry_run,              # Preview-only mode (no actual downloads)
        force_redownload=force,       # Re-download even if files already exist
        include_gen_ai=include_gen_ai # Allow gen-ai resources/models
    )

    # ------------------------------------------------------------
    # Create and return the downloader instance
    # ------------------------------------------------------------
    return ResourceDownloader(
        config=config,
        hailo_arch=hailo_arch,
        resource_root=Path(RESOURCES_ROOT_PATH_DEFAULT),
        download_config=download_config
    )


def download_resources(
    resource_config_path: str | None = None,
    arch: str | None = None,
    group: str | None = None,
    all_models: bool = False,
    resource_name: str | None = None,
    resource_type: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    parallel: bool = True,
    include_gen_ai: bool = False
):
    """
    Download resources based on the specified options.

    Args:
        resource_config_path: Path to resources config file
        arch: Hailo architecture override (hailo8, hailo8l, hailo10h)
        group: Specific group/app name to download resources for
        all_models: If True, download all models (default + extra) for all apps
        resource_name: Specific resource name to download
        resource_type: Type of the resource specified by `resource_name`, supported values: "model", "image", "video", "onnx".
        dry_run: If True, only show what would be downloaded
        force: If True, force re-download even if files exist
        parallel: If True, download files in parallel
        include_gen_ai: If True, include gen-ai models in downloads
    """
    load_environment()

    # ------------------------------------------------------------
    # Targeted mode:
    # Download exactly ONE resource (model OR image OR video).
    # This mode is strict and REQUIRES a valid group/app name.
    # ------------------------------------------------------------
    if resource_name:
        # resource_type must be provided
        if not resource_type:
            hailo_logger.error("resource_type must be specified with resource_name")
            return

        # Validate resource_type
        if resource_type not in RESOURCE_TYPES:
            hailo_logger.error(
                f"Invalid resource_type '{resource_type}'. "
                f"Supported types: {', '.join(RESOURCE_TYPES)}"
            )
            return

        # Group is mandatory for targeted mode
        if not group:
            hailo_logger.error("Targeted download requires a valid --group (app name).")
            return

        downloader = _create_downloader(
            resource_config_path,
            arch,
            dry_run,
            force,
            include_gen_ai
        )
        if downloader is None:
            return

        # Dispatch by resource type
        if resource_type == RESOURCE_TYPE_IMAGE:
            hailo_logger.info(f"Collecting specific image: {resource_name} (group={group})")
            downloader.collect_specific_image_for_app(group, resource_name)

        elif resource_type == RESOURCE_TYPE_VIDEO:
            hailo_logger.info(f"Collecting specific video: {resource_name} (group={group})")
            downloader.collect_specific_video_for_app(group, resource_name)

        elif resource_type == RESOURCE_TYPE_MODEL:
            hailo_logger.info(f"Collecting specific model: {resource_name} (group={group})")
            downloader.collect_specific_model_for_app(group, resource_name)

        elif resource_type == RESOURCE_TYPE_ONNX:
            hailo_logger.info(f"Collecting specific onnx: {resource_name} (group={group})")
            downloader.collect_specific_onnx_for_app(group, resource_name)

        downloader.execute(parallel=parallel)
        return

    # ------------------------------------------------------------
    # Group mode:
    # Download ALL resources for a specific group/app.
    # ------------------------------------------------------------
    if group and group.lower() != "default":
        download_group_resources(group, resource_config_path, arch)
        return

    # ------------------------------------------------------------
    # Bootstrap / default mode:
    # Download shared resources and default (or all) models.
    # ------------------------------------------------------------
    downloader = _create_downloader(
        resource_config_path,
        arch,
        dry_run,
        force,
        include_gen_ai
    )
    if downloader is None:
        return

    hailo_logger.info(f"Using Model Zoo version: {downloader.model_zoo_version}")

    # Collect shared resources used across apps
    hailo_logger.info("Collecting default resources: images, videos, and JSON files...")
    downloader.collect_all_videos()
    downloader.collect_all_images()
    downloader.collect_all_json_files()
    downloader.collect_all_npy_files()

    # Collect models according to the selected bootstrap mode
    if all_models:
        hailo_logger.info(f"Collecting all models for {downloader.hailo_arch}...")
        downloader.collect_all_default_models(
            include_extra=True,
            exclude_gen_ai_apps=not include_gen_ai
        )
    else:
        hailo_logger.info(f"Collecting default models for {downloader.hailo_arch}...")
        downloader.collect_all_default_models(
            include_extra=False,
            exclude_gen_ai_apps=not include_gen_ai
        )

    # Execute all queued downloads
    downloader.execute(parallel=parallel)


def list_models_for_arch(
    resource_config_path: str | None = None,
    arch: str | None = None,
    include_extra: bool = True
):
    """List all available models for a given architecture."""
    cfg_path = Path(resource_config_path or DEFAULT_RESOURCES_CONFIG_PATH)
    if not cfg_path.is_file():
        hailo_logger.error(f"Config file not found at {cfg_path}")
        return
    
    config = load_config(cfg_path)
    
    hailo_arch = arch or detect_hailo_arch()
    if not hailo_arch:
        hailo_logger.error("Could not detect Hailo architecture.")
        print(
            "\n❌ ERROR: Could not detect Hailo device architecture.\n"
            "   Please ensure:\n"
            "   - A Hailo device is connected\n"
            "   - The HailoRT driver is installed and loaded\n"
            "   - You have permissions to access the device\n"
            "\n   Alternatively, specify the architecture manually with --arch (e.g., --arch hailo8)\n",
            file=sys.stderr
        )
        sys.exit(1)
    
    print(f"\nAvailable models for architecture: {hailo_arch}\n")
    print("=" * 80)
    
    default_models = []
    extra_models = []
    
    for app_name, app_config in config.items():
        if not isinstance(app_config, dict) or "models" not in app_config:
            continue
        
        models_config = app_config["models"]
        if hailo_arch not in models_config:
            continue
        
        arch_models = models_config[hailo_arch]
        
        # Get default model
        if "default" in arch_models:
            default_model = arch_models["default"]
            if not is_none_value(default_model):
                if isinstance(default_model, list):
                    for entry in default_model:
                        if is_valid_model_entry(entry):
                            if isinstance(entry, dict):
                                source = entry.get("source", "mz")
                                name = entry.get("name")
                                default_models.append((app_name, name, source))
                            else:
                                default_models.append((app_name, entry, "mz"))
                elif isinstance(default_model, dict):
                    source = default_model.get("source", "mz")
                    name = default_model.get("name")
                    if name:
                        default_models.append((app_name, name, source))
                elif isinstance(default_model, str):
                    default_models.append((app_name, default_model, "mz"))
        
        # Get extra models
        if include_extra and "extra" in arch_models:
            for model_entry in arch_models["extra"]:
                if is_valid_model_entry(model_entry):
                    if isinstance(model_entry, dict):
                        source = model_entry.get("source", "mz")
                        name = model_entry.get("name")
                        extra_models.append((app_name, name, source))
                    elif isinstance(model_entry, str):
                        extra_models.append((app_name, model_entry, "mz"))
    
    # Print default models
    if default_models:
        print("\n📦 Default Models:")
        print("-" * 80)
        for app_name, model_name, source in sorted(default_models):
            print(f"  • {model_name:40s} [{source:10s}] (app: {app_name})")
    else:
        print("\n📦 Default Models: None")
    
    # Print extra models
    if include_extra and extra_models:
        print("\n📚 Extra Models:")
        print("-" * 80)
        for app_name, model_name, source in sorted(extra_models):
            print(f"  • {model_name:40s} [{source:10s}] (app: {app_name})")
    elif include_extra:
        print("\n📚 Extra Models: None")
    
    print("\n" + "=" * 80)
    total_msg = f"\nTotal: {len(default_models)} default model(s)"
    if include_extra:
        total_msg += f", {len(extra_models)} extra model(s)"
    print(total_msg)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download Hailo resources (models, videos, images, configs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default resources for detected architecture
  %(prog)s

  # Download all models (default + extra) for all apps
  %(prog)s --all

  # Download resources for a specific app
  %(prog)s --group detection

  # Download a specific resource (strict targeted mode)
  %(prog)s --group detection --resource-type model --resource-name yolov8m
  %(prog)s --group detection --resource-type image --resource-name bus.jpg
  %(prog)s --group detection --resource-type video --resource-name example.mp4

  # Download for a specific architecture
  %(prog)s --arch hailo10h

  # Preview what would be downloaded (dry run)
  %(prog)s --dry-run

  # Force re-download existing files
  %(prog)s --force

  # Download gen-ai app (auto-includes gen-ai models)
  %(prog)s --group vlm_chat --arch hailo10h

  # Include gen-ai apps in bulk download (downloads ALL apps including VLM, LLM, Whisper)
  %(prog)s --all --include-gen-ai --arch hailo10h
"""
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models (default + extra) for all apps"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_RESOURCES_CONFIG_PATH,
        help="Path to config file"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        choices=["hailo8", "hailo8l", "hailo10h"],
        help="Hailo architecture override"
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Group/app name to download resources for (e.g., detection, vlm_chat)"
    )
    parser.add_argument(
        "--resource-name",
        type=str,
        default=None,
        help="Specific resource name to download"
    )
    parser.add_argument(
        "--resource-type",
        type=str,
        default=None,
        choices=sorted(RESOURCE_TYPES),
        help="Type of the resource specified by --resource-name (model/image/video/onnx)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models for the detected/selected architecture"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel downloads (download sequentially)"
    )
    parser.add_argument(
        "--include-gen-ai",
        action="store_true",
        help="Include gen-ai apps in bulk downloads (not needed with --group, gen-ai models are auto-included when group has them)"
    )
    
    args = parser.parse_args()
    
    load_environment()
    
    # List models and exit
    if args.list_models:
        list_models_for_arch(
            resource_config_path=args.config,
            arch=args.arch,
            include_extra=True
        )
        return
    
    # Validate targeted mode arguments
    if args.resource_name and not args.resource_type:
        parser.error("--resource-name requires --resource-type")

    if args.resource_type and not args.resource_name:
        parser.error("--resource-type requires --resource-name")


    # Download resources
    download_resources(
        resource_config_path=args.config,
        arch=args.arch,
        group=args.group,
        all_models=args.all,
        resource_name=args.resource_name,
        resource_type=args.resource_type,
        dry_run=args.dry_run,
        force=args.force,
        parallel=not args.no_parallel,
        include_gen_ai=args.include_gen_ai
    )
    
    hailo_logger.info("Resource download completed.")


if __name__ == "__main__":
    main()
