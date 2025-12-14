"""
Audio Player module.

Handles audio playback using sounddevice OutputStream for continuous streaming.
"""

import logging
import os
import queue
import threading
import time
import wave
from typing import Optional, Union

import numpy as np
import sounddevice as sd

from hailo_apps.python.core.common.defines import TARGET_SR
from .audio_diagnostics import AudioDiagnostics

# Setup logger
logger = logging.getLogger(__name__)

# Audio chunk size for writing (smaller = more responsive, larger = less jitter)
WRITE_CHUNK_SIZE = 8192  # ~0.5 seconds at 16kHz


class AudioPlayer:
    """
    Handles audio playback using sounddevice OutputStream.
    Uses a persistent stream and queue to ensure gapless playback of chunks.
    """

    def __init__(self, device_id: Optional[int] = None):
        """
        Initialize the player.

        Args:
            device_id (Optional[int]): Device ID to use. If None, auto-detects best device.
        """
        self.stream = None
        self.queue = queue.Queue()
        self._playback_thread = None
        self._stop_event = threading.Event()
        self._reinit_event = threading.Event()
        self._stream_lock = threading.Lock()

        # Suppress stderr at startup and keep it suppressed for this player
        self._devnull_fd = None
        self._original_stderr_fd = None

        # Select device
        if device_id is None:
            _, self.device_id = AudioDiagnostics.auto_detect_devices()
            if self.device_id is None:
                logger.warning("No output device found during auto-detection. Will use system default.")
            else:
                # Verify device supports output
                try:
                    devices = sd.query_devices()
                    if self.device_id < len(devices):
                        device_info = devices[self.device_id]
                        if device_info['max_output_channels'] == 0:
                            logger.warning("Auto-detected device %d (%s) does not support output channels. "
                                         "Falling back to system default.",
                                         self.device_id, device_info.get('name', 'unknown'))
                            self.device_id = None
                        else:
                            logger.info("Verified output device %d (%s) supports %d output channels",
                                      self.device_id, device_info.get('name', 'unknown'),
                                      device_info['max_output_channels'])
                except Exception as e:
                    logger.warning("Failed to verify output device: %s", e)
        else:
            self.device_id = device_id

        logger.info("Initialized AudioPlayer with device_id=%s", self.device_id)

        # Start persistent playback worker
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()

    def play(self, audio_data: Union[str, np.ndarray], block: bool = False):
        """
        Queue audio data for playback.

        Args:
            audio_data (Union[str, np.ndarray]): Path to WAV file or numpy array.
            block (bool): Ignored in this streaming implementation. Kept for API compatibility.
        """
        if isinstance(audio_data, str):
            try:
                data, fs = self._read_wav(audio_data)
                if fs != TARGET_SR:
                    logger.warning("WAV sample rate %d does not match target %d.", fs, TARGET_SR)
            except Exception as e:
                logger.error("Failed to read WAV file: %s", e)
                return
        elif isinstance(audio_data, np.ndarray):
            data = audio_data
        else:
            logger.error("Unsupported audio data type: %s", type(audio_data))
            return

        # Ensure float32
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        # Enqueue for playback
        logger.debug("Queuing audio data for playback: %d samples", len(data))
        self.queue.put(data)

    def stop(self):
        """
        Clear the playback queue and stop audio immediately.
        """
        # Clear queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break

        # Abort stream
        with self._stream_lock:
            if self.stream:
                try:
                    self.stream.abort()
                except Exception:
                    pass

        # Global stop
        try:
            sd.stop()
        except Exception:
            pass

        # Signal reinit
        self._reinit_event.set()

        # Brief wait for hardware
        time.sleep(0.02)

    def close(self):
        """Shutdown the player and release resources."""
        self._stop_event.set()
        self._reinit_event.set()

        with self._stream_lock:
            if self.stream:
                try:
                    self.stream.abort()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None

        if self._playback_thread:
            self._playback_thread.join(timeout=1.0)

    def _suppress_stderr(self):
        """Redirect stderr to /dev/null."""
        try:
            self._original_stderr_fd = os.dup(2)
            self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self._devnull_fd, 2)
        except Exception:
            pass

    def _restore_stderr(self):
        """Restore stderr."""
        try:
            if self._original_stderr_fd is not None:
                os.dup2(self._original_stderr_fd, 2)
                os.close(self._original_stderr_fd)
                self._original_stderr_fd = None
            if self._devnull_fd is not None:
                os.close(self._devnull_fd)
                self._devnull_fd = None
        except Exception:
            pass

    def _create_stream(self):
        """Create a new output stream."""
        try:
            stream = sd.OutputStream(
                samplerate=TARGET_SR,
                device=self.device_id,
                channels=1,
                dtype='float32',
                blocksize=WRITE_CHUNK_SIZE,
                latency=0.3  # 300ms buffer to prevent underruns
            )
            stream.start()
            logger.debug("Audio output stream created successfully (device_id=%s, active=%s)",
                       self.device_id, stream.active)
            return stream
        except Exception as e:
            logger.error("Failed to create audio output stream (device_id=%s): %s", self.device_id, e)
            raise

    def _playback_worker(self):
        """Worker thread that writes to the OutputStream."""
        # Suppress stderr for the entire worker thread lifetime
        self._suppress_stderr()

        try:
            while not self._stop_event.is_set():
                # Create/recreate stream
                try:
                    with self._stream_lock:
                        if self.stream:
                            try:
                                self.stream.close()
                            except Exception:
                                pass
                        self.stream = self._create_stream()
                        self._reinit_event.clear()

                    logger.debug("Audio stream initialized successfully (device_id=%s)", self.device_id)

                except Exception as e:
                    logger.error("Failed to initialize audio stream (device_id=%s): %s", self.device_id, e)
                    logger.debug("Stream initialization error traceback:", exc_info=True)
                    time.sleep(0.5)
                    continue

                # Process audio queue
                while not self._stop_event.is_set() and not self._reinit_event.is_set():
                    try:
                        # Get audio data
                        try:
                            data = self.queue.get(timeout=0.05)
                        except queue.Empty:
                            time.sleep(0)  # Yield GIL
                            continue

                        # Write in chunks for responsiveness
                        offset = 0
                        data_len = len(data)
                        logger.debug("Playing audio chunk: %d samples", data_len)

                        while offset < len(data):
                            if self._reinit_event.is_set() or self._stop_event.is_set():
                                break

                            chunk = data[offset:offset + WRITE_CHUNK_SIZE]
                            offset += WRITE_CHUNK_SIZE

                            if self.stream and self.stream.active:
                                try:
                                    self.stream.write(chunk)
                                except Exception as write_error:
                                    logger.error("Error writing to audio stream: %s", write_error)
                                    raise
                            else:
                                logger.warning("Audio stream not active (stream=%s, active=%s)",
                                              self.stream is not None,
                                              self.stream.active if self.stream else False)

                            time.sleep(0)  # Yield GIL

                        logger.debug("Finished playing audio chunk: %d samples", data_len)
                        self.queue.task_done()

                    except Exception as e:
                        logger.error("Playback error (will reinit): %s", e)
                        logger.debug("Playback error traceback:", exc_info=True)
                        break

            # Cleanup
            with self._stream_lock:
                if self.stream:
                    try:
                        self.stream.close()
                    except Exception:
                        pass
                    self.stream = None
        finally:
            self._restore_stderr()

    def _read_wav(self, file_path: str) -> tuple[np.ndarray, int]:
        """Read WAV file to numpy array."""
        with wave.open(file_path, 'rb') as wf:
            fs = wf.getframerate()
            n_frames = wf.getnframes()
            data = wf.readframes(n_frames)

            width = wf.getsampwidth()
            if width == 2:
                dtype = np.int16
            elif width == 4:
                dtype = np.float32
            else:
                raise ValueError(f"Unsupported sample width: {width}")

            audio = np.frombuffer(data, dtype=dtype)

            if dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0

            return audio, fs
