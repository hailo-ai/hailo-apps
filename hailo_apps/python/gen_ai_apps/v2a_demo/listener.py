import time
import inspect
import logging
import threading
import queue
from collections import deque
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np
import scipy.signal
import sounddevice as sd
import soundfile as sf

import openwakeword
from vad import VoiceActivityDetector

# Audio
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_DTYPE = "float32"
CHUNK_SIZE = 480  # 30ms at 16kHz
AUDIO_QUEUE_SIZE = 100  # ~3s buffer between mic thread and processing
STREAM_CONFIG = dict(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=AUDIO_DTYPE, blocksize=CHUNK_SIZE)
EMPTY_AUDIO = np.array([], dtype=AUDIO_DTYPE)

# Wake word detection
WAKE_WORD_THRESHOLD = 0.8
WAKE_WORD_CONSECUTIVE = 2
WAKE_SMOOTHING_FRAMES = 4
WAKE_WARMUP_FRAMES = 10  # early frames produce spurious scores

# Recording
VAD_AGGRESSIVENESS = 3
SILENCE_DURATION_MS = 800
MAX_RECORDING_S = 10.0
MIN_RECORDING_S = 0.5
PRE_ROLL_MS = 300

# Derived frame counts
SILENCE_FRAMES = int(SILENCE_DURATION_MS / 1000.0 * SAMPLE_RATE / CHUNK_SIZE)
MAX_RECORD_FRAMES = int(MAX_RECORDING_S * SAMPLE_RATE / CHUNK_SIZE)
MIN_RECORD_FRAMES = int(MIN_RECORDING_S * SAMPLE_RATE / CHUNK_SIZE)
PRE_ROLL_FRAMES = int(PRE_ROLL_MS / 1000.0 * SAMPLE_RATE / CHUNK_SIZE)

logger = logging.getLogger("v2a_demo")


class WakeWordListener:
    """Low-latency wake-word listener with live mic and file input support."""

    def __init__(self, wake_word_model: str, wake_timeout_s: Optional[float] = None,
                 audio_device: Optional[int] = None):
        path = Path(wake_word_model)
        if not path.exists():
            raise FileNotFoundError(f"Wake word model not found: {wake_word_model}")

        models_dir = Path(openwakeword.__file__).parent / "resources" / "models"
        melspec = models_dir / "melspectrogram.onnx"
        embed = models_dir / "embedding_model.onnx"

        if not melspec.exists() or not embed.exists():
            openwakeword.utils.download_models()

        # OpenWakeWord 0.4 uses ONNX exclusively and the older argument name.
        if "wakeword_models" in inspect.signature(openwakeword.Model).parameters:
            self._wake_word_model = openwakeword.Model(
                wakeword_models=[wake_word_model],
                inference_framework="onnx",
            )
        else:
            self._wake_word_model = openwakeword.Model(
                wakeword_model_paths=[wake_word_model],
            )
        self._wake_word_name = path.stem
        self._vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE, aggressiveness=VAD_AGGRESSIVENESS)
        self._wake_timeout_s = wake_timeout_s
        self._audio_device = audio_device
        logger.info(f"Wake word model loaded: {self._wake_word_name}")

    def listen(self) -> np.ndarray:
        """Block until wake word + speech recorded, return audio (or empty array on failure)."""
        self._validate_audio_device()

        audio_queue: queue.Queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)
        stop_event = threading.Event()

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio status: {status}")
            try:
                audio_queue.put_nowait(indata[:, 0].copy())
            except queue.Full:
                logger.warning("Audio queue full — dropping frame")

        def frame_generator():
            while not stop_event.is_set():
                try:
                    yield audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

        stream_kwargs = {**STREAM_CONFIG}
        if self._audio_device is not None:
            stream_kwargs["device"] = self._audio_device

        logger.info(f"Listening for wake word '{self._wake_word_name}'...")
        try:
            with sd.InputStream(callback=audio_callback, **stream_kwargs):
                return self._process_stream(frame_generator(), stop_event)
        except sd.PortAudioError as e:
            logger.error(f"Audio device error: {e}")
            return EMPTY_AUDIO

    def listen_from_file(self, audio_path: str) -> np.ndarray:
        """Process a pre-recorded audio file through the wake + VAD pipeline."""
        audio = self._load_audio(audio_path)

        def frame_generator():
            for i in range(0, len(audio) - CHUNK_SIZE, CHUNK_SIZE):
                yield audio[i:i + CHUNK_SIZE]

        return self._process_stream(frame_generator(), threading.Event())

    def _process_stream(self, frames: Generator[np.ndarray, None, None],
                        stop_event: threading.Event) -> np.ndarray:
        """Two-phase pipeline: wait for wake word, then record speech until silence."""
        pre_roll = self._wait_for_wake(frames)

        if pre_roll is None:
            stop_event.set()
            return EMPTY_AUDIO

        audio = self._record_speech(frames, pre_roll)

        stop_event.set()

        if len(audio) == 0:
            logger.info("No usable audio recorded")
            return EMPTY_AUDIO

        audio = self._trim_leading_silence(audio)

        if len(audio) < MIN_RECORDING_S * SAMPLE_RATE:
            logger.info("No usable audio recorded after trimming silence")
            return EMPTY_AUDIO

        logger.info(f"Captured {len(audio) / SAMPLE_RATE:.2f}s of audio")
        return audio

    def _wake_warmup(self, frames: Generator[np.ndarray, None, None]) -> deque:
        """Feed initial frames to flush spurious scores. Returns pre-roll buffer."""
        pre_roll = deque(maxlen=PRE_ROLL_FRAMES)
        for frame_count, chunk in enumerate(frames, 1):
            pre_roll.append(chunk)
            self._wake_word_model.predict(self._to_int16(chunk))
            if frame_count >= WAKE_WARMUP_FRAMES:
                break
        return pre_roll

    def _wait_for_wake(self, frames: Generator[np.ndarray, None, None]) -> Optional[List[np.ndarray]]:
        """Listen for wake word. Returns pre-roll chunks on detection, None on timeout."""
        self._wake_word_model.reset()
        pre_roll = self._wake_warmup(frames)

        consecutive = 0
        score_history: List[float] = []
        wake_start = time.monotonic()

        for chunk in frames:
            pre_roll.append(chunk)
            scores = self._wake_word_model.predict(self._to_int16(chunk))

            # Smooth scores over a sliding window to filter noise
            score = max(scores.values())
            score_history.append(score)
            if len(score_history) > WAKE_SMOOTHING_FRAMES:
                score_history.pop(0)
            smoothed = np.mean(score_history)

            # Require consecutive high-confidence frames to trigger
            if smoothed >= WAKE_WORD_THRESHOLD:
                consecutive += 1
                if consecutive >= WAKE_WORD_CONSECUTIVE:
                    logger.info(f"Wake word detected (score={smoothed:.3f})")
                    return list(pre_roll)
            else:
                consecutive = 0

            if (self._wake_timeout_s
                    and time.monotonic() - wake_start >= self._wake_timeout_s):
                logger.info("Wake word timeout reached")
                return None

        return None

    def _record_speech(self, frames: Generator[np.ndarray, None, None],
                       pre_roll_chunks: List[np.ndarray]) -> np.ndarray:
        """Record speech until silence detected. Returns audio array or empty."""
        self._vad.reset()
        recorded_chunks = list(pre_roll_chunks)
        recorded_frame_count = len(recorded_chunks)
        speech_started = False
        silence_frames = 0

        for chunk in frames:
            recorded_chunks.append(chunk)
            recorded_frame_count += 1

            if self._vad.process(self._to_int16(chunk)):
                speech_started = True
                silence_frames = 0
            elif speech_started:
                silence_frames += 1
                if silence_frames >= SILENCE_FRAMES:
                    break

            if recorded_frame_count >= MAX_RECORD_FRAMES:
                break

        if recorded_frame_count < MIN_RECORD_FRAMES:
            return EMPTY_AUDIO

        return np.concatenate(recorded_chunks).astype(AUDIO_DTYPE)

    def _validate_audio_device(self) -> None:
        """Raise RuntimeError if no usable input device is found."""
        try:
            sd.check_input_settings(
                device=self._audio_device,
                channels=STREAM_CONFIG["channels"],
                dtype=STREAM_CONFIG["dtype"],
                samplerate=STREAM_CONFIG["samplerate"],
            )
        except sd.PortAudioError as e:
            raise RuntimeError(
                f"No usable audio input device (device={self._audio_device}). "
                f"Check that a microphone is connected. PortAudio: {e}"
            ) from e

    def _trim_leading_silence(self, audio: np.ndarray) -> np.ndarray:
        """Remove leading silence to prevent Whisper hallucinations on quiet audio."""
        self._vad.reset()
        for i in range(0, len(audio) - CHUNK_SIZE, CHUNK_SIZE):
            chunk = audio[i:i + CHUNK_SIZE]
            if self._vad.process(self._to_int16(chunk)):
                # Keep a small margin before the first speech frame
                start = max(0, i - CHUNK_SIZE)
                return audio[start:]
        return audio

    @staticmethod
    def _to_int16(chunk: np.ndarray) -> np.ndarray:
        return (chunk * np.iinfo(np.int16).max).astype(np.int16)

    @staticmethod
    def _load_audio(audio_path: str) -> np.ndarray:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio, sr = sf.read(audio_path, dtype=AUDIO_DTYPE)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != SAMPLE_RATE:
            gcd = np.gcd(sr, SAMPLE_RATE)
            up = SAMPLE_RATE // gcd
            down = sr // gcd
            audio = scipy.signal.resample_poly(audio, up, down)

        return audio.astype(AUDIO_DTYPE)
