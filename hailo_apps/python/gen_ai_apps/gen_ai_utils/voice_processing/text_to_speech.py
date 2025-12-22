"""
Text-to-Speech module for Hailo Voice Assistant.

This module handles speech synthesis using Piper TTS.
"""

import io
import logging
import os
import queue
import re
import threading
import time
import wave
from contextlib import redirect_stderr
from io import StringIO
from typing import Optional

import numpy as np
from piper import PiperVoice
from piper.voice import SynthesisConfig

from hailo_apps.python.core.common.defines import (
    TARGET_SR,
    TTS_JSON_PATH,
    TTS_LENGTH_SCALE,
    TTS_MODEL_NAME,
    TTS_NOISE_SCALE,
    TTS_ONNX_PATH,
    TTS_VOLUME,
    TTS_W_SCALE,
)
from .audio_player import AudioPlayer
from .audio_diagnostics import AudioDiagnostics

# Setup logger
logger = logging.getLogger(__name__)


class PiperModelNotFoundError(FileNotFoundError):
    """
    Exception raised when Piper TTS model files are not found.

    This exception should cause the application to exit, as TTS is a required
    component unless explicitly disabled with --no-tts flag.
    """
    pass


def check_piper_model_installed(onnx_path: str = TTS_ONNX_PATH, json_path: str = TTS_JSON_PATH) -> bool:
    """
    Check if Piper TTS model files are installed.

    Args:
        onnx_path (str): Path to the Piper TTS ONNX model file.
        json_path (str): Path to the Piper TTS JSON config file.

    Returns:
        bool: True if both model files exist, False otherwise.

    Raises:
        PiperModelNotFoundError: If model files are not found, with reference to documentation.
    """
    onnx_exists = os.path.exists(onnx_path)
    json_exists = os.path.exists(json_path)

    if not onnx_exists or not json_exists:
        missing_files = []
        if not onnx_exists:
            missing_files.append(onnx_path)
        if not json_exists:
            missing_files.append(json_path)

        error_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     PIPER TTS MODEL NOT FOUND                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
Please install the Piper TTS model before running this application.
For detailed installation instructions, see:
  hailo_apps/python/core/gen_ai_utils/voice_processing/README.md

Missing files:
{chr(10).join(f'  - {f}' for f in missing_files)}
"""
        raise PiperModelNotFoundError(error_msg)

    return True


def clean_text_for_tts(text: str) -> str:
    """
    Clean text for TTS to prevent artifacts and noise.

    Removes markdown formatting, special symbols, and characters that often cause
    issues with Piper TTS (like white noise).

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text safe for TTS.
    """
    if not text:
        return ""

    # 1. Remove Markdown formatting
    # Remove bold/italic asterisks/underscores (*, **, _, __)
    text = re.sub(r"[*_]{1,3}", "", text)
    # Remove code block backticks
    text = re.sub(r"`+", "", text)
    # Remove headers (#)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Remove links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # 2. Remove noisy characters
    # Filter out characters that are not:
    # - Alphanumeric (a-z, A-Z, 0-9, including accents/unicode letters)
    # - Basic punctuation (.,!?:;'-)
    # - Whitespace
    # - Currency symbols ($€£)
    # - Percent (%)
    # This regex keeps "word characters", spaces, and listed punctuation.
    # \w includes alphanumeric + underscore, but we stripped underscore above if it was markdown.
    # We allow underscore inside words if any remain, or we can be stricter.
    # Let's be permissive with \w but strip specific problematic symbols.

    # Common symbols causing noise: ~ @ ^ | \ < > { } [ ] #
    text = re.sub(r"[~@^|\\<>{}\[\]#]", " ", text)

    # 3. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


class TextToSpeechProcessor:
    """
    Handles text-to-speech synthesis and playback using Piper and AudioPlayer.
    """

    def __init__(self, onnx_path: str = TTS_ONNX_PATH, device_id: Optional[int] = None):
        """
        Initialize the TextToSpeechProcessor.

        Args:
            onnx_path (str): Path to the Piper TTS ONNX model.
            device_id (Optional[int]): Audio device ID for playback. If None, uses
                                     saved preferences or auto-detects best device.

        Raises:
            PiperModelNotFoundError: If Piper model files are not found.
        """
        # Check if Piper model is installed
        json_path = onnx_path + ".json"
        check_piper_model_installed(onnx_path, json_path)

        logger.debug("Loading Piper TTS model: %s", onnx_path)
        # Suppress Piper warning messages
        with redirect_stderr(StringIO()):
            self.piper_voice = PiperVoice.load(onnx_path)
            self.syn_config = SynthesisConfig(
                volume=TTS_VOLUME,
                length_scale=TTS_LENGTH_SCALE,
                noise_scale=TTS_NOISE_SCALE,
                noise_w_scale=TTS_W_SCALE,
                normalize_audio=True,
            )
        logger.debug("TTS initialized with volume=%.2f, length_scale=%.2f", TTS_VOLUME, TTS_LENGTH_SCALE)

        # Use preferred device if not explicitly provided
        if device_id is None:
            _, device_id = AudioDiagnostics.get_preferred_devices()
            if device_id is not None:
                logger.debug("Using preferred output device: %d", device_id)

        self.audio_player = AudioPlayer(device_id=device_id)

        self.speech_queue = queue.Queue()
        self._speech_lock = threading.Lock()
        self.generation_id = 0
        self._gen_id_lock = threading.Lock()
        self._interrupted = threading.Event()
        self._running = True

        # Start the background worker for speech synthesis and playback
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()

    def interrupt(self):
        """
        Interrupts any ongoing speech.

        Stops current playback, increments generation ID to invalidate stale chunks,
        and clears the queue.
        """
        logger.debug("Interrupting TTS")
        self._interrupted.set()
        with self._gen_id_lock:
            self.generation_id += 1

        with self._speech_lock:
            self.audio_player.stop()

        # Drain the queue of any stale audio chunks
        drained = 0
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
                drained += 1
            except queue.Empty:
                continue
        if drained > 0:
            logger.debug("Drained %d stale audio chunks from queue", drained)

    def queue_text(self, text: str, gen_id: Optional[int] = None):
        """
        Add text to the speech queue.

        Args:
            text (str): The text to speak.
            gen_id (Optional[int]): Generation ID. If None, uses current ID.
        """
        if gen_id is None:
            with self._gen_id_lock:
                gen_id = self.generation_id
        self.speech_queue.put((gen_id, text))

    def chunk_and_queue(self, buffer: str, gen_id: int, is_first_chunk: bool) -> str:
        """
        Chunk text buffer based on delimiters and queue for speech.

        Args:
            buffer (str): The accumulated text buffer.
            gen_id (int): The generation ID for the speech.
            is_first_chunk (bool): Whether this is the first chunk being processed.

        Returns:
            str: The remaining buffer after queuing complete chunks.
        """
        # For first chunk, use more delimiters for faster start
        # After first chunk, use only sentence-ending punctuation
        if is_first_chunk:
            delimiters = ['.', '?', '!', ',', ':', ';', '-']
            min_chars_before_force = 15  # ~3-4 words
        else:
            delimiters = ['.', '?', '!']
            min_chars_before_force = 0  # Don't force split after first chunk

        while True:
            # Find the first occurrence of any delimiter
            positions = {buffer.find(d): d for d in delimiters if buffer.find(d) != -1}

            # Latency optimization: Force split at word boundary if buffer is long enough
            if is_first_chunk and not positions and len(buffer) >= min_chars_before_force:
                last_space = buffer.rfind(' ')
                if last_space > 5:  # At least a few characters
                    positions[last_space] = ' '

            if not positions:
                break  # No delimiters found

            first_pos = min(positions.keys())
            chunk = buffer[:first_pos + 1]

            if chunk.strip():
                self.queue_text(chunk.strip(), gen_id)
                # After sending first chunk, switch to sentence-based chunking
                is_first_chunk = False
                delimiters = ['.', '?', '!']

            buffer = buffer[first_pos + 1:]

        return buffer

    def get_current_gen_id(self) -> int:
        """Get the current generation ID."""
        with self._gen_id_lock:
            return self.generation_id

    def clear_interruption(self):
        """Clear the interruption flag."""
        self._interrupted.clear()

    def stop(self):
        """Stop the worker thread and cleanup."""
        logger.debug("Stopping TTS processor")
        self._running = False
        if self.speech_thread.is_alive():
            self.speech_thread.join(timeout=1.0)

        self.audio_player.close()

    def _speech_worker(self):
        """
        Background thread that processes the speech queue.
        """
        logger.debug("TTS worker thread started")
        while self._running:
            try:
                gen_id, text = self.speech_queue.get(timeout=0.1)

                # If an interruption is signaled, discard this chunk
                if self._interrupted.is_set():
                    logger.debug("Discarding chunk due to interruption")
                    self.speech_queue.task_done()
                    continue

                # If this chunk is from a previous generation, discard it
                with self._gen_id_lock:
                    if gen_id != self.generation_id:
                        logger.debug("Discarding stale chunk (gen_id=%d, current=%d)", gen_id, self.generation_id)
                        self.speech_queue.task_done()
                        continue

                self._synthesize_and_play(text)
                self.speech_queue.task_done()

            except queue.Empty:
                time.sleep(0.1)
        logger.debug("TTS worker thread stopped")

    def _synthesize_and_play(self, text: str):
        """
        Synthesizes text to audio and plays it using AudioPlayer.

        Args:
            text (str): The text to be spoken.
        """
        # Clean text before synthesis to prevent artifacts
        text = clean_text_for_tts(text)
        if not text.strip():
            logger.debug("Empty text after cleaning, skipping synthesis")
            return

        logger.debug("Synthesizing text: %s", text[:50] + "..." if len(text) > 50 else text)

        try:
            # Synthesize to in-memory WAV buffer
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                with redirect_stderr(StringIO()):
                    self.piper_voice.synthesize_wav(
                        text, wav_file, self.syn_config
                    )

            # Convert WAV buffer to numpy array for playback
            wav_buffer.seek(0)

            # We can use soundfile or wave to read it back, or just use wave manually
            # Since AudioPlayer has _read_wav logic, let's just re-implement simple parsing here or use a helper
            # Actually, AudioPlayer.play supports reading from file path, but not bytes buffer directly yet.
            # But wait, AudioPlayer supports numpy array.

            # Parse WAV from buffer
            with wave.open(wav_buffer, 'rb') as wf:
                n_frames = wf.getnframes()
                data = wf.readframes(n_frames)
                width = wf.getsampwidth()
                fs = wf.getframerate()

                # Assume mono 16-bit based on Piper config
                if width == 2:
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    # Fallback or error
                    logger.error(f"Unexpected sample width from synthesis: {width}")
                    return

            with self._speech_lock:
                # Check if we should still play (might have been interrupted during synthesis)
                if self._interrupted.is_set():
                    logger.debug("Interrupted during synthesis, skipping playback")
                    return

                # Play synchronously (blocking this worker thread)
                logger.debug("Playing synthesized audio: %d samples (%.2f seconds)",
                           len(audio_data), len(audio_data) / TARGET_SR)
                self.audio_player.play(audio_data, block=True)
                logger.debug("Audio playback completed")

        except Exception as e:
            logger.error("TTS synthesis/playback failed: %s", e)
