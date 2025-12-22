"""
Speech-to-Text module for Hailo Voice Assistant.

This module handles the transcription of audio using Hailo's Speech2Text model.
"""

from pathlib import Path

import numpy as np
from hailo_platform import VDevice
from hailo_platform.genai import Speech2Text, Speech2TextTask

from hailo_apps.python.core.common.core import resolve_hef_path
from hailo_apps.python.core.common.defines import (
    HAILO10H_ARCH,
    WHISPER_CHAT_APP,
)
from hailo_apps.python.core.common.hailo_logger import get_logger

logger = get_logger(__name__)


class SpeechToTextProcessor:
    """
    Handles speech-to-text transcription using Hailo's AI models.

    This class encapsulates the Speech2Text functionality, providing a simplified
    interface for transcribing audio data.
    """

    def __init__(self, vdevice: VDevice, hef_path: str | Path | None = None):
        """
        Initialize the SpeechToTextProcessor.

        Args:
            vdevice: The Hailo VDevice instance to use.
            hef_path: Optional path to the Whisper HEF model. If not provided,
                     resolves the default Whisper-Base model from whisper_chat app.
        """
        if hef_path is None:
            # Resolve default Whisper model using whisper_chat app config
            hef_path = resolve_hef_path(
                hef_path=None,
                app_name=WHISPER_CHAT_APP,
                arch=HAILO10H_ARCH
            )
            if hef_path is None:
                raise ValueError(
                    "Could not resolve Whisper HEF path. "
                    "Please run: hailo-download-resources --group whisper_chat"
                )

        logger.debug("Initializing Speech2Text with model: %s", hef_path)
        self.speech2text = Speech2Text(vdevice, str(hef_path))

    def transcribe(
        self,
        audio_data: np.ndarray,
        language: str = "en",
        timeout_ms: int = 15000,
    ) -> str:
        """
        Transcribe audio data to text.

        Args:
            audio_data: The raw audio data to transcribe.
            language: The language of the audio. Defaults to "en".
            timeout_ms: Timeout in milliseconds for the transcription. Defaults to 15000.

        Returns:
            The transcribed text.
        """
        segments = self.speech2text.generate_all_segments(
            audio_data=audio_data,
            task=Speech2TextTask.TRANSCRIBE,
            language=language,
            timeout_ms=timeout_ms,
        )

        if not segments:
            logger.debug("No transcription segments returned")
            return ""

        # Log first segment for debugging/feedback
        logger.debug("Transcription: text='%s', time=%.2fs", segments[0].text, segments[0].end_sec)

        full_text = "".join([seg.text for seg in segments])
        logger.debug("Full transcription: %d segments, %d chars", len(segments), len(full_text))
        return full_text
