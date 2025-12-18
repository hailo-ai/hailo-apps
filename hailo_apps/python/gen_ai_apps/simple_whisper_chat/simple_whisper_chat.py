import argparse
import sys
import wave
from pathlib import Path

import numpy as np

from hailo_platform import VDevice
from hailo_platform.genai import Speech2Text, Speech2TextTask

from hailo_apps.python.core.common.core import handle_list_models_flag, resolve_hef_path
from hailo_apps.python.core.common.defines import WHISPER_CHAT_APP, SHARED_VDEVICE_GROUP_ID, HAILO10H_ARCH
from hailo_apps.python.core.common.hailo_logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def main():
    """Main function for Whisper Speech-to-Text Example."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Whisper Speech-to-Text Example")
    parser.add_argument("--hef-path", type=str, default=None, help="Path to HEF model file")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (default: audio.wav in same directory)")

    # Handle --list-models flag before full initialization
    handle_list_models_flag(parser, WHISPER_CHAT_APP)

    args = parser.parse_args()

    # Resolve HEF path with auto-download (Whisper is Hailo-10H only)
    hef_path = resolve_hef_path(args.hef_path, app_name=WHISPER_CHAT_APP, arch=HAILO10H_ARCH)
    if hef_path is None:
        logger.error("Failed to resolve HEF path for Whisper model.")
        sys.exit(1)

    logger.info(f"Using HEF: {hef_path}")
    print(f"✓ Model file found: {hef_path}")

    vdevice = None
    speech2text = None

    try:
        print("\n[1/5] Initializing Hailo device...")
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        vdevice = VDevice(params)
        print("✓ Hailo device initialized")

        print("[2/5] Loading Whisper model...")
        speech2text = Speech2Text(vdevice, str(hef_path))
        print("✓ Model loaded successfully")

        # Load audio file using wave module
        if args.audio:
            audio_path = Path(args.audio)
        else:
            # Default to audio.wav in the same directory as the script
            script_dir = Path(__file__).parent
            audio_path = script_dir / 'audio.wav'

        print(f"[3/5] Loading audio file: {audio_path}")
        with wave.open(str(audio_path), 'rb') as wav_file:
            # Get audio parameters
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            # Read raw audio data
            raw_audio = wav_file.readframes(frames)

        duration = frames / sample_rate
        print(f"✓ Audio loaded (duration: {duration:.2f}s, sample rate: {sample_rate}Hz, channels: {channels})")

        print("[4/5] Preprocessing audio...")
        # Convert to numpy array based on sample width
        audio_data = np.frombuffer(raw_audio, dtype=np.int16)
        # Convert 16-bit to float32 and normalize
        audio_data = audio_data.astype(np.float32) / 32768.0
        # Ensure little-endian format as expected by the model
        audio_data = audio_data.astype('<f4')
        print("✓ Audio preprocessed (converted to float32, normalized)")

        print("[5/5] Transcribing audio with Whisper...")
        # Create generator parameters and generate segments
        segments = speech2text.generate_all_segments(
            audio_data=audio_data,
            task=Speech2TextTask.TRANSCRIBE,
            language="en",
            timeout_ms=15000)

        if segments and len(segments) > 0:
            # Combine all segments into a single transcription
            transcription = ''.join([seg.text for seg in segments])
            print(f"\n✓ Transcription completed ({len(segments)} segment(s)):")
            print("-" * 60)
            print(transcription.strip())
            print("-" * 60)
            print("\n✓ Example completed successfully")
        else:
            print("\n⚠ No transcription generated")

    except FileNotFoundError as e:
        logger.error(f"Audio file not found: {e}")
        sys.exit(1)
    except wave.Error as e:
        logger.error(f"Error reading WAV file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Clean up resources
        if speech2text:
            try:
                speech2text.release()
            except Exception as e:
                logger.warning(f"Error releasing Speech2Text: {e}")

        if vdevice:
            try:
                vdevice.release()
            except Exception as e:
                logger.warning(f"Error releasing VDevice: {e}")


if __name__ == "__main__":
    main()

