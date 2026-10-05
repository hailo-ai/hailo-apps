"""Hailo Voice-to-Action Demo — entry point.

Listens for a wake word, captures speech, and processes it through
the full pipeline: STT -> Tool Selection -> LLM -> Tool Execution -> TTS.
"""

import argparse
import logging
import sys
from pathlib import Path

from pipeline import V2APipeline
from listener import WakeWordListener

repo_root = None
for p in Path(__file__).resolve().parents:
    if (p / "hailo_apps" / "config" / "config_manager.py").exists():
        repo_root = p
        break
if repo_root is not None:
    sys.path.insert(0, str(repo_root))

RESOURCES_DIR = Path(__file__).resolve().parent / "resources"


def configure_logger(debug: bool):
    logger = logging.getLogger("v2a_demo")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)

    fmt = "%(asctime)s | %(levelname)-5s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    logger.handlers.clear()
    logger.addHandler(handler)

    return logger


def create_parser():
    parser = argparse.ArgumentParser(description="Hailo Voice to Action Demo")

    parser.add_argument(
        "--wake-word-model",
        default=str(RESOURCES_DIR / "hey_hailo_v3.onnx"),
        help="Path to wake word model"
    )

    parser.add_argument(
        "--audio-input-path",
        help="Path to pre-recorded audio file"
    )

    parser.add_argument(
        "--audio-output-path",
        help="Path to save TTS output audio file"
    )

    parser.add_argument(
        "--audio-device",
        type=int,
        default=None,
        help="PortAudio device index for microphone input (see: python -m sounddevice)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logs"
    )

    return parser


def main():
    args = create_parser().parse_args()
    logger = configure_logger(args.debug)

    listener = WakeWordListener(
        wake_word_model=args.wake_word_model,
        audio_device=args.audio_device,
    )

    with V2APipeline(tts_output_path=args.audio_output_path) as pipeline:
        if args.audio_input_path:
            logger.info("Processing input audio file...")
            audio = listener.listen_from_file(args.audio_input_path)
            if len(audio) > 0:
                pipeline.process_audio(audio)
            else:
                logger.info("No wake word / speech found in the input file")
        else:
            logger.info("Starting continuous listening mode (Ctrl+C to exit)")
            try:
                while True:
                    audio = listener.listen()
                    if len(audio) > 0:
                        try:
                            pipeline.process_audio(audio)
                        except Exception as e:
                            logger.exception(f"Pipeline error: {e}")
            except KeyboardInterrupt:
                logger.info("Shutdown requested by user.")


if __name__ == "__main__":
    main()
