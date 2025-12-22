"""Main app for Hailo Whisper"""

import time
import os
import sys


def check_whisper_dependencies():
    """
    Check if all required Whisper dependencies are installed.
    
    Exits the program with installation instructions if any dependencies are missing.
    """
    missing_deps = []
    for dep_name in ["transformers", "sounddevice", "torch", "streamlit"]:
        try:
            __import__(dep_name)
        except ImportError:
            missing_deps.append(dep_name)

    if missing_deps:
        print("\n" + "="*70)
        print("❌ MISSING REQUIRED DEPENDENCIES")
        print("="*70)
        print("\nThe following dependencies are required but not installed:")
        for dep in missing_deps:
            print(f"  • {dep}")
        print("\n" + "-"*70)
        print("INSTALLATION INSTRUCTIONS:")
        print("-"*70)
        print("\nTo install all dependencies (recommended):")
        print("  1. Navigate to the repository root directory")
        print("  2. Run: pip install -e \".[speech-rec]\"")
        print("\n" + "="*70)
        sys.exit(1)


# Check dependencies before importing modules that depend on them
check_whisper_dependencies()

try:
    from hailo_apps.python.standalone_apps.speech_recognition.app.hailo_whisper_pipeline import (
        HailoWhisperPipeline,
    )
    from hailo_apps.python.standalone_apps.speech_recognition.common.audio_utils import load_audio
    from hailo_apps.python.standalone_apps.speech_recognition.common.preprocessing import (
        preprocess,
        improve_input_audio,
    )
    from hailo_apps.python.standalone_apps.speech_recognition.common.postprocessing import (
        clean_transcription,
    )
    from hailo_apps.python.standalone_apps.speech_recognition.common.record_utils import record_audio
    from hailo_apps.python.standalone_apps.speech_recognition.app.whisper_hef_registry import (
        HEF_REGISTRY,
    )
    from hailo_apps.python.core.common.parser import get_standalone_parser
    from hailo_apps.python.core.common.toolbox import resolve_arch
except ImportError:
    from pathlib import Path

    speech_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(speech_root))
    core_dir = Path(__file__).resolve().parents[3] / "core"
    sys.path.insert(0, str(core_dir))
    from app.hailo_whisper_pipeline import HailoWhisperPipeline
    from common.audio_utils import load_audio
    from common.preprocessing import preprocess, improve_input_audio
    from common.postprocessing import clean_transcription
    from common.record_utils import record_audio
    from app.whisper_hef_registry import HEF_REGISTRY
    from common.parser import get_standalone_parser
    from common.toolbox import resolve_arch


def get_args():
    """
    Initialize and run the argument parser.

    Return:
        argparse.Namespace: Parsed arguments.
    """
    parser = get_standalone_parser()
    parser.description = "Whisper Hailo Pipeline"
    # Don't set a default here - let resolve_arch() handle auto-detection
    parser.add_argument(
        "--reuse-audio", 
        action="store_true", 
        help="Reuse the previous audio file (sampled_audio.wav)"
    )
    parser.add_argument(
        "--hw-arch",
        dest="arch",
        type=str,
        choices=["hailo8", "hailo8l", "hailo10h"],
        default=None,
        help="Hardware architecture to use (alias for --arch, will auto-detect if not specified)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="base",
        choices=["base", "tiny", "tiny.en"],
        help="Whisper variant to use (default: base)"
    )
    parser.add_argument(
        "--multi-process-service", 
        action="store_true", 
        help="Enable multi-process service to run other models in addition to Whisper"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Recording duration in seconds (default: 10 seconds)"
    )
    args = parser.parse_args()
    # Preserve backwards compatibility with previous flag naming
    args.hw_arch = args.arch
    return args


def get_hef_path(model_variant: str, hw_arch: str, component: str) -> str:
    """
    Method to retrieve HEF path.

    Args:
        model_variant (str): e.g. "tiny", "base"
        hw_arch (str): e.g. "hailo8", "hailo8l"
        component (str): "encoder" or "decoder"

    Returns:
        str: Absolute path to the requested HEF file.
    """
    try:
        hef_path = HEF_REGISTRY[model_variant][hw_arch][component]
    except KeyError as e:
        raise FileNotFoundError(
            f"HEF not available for model '{model_variant}' on hardware '{hw_arch}'."
        ) from e

    if not os.path.exists(hef_path):
        from pathlib import Path
        download_script = Path(__file__).parent / "download_resources.py"
        raise FileNotFoundError(
            f"HEF file not found at: {hef_path}\n\n"
            f"To download the required HEF files, run:\n"
            f"  python3 {download_script} --hw-arch {hw_arch}"
        )
    return hef_path


def main():
    """
    Main function to run the Hailo Whisper pipeline.
    """
    # Get command line arguments
    args = get_args()
    
    # Resolve architecture (auto-detect if not specified)
    args.arch = resolve_arch(args.arch)
    args.hw_arch = args.arch

    variant = args.variant
    print(f"Selected variant: Whisper {variant}")
    print(f"Using hardware architecture: {args.arch}")
    encoder_path = get_hef_path(variant, args.arch, "encoder")
    decoder_path = get_hef_path(variant, args.arch, "decoder")

    whisper_hailo = HailoWhisperPipeline(encoder_path, decoder_path, variant, multi_process_service=args.multi_process_service)
    print("Hailo Whisper pipeline initialized.")
    audio_path = "sampled_audio.wav"
    is_nhwc = True

    chunk_length = whisper_hailo.get_model_input_audio_length()

    while True:
        if args.reuse_audio:
            # Reuse the previous audio file
            if not os.path.exists(audio_path):
                print(f"Audio file {audio_path} not found. Please record audio first.")
                break
        else:
            user_input = input("\nPress Enter to start recording, or 'q' to quit: ")
            if user_input.lower() == "q":
                break
            # Record audio
            sampled_audio = record_audio(args.duration, audio_path=audio_path)

        # Process audio
        sampled_audio = load_audio(audio_path)

        sampled_audio, start_time = improve_input_audio(sampled_audio, vad=True)
        if start_time is None:
            print("No speech detected in the audio. Please try recording again with clearer audio.")
            continue
        chunk_offset = start_time - 0.2
        if chunk_offset < 0:
            chunk_offset = 0

        mel_spectrograms = preprocess(
            sampled_audio,
            is_nhwc=is_nhwc,
            chunk_length=chunk_length,
            chunk_offset=chunk_offset
        )

        for mel in mel_spectrograms:
            whisper_hailo.send_data(mel)
            time.sleep(0.1)
            transcription = clean_transcription(whisper_hailo.get_transcription())
            print(f"\n{transcription}")

        if args.reuse_audio:
            break  # Exit the loop if reusing audio

    whisper_hailo.stop()


if __name__ == "__main__":
    main()
