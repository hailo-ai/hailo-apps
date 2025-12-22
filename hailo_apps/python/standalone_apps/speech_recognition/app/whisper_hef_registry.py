import os
from pathlib import Path

# Get the directory where this registry file is located (app/ directory)
APP_DIR = Path(__file__).resolve().parent

HEF_REGISTRY = {
    "base": {
        "hailo8": {
            "encoder": str(APP_DIR / "hefs/h8/base/base-whisper-encoder-5s.hef"),
            "decoder": str(APP_DIR / "hefs/h8/base/base-whisper-decoder-fixed-sequence-matmul-split.hef"),
        },
        "hailo8l": {
            "encoder": str(APP_DIR / "hefs/h8l/base/base-whisper-encoder-5s_h8l.hef"),
            "decoder": str(APP_DIR / "hefs/h8l/base/base-whisper-decoder-fixed-sequence-matmul-split_h8l.hef"),
        },
        "hailo10h": {
            "encoder": str(APP_DIR / "hefs/h10h/base/base-whisper-encoder-10s.hef"),
            "decoder": str(APP_DIR / "hefs/h10h/base/base-whisper-decoder-10s-out-seq-64.hef"),
        }

    },
    "tiny": {
        "hailo8": {
            "encoder": str(APP_DIR / "hefs/h8/tiny/tiny-whisper-encoder-10s_15dB.hef"),
            "decoder": str(APP_DIR / "hefs/h8/tiny/tiny-whisper-decoder-fixed-sequence-matmul-split.hef"),
        },
        "hailo8l": {
            "encoder": str(APP_DIR / "hefs/h8l/tiny/tiny-whisper-encoder-10s_15dB_h8l.hef"),
            "decoder": str(APP_DIR / "hefs/h8l/tiny/tiny-whisper-decoder-fixed-sequence-matmul-split_h8l.hef"),
        },
        "hailo10h": {
            "encoder": str(APP_DIR / "hefs/h10h/tiny/tiny-whisper-encoder-10s.hef"),
            "decoder": str(APP_DIR / "hefs/h10h/tiny/tiny-whisper-decoder-fixed-sequence.hef"),
        }
    },
    "tiny.en": {
        "hailo10h": {
                "encoder": str(APP_DIR / "hefs/h10h/tiny.en/tiny_en-whisper-encoder-10s.hef"),
                "decoder": str(APP_DIR / "hefs/h10h/tiny.en/tiny_en-whisper-decoder-fixed-sequence.hef"),
        }
    }
}