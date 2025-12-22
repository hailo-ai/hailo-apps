#!/usr/bin/env python3
import os, subprocess
import sys
from pathlib import Path

try:
    from hailo_apps.python.core.common.parser import get_standalone_parser
except ImportError:
    core_dir = Path(__file__).resolve().parents[3] / "core"
    sys.path.insert(0, str(core_dir))
    from common.parser import get_standalone_parser

BASE_HEF = "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/whisper"
BASE_ASSETS = "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/npy%20files/whisper/decoder_assets"

FILES = {
    "hefs": {
        "hailo8": {
            "tiny": [
                f"{BASE_HEF}/h8/tiny-whisper-decoder-fixed-sequence-matmul-split.hef",
                f"{BASE_HEF}/h8/tiny-whisper-encoder-10s_15dB.hef",
            ],
            "base": [
                f"{BASE_HEF}/h8/base-whisper-decoder-fixed-sequence-matmul-split.hef",
                f"{BASE_HEF}/h8/base-whisper-encoder-5s.hef",
            ],
        },
        "hailo8l": {
            "tiny": [
                f"{BASE_HEF}/h8l/tiny-whisper-decoder-fixed-sequence-matmul-split_h8l.hef",
                f"{BASE_HEF}/h8l/tiny-whisper-encoder-10s_15dB_h8l.hef",
            ],
            "base": [
                f"{BASE_HEF}/h8l/base-whisper-decoder-fixed-sequence-matmul-split_h8l.hef",
                f"{BASE_HEF}/h8l/base-whisper-encoder-5s_h8l.hef",
            ],
        },
        "hailo10h": {
            "base": [
                f"{BASE_HEF}/h10h/base-whisper-decoder-10s-out-seq-64.hef",
                f"{BASE_HEF}/h10h/base-whisper-encoder-10s.hef",
            ],
            "tiny": [
                f"{BASE_HEF}/h10h/tiny-whisper-decoder-fixed-sequence.hef",
                f"{BASE_HEF}/h10h/tiny-whisper-encoder-10s.hef",
            ],
            "tiny.en": [
                f"{BASE_HEF}/h10h/tiny_en-whisper-decoder-fixed-sequence.hef",
                f"{BASE_HEF}/h10h/tiny_en-whisper-encoder-10s.hef",
            ]
        }
    },
    "assets": {
        "tiny": [
            f"{BASE_ASSETS}/tiny/decoder_tokenization/onnx_add_input_tiny.npy",
            f"{BASE_ASSETS}/tiny/decoder_tokenization/token_embedding_weight_tiny.npy",
        ],
        "base": [
            f"{BASE_ASSETS}/base/decoder_tokenization/onnx_add_input_base.npy",
            f"{BASE_ASSETS}/base/decoder_tokenization/token_embedding_weight_base.npy",
        ],
        "tiny.en": [
            f"{BASE_ASSETS}/tiny.en/decoder_tokenization/onnx_add_input_tiny.en.npy",
            f"{BASE_ASSETS}/tiny.en/decoder_tokenization/token_embedding_weight_tiny.en.npy",
        ]
    },
}

def get_args():
    parser = get_standalone_parser()
    parser.description = "Whisper Downloader"
    parser.add_argument(
        "--hw-arch",
        dest="arch",
        type=str,
        default=None,
        choices=["hailo8", "hailo8l", "hailo10h"],
        help="Target hardware architecture to use (default: None)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=["base", "tiny", "tiny.en"],
        help="Whisper variant to download (default: None)"
    )
    args = parser.parse_args()
    args.hw_arch = args.arch
    return args


def remove_existing_file(path):
    if os.path.exists(path):
        print(f"Removing old HEF: {path}")
        os.remove(path)
    return

def download_hefs(arch=None, variant=None):
    if arch and variant:  # if both variant and arch are specified, check if the HEF exists for that architecture
        try:
          folder = FILES["hefs"][arch][variant]
        except KeyError as e:
          print(f"HEF not available for model '{variant}' on hardware '{arch}'.")
          sys.exit()
    for a, variants in FILES["hefs"].items():
        if arch and a != arch:
            continue
        for v, urls in variants.items():
            if variant and v != variant:
                continue
            target_dir = f"hefs/{a.replace('hailo','h')}/{v}"
            os.makedirs(target_dir, exist_ok=True)
            for url in urls:
                remove_existing_file(os.path.join(target_dir, url.split("/")[-1]))
                print(f"Downloading {url} -> {target_dir}")
                subprocess.run(["wget", "-P", target_dir, url], check=True)

def download_file(url, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    subprocess.run(["wget", "-nc", "-P", target_dir, url], check=True)

def download_assets(variant=None):
    for v, urls in FILES["assets"].items():
        if variant and v != variant:
            continue
        target_dir = f"decoder_assets/{v}/decoder_tokenization"
        if os.path.exists(target_dir):
            print(f"⏩ Skipping {target_dir}, already exists")
            continue
        print(f"📂 Creating {target_dir}")
        for url in urls:
            download_file(url, target_dir)

if __name__ == "__main__":
    args = get_args()
    arch = args.arch
    variant = args.variant
    download_hefs(arch, variant)
    download_assets(variant)
