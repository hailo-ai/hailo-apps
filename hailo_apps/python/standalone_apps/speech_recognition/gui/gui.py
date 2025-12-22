import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import base64
import threading
import time
import os
from PIL import Image


try:
    from hailo_apps.python.standalone_apps.speech_recognition.common.preprocessing import preprocess, improve_input_audio
    from hailo_apps.python.standalone_apps.speech_recognition.common.audio_utils import load_audio
    from hailo_apps.python.standalone_apps.speech_recognition.common.postprocessing import clean_transcription
    from hailo_apps.python.standalone_apps.speech_recognition.app.hailo_whisper_pipeline import HailoWhisperPipeline
    from hailo_apps.python.standalone_apps.speech_recognition.app.whisper_hef_registry import HEF_REGISTRY
    from hailo_apps.python.core.common.parser import get_standalone_parser
except ImportError:
    from pathlib import Path
    import sys

    speech_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(speech_root))
    core_dir = Path(__file__).resolve().parents[3] / "core"
    sys.path.insert(0, str(core_dir))
    from common.preprocessing import preprocess, improve_input_audio
    from common.audio_utils import load_audio
    from common.postprocessing import clean_transcription
    from app.hailo_whisper_pipeline import HailoWhisperPipeline
    from app.whisper_hef_registry import HEF_REGISTRY
    from common.parser import get_standalone_parser

# --- Constants ---
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5
FRAMES_PER_BUFFER = 4096
AUDIO_PATH = "sampled_audio.wav"
MIC_ICON_PATH = "./gui/microphone.svg"
LOGO_LEFT_PATH = "./gui/hailo_logo.png"

IS_NHWC = True

def get_args():
    """
    Initialize and run the argument parser.

    Return:
        argparse.Namespace: Parsed arguments.
    """
    parser = get_standalone_parser()
    parser.description = "Whisper Hailo Pipeline"
    parser.set_defaults(arch="hailo8")
    parser.add_argument(
        "--hw-arch",
        dest="arch",
        type=str,
        choices=["hailo8", "hailo8l", "hailo10h"],
        default=None,
        help="Hardware architecture to use (alias for --arch, default: hailo8)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="base",
        choices=["base", "tiny"],
        help="Whisper variant to use (default: base)"
    )
    args = parser.parse_args()
    if args.arch is None:
        args.arch = "hailo8"
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
        raise FileNotFoundError(f"HEF file not found at: {hef_path}")
    return hef_path

# --- Helper to inject CSS ---
def get_base64_svg(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def inject_custom_css(mic_icon_base64):
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: #f0f0f0; }}
        .centered {{ text-align: center; }}
        .stButton>button {{
            width: 100px; height: 100px; border-radius: 50%;
            font-size: 0px;
            background: url('data:image/svg+xml;base64,{mic_icon_base64}') no-repeat center;
            background-size: 60%;
        }}
        .fixed-bottom {{
            position: fixed; bottom: 30px; left: 50%;
            transform: translateX(-50%); z-index: 999;
        }}
        .top-section {{ margin-bottom: 20px; }}
        .button-section {{
            margin-bottom: 30px; display: flex; justify-content: center;
        }}
        .transcription-section {{ margin-top: 50px; }}
        .logo-container {{
            display: flex; justify-content: space-between; align-items: center; padding: 0px 0px;
        }}
        .logo {{ max-height: 60px; width: auto; }}
        </style>
        """,
        unsafe_allow_html=True
    )

def render_logo_column(image_path, fallback_text, width):
    try:
        img = Image.open(image_path)
        st.image(img, width=width)
    except Exception:
        st.markdown(fallback_text)

def render_logos():
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    logo_col1, _, _, logo_col4 = st.columns([1, 3, 3.5, 1])
    with logo_col1:
        render_logo_column(LOGO_LEFT_PATH, "Left Logo", 200)
    #with logo_col4:
    #    render_logo_column(LOGO_RIGHT_PATH, "Right Logo", 220)
    st.markdown('</div>', unsafe_allow_html=True)

def render_header():
    st.markdown('<div class="top-section">', unsafe_allow_html=True)
    st.markdown("<h1 class='centered'>Audio Transcription with Whisper on Hailo-8</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='centered'>Click the button below to record a {DURATION}s audio and transcribe it.</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_record_button():
    st.markdown('<div class="button-section">', unsafe_allow_html=True)
    _, col2, _ = st.columns([3.5, 1, 3])
    with col2:
        btn = st.button(" ", key="record_btn")
    st.markdown('</div>', unsafe_allow_html=True)
    return btn

def render_transcription_section():
    st.markdown('<div class="transcription-section">', unsafe_allow_html=True)
    status = st.empty()
    transcription = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    return status, transcription

def render_fixed_bottom():
    st.markdown('<div class="fixed-bottom"></div>', unsafe_allow_html=True)

# --- Audio Recorder Class ---
class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1, frames_per_buffer=4096, duration=5):
        self.SAMPLE_RATE = sample_rate
        self.CHANNELS = channels
        self.FRAMES = frames_per_buffer
        self.DURATION = duration
        self.audio_data = []
        self.recording = False
        self.thread = None
        self.finished_event = threading.Event()

    def _audio_recording(self):
        self.audio_data = []
        start_time = time.time()
        with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=self.CHANNELS) as stream:
            while self.recording and (time.time() - start_time < self.DURATION):
                frames, _ = stream.read(self.FRAMES)
                self.audio_data.append(frames)
        self.recording = False
        self.finished_event.set()  # signal finished

    def start_recording(self):
        if not self.recording:
            self.finished_event.clear()
            self.recording = True
            self.thread = threading.Thread(target=self._audio_recording, daemon=True)
            self.thread.start()

    def stop_recording(self):
        self.recording = False
        if self.thread:
            self.thread.join()
            self.thread = None

    def is_recording(self):
        return self.recording
    
    def wait_until_finished(self, countdown_placeholder=None):
        start_time = time.time()
        while not self.finished_event.is_set():
            if countdown_placeholder and self.DURATION:
                remaining = int(self.DURATION + 1 - (time.time() - start_time))
                if remaining < 0:
                    remaining = 0
                countdown_placeholder.markdown(
                    f"<h4 style='color:#0099c4; font-weight:bold;'>{remaining} seconds remaining</h4>",
                    unsafe_allow_html=True
                )
                if remaining > 0:
                    time.sleep(1)
        if countdown_placeholder:
            countdown_placeholder.empty()

    def save_to_wav(self, path):
        audio_np = np.concatenate(self.audio_data, axis=0)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        wav.write(path, self.SAMPLE_RATE, (audio_np * 32767).astype(np.int16))
        return audio_np


def process_and_transcribe():
    sampled_audio = load_audio(AUDIO_PATH)
    sampled_audio, start_time = improve_input_audio(sampled_audio, vad=True, low_audio_gain=True)
    mel_spectrograms = preprocess(sampled_audio, is_nhwc=IS_NHWC, chunk_length=st.session_state.chunk_length, chunk_offset=start_time - 0.2)
    results = []
    for mel in mel_spectrograms:
        st.session_state.whisper_hailo.send_data(mel)
        time.sleep(0.2)
        transcription = st.session_state.whisper_hailo.get_transcription()
        results.append(transcription)
    return results

# --- Streamlit App ---
st.set_page_config(page_title="Whisper Transcription", layout="wide")

if 'recorder' not in st.session_state:
    st.session_state.recorder = AudioRecorder()


if 'initialized' not in st.session_state:
    args = get_args()
    variant = args.variant
    print(f"Selected variant: Whisper {variant}")
    st.session_state.variant = variant
    st.session_state.chunk_length = 10 if ("tiny" in variant) else 5
    encoder_path = get_hef_path(variant, args.arch, "encoder")
    decoder_path = get_hef_path(variant, args.arch, "decoder")
    st.session_state.initialized = True
    print("Initializing whisper model...")
    st.session_state.whisper_hailo = HailoWhisperPipeline(encoder_path, decoder_path, variant=variant)
    print("Initialization complete!")

mic_icon_base64 = get_base64_svg(MIC_ICON_PATH)
inject_custom_css(mic_icon_base64)
render_logos()
render_header()

record_button = render_record_button()
status_indicator, transcription_placeholder = render_transcription_section()

render_fixed_bottom()

countdown_placeholder = st.empty()

if record_button:
    if not st.session_state.recorder.is_recording():
        status_indicator.info("Recording started.. Press the button again to stop.")
        st.session_state.recorder.start_recording()
        st.session_state.recorder.wait_until_finished(countdown_placeholder)
        
        status_indicator.success("Recording finished.")
        audio_data = st.session_state.recorder.save_to_wav(AUDIO_PATH)
        st.audio(AUDIO_PATH, format="audio/wav")
        st.session_state.recorded_audio = audio_data
    else:
        status_indicator.info("Stopping recording...")
        st.session_state.recorder.stop_recording()
        st.session_state.recorder.wait_until_finished(countdown_placeholder)
        status_indicator.success("Recording finished.")
        audio_data = st.session_state.recorder.save_to_wav(AUDIO_PATH)
        st.audio(AUDIO_PATH, format="audio/wav")
        st.session_state.recorded_audio = audio_data

    results = process_and_transcribe()
    success = bool(results and results[0])

    if not success:
            results = ["Sorry, I haven't understood what you said. Please try again."]
    transcription_placeholder.markdown(
        f"""### Transcription Result
        {clean_transcription(results[0])}
        """, unsafe_allow_html=True
    )
    if success:
        status_indicator.success("Transcription complete!")
    else:
        status_indicator.error("Transcription failed.")

# Show real-time status
if st.session_state.recorder.is_recording():
    status_indicator.info("Recording in progress...")
else:
    if 'recorded_audio' not in st.session_state:
        status_indicator.info("Not recording.")
