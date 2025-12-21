"""Preprocessing functions for Whisper audio data."""

try:
    from hailo_apps.python.standalone_apps.speech_recognition.common import audio_utils
except ImportError:
    from pathlib import Path
    import sys

    speech_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(speech_root))
    import common.audio_utils as audio_utils
import numpy as np
import logging


def preprocess(audio, is_nhwc=False, chunk_length = 10, chunk_offset=0, max_duration = 60, overlap=0.0):
    """
    Generate the mel spectrograms
    
    Parameters:
    - audio: The audio sample.
    - chunk_length: Length in seconds of each audio chunk to process. This must match the input length of the model.
    - chunk_offset: Position - in seconds - to start processing the audio. This is useful for skipping silence at the beginning of the audio.
    - max_duration: Max duration of the audio sample to process.
    - overlap: Overlap between chunks. This is useful for continuous audio processing. Add some overlap (e.g. 0.2) when processing an audio longer than 10 seonds.
    """
    # Limit the audio duration
    sample_rate = audio_utils.SAMPLE_RATE
    max_samples = max_duration * sample_rate
    offset = int(chunk_offset * sample_rate)

    # Define parameters for chunking
    segment_duration = chunk_length  # in seconds
    segment_samples = segment_duration * sample_rate
    step = int(segment_samples * (1 - overlap))

    audio = audio[offset:max_samples]
    mel_spectrograms = []

    for start in range(0, len(audio), step):
        end = int(start + segment_samples)
        if start >= len(audio):
            break
        chunk = audio[start:end]

        # Ensure the chunk is 10s long (Whisper requires this)
        chunk = audio_utils.pad_or_trim(chunk, int(segment_duration * sample_rate))

        # Convert to Mel spectrogram
        mel = audio_utils.log_mel_spectrogram(chunk).to("cpu")
        # Run the encoder

        mel = np.expand_dims(mel, axis=0)  # Add new axis to match shape (1, 80, 1, 1000)
        #print(mel.shape)
        mel = np.expand_dims(mel, axis=2) 
        #print(mel.shape)

        if is_nhwc:
            mel = np.transpose(mel, [0, 2, 3, 1])

        mel_spectrograms.append(mel)

    return mel_spectrograms


def apply_gain(audio, gain_db):
    """
    Apply gain to the audio signal.
    Parameters:
    - audio: The audio sample.
    - gain_db: Gain in decibels (dB).
    """
    gain_linear = 10 ** (gain_db / 20)
    return audio * gain_linear


def improve_input_audio(audio, vad=True, low_audio_gain = True):
    """
    Improve the input audio by applying gain and detecting speech.
    Parameters:
    - audio: The audio sample.
    - vad: Boolean indicating whether to apply voice activity detection (VAD).
    - low_audio_gain: Boolean indicating whether to apply gain if the audio level is low.
    """
    
    # print(f"Max audio level: {np.max(audio)}")
    if (low_audio_gain == True) and (np.max(audio) < 0.1):
        if np.max(audio) < 0.1:
            audio = apply_gain(audio, gain_db=20)  # Increase by 15 dB
        elif np.max(audio) < 0.2:
            audio = apply_gain(audio, gain_db=10)  # Increase by 10 dB
        print(f"New max audio level: {np.max(audio)}")

    start_time = 0
    if vad:
        start_time = detect_first_speech(audio, audio_utils.SAMPLE_RATE, threshold=0.2, frame_duration=0.2)
        if start_time is not None:
            logging.info(f"Speech detected at {start_time:.2f} seconds.")
        else:
            logging.info("No speech detected.")
    return audio, start_time


def detect_first_speech(audio_data, sample_rate, threshold=0.2, frame_duration=0.02):
    """
    Detect the first time when human speech occurs in preloaded audio data.

    Parameters:
    - audio_data: NumPy array containing the audio samples.
    - sample_rate: Sample rate of the audio data.
    - threshold: Energy threshold for detecting speech (default: 0.02).
    - frame_duration: Duration of each frame in seconds (default: 0.02).

    Returns:
    - start_time: The time (in seconds) when speech is first detected.
    """
    # Convert stereo to mono if necessary
    if len(audio_data.shape) == 2:
        audio_data = np.mean(audio_data, axis=1)

    # Calculate frame size in samples
    frame_size = int(frame_duration * sample_rate)

    # Split the audio into frames
    frames = [audio_data[i:i + frame_size] for i in range(0, len(audio_data), frame_size)]

    # Calculate the energy of each frame
    energy = [np.sum(np.abs(frame)**2) / len(frame) for frame in frames]

    # Normalize energy to [0, 1]
    max_energy = max(energy)
    if max_energy > 0:
        energy = [e / max_energy for e in energy]
    #print(energy)
    # Detect the first frame with energy above the threshold. TODO add noise floor estimation for the threshold.
    for i, e in enumerate(energy):
        if e > threshold:
            start_time = i * frame_duration
            #start_time_rounded = math.floor(start_time)
            start_time_rounded = round(start_time, 1)
            return start_time_rounded

    return None  # No speech detected
