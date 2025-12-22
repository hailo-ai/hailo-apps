import getpass
import grp
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Default required dependencies for voice processing modules
_VOICE_DEPENDENCIES = ['sounddevice', 'numpy']


def check_voice_dependencies(required_deps: List[str] = None) -> None:
    """
    Check for required voice processing dependencies and exit with helpful error if missing.

    Args:
        required_deps: List of dependency names to check. Defaults to _VOICE_DEPENDENCIES.

    This function will:
    - Check if each dependency can be imported
    - Display a formatted error message if any are missing
    - Exit with code 1 if dependencies are missing
    - Return normally if all dependencies are available
    """
    if required_deps is None:
        required_deps = _VOICE_DEPENDENCIES

    missing_deps = []
    for dep_name in required_deps:
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
        print("\nTo install all GenAI dependencies (recommended):")
        print("  1. Navigate to the repository root directory")
        print("  2. Run: pip install -e \".[gen-ai]\"")
        print("\nThis will install:")
        print("  • sounddevice (for audio I/O)")
        print("  • piper-tts (for text-to-speech)")
        print("  • All other GenAI dependencies")
        print("\nFor detailed installation instructions, see:")
        print("  hailo_apps/python/gen_ai_apps/README.md")
        print("\n" + "="*70)
        sys.exit(1)


# Check for required dependencies and provide helpful error messages if missing
check_voice_dependencies()

import numpy as np
import sounddevice as sd

from hailo_apps.python.core.common.defines import REPO_ROOT, TARGET_SR

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class AudioDeviceInfo:
    """Dataclass to store audio device information."""
    id: int
    name: str
    host_api: int
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    is_default: bool = False
    score: int = 0
    test_result: bool = False
    error_msg: str = ""
    is_usb: bool = False
    pulse_name: Optional[str] = None
    alsa_card: Optional[int] = None


class AudioDiagnostics:
    """
    Provides tools for diagnosing audio issues, enumerating devices,
    and auto-detecting the best available hardware.
    """

    @staticmethod
    def list_audio_devices() -> Tuple[List[AudioDeviceInfo], List[AudioDeviceInfo]]:
        """
        Enumerate all audio devices.

        Returns:
            Tuple[List[AudioDeviceInfo], List[AudioDeviceInfo]]:
                Lists of (input_devices, output_devices).
        """
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]
        except Exception as e:
            logger.error(f"Failed to query audio devices: {e}")
            return [], []

        input_devices = []
        output_devices = []

        for i, dev in enumerate(devices):
            try:
                name_lower = dev['name'].lower()
                is_usb = 'usb' in name_lower

                device_info = AudioDeviceInfo(
                    id=i,
                    name=dev['name'],
                    host_api=dev['hostapi'],
                    max_input_channels=dev['max_input_channels'],
                    max_output_channels=dev['max_output_channels'],
                    default_samplerate=dev['default_samplerate'],
                    is_default=(i == default_input if dev['max_input_channels'] > 0 else i == default_output),
                    is_usb=is_usb
                )

                if dev['max_input_channels'] > 0:
                    input_devices.append(device_info)
                if dev['max_output_channels'] > 0:
                    output_devices.append(device_info)

            except Exception as e:
                logger.warning(f"Error parsing device {i}: {e}")

        return input_devices, output_devices

    @staticmethod
    def resample_audio(audio: np.ndarray, original_sr: float, target_sr: float) -> np.ndarray:
        """
        Resample audio from original_sr to target_sr using numpy linear interpolation.

        Args:
            audio (np.ndarray): Input audio data (1D array).
            original_sr (float): Original sample rate.
            target_sr (float): Target sample rate.

        Returns:
            np.ndarray: Resampled audio data.
        """
        if abs(original_sr - target_sr) < 1.0:
            return audio  # No resampling needed

        # Calculate the number of samples in the output
        num_samples = len(audio)
        duration = num_samples / original_sr
        target_num_samples = int(duration * target_sr)

        # Create indices for the original and target arrays
        original_indices = np.arange(num_samples)
        target_indices = np.linspace(0, num_samples - 1, target_num_samples)

        # Use numpy's interp function for linear interpolation
        resampled = np.interp(target_indices, original_indices, audio)

        return resampled.astype(audio.dtype)

    @staticmethod
    def test_microphone(device_id: int, duration: float = 1.0, threshold: float = 0.001) -> Tuple[bool, str, float, Optional[np.ndarray]]:
        """
        Test a microphone device by recording a short clip.

        Args:
            device_id (int): Device ID to test.
            duration (float): Duration of test recording in seconds.
            threshold (float): RMS amplitude threshold to consider signal valid.

        Returns:
            Tuple[bool, str, float, Optional[np.ndarray]]: (Success, Message, Max Amplitude, Recorded Data)
        """
        try:
            logger.debug(f"Testing microphone device {device_id}...")

            # Get device info to determine supported sample rate
            devices = sd.query_devices()
            device_info = devices[device_id]
            device_sr = device_info['default_samplerate']

            # Try to record at device's native sample rate first
            # If that fails, try TARGET_SR as fallback
            recording = None
            actual_sr = device_sr

            try:
                # Record at device's native sample rate
                num_samples = int(duration * device_sr)
                recording = sd.rec(
                    num_samples,
                    samplerate=device_sr,
                    channels=1,
                    device=device_id,
                    dtype='float32',
                    blocking=True
                )
                logger.debug(f"Recorded at device sample rate: {device_sr} Hz")
            except Exception as e:
                # Fallback: try TARGET_SR
                logger.debug(f"Failed to record at {device_sr} Hz, trying {TARGET_SR} Hz: {e}")
                try:
                    num_samples = int(duration * TARGET_SR)
                    recording = sd.rec(
                        num_samples,
                samplerate=TARGET_SR,
                channels=1,
                device=device_id,
                dtype='float32',
                blocking=True
            )
                    actual_sr = TARGET_SR
                    logger.debug(f"Recorded at fallback sample rate: {TARGET_SR} Hz")
                except Exception as fallback_error:
                    raise fallback_error  # Re-raise if fallback also fails

            # Ensure mono (take first channel if stereo)
            if recording.ndim > 1:
                recording = recording[:, 0]

            # Resample to TARGET_SR if needed
            if abs(actual_sr - TARGET_SR) > 1.0:
                logger.debug(f"Resampling from {actual_sr} Hz to {TARGET_SR} Hz")
                recording = AudioDiagnostics.resample_audio(recording, actual_sr, TARGET_SR)

            # Calculate levels
            max_amp = float(np.max(np.abs(recording)))
            rms = float(np.sqrt(np.mean(recording**2)))

            logger.debug(f"Mic test result - Max: {max_amp:.4f}, RMS: {rms:.4f}")

            if max_amp < threshold:
                return False, f"Signal too low (Max: {max_amp:.4f}). Check mute/volume.", max_amp, recording

            return True, "Microphone working correctly", max_amp, recording

        except Exception as e:
            msg = f"Recording failed: {str(e)}"
            logger.error(msg)
            return False, msg, 0.0, None

    @staticmethod
    def test_speaker(device_id: int, duration: float = 1.0, audio_data: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """
        Test a speaker device by playing a generated tone or provided audio using AudioPlayer.

        Args:
            device_id (int): Device ID to test.
            duration (float): Duration of test tone (ignored if audio_data provided).
            audio_data (Optional[np.ndarray]): Audio data to play. If None, generates a tone.

        Returns:
            Tuple[bool, str]: (Success, Message)
        """
        player = None
        try:
            logger.debug(f"Testing speaker device {device_id}...")

            # Prepare audio
            if audio_data is not None:
                to_play = audio_data
            else:
                # Generate 440Hz sine wave at TARGET_SR (16kHz)
                t = np.linspace(0, duration, int(duration * TARGET_SR), False)
                to_play = 0.5 * np.sin(2 * np.pi * 440 * t)

            # Ensure float32
            if to_play.dtype != np.float32:
                to_play = to_play.astype(np.float32)

            # Local import to avoid circular dependency
            from .audio_player import AudioPlayer

            # Initialize player
            player = AudioPlayer(device_id=device_id)

            # Play audio
            # AudioPlayer handles resampling and channel expansion internally
            player.play(to_play)

            # Wait for playback to complete (AudioPlayer is async)
            # Calculate duration based on input length and rate
            play_duration = len(to_play) / TARGET_SR
            time.sleep(play_duration + 0.5) # Add buffer

            return True, "Audio playback successful"

        except Exception as e:
            msg = f"Playback failed: {str(e)}"
            logger.error(msg)
            return False, msg
        finally:
            if player:
                player.close()

    @staticmethod
    def score_device(device: AudioDeviceInfo, is_input: bool) -> int:
        """
        Calculate a suitability score for a device.
        Higher score = better candidate.
        """
        score = 0

        # Prefer default devices
        if device.is_default:
            score += 100

        # Penalize "default", "sysdefault", "dmix" virtual devices to prefer hardware names
        # But only if we have other options. For now, let's just prefer hardware-looking names
        name_lower = device.name.lower()

        if "usb" in name_lower:
            score += 50  # Prefer USB devices (likely the plugged in mic/speaker)

        if "hdmi" in name_lower and is_input:
            score -= 50  # HDMI inputs are rarely used for voice

        # Prefer devices that support our target sample rate naturally (though sounddevice resamples)
        if abs(device.default_samplerate - TARGET_SR) < 1.0:
            score += 20

        return score

    @classmethod
    def auto_detect_devices(cls) -> Tuple[Optional[int], Optional[int]]:
        """
        Automatically detect best input and output devices.

        Returns:
            Tuple[Optional[int], Optional[int]]: (Best Input ID, Best Output ID)
        """
        input_devices, output_devices = cls.list_audio_devices()

        best_input = None
        best_input_score = -9999

        best_output = None
        best_output_score = -9999

        # Find best input
        for dev in input_devices:
            score = cls.score_device(dev, is_input=True)
            # Optional: actively test device if needed, but that takes time.
            # We'll rely on static properties for auto-selection to be fast.

            if score > best_input_score:
                best_input_score = score
                best_input = dev.id

        # Find best output
        for dev in output_devices:
            score = cls.score_device(dev, is_input=False)

            if score > best_output_score:
                best_output_score = score
                best_output = dev.id

        logger.debug(f"Auto-detected devices - Input: {best_input}, Output: {best_output}")
        return best_input, best_output

    @staticmethod
    def _get_preferences_path() -> Path:
        """
        Get the path to the device preferences file.

        Returns:
            Path: Path to the preferences JSON file in local_resources directory.
        """
        # Store preferences in local_resources directory (same location as TTS models)
        # This ensures path resolution works from anywhere (package -m or direct Python)
        prefs_path = REPO_ROOT / "local_resources" / "audio_device_preferences.json"
        # Ensure directory exists
        prefs_path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        return prefs_path

    @classmethod
    def save_device_preferences(cls, input_device_id: Optional[int], output_device_id: Optional[int]) -> Tuple[bool, str]:
        """
        Save device preferences to a persistent configuration file.

        Args:
            input_device_id (Optional[int]): Preferred input device ID, or None to clear.
            output_device_id (Optional[int]): Preferred output device ID, or None to clear.

        Returns:
            Tuple[bool, str]: (Success, Message)
        """
        try:
            prefs_path = cls._get_preferences_path()

            # Get device names for reference
            input_name = None
            output_name = None
            if input_device_id is not None:
                input_devices, _ = cls.list_audio_devices()
                input_dev = next((d for d in input_devices if d.id == input_device_id), None)
                if input_dev:
                    input_name = input_dev.name

            if output_device_id is not None:
                _, output_devices = cls.list_audio_devices()
                output_dev = next((d for d in output_devices if d.id == output_device_id), None)
                if output_dev:
                    output_name = output_dev.name

            preferences = {
                "input_device_id": input_device_id,
                "input_device_name": input_name,
                "output_device_id": output_device_id,
                "output_device_name": output_name,
                "timestamp": time.time()
            }

            with open(prefs_path, 'w') as f:
                json.dump(preferences, f, indent=2)

            logger.info(f"Saved device preferences to {prefs_path}")
            return True, f"Device preferences saved to {prefs_path}"

        except Exception as e:
            msg = f"Failed to save device preferences: {str(e)}"
            logger.error(msg)
            return False, msg

    @classmethod
    def load_device_preferences(cls) -> Tuple[Optional[int], Optional[int]]:
        """
        Load device preferences from the configuration file.

        Returns:
            Tuple[Optional[int], Optional[int]]: (Input Device ID, Output Device ID)
        """
        try:
            prefs_path = cls._get_preferences_path()

            if not prefs_path.exists():
                logger.debug("No device preferences file found")
                return None, None

            with open(prefs_path, 'r') as f:
                preferences = json.load(f)

            input_id = preferences.get("input_device_id")
            output_id = preferences.get("output_device_id")

            # Validate that devices still exist
            input_devices, output_devices = cls.list_audio_devices()

            if input_id is not None:
                if not any(d.id == input_id for d in input_devices):
                    logger.warning(f"Saved input device {input_id} no longer exists")
                    input_id = None

            if output_id is not None:
                if not any(d.id == output_id for d in output_devices):
                    logger.warning(f"Saved output device {output_id} no longer exists")
                    output_id = None

            if input_id is not None or output_id is not None:
                logger.debug(f"Loaded device preferences - Input: {input_id}, Output: {output_id}")

            return input_id, output_id

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid preferences file format: {e}")
            return None, None
        except Exception as e:
            logger.warning(f"Failed to load device preferences: {e}")
            return None, None

    @classmethod
    def get_preferred_devices(cls) -> Tuple[Optional[int], Optional[int]]:
        """
        Get preferred devices, falling back to auto-detection if no preferences exist.

        Returns:
            Tuple[Optional[int], Optional[int]]: (Input Device ID, Output Device ID)
        """
        input_id, output_id = cls.load_device_preferences()

        # Fall back to auto-detection for any missing preferences
        if input_id is None or output_id is None:
            auto_input, auto_output = cls.auto_detect_devices()
            if input_id is None:
                input_id = auto_input
            if output_id is None:
                output_id = auto_output

        return input_id, output_id

    @staticmethod
    def is_raspberry_pi() -> bool:
        """
        Check if running on Raspberry Pi.

        Returns:
            bool: True if running on Raspberry Pi.
        """
        try:
            machine = platform.machine()
            release = platform.release().lower()
            return machine.startswith(('arm', 'aarch')) or 'raspberry' in release
        except Exception:
            return False

    @staticmethod
    def get_audio_server() -> Optional[str]:
        """
        Detect which audio server is running (PulseAudio or PipeWire).

        Returns:
            Optional[str]: 'pulseaudio', 'pipewire', or None if neither detected.
        """
        try:
            # Check for PipeWire first (it can run PulseAudio compatibility layer)
            result = subprocess.run(
                ['pactl', 'info'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                output = result.stdout.lower()
                if 'pipewire' in output:
                    return 'pipewire'
                return 'pulseaudio'
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f"Could not detect audio server: {e}")
        return None

    @staticmethod
    def get_pulseaudio_usb_devices() -> List[Dict[str, str]]:
        """
        Get USB audio devices from PulseAudio/PipeWire.

        Returns:
            List[Dict[str, str]]: List of USB device info dicts with keys:
                'index', 'name', 'description', 'profiles'
        """
        devices = []
        try:
            result = subprocess.run(
                ['pactl', 'list', 'short', 'sinks'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                return devices

            # Also get sources (inputs)
            result_sources = subprocess.run(
                ['pactl', 'list', 'short', 'sources'],
                capture_output=True,
                text=True,
                timeout=5
            )

            # Parse sinks
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    index = parts[0]
                    name = parts[1]
                    # Get detailed info
                    detail_result = subprocess.run(
                        ['pactl', 'list', 'sinks', 'short'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    # Try to get full description
                    desc_result = subprocess.run(
                        ['pactl', 'info', '-s', name],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    description = name
                    if desc_result.returncode == 0:
                        for desc_line in desc_result.stdout.split('\n'):
                            if 'Description:' in desc_line:
                                description = desc_line.split('Description:')[1].strip()
                                break

                    if 'usb' in name.lower() or 'usb' in description.lower():
                        devices.append({
                            'index': index,
                            'name': name,
                            'description': description,
                            'type': 'sink'
                        })

            # Parse sources similarly
            if result_sources.returncode == 0:
                for line in result_sources.stdout.strip().split('\n'):
                    if not line.strip():
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        name = parts[1]
                        if 'usb' in name.lower():
                            devices.append({
                                'index': parts[0],
                                'name': name,
                                'description': name,
                                'type': 'source'
                            })

        except Exception as e:
            logger.warning(f"Failed to query PulseAudio devices: {e}")
        return devices

    @staticmethod
    def set_pulseaudio_profile(device_name: str, profile: str = "pro-audio") -> Tuple[bool, str]:
        """
        Set PulseAudio/PipeWire profile for a USB device.

        Args:
            device_name (str): PulseAudio device name or index.
            profile (str): Profile to set. Common: 'pro-audio', 'analog-stereo-duplex', 'off'.

        Returns:
            Tuple[bool, str]: (Success, Message)
        """
        try:
            # First, list available profiles
            result = subprocess.run(
                ['pactl', 'list', 'cards', 'short'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                return False, "Could not list audio cards"

            # Find the card index for this device
            card_index = None
            for line in result.stdout.strip().split('\n'):
                if device_name.lower() in line.lower() or 'usb' in line.lower():
                    parts = line.split('\t')
                    if parts:
                        card_index = parts[0]
                        break

            if card_index is None:
                # Try to find USB card
                detail_result = subprocess.run(
                    ['pactl', 'list', 'cards'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if detail_result.returncode == 0:
                    current_card = None
                    for line in detail_result.stdout.split('\n'):
                        if line.startswith('Card #'):
                            current_card = line.split('#')[1].split()[0]
                        elif 'usb' in line.lower() and current_card:
                            card_index = current_card
                            break

            if card_index is None:
                return False, f"Could not find USB audio card for device {device_name}"

            # Set the profile
            set_result = subprocess.run(
                ['pactl', 'set-card-profile', card_index, profile],
                capture_output=True,
                text=True,
                timeout=5
            )

            if set_result.returncode == 0:
                return True, f"Successfully set profile '{profile}' for card {card_index}"
            else:
                # Try alternative profile names
                alternatives = ['analog-stereo-duplex', 'analog-stereo-input', 'analog-stereo-output']
                for alt_profile in alternatives:
                    alt_result = subprocess.run(
                        ['pactl', 'set-card-profile', card_index, alt_profile],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if alt_result.returncode == 0:
                        return True, f"Successfully set profile '{alt_profile}' for card {card_index}"

                return False, f"Failed to set profile: {set_result.stderr}"

        except Exception as e:
            return False, f"Error setting profile: {str(e)}"

    @staticmethod
    def set_pulseaudio_default(device_name: str, is_input: bool = True) -> Tuple[bool, str]:
        """
        Set PulseAudio default sink/source.

        Args:
            device_name (str): PulseAudio device name or index.
            is_input (bool): True for input (source), False for output (sink).

        Returns:
            Tuple[bool, str]: (Success, Message)
        """
        try:
            cmd_type = 'set-default-source' if is_input else 'set-default-sink'
            result = subprocess.run(
                ['pactl', cmd_type, device_name],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                return True, f"Successfully set default {'input' if is_input else 'output'} to {device_name}"
            else:
                return False, f"Failed to set default: {result.stderr}"

        except Exception as e:
            return False, f"Error setting default: {str(e)}"

    @staticmethod
    def get_alsa_cards() -> List[Dict[str, str]]:
        """
        Get ALSA card information.

        Returns:
            List[Dict[str, str]]: List of card info dicts with 'card', 'name', 'id'
        """
        cards = []
        try:
            result = subprocess.run(
                ['arecord', '-l'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'card' in line.lower() and ':' in line:
                        # Parse: "card 1: Device [USB Audio Device], device 0: USB Audio [USB Audio]"
                        parts = line.split(':')
                        if len(parts) >= 2:
                            card_part = parts[0].strip()
                            card_num = card_part.split()[1] if len(card_part.split()) > 1 else None
                            name_part = parts[1].split(',')[0].strip()
                            if card_num and 'usb' in line.lower():
                                cards.append({
                                    'card': card_num,
                                    'name': name_part,
                                    'id': name_part.lower().replace(' ', '_')
                                })
        except Exception as e:
            logger.warning(f"Failed to query ALSA cards: {e}")
        return cards

    @staticmethod
    def configure_alsa_default(card_number: int) -> Tuple[bool, str]:
        """
        Configure ALSA default device via ~/.asoundrc.

        Args:
            card_number (int): ALSA card number to set as default.

        Returns:
            Tuple[bool, str]: (Success, Message)
        """
        try:
            asoundrc_path = Path.home() / '.asoundrc'
            config_content = f"""pcm.!default {{
    type hw
    card {card_number}
}}

ctl.!default {{
    type hw
    card {card_number}
}}
"""

            # Backup existing config if present
            if asoundrc_path.exists():
                backup_path = Path.home() / '.asoundrc.backup'
                shutil.copy2(asoundrc_path, backup_path)
                logger.debug(f"Backed up existing .asoundrc to {backup_path}")

            # Write new config
            asoundrc_path.write_text(config_content)
            return True, f"Successfully configured ALSA default to card {card_number}"

        except Exception as e:
            return False, f"Error configuring ALSA: {str(e)}"

    @staticmethod
    def check_audio_permissions() -> Tuple[bool, List[str]]:
        """
        Check if user has proper audio permissions.

        Returns:
            Tuple[bool, List[str]]: (Has permissions, List of issues)
        """
        issues = []
        try:
            user = getpass.getuser()
            user_groups = [g.gr_name for g in grp.getgrall() if user in g.gr_mem]
            user_groups.append(grp.getgrgid(os.getgid()).gr_name)

            if 'audio' not in user_groups:
                issues.append(f"User '{user}' is not in 'audio' group")
                return False, issues

            return True, []

        except Exception as e:
            logger.warning(f"Could not check permissions: {e}")
            return True, []  # Assume OK if we can't check

    @staticmethod
    def fix_audio_permissions() -> Tuple[bool, str]:
        """
        Attempt to add user to audio group (requires sudo).

        Returns:
            Tuple[bool, str]: (Success, Message)
        """
        try:
            user = getpass.getuser()

            result = subprocess.run(
                ['sudo', 'usermod', '-aG', 'audio', user],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                return True, f"Added user '{user}' to audio group. Please log out and back in for changes to take effect."
            else:
                return False, f"Failed to add user to audio group: {result.stderr}"

        except Exception as e:
            return False, f"Error fixing permissions: {str(e)}"

    @classmethod
    def configure_usb_audio_rpi(cls, auto_fix: bool = False) -> Dict[str, any]:
        """
        Comprehensive USB audio configuration for Raspberry Pi.

        This method attempts to:
        1. Detect USB audio devices
        2. Configure PulseAudio/PipeWire profiles
        3. Set ALSA defaults
        4. Check/fix permissions
        5. Test the configuration

        Args:
            auto_fix (bool): If True, attempts to automatically fix issues.

        Returns:
            Dict[str, any]: Configuration results with status and details.
        """
        results = {
            'success': False,
            'steps': [],
            'warnings': [],
            'errors': [],
            'device_info': {}
        }

        # Step 1: Check if RPi
        if not cls.is_raspberry_pi():
            results['warnings'].append("Not running on Raspberry Pi - some fixes may not apply")
        else:
            results['steps'].append("✓ Running on Raspberry Pi")

        # Step 2: Detect audio server
        audio_server = cls.get_audio_server()
        if audio_server:
            results['steps'].append(f"✓ Detected audio server: {audio_server}")
        else:
            results['warnings'].append("Could not detect audio server (PulseAudio/PipeWire)")

        # Step 3: Check permissions
        has_perms, perm_issues = cls.check_audio_permissions()
        if not has_perms:
            results['errors'].extend(perm_issues)
            if auto_fix:
                success, msg = cls.fix_audio_permissions()
                if success:
                    results['steps'].append(f"✓ {msg}")
                else:
                    results['errors'].append(f"Failed to fix permissions: {msg}")
            else:
                results['warnings'].append("User not in audio group - run: sudo usermod -aG audio $USER")
        else:
            results['steps'].append("✓ User has audio permissions")

        # Step 4: Detect USB devices
        input_devices, output_devices = cls.list_audio_devices()
        usb_inputs = [d for d in input_devices if d.is_usb]
        usb_outputs = [d for d in output_devices if d.is_usb]

        if not usb_inputs and not usb_outputs:
            results['errors'].append("No USB audio devices detected")
            return results

        results['device_info'] = {
            'usb_inputs': [{'id': d.id, 'name': d.name} for d in usb_inputs],
            'usb_outputs': [{'id': d.id, 'name': d.name} for d in usb_outputs]
        }
        results['steps'].append(f"✓ Found {len(usb_inputs)} USB input(s) and {len(usb_outputs)} USB output(s)")

        # Step 5: Configure PulseAudio if available
        if audio_server:
            pulse_devices = cls.get_pulseaudio_usb_devices()
            if pulse_devices:
                results['steps'].append(f"✓ Found {len(pulse_devices)} USB device(s) in PulseAudio")

                # Try to set profile for each USB card
                for device in pulse_devices:
                    # Try Pro Audio first (best for low latency)
                    success, msg = cls.set_pulseaudio_profile(device['name'], 'pro-audio')
                    if success:
                        results['steps'].append(f"✓ {msg}")
                    else:
                        # Try analog-stereo-duplex as fallback
                        success2, msg2 = cls.set_pulseaudio_profile(device['name'], 'analog-stereo-duplex')
                        if success2:
                            results['steps'].append(f"✓ {msg2}")
                        else:
                            results['warnings'].append(f"Could not set profile for {device['name']}: {msg}")

                    # Set as default if it's an input/output device
                    if device.get('type') == 'source' and usb_inputs:
                        success, msg = cls.set_pulseaudio_default(device['name'], is_input=True)
                        if success:
                            results['steps'].append(f"✓ {msg}")
                    elif device.get('type') == 'sink' and usb_outputs:
                        success, msg = cls.set_pulseaudio_default(device['name'], is_input=False)
                        if success:
                            results['steps'].append(f"✓ {msg}")
            else:
                results['warnings'].append("No USB devices found in PulseAudio - device may need to be plugged in")

        # Step 6: Configure ALSA defaults
        alsa_cards = cls.get_alsa_cards()
        if alsa_cards:
            results['steps'].append(f"✓ Found {len(alsa_cards)} USB ALSA card(s)")
            # Use first USB card
            card_num = int(alsa_cards[0]['card'])
            if auto_fix:
                success, msg = cls.configure_alsa_default(card_num)
                if success:
                    results['steps'].append(f"✓ {msg}")
                else:
                    results['warnings'].append(f"Could not configure ALSA: {msg}")
        else:
            results['warnings'].append("No USB ALSA cards found")

        # Step 7: Test configuration
        best_input, best_output = cls.auto_detect_devices()
        if best_input is not None:
            input_dev = next((d for d in input_devices if d.id == best_input), None)
            if input_dev and input_dev.is_usb:
                results['steps'].append(f"✓ USB input device auto-detected: {input_dev.name}")
                # Quick test
                success, msg, max_amp, _ = cls.test_microphone(best_input, duration=0.5)
                if success:
                    results['steps'].append(f"✓ Microphone test passed (max amplitude: {max_amp:.4f})")
                else:
                    results['warnings'].append(f"Microphone test failed: {msg}")

        if best_output is not None:
            output_dev = next((d for d in output_devices if d.id == best_output), None)
            if output_dev and output_dev.is_usb:
                results['steps'].append(f"✓ USB output device auto-detected: {output_dev.name}")

        # Determine overall success
        if not results['errors'] and (usb_inputs or usb_outputs):
            results['success'] = True

        return results

