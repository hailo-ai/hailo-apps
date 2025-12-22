#!/usr/bin/env python3
"""
Audio Troubleshooting Tool for Hailo Voice Apps.

Diagnoses microphone and speaker issues, tests hardware, and recommends fixes.
"""

import argparse
import logging
import platform
import re
import sys
from typing import Optional

# Dependencies are checked in audio_diagnostics.py when it's imported
from .audio_diagnostics import AudioDiagnostics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def is_hdmi_device(device_name: str) -> bool:
    """
    Check if a device is an HDMI audio device.

    Args:
        device_name (str): Device name to check.

    Returns:
        bool: True if device appears to be HDMI audio.
    """
    name_lower = device_name.lower()
    return 'hdmi' in name_lower


def is_real_hardware_device(device_name: str) -> bool:
    """
    Check if a device name represents real hardware vs virtual device.

    Args:
        device_name (str): Device name to check.

    Returns:
        bool: True if device appears to be real hardware.
    """
    name_lower = device_name.lower().strip()

    # Explicitly virtual device names (exact matches)
    virtual_exact = ['default', 'pulse', 'dmix', 'sysdefault']
    if name_lower in virtual_exact:
        return False

    # Real hardware indicators (presence indicates hardware)
    hardware_keywords = ['usb', 'alsa', 'card', 'hdmi', 'analog', 'iec958', 'hw:', 'plughw:', 'i2s', 'pcm']

    # If it contains hardware indicators, it's likely real hardware
    for keyword in hardware_keywords:
        if keyword in name_lower:
            return True

    # If name contains "hw:" or "card" followed by a number, it's hardware
    if re.search(r'hw:\d+|card\s*\d+', name_lower):
        return True

    # If it's just "default" or "pulse" (already checked above), it's virtual
    # Otherwise, if it has any complexity, assume it might be hardware
    # (conservative approach - better to test than skip)
    return len(name_lower) > 10  # Simple names are more likely virtual


def filter_real_hardware_devices(devices):
    """
    Filter out virtual devices, keeping only real hardware.

    Args:
        devices: List of AudioDeviceInfo objects.

    Returns:
        List of AudioDeviceInfo objects representing real hardware.
    """
    return [d for d in devices if is_real_hardware_device(d.name)]


def print_device_table(devices, title, show_preferred=False, preferred_input_id=None, preferred_output_id=None):
    print(f"\n--- {title} ---")
    if not devices:
        print("No devices found.")
        return

    print(f"{'ID':<4} {'Name':<40} {'Ch':<5} {'Rate':<8} {'Def':<5} {'Score':<5} {'Pref':<5}")
    print("-" * 85)
    for dev in devices:
        is_def = "*" if dev.is_default else ""
        is_pref = ""
        if show_preferred:
            if 'Input' in title and preferred_input_id == dev.id:
                is_pref = "✓"
            elif 'Output' in title and preferred_output_id == dev.id:
                is_pref = "✓"
        print(f"{dev.id:<4} {dev.name[:38]:<40} {dev.max_input_channels if 'Input' in title else dev.max_output_channels:<5} {int(dev.default_samplerate):<8} {is_def:<5} {dev.score:<5} {is_pref:<5}")


def select_device_interactive(devices, device_type: str, current_preferred_id: Optional[int] = None) -> Optional[int]:
    """
    Interactively select a device from a list.

    Args:
        devices: List of AudioDeviceInfo objects.
        device_type: Type of device ("Input" or "Output").
        current_preferred_id: Currently preferred device ID, if any.

    Returns:
        Optional[int]: Selected device ID, or None if cancelled.
    """
    if not devices:
        print(f"\n⚠️  No {device_type.lower()} devices available.")
        return None

    print(f"\n--- Select {device_type} Device ---")
    print(f"{'ID':<4} {'Name':<40} {'Ch':<5} {'Rate':<8} {'Def':<5} {'Score':<5}")
    print("-" * 75)
    for dev in devices:
        is_def = "*" if dev.is_default else ""
        is_pref = " [CURRENT]" if current_preferred_id == dev.id else ""
        print(f"{dev.id:<4} {dev.name[:38]:<40} {dev.max_input_channels if device_type == 'Input' else dev.max_output_channels:<5} {int(dev.default_samplerate):<8} {is_def:<5} {dev.score:<5}{is_pref}")

    print("\nOptions:")
    print("  - Enter device ID to select")
    print("  - Enter 't' to test a device")
    print("  - Enter 'a' to use auto-detected device")
    print("  - Enter 'c' to cancel (keep current preference)")

    while True:
        try:
            choice = input(f"\nSelect {device_type.lower()} device: ").strip().lower()

            if choice == 'c':
                return current_preferred_id  # Keep current

            if choice == 'a':
                # Auto-detect
                if device_type == "Input":
                    best_in, _ = AudioDiagnostics.auto_detect_devices()
                    return best_in
                else:
                    _, best_out = AudioDiagnostics.auto_detect_devices()
                    return best_out

            if choice == 't':
                # Test mode - let user pick device to test
                test_id_str = input("Enter device ID to test: ").strip()
                try:
                    test_id = int(test_id_str)
                    test_dev = next((d for d in devices if d.id == test_id), None)
                    if test_dev is None:
                        print(f"❌ Device ID {test_id} not found.")
                        continue

                    print(f"\nTesting device [{test_id}] {test_dev.name}...")
                    if device_type == "Input":
                        success, msg, max_amp, _ = AudioDiagnostics.test_microphone(test_id, duration=2.0)
                        if success:
                            print(f"✅ Test passed! Max amplitude: {max_amp:.4f}")
                            if input(f"Use this device? [Y/n]: ").lower() != 'n':
                                return test_id
                        else:
                            print(f"❌ Test failed: {msg}")
                    else:
                        success, msg = AudioDiagnostics.test_speaker(test_id, duration=1.0)
                        if success:
                            print("✅ Test tone played.")
                            if input("Did you hear the tone? Use this device? [y/N]: ").lower() == 'y':
                                return test_id
                        else:
                            print(f"❌ Test failed: {msg}")
                except ValueError:
                    print("❌ Invalid device ID.")
                continue

            # Try to parse as device ID
            device_id = int(choice)
            selected_dev = next((d for d in devices if d.id == device_id), None)
            if selected_dev is None:
                print(f"❌ Device ID {device_id} not found. Please try again.")
                continue

            return device_id

        except ValueError:
            print("❌ Invalid input. Please enter a device ID, 't', 'a', or 'c'.")
        except KeyboardInterrupt:
            print("\n\nCancelled.")
            return current_preferred_id


def run_diagnostics(args):
    print_header("Hailo Audio Troubleshooter")

    print(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {sys.version.split()[0]}")

    # 1. Enumerate Devices
    print_header("1. Device Enumeration")
    input_devs, output_devs = AudioDiagnostics.list_audio_devices()

    # Score devices
    for d in input_devs:
        d.score = AudioDiagnostics.score_device(d, is_input=True)
    for d in output_devs:
        d.score = AudioDiagnostics.score_device(d, is_input=False)

    print_device_table(input_devs, "Input Devices")
    print_device_table(output_devs, "Output Devices")

    # Check for real hardware devices
    real_input_devs = filter_real_hardware_devices(input_devs)
    real_output_devs = filter_real_hardware_devices(output_devs)

    # Check for HDMI devices
    hdmi_outputs = [d for d in real_output_devs if is_hdmi_device(d.name)]
    has_hdmi_only = len(hdmi_outputs) > 0 and len([d for d in real_output_devs if not is_hdmi_device(d.name)]) == 0

    if not real_input_devs and not real_output_devs:
        print("\n⚠️  WARNING: No real hardware audio devices detected!")
        print("   Only virtual devices (default, pulse) were found.")
        print("   This usually means:")
        print("   - No USB audio device is connected")
        print("   - USB device is not recognized by the system")
        print("   - Audio drivers are not properly loaded")
        print("\n   Try:")
        print("   - Plugging in your USB audio device")
        print("   - Running: lsusb (to verify USB device is detected)")
        print("   - Running: arecord -l (to list ALSA capture devices)")
        print("   - Running: aplay -l (to list ALSA playback devices)")
        if not args.no_interactive:
            if input("\nContinue with virtual devices anyway? [y/N]: ").lower() != 'y':
                print("Exiting.")
                return
    elif not real_input_devs:
        print("\n⚠️  WARNING: No real hardware input devices detected!")
        print("   Only virtual input devices found. Your microphone may not be connected or recognized.")
    elif not real_output_devs:
        print("\n⚠️  WARNING: No real hardware output devices detected!")
        print("   Only virtual output devices found. Your speakers/headphones may not be connected or recognized.")

    # Warn about HDMI-only audio
    if has_hdmi_only:
        print("\n⚠️  NOTE: Only HDMI audio output detected!")
        print("   HDMI audio only works if your display/monitor supports audio passthrough.")
        print("   Many monitors and some TVs do not support HDMI audio.")
        print("   If you don't hear audio, try:")
        print("   - Connecting USB headphones or speakers")
        print("   - Using a 3.5mm audio jack if available")
        print("   - Checking your display's audio settings")

    # 2. Auto-detection & Current Preferences
    print_header("2. Auto-Detection & Current Preferences")

    # Show saved preferences
    saved_input_id, saved_output_id = AudioDiagnostics.load_device_preferences()
    if saved_input_id is not None or saved_output_id is not None:
        print("\n📋 Saved Preferences:")
        if saved_input_id is not None:
            in_dev = next((d for d in input_devs if d.id == saved_input_id), None)
            if in_dev:
                is_real = is_real_hardware_device(in_dev.name)
                marker = " (virtual)" if not is_real else ""
                print(f"  Input:  [{saved_input_id}] {in_dev.name}{marker}")
            else:
                print(f"  Input:  [{saved_input_id}] (device no longer exists)")
        if saved_output_id is not None:
            out_dev = next((d for d in output_devs if d.id == saved_output_id), None)
            if out_dev:
                is_real = is_real_hardware_device(out_dev.name)
                marker = " (virtual)" if not is_real else ""
                print(f"  Output: [{saved_output_id}] {out_dev.name}{marker}")
            else:
                print(f"  Output: [{saved_output_id}] (device no longer exists)")

    # Show auto-detected devices
    best_in, best_out = AudioDiagnostics.auto_detect_devices()
    print("\n🔍 Auto-Detected (if no preferences):")
    if best_in is not None:
        in_dev = next((d for d in input_devs if d.id == best_in), None)
        is_real = in_dev and is_real_hardware_device(in_dev.name) if in_dev else False
        marker = " (virtual)" if not is_real else ""
        print(f"  Input:  [{best_in}] {in_dev.name if in_dev else 'Unknown'}{marker}")
    else:
        print("  Input:  ❌ No suitable input device found!")

    if best_out is not None:
        out_dev = next((d for d in output_devs if d.id == best_out), None)
        is_real = out_dev and is_real_hardware_device(out_dev.name) if out_dev else False
        marker = " (virtual)" if not is_real else ""
        print(f"  Output: [{best_out}] {out_dev.name if out_dev else 'Unknown'}{marker}")
    else:
        print("  Output: ❌ No suitable output device found!")

    # Show what will actually be used
    preferred_input, preferred_output = AudioDiagnostics.get_preferred_devices()
    print("\n✅ Will Use (preferences + auto-detection fallback):")
    if preferred_input is not None:
        in_dev = next((d for d in input_devs if d.id == preferred_input), None)
        source = "saved preference" if preferred_input == saved_input_id else "auto-detected"
        print(f"  Input:  [{preferred_input}] {in_dev.name if in_dev else 'Unknown'} ({source})")
    else:
        print("  Input:  ❌ None available")

    if preferred_output is not None:
        out_dev = next((d for d in output_devs if d.id == preferred_output), None)
        source = "saved preference" if preferred_output == saved_output_id else "auto-detected"
        print(f"  Output: [{preferred_output}] {out_dev.name if out_dev else 'Unknown'} ({source})")
    else:
        print("  Output: ❌ None available")

    # Update best_in/best_out to use preferred devices for testing
    if preferred_input is not None:
        best_in = preferred_input
    if preferred_output is not None:
        best_out = preferred_output

    # 3. Interactive Testing
    if not args.no_interactive:
        print_header("3. Interactive Testing")

        # Test Microphone
        recorded_audio = None
        recording_successful = False
        mic_test_failed = False
        if best_in is not None:
            if input(f"\nTest Microphone (ID {best_in})? [Y/n]: ").lower() != 'n':
                print("Recording 3 seconds... Speak now!")
                success, msg, max_amp, recorded_audio = AudioDiagnostics.test_microphone(best_in, duration=3.0)
                if success:
                    print(f"✅ Success! Max Amplitude: {max_amp:.4f}")
                    recording_successful = True
                else:
                    print(f"❌ Failed: {msg}")
                    recording_successful = False
                    mic_test_failed = True
                    # Clear recorded_audio if recording failed
                    recorded_audio = None
        else:
            print("\n⚠️  No input device available for testing.")
            mic_test_failed = True

        # Test Speaker
        speaker_test_failed = False
        if best_out is not None:
            if input(f"\nTest Speaker (ID {best_out})? [Y/n]: ").lower() != 'n':
                playback_attempted = False

                # Only try playing back recording if it was successful
                if recording_successful and recorded_audio is not None and len(recorded_audio) > 0:
                    if input("Play back recorded audio? [Y/n]: ").lower() != 'n':
                        print("Playing back recorded audio...")
                        success, msg = AudioDiagnostics.test_speaker(best_out, audio_data=recorded_audio)
                        if success:
                            print("✅ Playback command sent.")
                            if input("Did you hear your recording? [y/N]: ").lower() == 'y':
                                print("✅ Speaker confirmed working.")
                                playback_attempted = True
                            else:
                                print("❌ User did not hear audio.")
                                speaker_test_failed = True
                        else:
                            print(f"⚠️  Playback of recording failed: {msg}")
                            speaker_test_failed = True

                if not playback_attempted:
                    print("Playing test tone...")
                    success, msg = AudioDiagnostics.test_speaker(best_out)
                    if success:
                        print("✅ Playback command sent.")
                        if input("Did you hear the tone? [y/N]: ").lower() == 'y':
                            print("✅ Speaker confirmed working.")
                        else:
                            print("❌ User did not hear audio.")
                            speaker_test_failed = True
                    else:
                        print(f"❌ Playback failed: {msg}")
                        speaker_test_failed = True
        else:
            print("\n⚠️  No output device available for testing.")
            speaker_test_failed = True

        # Check if we're using virtual devices for the tested devices
        using_virtual_input = False
        using_virtual_output = False

        if best_in is not None:
            in_dev = next((d for d in input_devs if d.id == best_in), None)
            using_virtual_input = in_dev is not None and not is_real_hardware_device(in_dev.name)

        if best_out is not None:
            out_dev = next((d for d in output_devs if d.id == best_out), None)
            using_virtual_output = out_dev is not None and not is_real_hardware_device(out_dev.name)

        # Check if only virtual devices exist (no real hardware at all)
        no_real_hardware = not real_input_devs and not real_output_devs

        # Notify if tests failed and virtual devices are being used
        if (mic_test_failed or speaker_test_failed) and (no_real_hardware or (mic_test_failed and using_virtual_input) or (speaker_test_failed and using_virtual_output)):
            print("\n" + "="*60)
            print("⚠️  IMPORTANT: Tests failed and you're using virtual devices!")
            print("   Virtual devices (like 'default' or 'pulse') often don't work properly")
            print("   for actual audio input/output. This is likely why your tests failed.")
            print("\n   To fix this:")
            print("   1. Connect a real USB audio device (microphone/headphones)")
            print("   2. Run: hailo-audio-troubleshoot --configure --auto-fix")
            print("   3. Or manually configure your USB device in PulseAudio Volume Control")
            print("="*60)
        # Suggest auto-configuration if devices exist but tests failed
        elif (mic_test_failed or speaker_test_failed) and (real_input_devs or real_output_devs):
            print("\n" + "="*60)
            print("💡 SUGGESTION: Devices are detected but not working properly.")
            print("   Try running the auto-configuration tool:")
            print("   hailo-audio-troubleshoot --configure")
            print("   or with automatic fixes:")
            print("   hailo-audio-troubleshoot --configure --auto-fix")
            print("="*60)

    # 4. Troubleshooting Tips
    print_header("4. Troubleshooting Tips")

    is_rpi = platform.machine().startswith(('arm', 'aarch')) or "raspberry" in platform.release().lower()

    if is_rpi:
        print("RASPBERRY PI SPECIFIC:")
        print("  - If USB mic/speaker not working, try auto-configuration:")
        print("    hailo-audio-troubleshoot --configure --auto-fix")
        print("  - Or manually check 'Device Profiles' in Volume Control (pavucontrol).")
        print("  - Select 'Pro Audio' or 'Analog Stereo Duplex'.")
        print("  - Ensure current user is in 'audio' group: `sudo usermod -aG audio $USER`")

    print("\nGENERAL:")
    print("  - If devices are detected but not working, try auto-configuration:")
    print("    hailo-audio-troubleshoot --configure")
    print("  - Check system volume and mute status (`alsamixer` on Linux).")
    print("  - Ensure no other app is blocking the audio device.")
    print("  - If using PulseAudio, try restarting it: `pulseaudio -k && pulseaudio --start`")

    print("\n" + "="*60)


def run_device_selection(args):
    """Run interactive device selection and save preferences."""
    print_header("Device Selection & Configuration")

    print(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {sys.version.split()[0]}")

    # Load current preferences
    current_input_id, current_output_id = AudioDiagnostics.load_device_preferences()
    if current_input_id is not None or current_output_id is not None:
        print("\n📋 Current Preferences:")
        if current_input_id is not None:
            input_devs, _ = AudioDiagnostics.list_audio_devices()
            input_dev = next((d for d in input_devs if d.id == current_input_id), None)
            if input_dev:
                print(f"  Input:  [{current_input_id}] {input_dev.name}")
            else:
                print(f"  Input:  [{current_input_id}] (device no longer exists)")
        if current_output_id is not None:
            _, output_devs = AudioDiagnostics.list_audio_devices()
            output_dev = next((d for d in output_devs if d.id == current_output_id), None)
            if output_dev:
                print(f"  Output: [{current_output_id}] {output_dev.name}")
            else:
                print(f"  Output: [{current_output_id}] (device no longer exists)")
    else:
        print("\n📋 No saved preferences. Using auto-detection.")

    # Enumerate devices
    print_header("Available Devices")
    input_devs, output_devs = AudioDiagnostics.list_audio_devices()

    # Score devices
    for d in input_devs:
        d.score = AudioDiagnostics.score_device(d, is_input=True)
    for d in output_devs:
        d.score = AudioDiagnostics.score_device(d, is_input=False)

    # Show current preferences in table
    print_device_table(input_devs, "Input Devices", show_preferred=True,
                      preferred_input_id=current_input_id, preferred_output_id=None)
    print_device_table(output_devs, "Output Devices", show_preferred=True,
                      preferred_input_id=None, preferred_output_id=current_output_id)

    # Device selection
    selected_input_id = None
    selected_output_id = None

    # Handle command-line arguments first
    if args.input_device is not None:
        try:
            selected_input_id = int(args.input_device)
            # Validate
            if not any(d.id == selected_input_id for d in input_devs):
                print(f"⚠️  Warning: Input device {selected_input_id} not found.")
                selected_input_id = None
            else:
                input_dev = next((d for d in input_devs if d.id == selected_input_id), None)
                print(f"\n✅ Input device set to: [{selected_input_id}] {input_dev.name if input_dev else 'Unknown'}")
        except ValueError:
            print(f"⚠️  Warning: Invalid input device ID: {args.input_device}")

    if args.output_device is not None:
        try:
            selected_output_id = int(args.output_device)
            # Validate
            if not any(d.id == selected_output_id for d in output_devs):
                print(f"⚠️  Warning: Output device {selected_output_id} not found.")
                selected_output_id = None
            else:
                output_dev = next((d for d in output_devs if d.id == selected_output_id), None)
                print(f"\n✅ Output device set to: [{selected_output_id}] {output_dev.name if output_dev else 'Unknown'}")
        except ValueError:
            print(f"⚠️  Warning: Invalid output device ID: {args.output_device}")

    # Interactive selection (if not using command-line args and interactive mode enabled)
    if not args.no_interactive:
        print_header("Device Selection")

        # Select input device interactively if not set via command line
        if args.input_device is None:
            if input("\nSelect input device? [Y/n]: ").lower() != 'n':
                selected_input_id = select_device_interactive(input_devs, "Input", current_input_id)

        # Select output device interactively if not set via command line
        if args.output_device is None:
            if input("\nSelect output device? [Y/n]: ").lower() != 'n':
                selected_output_id = select_device_interactive(output_devs, "Output", current_output_id)

    # Save preferences
    # Use current preferences if not changed
    final_input_id = selected_input_id if selected_input_id is not None else current_input_id
    final_output_id = selected_output_id if selected_output_id is not None else current_output_id

    # Only save if something changed or if explicitly set via command line
    if (selected_input_id is not None or selected_output_id is not None) or \
       (args.input_device is not None or args.output_device is not None):
        success, msg = AudioDiagnostics.save_device_preferences(final_input_id, final_output_id)
        if success:
            print_header("Preferences Saved")
            print("✅ Device preferences saved successfully!")
            if final_input_id is not None:
                input_dev = next((d for d in input_devs if d.id == final_input_id), None)
                print(f"  Input:  [{final_input_id}] {input_dev.name if input_dev else 'Unknown'}")
            if final_output_id is not None:
                output_dev = next((d for d in output_devs if d.id == final_output_id), None)
                print(f"  Output: [{final_output_id}] {output_dev.name if output_dev else 'Unknown'}")
            print(f"\nThese preferences will be used by AudioRecorder and AudioPlayer.")
        else:
            print(f"\n❌ Failed to save preferences: {msg}")
    else:
        print("\n⚠️  No changes made to device preferences.")

    print("\n" + "="*60)


def run_auto_configure(args):
    """Run automatic USB audio configuration for Raspberry Pi."""
    print_header("USB Audio Auto-Configuration (Raspberry Pi)")

    if not AudioDiagnostics.is_raspberry_pi():
        print("⚠️  Warning: This tool is designed for Raspberry Pi.")
        if input("Continue anyway? [y/N]: ").lower() != 'y':
            return

    print("\nThis will attempt to configure your USB audio device.")
    if not args.yes:
        if input("Continue? [y/N]: ").lower() != 'y':
            print("Cancelled.")
            return

    print("\nRunning configuration...")
    results = AudioDiagnostics.configure_usb_audio_rpi(auto_fix=args.auto_fix)

    print_header("Configuration Results")

    if results['success']:
        print("✅ Configuration completed successfully!\n")
    else:
        print("⚠️  Configuration completed with warnings/errors.\n")

    if results['steps']:
        print("Steps completed:")
        for step in results['steps']:
            print(f"  {step}")

    if results['warnings']:
        print("\n⚠️  Warnings:")
        for warning in results['warnings']:
            print(f"  - {warning}")

    if results['errors']:
        print("\n❌ Errors:")
        for error in results['errors']:
            print(f"  - {error}")

    if results['device_info']:
        print("\n📱 Detected USB Devices:")
        if results['device_info'].get('usb_inputs'):
            print("  Input devices:")
            for dev in results['device_info']['usb_inputs']:
                print(f"    - [{dev['id']}] {dev['name']}")
        if results['device_info'].get('usb_outputs'):
            print("  Output devices:")
            for dev in results['device_info']['usb_outputs']:
                print(f"    - [{dev['id']}] {dev['name']}")

    print("\n" + "="*60)
    print("\nNext steps:")
    print("  1. If permissions were changed, log out and back in")
    print("  2. Test your microphone: hailo-audio-troubleshoot")
    print("  3. If issues persist, check PulseAudio Volume Control (pavucontrol)")
    print("     and ensure USB device profile is set to 'Pro Audio' or 'Analog Stereo Duplex'")


def main():
    parser = argparse.ArgumentParser(
        description="Hailo Audio Troubleshooting Tool",
        epilog="For installation instructions, see: hailo_apps/python/gen_ai_apps/README.md"
    )
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive tests")
    parser.add_argument("--configure", action="store_true", help="Run USB audio auto-configuration for RPi")
    parser.add_argument("--select-devices", action="store_true", help="Interactively select and save audio devices")
    parser.add_argument("--input-device", type=int, metavar="ID", help="Set input device ID (use with --select-devices)")
    parser.add_argument("--output-device", type=int, metavar="ID", help="Set output device ID (use with --select-devices)")
    parser.add_argument("--auto-fix", action="store_true", help="Automatically fix issues (requires sudo)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts")
    args = parser.parse_args()

    # Dependencies are checked at import time - if we got here, they're available
    try:
        if args.configure:
            run_auto_configure(args)
        elif args.select_devices or args.input_device is not None or args.output_device is not None:
            run_device_selection(args)
        else:
            run_diagnostics(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except ImportError as e:
        # Catch any other import errors that might occur
        print("\n" + "="*70)
        print("❌ IMPORT ERROR")
        print("="*70)
        print(f"\nFailed to import required module: {e}")
        print("\nThis usually means dependencies are missing.")
        print("\nPlease install GenAI dependencies:")
        print("  pip install -e \".[gen-ai]\"")
        print("\nFor detailed installation instructions, see:")
        print("  hailo_apps/python/gen_ai_apps/README.md")
        print("\n" + "="*70)
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()

