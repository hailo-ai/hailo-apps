import os
import signal
import subprocess
import time

from .defines import UDEV_CMD
from .hailo_logger import get_logger

hailo_logger = get_logger(__name__)

# if udevadm is not installed, install it using the following command:
# sudo apt-get install udev


# Checks if a Raspberry Pi camera is connected and responsive.
def is_rpi_camera_available():
    """Returns True if the RPi camera is connected."""
    hailo_logger.debug("Checking if Raspberry Pi camera is available...")
    try:
        process = subprocess.Popen(
            ["rpicam-hello", "-t", "0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        hailo_logger.debug("Started rpicam-hello process.")
        time.sleep(5)
        process.send_signal(signal.SIGTERM)
        hailo_logger.debug("Sent SIGTERM to rpicam-hello process.")
        process.wait(timeout=2)
        stdout, stderr = process.communicate()
        stderr_str = stderr.decode().lower()
        hailo_logger.debug(f"rpicam-hello stderr: {stderr_str}")
        if "no cameras available" in stderr_str:
            hailo_logger.info("No Raspberry Pi cameras detected.")
            return False
        hailo_logger.info("Raspberry Pi camera is available.")
        return True
    except Exception as e:
        hailo_logger.error(f"Error checking Raspberry Pi camera: {e}")
        return False

# Checks if a USB camera is connected and responsive.
def get_usb_video_devices():
    """Get a list of video devices that are connected via USB and have video capture capability."""
    hailo_logger.debug("Scanning /dev for video devices...")
    video_devices = [
        f"/dev/{device}" for device in os.listdir("/dev") if device.startswith("video")
    ]
    usb_video_devices = []
    hailo_logger.debug(f"Found video devices: {video_devices}")

    for device in video_devices:
        try:
            hailo_logger.debug(f"Checking device: {device}")
            # Use udevadm to get detailed information about the device
            udevadm_cmd = [UDEV_CMD, "info", "--query=all", "--name=" + device]
            hailo_logger.debug(f"Running command: {' '.join(udevadm_cmd)}")
            result = subprocess.run(udevadm_cmd, check=False, capture_output=True)
            output = result.stdout.decode("utf-8")
            hailo_logger.debug(f"udevadm output for {device}: {output}")

            # Check if the device is connected via USB and has video capture capabilities
            if "ID_BUS=usb" in output and ":capture:" in output:
                hailo_logger.info(f"USB camera detected: {device}")
                usb_video_devices.append(device)
        except Exception as e:
            hailo_logger.error(f"Error checking device {device}: {e}")

    hailo_logger.debug(f"USB video devices found: {usb_video_devices}")
    return usb_video_devices


def main():
    hailo_logger.debug("Running main() to check for USB cameras.")
    usb_video_devices = get_usb_video_devices()

    if usb_video_devices:
        hailo_logger.info(f"USB cameras found on: {', '.join(usb_video_devices)}")
        print(f"USB cameras found on: {', '.join(usb_video_devices)}")
    else:
        hailo_logger.info("No available USB cameras found.")
        print("No available USB cameras found.")


if __name__ == "__main__":
    main()
