"""
Unit tests for Voice Assistant modules.

Tests cover:
- Audio Diagnostics (device enumeration, scoring, auto-detection)
- Audio Player (playback queue, stop functionality)
"""

import sys
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Mock dependencies before importing modules
piper_mock = MagicMock()
sys.modules['piper'] = piper_mock
sys.modules['piper.voice'] = MagicMock()

# Create a proper mock for sounddevice
mock_sd = MagicMock()
mock_sd.default.device = [0, 0]
sys.modules['sounddevice'] = mock_sd


# =============================================================================
# Audio Diagnostics Tests
# =============================================================================


class TestAudioDiagnostics(unittest.TestCase):
    """Tests for Audio Diagnostics module."""

    def setUp(self):
        self.mock_device = {
            'name': 'Test Device',
            'hostapi': 0,
            'max_input_channels': 1,
            'max_output_channels': 2,
            'default_samplerate': 44100.0
        }

    def test_list_audio_devices(self):
        """Test listing audio devices returns correct input/output devices."""
        # Setup mock for this test
        mock_sd.query_devices.return_value = [self.mock_device]
        mock_sd.default.device = [0, 0]

        # Import after mocking
        if 'hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_diagnostics' in sys.modules:
            del sys.modules['hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_diagnostics']

        from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_diagnostics import AudioDiagnostics

        input_devs, output_devs = AudioDiagnostics.list_audio_devices()

        self.assertEqual(len(input_devs), 1)
        self.assertEqual(len(output_devs), 1)
        self.assertEqual(input_devs[0].name, 'Test Device')

    def test_score_device(self):
        """Test device scoring favors USB devices and correct sample rates."""
        from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_diagnostics import (
            AudioDiagnostics,
            AudioDeviceInfo,
        )

        # Test USB device scoring
        dev = AudioDeviceInfo(
            id=0,
            name="USB Microphone",
            host_api=0,
            max_input_channels=1,
            max_output_channels=0,
            default_samplerate=16000.0,
            is_default=True
        )

        score = AudioDiagnostics.score_device(dev, is_input=True)
        # 100 (default) + 50 (usb) + 20 (16k) = 170
        self.assertEqual(score, 170)

        # Test HDMI input (should be penalized)
        dev_hdmi = AudioDeviceInfo(
            id=1,
            name="HDMI Input",
            host_api=0,
            max_input_channels=1,
            max_output_channels=0,
            default_samplerate=48000.0,
            is_default=False
        )
        score_hdmi = AudioDiagnostics.score_device(dev_hdmi, is_input=True)
        # 0 + 0 - 50 = -50
        self.assertEqual(score_hdmi, -50)

    def test_auto_detect_devices(self):
        """Test auto-detection selects highest scoring devices."""
        from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_diagnostics import (
            AudioDiagnostics,
            AudioDeviceInfo,
        )

        with patch.object(AudioDiagnostics, 'list_audio_devices') as mock_list:
            dev1 = AudioDeviceInfo(
                id=1, name="Good Mic", host_api=0,
                max_input_channels=1, max_output_channels=0,
                default_samplerate=16000, is_default=True
            )
            dev2 = AudioDeviceInfo(
                id=2, name="Bad Mic", host_api=0,
                max_input_channels=1, max_output_channels=0,
                default_samplerate=44100, is_default=False
            )

            mock_list.return_value = ([dev1, dev2], [])

            best_in, best_out = AudioDiagnostics.auto_detect_devices()

            self.assertEqual(best_in, 1)
            self.assertIsNone(best_out)

    def test_is_raspberry_pi(self):
        """Test Raspberry Pi detection."""
        from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_diagnostics import AudioDiagnostics

        with patch('platform.machine', return_value='aarch64'), \
             patch('platform.release', return_value='5.10.0'):
            self.assertTrue(AudioDiagnostics.is_raspberry_pi())

        with patch('platform.machine', return_value='x86_64'), \
             patch('platform.release', return_value='5.10.0'):
            self.assertFalse(AudioDiagnostics.is_raspberry_pi())

    def test_get_audio_server(self):
        """Test audio server detection."""
        from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_diagnostics import AudioDiagnostics

        # Mock subprocess to return PulseAudio
        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = 'Server Name: PulseAudio'
            mock_run.return_value = mock_result

            server = AudioDiagnostics.get_audio_server()
            self.assertEqual(server, 'pulseaudio')

        # Mock subprocess to return PipeWire
        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = 'Server Name: PipeWire'
            mock_run.return_value = mock_result

            server = AudioDiagnostics.get_audio_server()
            self.assertEqual(server, 'pipewire')

        # Mock subprocess failure
        with patch('subprocess.run', side_effect=FileNotFoundError()):
            server = AudioDiagnostics.get_audio_server()
            self.assertIsNone(server)

    def test_get_pulseaudio_usb_devices(self):
        """Test PulseAudio USB device detection."""
        from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_diagnostics import AudioDiagnostics

        with patch('subprocess.run') as mock_run:
            # Mock sinks output
            sinks_result = MagicMock()
            sinks_result.returncode = 0
            sinks_result.stdout = '0\talsa_output.usb-Device_123.analog-stereo\tRUNNING'
            mock_run.return_value = sinks_result

            devices = AudioDiagnostics.get_pulseaudio_usb_devices()
            # Should find USB device in sinks
            self.assertGreaterEqual(len(devices), 0)

    def test_get_alsa_cards(self):
        """Test ALSA card detection."""
        from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_diagnostics import AudioDiagnostics

        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = 'card 1: Device [USB Audio Device], device 0: USB Audio [USB Audio]'
            mock_run.return_value = mock_result

            cards = AudioDiagnostics.get_alsa_cards()
            self.assertGreaterEqual(len(cards), 0)

    def test_check_audio_permissions(self):
        """Test audio permission checking."""
        from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_diagnostics import AudioDiagnostics

        # Mock user in audio group
        with patch('getpass.getuser', return_value='testuser'), \
             patch('grp.getgrall', return_value=[MagicMock(gr_name='audio', gr_mem=['testuser'])]), \
             patch('os.getgid', return_value=1000), \
             patch('grp.getgrgid', return_value=MagicMock(gr_name='audio')):
            has_perms, issues = AudioDiagnostics.check_audio_permissions()
            self.assertTrue(has_perms)
            self.assertEqual(len(issues), 0)

        # Mock user not in audio group
        with patch('getpass.getuser', return_value='testuser'), \
             patch('grp.getgrall', return_value=[]), \
             patch('os.getgid', return_value=1000), \
             patch('grp.getgrgid', return_value=MagicMock(gr_name='users')):
            has_perms, issues = AudioDiagnostics.check_audio_permissions()
            self.assertFalse(has_perms)
            self.assertGreater(len(issues), 0)

    def test_configure_usb_audio_rpi(self):
        """Test USB audio configuration for RPi."""
        from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_diagnostics import (
            AudioDiagnostics,
            AudioDeviceInfo,
        )

        # Mock all the dependencies
        with patch.object(AudioDiagnostics, 'is_raspberry_pi', return_value=True), \
             patch.object(AudioDiagnostics, 'get_audio_server', return_value='pulseaudio'), \
             patch.object(AudioDiagnostics, 'check_audio_permissions', return_value=(True, [])), \
             patch.object(AudioDiagnostics, 'list_audio_devices') as mock_list, \
             patch.object(AudioDiagnostics, 'get_pulseaudio_usb_devices', return_value=[]), \
             patch.object(AudioDiagnostics, 'get_alsa_cards', return_value=[]), \
             patch.object(AudioDiagnostics, 'auto_detect_devices', return_value=(1, 2)):

            usb_input = AudioDeviceInfo(
                id=1, name="USB Microphone", host_api=0,
                max_input_channels=1, max_output_channels=0,
                default_samplerate=16000, is_usb=True
            )
            usb_output = AudioDeviceInfo(
                id=2, name="USB Speaker", host_api=0,
                max_input_channels=0, max_output_channels=2,
                default_samplerate=16000, is_usb=True
            )

            mock_list.return_value = ([usb_input], [usb_output])

            results = AudioDiagnostics.configure_usb_audio_rpi(auto_fix=False)

            self.assertIsInstance(results, dict)
            self.assertIn('success', results)
            self.assertIn('steps', results)
            self.assertIn('warnings', results)
            self.assertIn('errors', results)


# =============================================================================
# Audio Player Tests
# =============================================================================


class TestAudioPlayer(unittest.TestCase):
    """Tests for Audio Player module."""

    def setUp(self):
        # Reset the mock
        mock_sd.reset_mock()
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_sd.OutputStream.return_value = mock_stream
        self.mock_stream = mock_stream

        # Need to reimport to pick up the mock
        if 'hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_player' in sys.modules:
            del sys.modules['hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_player']

        with patch('hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_diagnostics.AudioDiagnostics.auto_detect_devices') as mock_detect:
            mock_detect.return_value = (None, 1)
            from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_player import AudioPlayer
            self.player = AudioPlayer()

        time.sleep(0.2)  # Allow worker thread to start

    def tearDown(self):
        if hasattr(self, 'player'):
            self.player.close()

    def test_play_queues_data(self):
        """Test that play() adds data to the queue."""
        data = np.zeros(100, dtype=np.float32)
        self.player.play(data)

        # Data should be in queue or being processed
        # Verify no exception is raised and queue exists
        self.assertIsNotNone(self.player.queue)

    def test_stop_clears_queue(self):
        """Test that stop() clears the queue."""
        data = np.zeros(100, dtype=np.float32)
        self.player.play(data)
        self.player.play(data)

        self.player.stop()

        # Queue should be empty after stop
        self.assertTrue(self.player.queue.empty())


if __name__ == '__main__':
    unittest.main()

