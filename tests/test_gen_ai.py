import os
import sys
import pytest
import subprocess
import time
import signal
from pathlib import Path

# Apply genai marker to all tests in this module
pytestmark = pytest.mark.genai

# Constants
TIMEOUT_DEFAULT = 60
TIMEOUT_LONG = 120
REPO_ROOT = Path(__file__).parent.parent

# Skip all gen_ai tests unless running on Hailo-10H
try:
    from hailo_apps.python.core.common.installation_utils import detect_hailo_arch
    from hailo_apps.python.core.common.defines import HAILO10H_ARCH
    _is_hailo10h = detect_hailo_arch() == HAILO10H_ARCH
except Exception:
    _is_hailo10h = False

pytestmark = pytest.mark.skipif(
    not _is_hailo10h,
    reason="Gen AI tests require Hailo-10H architecture",
)

def run_app_subprocess(module_path, args=None, input_text=None, timeout=TIMEOUT_DEFAULT, check_retcode=True):
    """
    Runs a python module as a subprocess.
    """
    cmd = [sys.executable, str(module_path)]
    if args:
        cmd.extend(args)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    print(f"Running command: {' '.join(cmd)}")

    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=0,
            preexec_fn=os.setsid  # Create new session for easier group killing
        )

        stdout, stderr = process.communicate(input=input_text, timeout=timeout)

        print("STDOUT:", stdout)
        print("STDERR:", stderr)

        if check_retcode and process.returncode != 0:
            pytest.fail(f"Process failed with return code {process.returncode}.\nStderr: {stderr}")

        return stdout, stderr

    except subprocess.TimeoutExpired:
        print(f"Process timed out after {timeout} seconds. Killing...")
        kill_process_group(process)
        stdout, stderr = process.communicate()
        print("STDOUT (partial):", stdout)
        print("STDERR (partial):", stderr)
        pytest.fail(f"Process timed out after {timeout} seconds")

    except Exception as e:
        if process:
            kill_process_group(process)
        pytest.fail(f"Failed to run process: {e}")
    finally:
        if process and process.poll() is None:
            kill_process_group(process)
        # Give some time for the process to release hardware resources (VDevice)
        time.sleep(2)

def kill_process_group(process):
    """Robustly kill a process and its process group."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        time.sleep(0.5)
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass # Process already gone
    except Exception as e:
        print(f"Error killing process group: {e}")

@pytest.fixture
def gen_ai_apps_dir():
    return REPO_ROOT / "hailo_apps" / "python" / "gen_ai_apps"

def test_simple_llm_chat(gen_ai_apps_dir):
    script_path = gen_ai_apps_dir / "simple_llm_chat" / "simple_llm_chat.py"
    stdout, _ = run_app_subprocess(script_path)
    # Check for either success marker or just non-failure exit
    assert "Response received" in stdout
    assert "Example completed successfully" in stdout

def test_simple_vlm_chat(gen_ai_apps_dir):
    script_path = gen_ai_apps_dir / "simple_vlm_chat" / "simple_vlm_chat.py"
    stdout, _ = run_app_subprocess(script_path)
    assert "Response received" in stdout
    assert "Example completed successfully" in stdout

def test_simple_whisper_chat(gen_ai_apps_dir):
    script_path = gen_ai_apps_dir / "simple_whisper_chat" / "simple_whisper_chat.py"
    stdout, _ = run_app_subprocess(script_path)
    assert "Transcription completed" in stdout
    assert "Example completed successfully" in stdout

@pytest.mark.skip(reason="Interactive test with GUI hangs in headless environment")
def test_vlm_chat_interactive(gen_ai_apps_dir):
    script_path = gen_ai_apps_dir / "vlm_chat" / "vlm_chat.py"
    input_str = "\nDescribe this image\nq\n"
    stdout, stderr = run_app_subprocess(
        script_path,
        args=["--input", "usb"],
        input_text=input_str,
        timeout=TIMEOUT_LONG
    )
    assert "RESULT READY" in stdout or "Answer:" in stdout

def test_voice_assistant_startup(gen_ai_apps_dir):
    script_path = gen_ai_apps_dir / "voice_assistant" / "voice_assistant.py"
    cmd = [sys.executable, str(script_path), "--no-tts"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    print(f"Running command: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=0,
        preexec_fn=os.setsid
    )

    try:
        start_time = time.time()
        max_duration = 60
        initialized = False

        while time.time() - start_time < max_duration:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print("STDOUT:", stdout)
                print("STDERR:", stderr)
                pytest.fail(f"Voice assistant exited prematurely with code {process.returncode}")

            line = process.stdout.readline()
            if line:
                print(f"[App]: {line.strip()}")
                if "AI components ready" in line or "Listening" in line:
                    initialized = True
                    break
            time.sleep(0.1)

        if not initialized:
            pytest.fail("Voice assistant did not initialize within timeout")

        time.sleep(5)
        if process.poll() is not None:
             pytest.fail(f"Voice assistant crashed after initialization. Return code: {process.returncode}")

    except Exception:
        kill_process_group(process)
        raise
    finally:
        kill_process_group(process)
        time.sleep(2) # Release resources

def test_agent_tools_example_math(gen_ai_apps_dir):
    script_path = gen_ai_apps_dir / "agent_tools_example" / "agent.py"
    input_str = "Calculate 5 + 3\n/exit\n"
    stdout, _ = run_app_subprocess(
        script_path,
        args=["--tool", "math"],
        input_text=input_str,
        timeout=TIMEOUT_LONG
    )
    assert "Tool call detected" in stdout or "tool_call" in stdout or "Tool execution result" in stdout
    assert "8" in stdout
