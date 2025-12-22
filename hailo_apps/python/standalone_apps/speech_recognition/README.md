# Automatic Speech Recognition with Whisper model

This application performs a speech-to-text transcription using OpenAI's *Whisper-tiny* and *Whisper-base* model on the Hailo-8/8L/10H AI accelerators.

## Prerequisites

Ensure your system matches the following requirements before proceeding:

- Platforms tested: x86, Raspberry Pi 5
- OS: Ubuntu 22 (x86) or Raspberry OS.
- **HailoRT 4.20 or 4.21** and the corresponding **PCIe driver** must be installed. You can download them from the [Hailo Developer Zone](https://hailo.ai/developer-zone/)
- **ffmpeg** and **libportaudio2** installed for audio processing.
  ```
  sudo apt update
  sudo apt install ffmpeg
  sudo apt install libportaudio2
  ```
- **Python 3.10 or 3.11** installed.

## Before running the app

- Make sure you have a microphone connected to your system. If you have multiple microphones connected, please make sure the proper one is selected in the system configuration, and that the input volume is set to a medium/high level.  
  A good quality microphone (or a USB camera) is suggested to acquire the audio.
- The application allows the user to acquire and process an audio sample up to 5 seconds long. The duration can be modified in the application code.
- The current pipeline supports **English language only**.

## Installation and Usage

Run this app in one of two ways:
1. Standalone installation in a clean virtual environment (no TAPPAS required) — see [Option 1](#option-1-standalone-installation)
2. From an installed `hailo-apps` repository — see [Option 2](#option-2-inside-an-installed-hailo-apps-repository)

## Option 1: Standalone Installation

To avoid compatibility issues, it's recommended to use a clean virtual environment.

0. Clone the repository and install Python dependencies:
    ```shell script
    git clone https://github.com/hailo-ai/hailo-apps.git
    cd hailo-apps
    pip install -e ".[speech-rec]"
    ```

1. Download resources:
    ```shell script
    python3 hailo_apps/python/standalone_apps/speech_recognition/app/download_resources.py --hw-arch <hailo device type>
    ```

## Option 2: Inside an Installed hailo-apps Repository
If you installed the full repository:
```shell script
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
sudo ./install.sh
source setup_env.sh
```
Then the app is already ready for usage:
```shell script
cd hailo_apps/python/standalone_apps/speech_recognition/app
```

## Run

Run from the application folder.

CLI:
```shell script
python3 -m app.app_hailo_whisper [--hw-arch hailo8l] [--variant base|tiny]
```

Script:
```shell script
python3 app_hailo_whisper.py [--hw-arch hailo8l] [--variant base|tiny]
```

GUI:
```shell script
streamlit run gui/gui.py -- [--hw-arch hailo8l] [--variant base|tiny]
```

To see all possible arguments:
```shell script
python3 app_hailo_whisper.py --help
```

### Command line arguments
Use the `python3 -m app.app_hailo_whisper --help` command to print the helper.

The following command line options are available:

- **--reuse-audio**: Reloads the audio from the previous run.
- **--hw-arch**: Selects the Whisper models compiled for the target architecture (*hailo8* / *hailo8l / hailo10h*). If not specified, the *hailo8* architecture is selected.
- **--variant**: Variant of the Whisper model to use (*tiny* / *base*). If not specified, the *base* model is used.
- **--multi-process-service**: Enables the multi-process service, to run other models on the same chip in addition to Whisper

## Usage from GUI
1. Activate the virtual environment from the repository root folder:

   ```sh
   source whisper_env/bin/activate
   ```
2. Install **streamlit**:
   ```sh
   pip install streamlit
   ```
3. Set the PYTHONPATH to the repository root folder:
   ```sh
   export PYTHONPATH=$(pwd)
   ```
4. Run the GUI:
   ```
   streamlit run gui/gui.py
   ```
5. The *--hw-arch* and *--variant* arguments are available for the GUI as well.
   Please use **--** as a separator between the streamlit command and the arguments, for example:
   ```
   streamlit run gui/gui.py -- --hw-arch hailo8l
   ```


## Additional notes

- This application is just an example to show how to run a Whisper-based pipeline on the Hailo-8/8L/10H AI accelerator, and it is not focused on optimal pre/post-processing.
- Torch is still required for pre-processing. It will be removed in the next release.
- We are considering future improvements, like:
  - Release scripts for model conversion
  - Optimized post-processing to improve transcription's accuracy
  - Additional models support
  - Dedicated C++ implementation  

  Feel free to share your suggestions in the [Hailo Community](https://community.hailo.ai/) regarding how this application can be improved.

## Troubleshooting

- Make sure that the microphone is connected to your host and that it can be detected by the system.
- Post-processing is being applied to improve the quality of the transcription, e.g. applying peanlty on repeated tokens and removing model's hallucinations (if any). These methods can be modified by the user to find an optimal solution.
- In the CLI application, the `--reuse-audio` flag can be used to load the audio acquired during the previous run, for debugging purposes.
- If the transcription is not generated, listen to the saved audio record to make sure that the audio was actually recorded and that the quality is good.

## Disclaimer
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.
