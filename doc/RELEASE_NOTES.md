# Hailo Apps – v25.12.0 Release Notes

**Release Date:** December 2025

---

## Welcome to Our Biggest Release Yet!

We’re thrilled to announce version 25.12.0 of the Hailo Applications Infrastructure — a complete transformation in how you build and deploy AI applications on Hailo hardware. This release focuses on making cutting-edge AI both accessible and practical, with a particular emphasis on **Generative AI** applications you can actually use.

---

## 🌟 Introducing GenAI Applications

### Voice Assistant – Your AI Companion

Talk naturally with AI using our complete voice assistant implementation. It listens, understands, and responds with natural speech — all running locally on your Hailo hardware.

**What it does:**

* Real-time speech recognition using Whisper
* Natural language understanding with LLMs
* Text-to-speech with Piper for human-like responses
* Continuous conversation with context awareness

**Perfect for:**

* Smart home control
* Interactive kiosks
* Accessibility applications
* Hands-free information systems

```bash
python -m hailo_apps.python.gen_ai_apps.voice_assistant.voice_assistant
```

---

### Agent Tools Example – AI That Controls Hardware

This isn’t just a chatbot — it’s an AI that can interact with the physical world. Our agent framework demonstrates how to give AI control over real hardware components.

**Capabilities:**

* **Servo Control** – Position servos using natural language (“move to 90 degrees”)
* **RGB LED Control** – Change colors and patterns (“make it blink red”)
* **Elevator Simulator** – Complex, multi-step operations (“go to floor 3”)
* **Extensible Tool System** – Easily add your own hardware integrations

**Use cases:**

* Robotics control
* Industrial automation
* Educational demonstrations
* Prototype testing

The framework automatically discovers available tools and lets the AI decide when and how to use them. Add your own tools by simply implementing the interface.

---

### VLM Chat – AI That Sees

Combine computer vision with language understanding. Point your camera at something and ask questions about what it sees.

**Examples:**

* “What objects do you see?”
* “Read the text in this image.”
* “How many people are in the room?”
* “What color is the car?”

The camera feed is processed in real time, and the AI maintains conversational context so you can ask follow-up questions naturally.

---

## 🎯 CLIP Zero-Shot Classification

Classify images without training a custom model. Just describe what you're looking for in plain English.

**How it works:**

* Type descriptions like “a red apple,” “a person waving,” “a stop sign”
* CLIP compares your text to the video feed in real time
* Get confidence scores for each description
* Interactive GUI for easy experimentation

**Real-world applications:**

* Quality control — “defective part vs good part”
* Inventory management — describe items you're looking for
* Security — detect unusual objects or situations
* Rapid prototyping — test ideas without training

```bash
hailo-clip
```

The GUI lets you add and remove descriptions on the fly — perfect for exploring new use cases or demonstrating concepts to stakeholders.

---

## 🚀 Standalone Applications

Sometimes you don’t need a full pipeline — you just need something that works. Our standalone applications are self-contained and easy to integrate.

### Speech Recognition

Transform spoken words into text with a clean GUI.

**Features:**

* Real-time transcription using Whisper
* Automatic microphone detection
* Audio level visualization
* Save transcripts to file

**Use it for:**

* Meeting transcription
* Voice note-taking
* Dictation systems
* Audio data collection

---

### Paddle OCR

Extract text from images and documents with high accuracy.

**Capabilities:**

* Text detection and recognition
* Multilingual support
* Rotated text handling
* Structured output

**Applications:**

* Document digitization
* License plate recognition
* Sign reading
* Form processing

---

### Lane Detection (Beta)

Real-time road lane detection for automotive applications.

**Features:**

* Multiple lane detection
* Curve estimation
* Distance calculation
* Overlay visualization

**Ideal for:**

* ADAS development
* Autonomous vehicle prototypes
* Driver assistance systems
* Traffic analysis

---

### Super Resolution

Enhance image quality and upscale resolution using AI.

**Benefits:**

* Improved detail in low-resolution images
* Real-time processing
* Configurable upscaling factors
* Minimal artifacts

---

### Oriented Object Detection

Detect objects using rotated bounding boxes — perfect for aerial imagery and complex scenes.

**Why it matters:**

* Handles objects at any angle
* Based on the DOTA dataset (ships, planes, vehicles)
* More accurate than axis-aligned boxes
* Essential for satellite/drone imagery

---

## ⚡ Native C++ Applications

For performance-critical or embedded systems, we now provide complete C++ implementations.

### What’s Included

**Core Vision Tasks:**

* **Object Detection** — YOLO-based detection with configurable models
* **Instance Segmentation** — Pixel-perfect object boundaries
* **Pose Estimation** — YOLOv8 human pose detection
* **Semantic Segmentation** — Scene understanding with Cityscape models
* **Classification** — ImageNet-based image classification

**Advanced Capabilities:**

* **Depth Estimation** — Monocular and stereo variants
* **Zero-Shot Classification** — CLIP in native C++
* **Oriented Object Detection** — Rotated bounding boxes

**Special Features:**

* **ONNX Runtime Integration** — Example pipeline using ONNX-RT with Hailo
* **Custom Tokenizer** — Full CLIP tokenization in C++
* **Optimized Performance** — Native implementations for minimal overhead

Each application includes build scripts and clear examples:

```bash
cd hailo_apps/cpp/object_detection
./build.sh
./build/object_detection
```

---

## 🛠️ Getting Started Examples

We know the hardest part is getting started. That’s why we’ve included simple, focused examples:

### Simple LLM Chat

Your first text conversation with an AI. No camera, no audio — just pure language interaction.

```bash
python -m hailo_apps.python.gen_ai_apps.simple_llm_chat.simple_llm_chat
```

Perfect for understanding how LLMs work and testing prompt strategies.

---

### Simple VLM Chat

Like LLM chat, but with vision. Show your camera something and ask about it.

**Try asking:**

* “What do you see?”
* “Describe this object.”
* “What’s unusual about this scene?”

---

### Simple Whisper Chat

The simplest speech recognition example. Speak and see your words transcribed.

Great for testing your microphone setup and understanding how speech-to-text works before building something more complex.

---

## 💡 Practical Tools and Utilities

### GenAI Utils Library

We’ve built the infrastructure so you don’t have to.

**Context Management:**

* Automatic conversation history tracking
* Token counting and optimization
* Context window management
* State persistence

**Streaming Support:**

* Real-time token generation
* Progressive UI updates
* Graceful error handling
* Cancellation support

**Tool System:**

* Automatic tool discovery
* JSON schema validation
* Execution sandboxing
* Result formatting

**Message Handling:**

* Multi-modal message formatting
* Image encoding/decoding
* Audio preprocessing
* Metadata management

---

### Voice Processing Suite

Everything you need for audio applications.

**Audio I/O:**

* Cross-platform device detection
* Automatic sample rate handling
* Buffer management
* Low-latency streaming

**Speech Recognition:**

* Whisper model integration
* Multiple model sizes
* Language detection
* Confidence scoring

**Text-to-Speech:**

* Piper TTS integration
* Natural-sounding voices
* Customizable speech rate
* Multiple voice options

**Diagnostics:**

* Audio device testing
* Latency measurement
* Quality analysis
* Troubleshooting tools

---

## 🎨 Developer Experience Improvements

### Unified Command-Line Interface

Every application now shares consistent CLI options:

```bash
--input-source    # Camera, video file, or image  
--hef-path        # Model file location  
--labels-json     # Class labels  
--disable-sync    # Performance tuning  
--show-fps        # Display performance  
--debug           # Detailed logging  
```

No more memorizing different flags for different applications.

---

### Automatic Hardware Detection

The system automatically detects your Hailo hardware and configures accordingly:

* Hailo-8
* Hailo-8L
* Hailo-10H

No manual configuration needed — it just works.

---

### Better Logging

Clean, informative logs that help you debug effectively:

* Concise, readable format
* Adjustable verbosity levels
* Performance metrics in debug mode
* Clear error messages with suggestions

---

### Pipeline Watchdog

Your applications are now more robust. If a pipeline gets stuck, the watchdog detects and recovers automatically — no more hanging applications requiring manual restart.

---

## 📚 Enhanced Documentation

We’ve completely rewritten our documentation with you in mind.

**New Guides:**

* **GStreamer Helpers** — Understand our pipeline architecture
* **GST-Shark Debugging** — Profile and optimize your pipelines
* **Model Compilation** — Jupyter notebook walkthrough
* **Model Retraining** — Step-by-step guide

**Updated Documentation:**

* Application running guide with new structure
* Parallel execution guide for multi-process workflows
* Repository structure overview
* Complete API reference

---

## 🔧 Installation and Configuration

### Cleaner Installation Process

We’ve streamlined installation to be more reliable and maintainable.

* Separated Python dependency management
* Improved resource download with retry logic
* Better error messages
* Faster downloads via parallel fetching

---

### Resource Management

New configuration system for models and resources:

* JSON configs for bash downloader
* YAML configs for Python applications
* Automatic version management
* Missing resource detection

---

### Easy Updates

Updating is now straightforward:

```bash
git pull origin dev
sudo ./scripts/cleanup_installation.sh
sudo ./install.sh
```

The cleanup script ensures removal of old files for a clean installation.

---

## 🎯 Use Case Examples

Let’s talk about what you can actually build with this release.

### Smart Retail

**Inventory Assistant:**

* Voice assistant for employees: “How many red shirts do we have?”
* CLIP for finding specific items: “blue sneakers, size 10”
* OCR for automatic label reading
* Object detection for shelf monitoring

---

### Industrial Automation

**Quality Control Station:**

* Object detection for part identification
* CLIP for defect classification: “cracked surface” vs “good surface”
* Agent tools for robotic positioning
* Speech recognition for hands-free operation

---

### Healthcare

**Patient Assistance:**

* Voice assistant for patient information
* VLM for reading medication labels
* OCR for form digitization
* Speech recognition for clinical notes

---

### Education

**Interactive Learning:**

* VLM chat for object identification lessons
* Agent tools for physics demonstrations
* Speech recognition for language practice
* CLIP for educational material search

---

### Automotive

**ADAS Prototyping:**

* Lane detection for path tracking
* Object detection for obstacle avoidance
* Depth estimation for distance measurement
* C++ implementations for ECU integration

---

## 🔄 Migration Guide

### If You’re Upgrading

**Import Path Changes:**

Old:

```python
from standalone_apps.chat_agent import ChatAgent
```

New:

```python
from hailo_apps.python.gen_ai_apps.agent_tools_example import Agent
```

**Configuration Changes:**

The old `agent_config.yaml` is now replaced with a Python-based configuration:

```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils import Config

config = Config(
    system_prompt="You are a helpful assistant",
    max_tokens=1000,
    temperature=0.7
)
```

**Hardware Naming Update:**

```python
# Old
device = "hailo10"

# New
device = "hailo10h"
```

---

## 🤝 Community and Support

### We Want Your Feedback

This release represents our vision — but we want to hear yours:

* What applications are you building?
* What features would make your work easier?
* Where did you get stuck?
* What worked surprisingly well?

---

### Get Help

**Documentation Portal:**
[https://hailo.ai/developer-zone](https://hailo.ai/developer-zone)

**Community Forum:**
[https://community.hailo.ai](https://community.hailo.ai) — Join discussions, share projects, get help

**GitHub Issues:**
Report bugs and request features on our repository

**Direct Examples:**
Every application includes a README with usage examples and tips

---

## 🙏 Thank You

This release was made possible by our amazing team and community.

**Core Contributors:**
Gilad Nahor, OmriAx, mikehailodev, nina-vilela, Marina Vilela Bento, and the entire Hailo engineering team

**Community:**
Thank you to everyone who reported issues, suggested features, and shared their projects. You inspire us to build better tools.

---

## 🚀 What’s Next?

This is just the beginning. We’re already working on:

* More GenAI application templates
* Additional hardware tool integrations
* Performance optimizations
* An expanded model zoo
* Enhanced debugging tools

Stay tuned — and happy building!

---

## ✅ Quick Start Checklist

1. **Install the release:**

   ```bash
   git clone https://github.com/hailo-ai/hailo-apps-infra.git
   cd hailo-apps-infra
   sudo ./install.sh
   ```

2. **Try the voice assistant:**

   ```bash
   python -m hailo_apps.python.gen_ai_apps.voice_assistant.voice_assistant
   ```

3. **Experiment with CLIP:**

   ```bash
   hailo-clip
   ```

4. **Build a C++ application:**

   ```bash
   cd hailo_apps/cpp/object_detection
   ./build.sh && ./build/object_detection
   ```

5. **Read the docs and start building!**

---

**Version:** 25.12.0
**Release Branch:** `dev`
**Release Team:** Hailo Technologies

*Built with ❤️ for the AI community*
