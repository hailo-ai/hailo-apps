# Hailo Ollama with Open WebUI

A complete guide for setting up and using Hailo-Ollama with Open WebUI for an interactive AI chat interface.

## Overview

The Hailo Model Zoo GenAI is a curated collection of pre-trained models and example applications optimized for Hailo's AI processors, designed to accelerate GenAI application development. It includes Hailo-Ollama, an Ollama-compatible API written in C++ on top of HailoRT, enabling seamless integration with various external tools and frameworks.

Ollama simplifies running large language models locally by managing model downloads, deployments, and interactions through a convenient REST API. Models are specifically optimized for Hailo hardware, providing efficient, high-performance inference tailored for GenAI tasks.

Open WebUI is an extensible, feature-rich, and user-friendly self-hosted AI platform designed to operate entirely offline. Once Hailo-Ollama is running, you can use Open WebUI to interact with your models through a modern web interface.

For full details on Hailo Model Zoo GenAI, see: https://github.com/hailo-ai/hailo_model_zoo_genai?tab=readme-ov-file#basic-usage

## Part 1: Hailo-Ollama Installation and Setup

### Step 1: Download and Install Hailo GenAI Model Zoo

1. Visit: https://hailo.ai/developer-zone/
2. Download the appropriate package for your architecture
3. Install the package:
   ```bash
   sudo apt install hailo_gen_ai_model_zoo_<ver>_<arch>.deb
   ```

### Step 2: Start Hailo-Ollama Service

In a terminal window, start the Hailo-Ollama service:

```bash
hailo-ollama
```

The service will start and listen on `http://localhost:8000` by default.

### Step 3: Pull a Model

In another terminal window, pull a model:

```bash
curl --silent http://localhost:8000/api/pull \
     -H 'Content-Type: application/json' \
     -d '{ "model": "qwen2.5-instruct:1.5b", "stream" : true }'
```

The models will be downloaded to: `~/usr/share/hailo-ollama/models/blob/`

### Step 4: Test the Model

Test the model via API:

```bash
curl --silent http://localhost:8000/api/chat \
     -H 'Content-Type: application/json' \
     -d '{"model": "qwen2.5-instruct:1.5b", "messages": [{"role": "user", "content": "Translate to French: The cat is on the table."}]}'
```

If successful, you should receive a response from the model.

## Part 2: Open WebUI Integration

Once Hailo-Ollama is up and running, you can consume it with the popular Open WebUI for a user-friendly web interface.

### Prerequisites

- Docker must be installed and running
- For Docker installation instructions, see: https://docs.docker.com/engine/

### Installation

Based on the Open WebUI quick start guide: https://docs.openwebui.com/getting-started/quick-start

1. Download and run the **slim** variant
2. **Important:** Run with host network

```bash
docker pull ghcr.io/open-webui/open-webui:main-slim

# Run with host network (container shares host's network)
docker run -d --network host \
  -v open-webui:/app/backend/data \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main-slim
```

### Configure Open WebUI

1. Open your browser and navigate to the Open WebUI interface at: **http://localhost:8080**

2. In **Settings → Admin Settings → Connections**, add the Hailo-Ollama API URL:
   ```
   http://localhost:8000
   ```

3. Under the "Ollama API" section:
   - Set "Connection Type" to "Local"
   - Set "Auth" to "None"

4. Now in the chat, select one of the models served by Hailo-Ollama from the available models.

## Usage

Once configured, you can:

- Chat with models through the Open WebUI web interface
- Access models via the Hailo-Ollama REST API directly
- Use any tool that supports the Ollama API format

## Troubleshooting

### Service not starting
- Ensure Hailo-Ollama service is running: `hailo-ollama`
- Check that port 8000 is not already in use
- Verify Hailo GenAI Model Zoo is properly installed

### Model not found
- Verify the model was pulled successfully using the curl command
- Check that models are in: `~/usr/share/hailo-ollama/models/blob/`
- Ensure the model name matches exactly (e.g., "qwen2.5-instruct:1.5b")

### Connection issues in Open WebUI
- Check that the API URL in Open WebUI settings matches the hailo-ollama service URL
- Verify both services are running:
  - Hailo-Ollama: `http://localhost:8000`
  - Open WebUI: `http://localhost:8080`
- Ensure Docker container is using host network mode

### Port conflicts
- Default ports are:
  - 8000 (hailo-ollama)
  - 8080 (open-webui)
- If ports are in use, you may need to stop conflicting services or configure different ports

### Docker issues
- Ensure Docker is installed and running: `docker --version`
- Check Docker container status: `docker ps`
- View container logs: `docker logs open-webui`

## Additional Resources

- [Hailo Model Zoo GenAI GitHub](https://github.com/hailo-ai/hailo_model_zoo_genai)
- [Open WebUI Documentation](https://docs.openwebui.com/)
- [Docker Documentation](https://docs.docker.com/engine/)

