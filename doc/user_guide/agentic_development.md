# AI-Powered Development (Beta)

> **Status:** Beta — Available for early adopters. Feedback welcome on the [Hailo Community Forum](https://community.hailo.ai/).

Build complete Hailo AI applications using natural language and AI coding agents. The agents understand the Hailo SDK, GStreamer pipelines, model architectures, and code conventions — so you describe **what** you want and they build **how**.

<img src="../images/agentic_ai.gif" width="600"/>

🎮 See it in action: our [Easter Eggs game](../../hailo_apps/python/pipeline_apps/easter_game/) was built entirely autonomously by AI using this workflow.

## Prerequisites

- **Hailo development environment** set up ([Installation Guide](./installation.md))
- **One of these IDEs** with AI agent support:
  - [GitHub Copilot](https://github.com/features/copilot) in VS Code (agents via `@agent-name`)
  - [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (agents via `CLAUDE.md` + `.claude/`)
  - [Cursor](https://cursor.sh/) (rules via `.cursor/rules/`)

## Quick Start

### Example 1: Build a Detection Pipeline (via App Builder)

The **app builder** is the master router — describe any app and it picks the right specialist.

```
@hl-app-builder "person detection with tracking on USB camera"
```

The agent will:
1. Ask clarifying questions (app type, model, features, input source)
2. Present a build plan for your approval
3. Route to the **pipeline builder**
4. Generate all code in `hailo_apps/python/<type>/<app_name>/`
5. Validate and report

### Example 2: Build a VLM Monitor (directly)

If you know you want a VLM app, go straight to the specialist:

```
@hl-vlm-builder "dog monitoring camera that alerts when the dog is eating shoes"
```

The agent will:
1. Ask key decisions (continuous monitor vs interactive, camera type, events to track)
2. Load VLM patterns and conventions
3. Build the complete app with backend reuse, event tracking, graceful shutdown
4. Validate conventions and deliver with run instructions

## Available Agents

| Agent | Use Case | Hardware |
|-------|----------|----------|
| **hl-app-builder** | Master router — describe any app, get routed to the right specialist | All |
| **hl-vlm-builder** | Vision-Language Model apps (monitoring, scene analysis, visual Q&A) | Hailo-10H |
| **hl-pipeline-builder** | GStreamer video pipelines (detection, pose, segmentation, tracking) | All |
| **hl-standalone-builder** | OpenCV + HailoInfer apps (batch processing, custom inference) | All |
| **hl-llm-builder** | LLM chat and text generation apps | Hailo-10H |
| **hl-agent-builder** | Agents with LLM tool calling (smart assistants, API integrators) | Hailo-10H |
| **hl-voice-builder** | Voice assistants with Whisper STT + Piper TTS | All |

### When to Use the App Builder vs. a Specialist

- **Use `hl-app-builder`** when you're not sure which architecture fits, or when describing a complex app that might span categories.
- **Use a specialist directly** when you know exactly what type of app you want — it saves one routing step.

## How Agents Work

All agents follow an **interactive workflow** — they walk through key decisions with you before building, even when the request seems clear. This catches misunderstandings early and creates a collaborative experience.

Each agent follows a structured workflow:

1. **Phase 1: Understand** — The agent responds immediately, asks 2-3 key questions, presents a plan for approval
2. **Phase 2: Load Context** — After you approve, it reads relevant skill files, patterns, and conventions
3. **Phase 3: Build** — Creates all files in `hailo_apps/python/<type>/<app_name>/`
4. **Phase 4: Validate** — Checks conventions, imports, CLI, runs automated validation
5. **Phase 5: Report** — Presents what was built, how to run it, what it does
6. **Phase 6: Launch** — If you provide a video file or say "launch", runs the app automatically

### What Gets Created

```
hailo_apps/python/<type>/<app_name>/
├── __init__.py    # Package marker
├── <app_name>.py  # Main application code
├── app.yaml       # App manifest (name, type, hardware, model, tags)
├── run.sh         # Launch wrapper (sets PYTHONPATH)
├── README.md      # Usage and architecture docs
└── ...            # Additional modules as needed
```

### Running Your App

```bash
python hailo_apps/python/<type>/<app_name>/<app_name>.py --input usb
```

### Validating Your App

Before submitting, run the automated validator:

```bash
# Static checks only (11 convention checks)
python .hailo/scripts/validate_app.py hailo_apps/python/<type>/<app_name>

# Static checks + runtime smoke tests (CLI --help, module import)
python .hailo/scripts/validate_app.py hailo_apps/python/<type>/<app_name> --smoke-test
```

Smoke tests gracefully skip if Hailo hardware or GStreamer aren't available.

## Knowledge Base

The agentic knowledge lives in `.hailo/` and is automatically adapted for each IDE:

| Directory | Content |
|-----------|---------|
| `.hailo/agents/` | Agent definitions (7 agents) |
| `.hailo/skills/` | Detailed build skills per app type |
| `.hailo/instructions/` | Coding standards, architecture, testing |
| `.hailo/toolsets/` | SDK and API references |
| `.hailo/memory/` | Persistent patterns and pitfall avoidance |
| `.hailo/scripts/` | Automation tools (see below) |

### Scripts

| Script | Purpose |
|--------|---------|
| `validate_app.py` | Validate app conventions (11 static checks + 2 `--smoke-test` runtime checks) |
| `validate_framework.py` | Cross-reference integrity: routing table paths, file tree accuracy, `.hailo/` leak detection, agent handoffs, required sections, platform structural checks |
| `generate_platforms.py` | Sync `.hailo/` → `.github/`, `.claude/`, `.cursor/` (includes cross-ref validation via `--check`) |

All scripts live in `.hailo/scripts/` (source of truth) and are mirrored to `.github/scripts/`.

### Platform Sync

Platform-specific files are generated from `.hailo/` by running:

```bash
python .hailo/scripts/generate_platforms.py --generate
```

This produces:

| Platform | Entry Point | Output | Files | Strategy |
|---|---|---|---|---|
| GitHub Copilot | `copilot-instructions.md` | `.github/` | 59 | Full copies (auto-loaded by IDE) |
| Claude Code | `CLAUDE.md` | `.claude/` | 50 | Thin redirects → `.hailo/` |
| Cursor | `.cursor/rules/` | `.cursor/` | 48 | Thin `.mdc` redirects → `.hailo/` |

Copilot needs full copies because its auto-load mechanism can't read arbitrary files. Claude and Cursor read `.hailo/` directly at runtime, so thin redirects avoid duplication.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Agent doesn't know Hailo APIs | Check that `.hailo/` directory exists and is populated |
| Agent writes relative imports | Convention bug — agent should always use absolute imports |
| App fails `--help` | Check `run.sh` sets PYTHONPATH correctly |
| Agent builds in wrong directory | Should always be in `hailo_apps/python/<type>/`, using absolute imports |
| Platform configs are stale | Run `python .hailo/scripts/generate_platforms.py --generate` |
| Cross-references broken | Run `python .hailo/scripts/validate_framework.py -v` to find broken paths |
| Validation fails on hardware checks | Use `--smoke-test` — it gracefully skips if Hailo/GStreamer unavailable |
