# .hailo/ — Shared Agentic Knowledge

Platform-neutral knowledge base for building Hailo AI applications. This directory is the **canonical source of truth** for all AI coding agents (GitHub Copilot, Claude Code, Cursor, or any LLM-based agent).

Platform-specific configurations (`.github/`, `.claude/`, `.cursor/`) are **generated** from this directory using the generator script. Never edit generated files directly — edit here, then regenerate.

## Directory Structure

```
.hailo/
├── README.md                          ← This file (master index)
├── agents/                            ← Platform-neutral agent definitions
├── contextual-rules/                  ← File-pattern-triggered context rules
├── instructions/                      ← Architecture & development standards
├── knowledge/                         ← Structured knowledge bases (YAML)
├── memory/                            ← Persistent cross-session knowledge
├── prompts/                           ← Build prompt templates
├── scripts/                           ← Generator, validation & curation scripts
├── skills/                            ← Step-by-step workflow guides (hl- prefix)
├── templates/                         ← Platform-specific entry point templates
│   └── copilot-instructions.md        ← Copilot global instructions template
└── toolsets/                          ← API references
```

## Instructions (`instructions/`)

Architecture and development standards for building Hailo apps.

| File | Description |
|---|---|
| `architecture.md` | Three-tier app architecture (pipeline, standalone, gen-ai), module dependency graph |
| `coding-standards.md` | Mandatory coding conventions: imports, logging, HEF resolution, CLI parsers, error handling |
| `gen-ai-development.md` | Hailo-10H gen AI guide: VLM/LLM/Whisper patterns, multiprocessing backend |
| `gstreamer-pipelines.md` | GStreamer pipeline composition: source/inference/display fragments, callback patterns |
| `testing-patterns.md` | Test framework, markers, fixtures, test patterns |
| `orchestration.md` | Multi-agent orchestration: plan-and-execute loops, phase gates, sub-agent delegation |
| `agent-protocols.md` | Behavioral contracts for agents: context-first execution, phase gates, todo management |

## Skills (`skills/`)

Step-by-step workflow guides for agents. All use the `hl-` prefix.

### Build Skills
| File | Description |
|---|---|
| `hl-build-vlm-app.md` | Build Vision-Language Model apps for Hailo-10H |
| `hl-build-pipeline-app.md` | Build GStreamer pipeline apps for real-time video |
| `hl-build-standalone-app.md` | Build standalone HailoRT inference apps |
| `hl-build-agent-app.md` | Build AI agent apps with LLM tool calling |
| `hl-build-llm-app.md` | Build LLM chat and text generation apps |
| `hl-build-voice-app.md` | Build voice assistant apps (Whisper STT + Piper TTS) |

### Pattern Skills (Reference)
| File | Description |
|---|---|
| `hl-monitoring.md` | Continuous video monitoring with periodic VLM analysis |
| `hl-event-detection.md` | Detect specific events from VLM responses |
| `hl-camera.md` | Camera setup, discovery, and management (USB, RPi, RTSP) |
| `hl-model-management.md` | HEF model resolution, download, and configuration |

### Meta Skills
| File | Description |
|---|---|
| `hl-plan-and-execute.md` | Orchestrated multi-phase workflow: plan, delegate, execute, gate |
| `hl-validate.md` | 5-level validation: structural, imports, functional, conventions, lint |

## Toolsets (`toolsets/`)

API references for frameworks and libraries used in Hailo apps.

| File | Description |
|---|---|
| `hailort-api.md` | HailoRT API: VDevice, VLM, LLM, Speech2Text, GStreamer buffer API |
| `gstreamer-elements.md` | GStreamer elements catalog: Hailo-specific and standard elements |
| `vlm-backend-api.md` | VLM Backend class: constructor, vlm_inference(), thread safety |
| `core-framework-api.md` | Core framework: resolve_hef_path, parsers, logger, HailoInfer, GStreamerApp |
| `gen-ai-utilities.md` | Gen AI utilities: LLM streaming, voice processing, agent tools |
| `yolo-coco-classes.md` | YOLO COCO 80-class label set: IDs, names, groupings, filtering patterns |
| `pose-keypoints.md` | COCO 17 pose keypoints: indices, skeleton connections, coordinate transform |

## Knowledge (`knowledge/`)

Structured knowledge bases in YAML format for agent decision-making.

| File | Description |
|---|---|
| `knowledge_base.yaml` | Operational knowledge: tuning recipes, bottleneck patterns, gen AI recipes |

## Memory (`memory/`)

Persistent cross-session knowledge base. Read at task start, update when discovering new patterns.

| File | When to Read |
|---|---|
| `MEMORY.md` | Always — unified index of all memory files |
| `gen_ai_patterns.md` | Building Gen AI apps — VLM/LLM architecture, multiprocessing, gotchas |
| `pipeline_optimization.md` | Profiling or optimizing pipelines — bottleneck fixes, FPS strategy |
| `camera_and_display.md` | Camera or display work — USB/RPi/RTSP setup, BGR/RGB, OpenCV patterns |
| `hailo_platform_api.md` | Using HailoRT directly — VDevice, HEF resolution, SDK API patterns |
| `common_pitfalls.md` | Always useful — import errors, signal handling, multiprocessing gotchas |

## Agents (`agents/`)

Platform-neutral agent definitions. The generator converts these to platform-specific formats.

| File | Description |
|---|---|
| `hl-app-builder.md` | Master router — classifies user request and routes to specialist |
| `hl-vlm-builder.md` | VLM app builder specialist |
| `hl-pipeline-builder.md` | GStreamer pipeline app builder specialist |
| `hl-standalone-builder.md` | Standalone inference app builder specialist |
| `hl-llm-builder.md` | LLM chat app builder specialist |
| `hl-agent-builder.md` | Agent with tool calling builder specialist |
| `hl-voice-builder.md` | Voice assistant builder specialist |

### Neutral Format

Agent files use platform-neutral YAML frontmatter:
- `name`, `description`, `argument-hint` — identity
- `capabilities` — abstract capability list (read, edit, execute, search, etc.)
- `routes-to` — sub-agent routing targets

Body content uses `<!-- INTERACTION: question \n OPTIONS: opt1 | opt2 -->` markers for platform-specific question UI. The generator converts these to:
- **Copilot**: `askQuestions:` blocks
- **Claude**: Natural language prompts
- **Cursor**: Inline guidance text

## Contextual Rules (`contextual-rules/`)

File-pattern-triggered context rules. Each file has a `glob` in YAML frontmatter and body content that's injected when the user edits matching files.

| File | Glob Pattern | Triggers When Editing |
|---|---|---|
| `core-framework.md` | `**/core/**` | Core framework files |
| `gen-ai-apps.md` | `**/gen_ai_apps/**` | Gen AI app files |
| `pipeline-apps.md` | `**/pipeline_apps/**` | Pipeline app files |
| `standalone-apps.md` | `**/standalone_apps/**` | Standalone app files |
| `tests.md` | `tests/**` | Test files |

## Prompts (`prompts/`)

Build prompt templates. Platform-neutral markdown — the generator wraps them for each platform.

| File | Description |
|---|---|
| `new-vlm-variant.md` | Create a VLM app variant |
| `new-pipeline-app.md` | Create a pipeline app |
| `new-standalone-app.md` | Create a standalone app |
| `new-llm-app.md` | Create an LLM app |
| `new-agent-tool.md` | Create an agent tool |
| `new-voice-app.md` | Create a voice app |
| `new-pose-game.md` | Create a pose estimation game |
| `orchestrated-build.md` | Meta-template for orchestrated builds |

## Platform Integration

This directory is the source of truth. Platform-specific files are **generated** from it:

| Platform | Entry Point | Generated To | Files | Strategy |
|---|---|---|---|---|
| GitHub Copilot | `.github/copilot-instructions.md` | `.github/` | 59 | Full copies (auto-loaded, can't read `.hailo/`) |
| Claude Code | `CLAUDE.md` | `.claude/` | 50 | Thin redirects → `.hailo/` (reads files at runtime) |
| Cursor | `.cursor/rules/` | `.cursor/` | 48 | Thin redirect `.mdc` rules with descriptions (reads `.hailo/` at runtime) |
| Any AI agent | Direct | Read `.hailo/` files | — | N/A |

> **Note**: A `.copilotignore` file at the repo root excludes `.claude/` from Copilot's file discovery. Without it, Copilot detects agents from both `.github/` and `.claude/`, causing duplicates in the agent dropdown. This file is auto-generated by `generate_platforms.py`.

### What Each Platform Gets

| Content | Copilot | Claude | Cursor |
|---|---|---|---|
| Global instructions + routing table | Full (template) | Full (`CLAUDE.md`) | Full (`hailo-global.mdc`) |
| Interactive workflow rule | ✅ | ✅ | ✅ |
| Agent behavioral files (7) | `.agent.md` (transformed) | agents/ (transformed) | N/A (no agent system) |
| Build skills (6) | Full content (path-rewritten) | Thin wrappers | `.mdc` (full, description-matched) |
| Utility skills (6) | Full content | Thin wrappers | `.mdc` thin redirects |
| Contextual rules (5) | `.instructions.md` (applyTo) | rules/ (paths:) | `.mdc` (globs:) |
| Toolsets (7) | Full copies | Thin redirects | `.mdc` thin redirects |
| Memory (6) | Full copies | 1 redirect | `.mdc` thin redirects |
| Instructions (7) | Full copies | Thin redirects | `.mdc` thin redirects |
| Prompts (10) | `.prompt.md` copies | Thin redirects | `.mdc` thin redirects |
| Scripts (3) | Full copies | — | — |

> **Note**: `CLAUDE.md` at the repo root is a generated entry point that points Claude Code to `.hailo/` and `.claude/`. It is NOT hand-maintained — it is regenerated by `generate_platforms.py`. The actual Claude Code agent configs live in `.claude/` (also generated from `.hailo/`).

### Developer Workflow

```bash
# Edit the source of truth
vim .hailo/skills/hl-build-vlm-app.md

# Regenerate all platform configs
python3 .hailo/scripts/generate_platforms.py --generate

# Verify nothing is stale
python3 .hailo/scripts/generate_platforms.py --check

# Commit both source and generated output
git add .hailo/ .github/ .claude/ .cursor/ CLAUDE.md .copilotignore
git commit -m "Update VLM skill"
```

---

## Scripts (`scripts/`)

Automation scripts for validation, curation, platform sync, and publishing.

| Script | Who runs it | Description |
|--------|-----------|-------------|
| `validate_app.py` | Maintainer / agent | Static convention checks (11 checks: files, syntax, imports, logger, SIGINT, unused imports, unreachable code, README quality) |
| `validate_app.py --smoke-test` | Maintainer / agent | Adds runtime checks: CLI `--help` and module import (gracefully skips on non-Hailo systems) |
| `generate_platforms.py --generate` | Maintainer | Syncs `.hailo/` → `.github/` + `.claude/` + `.cursor/` |
| `generate_platforms.py --check` | CI / Maintainer | Verifies generated files are in sync with `.hailo/` + runs cross-reference validation |
| `validate_framework.py` | CI / Maintainer | Cross-reference integrity: routing table paths, file tree accuracy, .hailo/ leak detection, agent handoffs, required sections, platform structural checks, source file integrity |
