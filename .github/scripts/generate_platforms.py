#!/usr/bin/env python3
"""
Platform Configuration Generator for Hailo Agentic Development.

Reads the canonical knowledge base in .hailo/ and generates platform-specific
configurations for GitHub Copilot (.github/), Claude Code (.claude/), and
Cursor (.cursor/).

Usage:
    python3 .hailo/scripts/generate_platforms.py --generate [--platform copilot|claude|cursor|all]
    python3 .hailo/scripts/generate_platforms.py --check [--platform copilot|claude|cursor|all]

The --check mode compares generated output against committed files and exits
non-zero if they differ (for CI validation).
"""

import argparse
import hashlib
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # .hailo/scripts/ -> repo root
HAILO_DIR = REPO_ROOT / ".hailo"
GITHUB_DIR = REPO_ROOT / ".github"
CLAUDE_DIR = REPO_ROOT / ".claude"
CURSOR_DIR = REPO_ROOT / ".cursor"

GENERATED_HEADER = (
    "<!-- AUTO-GENERATED from .hailo/ — DO NOT EDIT DIRECTLY -->\n"
    "<!-- Source: {source} -->\n"
    "<!-- Run: python3 .hailo/scripts/generate_platforms.py --generate -->\n\n"
)

# Copilot tool IDs by capability
COPILOT_TOOLS = {
    "read": [
        "read/readFile",
        "read/problems",
        "read/terminalSelection",
        "read/terminalLastCommand",
    ],
    "edit": [
        "edit/createDirectory",
        "edit/createFile",
        "edit/editFiles",
    ],
    "search": [
        "search/codebase",
        "search/fileSearch",
        "search/listDirectory",
        "search/textSearch",
        "search/usages",
        "search/changes",
        "search/searchResults",
    ],
    "execute": [
        "execute/runInTerminal",
        "execute/getTerminalOutput",
        "execute/awaitTerminal",
        "execute/killTerminal",
        "execute/createAndRunTask",
    ],
    "sub-agent": ["agent/runSubagent"],
    "ask-user": ["vscode/askQuestions"],
    "web": ["web/fetch", "web/githubRepo"],
    "todo": ["todo"],
}

# Claude tool names by capability
CLAUDE_TOOLS = {
    "read": ["Read", "Grep", "Glob"],
    "edit": ["Write", "Edit"],
    "search": ["Grep", "Glob"],
    "execute": ["Bash"],
    "sub-agent": ["Agent"],
    "ask-user": ["AskUserQuestion"],
    "web": ["WebFetch"],
    "todo": [],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def strip_yaml_frontmatter(content: str) -> tuple:
    """Return (frontmatter_dict, body_str) from a file with --- delimiters."""
    m = re.match(r"^---\n(.*?)\n---\n*", content, re.DOTALL)
    if m:
        fm = yaml.safe_load(m.group(1)) or {}
        return fm, content[m.end():]
    return {}, content


def header(source_rel: str) -> str:
    return GENERATED_HEADER.format(source=source_rel)


def path_hailo_to_github(body: str) -> str:
    """Convert .hailo/ paths in body text to .github/ paths."""
    # Build skill files: .hailo/skills/hl-build-X.md → .github/skills/hl-build-X/SKILL.md
    # This must run BEFORE the generic skills/ prefix replacement
    body = re.sub(
        r"\.hailo/skills/(hl-build-[\w-]+)\.md",
        r".github/skills/\1/SKILL.md",
        body,
    )
    body = body.replace(".hailo/skills/", ".github/skills/")
    body = body.replace(".hailo/instructions/", ".github/instructions/")
    body = body.replace(".hailo/toolsets/", ".github/toolsets/")
    body = body.replace(".hailo/memory/", ".github/memory/")
    body = body.replace(".hailo/knowledge/", ".github/knowledge/")
    body = body.replace(".hailo/scripts/", ".github/scripts/")
    body = body.replace(".hailo/prompts/", ".github/prompts/")
    # Also convert prose references to the directory name
    body = body.replace("relative to `.hailo/`", "relative to `.github/`")
    body = body.replace("live in `.hailo/`", "live in `.github/`")
    body = body.replace("Read `.hailo/README.md`", "Read `.github/copilot-instructions.md`")
    return body


def interaction_to_ask_questions(body: str) -> str:
    """Convert <!-- INTERACTION --> markers to Copilot askQuestions blocks."""
    def _convert(match):
        block = match.group(0)
        q_match = re.search(r"INTERACTION:\s*(.*?)(?:\n|-->)", block)
        opts_match = re.search(r"OPTIONS:\s*(.*?)-->", block, re.DOTALL)
        multi_match = re.search(r"MULTISELECT:\s*(true|yes)", block, re.IGNORECASE)
        question = q_match.group(1).strip() if q_match else "Question"
        if opts_match:
            options = [o.strip() for o in opts_match.group(1).split("|") if o.strip()]
            opts_yaml = "\n".join(
                f'    - label: "{opt}"' for opt in options
            )
            multi_line = "\n  multiSelect: true" if multi_match else ""
            return (
                f"```\naskQuestions:\n"
                f'  header: "Choice"\n'
                f'  question: "{question}"\n'
                f"  options:\n{opts_yaml}{multi_line}\n```"
            )
        return f"**Ask the user:** {question}"

    return re.sub(r"<!-- INTERACTION:.*?-->", _convert, body, flags=re.DOTALL)


def interaction_to_natural_language(body: str) -> str:
    """Convert <!-- INTERACTION --> markers to natural language for Claude."""
    def _convert(match):
        block = match.group(0)
        q_match = re.search(r"INTERACTION:\s*(.*?)(?:\n|-->)", block)
        opts_match = re.search(r"OPTIONS:\s*(.*?)-->", block, re.DOTALL)
        multi_match = re.search(r"MULTISELECT:\s*(true|yes)", block, re.IGNORECASE)
        question = q_match.group(1).strip() if q_match else "Question"
        if opts_match:
            options = [o.strip() for o in opts_match.group(1).split("|") if o.strip()]
            opts_list = "\n".join(f"  - {opt}" for opt in options)
            multi_hint = " (Multiple selections allowed)" if multi_match else ""
            return f"**Ask the user:** {question}{multi_hint}\n\nOptions:\n{opts_list}"
        return f"**Ask the user:** {question}"

    return re.sub(r"<!-- INTERACTION:.*?-->", _convert, body, flags=re.DOTALL)


def interaction_to_inline(body: str) -> str:
    """Convert <!-- INTERACTION --> markers to inline guidance for Cursor."""
    def _convert(match):
        block = match.group(0)
        q_match = re.search(r"INTERACTION:\s*(.*?)(?:\n|-->)", block)
        opts_match = re.search(r"OPTIONS:\s*(.*?)-->", block, re.DOTALL)
        multi_match = re.search(r"MULTISELECT:\s*(true|yes)", block, re.IGNORECASE)
        question = q_match.group(1).strip() if q_match else "Question"
        if opts_match:
            options = [o.strip() for o in opts_match.group(1).split("|") if o.strip()]
            multi_tag = " [multi-select]" if multi_match else ""
            return f"Consider asking: {question}{multi_tag} (options: {', '.join(options)})"
        return f"Consider asking: {question}"

    return re.sub(r"<!-- INTERACTION:.*?-->", _convert, body, flags=re.DOTALL)


def first_paragraph(content: str, prefer_heading: bool = False) -> str:
    """Extract a clean description: first prose line, or heading if prefer_heading=True."""
    in_frontmatter = False
    heading = None
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            continue
        # Capture the first heading as fallback description
        if stripped.startswith("# ") and heading is None:
            heading = stripped.lstrip("# ").strip()
            if prefer_heading:
                return heading[:200].replace('"', '\\"')
            continue
        if stripped and not stripped.startswith("#") and not stripped.startswith(">") and not stripped.startswith("`") and not stripped.startswith("|") and not stripped.startswith("- "):
            # Escape double quotes for safe YAML embedding
            return stripped[:200].replace('"', '\\"')
    # Fall back to heading if no prose found
    if heading:
        return heading[:200].replace('"', '\\"')
    return "Hailo development reference"


# Routing table (canonical, using .hailo/ paths) — embedded in Cursor global rule and CLAUDE.md
ROUTING_TABLE = """
### Context Routing Table

Based on what the task involves, read **only** the matching rows:

| If the task mentions... | Read these files |
|---|---|
| **VLM, vision, image understanding** | `{pfx}skills/hl-build-vlm-app.md`, `{pfx}toolsets/vlm-backend-api.md`, `{pfx}memory/gen_ai_patterns.md` |
| **LLM, chat, text generation** | `{pfx}skills/hl-build-llm-app.md`, `{pfx}instructions/gen-ai-development.md`, `{pfx}toolsets/gen-ai-utilities.md`, `{pfx}memory/gen_ai_patterns.md` |
| **Agent, tools, function calling** | `{pfx}skills/hl-build-agent-app.md`, `{pfx}toolsets/gen-ai-utilities.md`, `{pfx}memory/gen_ai_patterns.md` |
| **Voice, STT, TTS, Whisper, speech** | `{pfx}skills/hl-build-voice-app.md`, `{pfx}toolsets/gen-ai-utilities.md` |
| **Pipeline, GStreamer, video, stream** | `{pfx}skills/hl-build-pipeline-app.md`, `{pfx}instructions/gstreamer-pipelines.md`, `{pfx}toolsets/gstreamer-elements.md`, `{pfx}memory/pipeline_optimization.md` |
| **Game, interactive, pose game** | `{pfx}skills/hl-build-pipeline-app.md`, `{pfx}toolsets/pose-keypoints.md`, `{pfx}toolsets/core-framework-api.md`, `{pfx}memory/common_pitfalls.md` |
| **Standalone, OpenCV, HailoInfer** | `{pfx}skills/hl-build-standalone-app.md`, `{pfx}toolsets/core-framework-api.md` |
| **Camera, USB, RPi, capture** | `{pfx}skills/hl-camera.md`, `{pfx}memory/camera_and_display.md` |
| **HEF, model, download, config** | `{pfx}skills/hl-model-management.md`, `{pfx}toolsets/hailort-api.md`, `{pfx}memory/hailo_platform_api.md` |
| **Monitoring, events, alerts** | `{pfx}skills/hl-monitoring.md`, `{pfx}skills/hl-event-detection.md` |
| **Testing, validation, pytest** | `{pfx}skills/hl-validate.md`, `{pfx}instructions/testing-patterns.md` |
| **ALWAYS read (every task)** | `{pfx}memory/common_pitfalls.md`, `{pfx}instructions/coding-standards.md` |
"""


# ---------------------------------------------------------------------------
# Copilot Generator
# ---------------------------------------------------------------------------


def generate_copilot():
    """Generate .github/ from .hailo/."""
    generated_files = {}

    # 1. copilot-instructions.md — generated from .hailo/templates/copilot-instructions.md
    ci_template = HAILO_DIR / "templates" / "copilot-instructions.md"
    if ci_template.exists():
        content = read_file(ci_template)
        # The template already uses .github/ paths (it's Copilot-specific)
        out_path = GITHUB_DIR / "copilot-instructions.md"
        generated_files[out_path] = content

    # 2. Agents — convert neutral format to chatagent
    agents_src = HAILO_DIR / "agents"
    if agents_src.is_dir():
        for src_file in sorted(agents_src.glob("*.md")):
            content = read_file(src_file)
            fm, body = strip_yaml_frontmatter(content)

            # Build Copilot tool list from capabilities
            tools = []
            for cap in fm.get("capabilities", []):
                tools.extend(COPILOT_TOOLS.get(cap, []))
            tools = sorted(set(tools))

            # Build handoffs from routes-to
            handoffs = []
            for route in fm.get("routes-to", []):
                handoffs.append({
                    "label": route.get("label", ""),
                    "agent": route.get("target", ""),
                    "prompt": route.get("description", ""),
                    "send": False,
                })

            # Build Copilot frontmatter
            copilot_fm = {
                "name": fm.get("name", ""),
                "description": fm.get("description", ""),
                "argument-hint": fm.get("argument-hint", ""),
                "tools": tools,
            }
            if handoffs:
                copilot_fm["handoffs"] = handoffs

            fm_str = yaml.dump(
                copilot_fm,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

            # Convert interaction markers and paths
            body = interaction_to_ask_questions(body)
            body = path_hailo_to_github(body)

            out_name = src_file.stem + ".agent.md"
            out_content = f"---\n{fm_str}---\n{body}"
            out_path = GITHUB_DIR / "agents" / out_name
            generated_files[out_path] = out_content

    # 3. Skills — wrap in Copilot skill format
    skills_src = HAILO_DIR / "skills"
    if skills_src.is_dir():
        # Only the main build skills (not instructions variants)
        for src_file in sorted(skills_src.glob("hl-build-*.md")):
            if "-instructions" in src_file.name:
                continue
            content = read_file(src_file)
            skill_name = src_file.stem  # e.g., hl-build-vlm-app

            # Extract first line as description
            first_line = ""
            for line in content.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    first_line = line[:200]
                    break

            body = path_hailo_to_github(content)

            out_content = f"---\nname: {skill_name}\ndescription: {first_line}\n---\n\n{body}"
            out_dir = GITHUB_DIR / "skills" / skill_name
            out_path = out_dir / "SKILL.md"
            generated_files[out_path] = out_content

        # Also the voice skill
        voice_skill = skills_src / "hl-build-voice-app.md"
        if voice_skill.exists():
            content = read_file(voice_skill)
            first_line = ""
            for line in content.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    first_line = line[:200]
                    break
            body = path_hailo_to_github(content)
            out_content = f"---\nname: hl-build-voice-app\ndescription: {first_line}\n---\n\n{body}"
            out_path = GITHUB_DIR / "skills" / "hl-build-voice-app" / "SKILL.md"
            generated_files[out_path] = out_content

        # Utility skill files (not hl-build-* pattern) — copy as flat files
        utility_skills = [
            "hl-monitoring.md",
            "hl-event-detection.md",
            "hl-camera.md",
            "hl-model-management.md",
            "hl-plan-and-execute.md",
            "hl-validate.md",
        ]
        for skill_name in utility_skills:
            src_file = skills_src / skill_name
            if src_file.exists():
                content = read_file(src_file)
                body = path_hailo_to_github(content)
                out_path = GITHUB_DIR / "skills" / skill_name
                generated_files[out_path] = body

    # 4. Contextual instructions — wrap with applyTo
    rules_src = HAILO_DIR / "contextual-rules"
    if rules_src.is_dir():
        name_map = {
            "core-framework.md": "core-framework.instructions.md",
            "gen-ai-apps.md": "gen-ai-apps.instructions.md",
            "pipeline-apps.md": "pipeline-apps.instructions.md",
            "standalone-apps.md": "standalone-apps.instructions.md",
            "tests.md": "tests.instructions.md",
        }
        for src_file in sorted(rules_src.glob("*.md")):
            content = read_file(src_file)
            fm, body = strip_yaml_frontmatter(content)
            glob_pattern = fm.get("glob", "")
            body = path_hailo_to_github(body)
            out_content = f'---\napplyTo: "{glob_pattern}"\n---\n\n{body}'
            out_name = name_map.get(src_file.name, src_file.name)
            out_path = GITHUB_DIR / "instructions" / out_name
            generated_files[out_path] = out_content

    # 5. Instructions, toolsets, memory, knowledge — copy verbatim with path transform
    for subdir in ["instructions", "toolsets", "memory", "knowledge"]:
        src = HAILO_DIR / subdir
        if not src.is_dir():
            continue
        for src_file in sorted(src.rglob("*")):
            if src_file.is_file() and src_file.suffix in (".md", ".yaml", ".yml"):
                content = read_file(src_file)
                content = path_hailo_to_github(content)
                rel = src_file.relative_to(src)
                out_path = GITHUB_DIR / subdir / rel
                generated_files[out_path] = content

    # 6. Prompts — wrap in ```prompt
    prompts_src = HAILO_DIR / "prompts"
    if prompts_src.is_dir():
        for src_file in sorted(prompts_src.glob("*.md")):
            content = read_file(src_file)
            content = path_hailo_to_github(content)
            out_name = src_file.stem + ".prompt.md"
            out_path = GITHUB_DIR / "prompts" / out_name
            generated_files[out_path] = content

    # 7. Scripts — copy to .github/scripts/ (including the generator itself)
    scripts_src = HAILO_DIR / "scripts"
    if scripts_src.is_dir():
        for src_file in sorted(scripts_src.glob("*.py")):
            content = read_file(src_file)
            out_path = GITHUB_DIR / "scripts" / src_file.name
            generated_files[out_path] = content

    # 8. .copilotignore — prevent Copilot from scanning .claude/ (which it now
    #    recognizes natively), avoiding duplicate agents since the same agents
    #    already exist in .github/agents/.
    copilotignore_content = (
        "# Prevent Copilot from scanning .claude/ directory\n"
        "# Copilot now recognizes .claude/agents/ format, which causes duplicate agents\n"
        "# since the same agents already exist in .github/agents/ (Copilot's native format).\n"
        "# Both .claude/ and .github/ are generated from .hailo/ — only .github/ should be\n"
        "# used by Copilot.\n"
        ".claude/\n"
    )
    generated_files[REPO_ROOT / ".copilotignore"] = copilotignore_content

    return generated_files


# ---------------------------------------------------------------------------
# Claude Generator
# ---------------------------------------------------------------------------


def generate_claude():
    """Generate .claude/ and CLAUDE.md from .hailo/."""
    generated_files = {}

    # 1. CLAUDE.md at repo root — build from .hailo/README.md + additions
    readme_content = read_file(HAILO_DIR / "README.md") if (HAILO_DIR / "README.md").exists() else ""

    claude_md = f"""# Hailo Apps — Claude Code Entry Point

> Auto-generated from `.hailo/`. Do not edit directly.

## Shared Knowledge

All skills, instructions, toolsets, knowledge bases, and memory live in `.hailo/`.
Read `.hailo/README.md` for the complete master index.

## Interactive Workflow (MUST FOLLOW)

**Always walk through key decisions with the user before building.** Ask 2-3 targeted
questions to confirm app type, features, and input source. This creates a collaborative
workflow and catches misunderstandings early. Only skip questions if the user explicitly
says "just build it" or "use defaults".

## Quick Reference

```bash
source setup_env.sh                    # Activate environment (always do this first)
pip install -e .                       # Install in editable mode
```

## Skills (slash commands)

| Command | Description |
|---------|-------------|
| `/hl-build-vlm-app` | Build VLM image understanding apps |
| `/hl-build-pipeline-app` | Build GStreamer pipeline apps |
| `/hl-build-standalone-app` | Build standalone HailoRT apps |
| `/hl-build-agent-app` | Build AI agent apps with tool calling |
| `/hl-build-llm-app` | Build LLM chat apps |
| `/hl-build-voice-app` | Build voice assistant apps |

{ROUTING_TABLE.replace("{pfx}", ".hailo/")}

## Python Imports

```python
from hailo_apps.python.core.common.defines import *
from hailo_apps.python.core.common.core import resolve_hef_path
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import GStreamerApp
```

## Critical Conventions

1. **Imports are always absolute**: `from hailo_apps.python.core.common.xyz import ...`
2. **HEF resolution**: Always use `resolve_hef_path(path, app_name, arch)`
3. **Device sharing**: Always use `SHARED_VDEVICE_GROUP_ID` when creating VDevice
4. **Logging**: Use `get_logger(__name__)`
5. **CLI parsers**: Use `get_pipeline_parser()` or `get_standalone_parser()`
6. **Architecture detection**: Use `detect_hailo_arch()` or `--arch` flag
7. **USB camera**: Always `--input usb` for auto-detection. Never hardcode `/dev/video0` (typically integrated webcam).
8. **SKILL.md is sufficient**: Read SKILL.md + common_pitfalls.md. Do NOT read source code files.
9. **Custom background**: When user provides a background image, use `background.copy()` — never blend camera feed.

## Hardware

| Architecture | Value | Use case |
|---|---|---|
| Hailo-8 | `hailo8` | Full performance, all pipeline + standalone apps |
| Hailo-8L | `hailo8l` | Lower power, compatible model subset |
| Hailo-10H | `hailo10h` | GenAI (LLM, VLM, Whisper) + vision pipelines |

## Memory

Persistent knowledge in `.hailo/memory/`. Read at task start, update when learning.
"""
    generated_files[REPO_ROOT / "CLAUDE.md"] = claude_md

    # 2. Claude skills — thin wrappers with allowed-tools
    skills_src = HAILO_DIR / "skills"
    if skills_src.is_dir():
        skill_configs = {
            "hl-build-vlm-app": {
                "description": "Build a Vision-Language Model application for Hailo-10H.",
                "tools": "Bash(python *), Read, Write, Edit, Grep, Glob, Agent, AskUserQuestion",
                "refs": [".hailo/skills/hl-monitoring.md", ".hailo/skills/hl-event-detection.md",
                         ".hailo/toolsets/vlm-backend-api.md"],
            },
            "hl-build-pipeline-app": {
                "description": "Build a GStreamer pipeline application for Hailo accelerators.",
                "tools": "Bash(python *), Read, Write, Edit, Grep, Glob, Agent, AskUserQuestion",
                "refs": [".hailo/toolsets/gstreamer-elements.md", ".hailo/toolsets/core-framework-api.md"],
            },
            "hl-build-standalone-app": {
                "description": "Build a standalone HailoRT inference application.",
                "tools": "Bash(python *), Read, Write, Edit, Grep, Glob, Agent, AskUserQuestion",
                "refs": [".hailo/toolsets/core-framework-api.md", ".hailo/skills/hl-camera.md"],
            },
            "hl-build-agent-app": {
                "description": "Build an AI agent application with LLM tool calling for Hailo-10H.",
                "tools": "Bash(python *), Read, Write, Edit, Grep, Glob, Agent, AskUserQuestion",
                "refs": [".hailo/toolsets/gen-ai-utilities.md"],
            },
            "hl-build-llm-app": {
                "description": "Build an LLM chat application for Hailo-10H.",
                "tools": "Bash(python *), Read, Write, Edit, Grep, Glob, Agent, AskUserQuestion",
                "refs": [".hailo/toolsets/hailort-api.md"],
            },
            "hl-build-voice-app": {
                "description": "Build a voice assistant with Whisper STT and Piper TTS for Hailo-10H.",
                "tools": "Bash(python *), Read, Write, Edit, Grep, Glob, Agent, AskUserQuestion",
                "refs": [".hailo/toolsets/gen-ai-utilities.md"],
            },
        }

        for skill_name, config in skill_configs.items():
            src_file = skills_src / f"{skill_name}.md"
            if not src_file.exists():
                continue

            refs_md = "\n".join(f"- `{r}`" for r in config["refs"])
            out_content = (
                f"---\n"
                f'name: {skill_name}\n'
                f'description: "{config["description"]}"\n'
                f'argument-hint: "[app-description]"\n'
                f'allowed-tools: {config["tools"]}\n'
                f"---\n\n"
                f"<!-- Thin Claude Code wrapper — canonical skill doc lives in .hailo/ -->\n\n"
                f"Read and follow the complete skill documentation at `.hailo/skills/{skill_name}.md`.\n\n"
                f"Also consult:\n{refs_md}\n"
            )
            out_path = CLAUDE_DIR / "skills" / skill_name / "SKILL.md"
            generated_files[out_path] = out_content

    # 3. Claude agents — convert neutral format
    agents_src = HAILO_DIR / "agents"
    if agents_src.is_dir():
        for src_file in sorted(agents_src.glob("*.md")):
            content = read_file(src_file)
            fm, body = strip_yaml_frontmatter(content)

            # Build Claude tool list from capabilities
            tools = set()
            for cap in fm.get("capabilities", []):
                tools.update(CLAUDE_TOOLS.get(cap, []))
            tools = sorted(tools)

            claude_fm = {
                "name": fm.get("name", ""),
                "description": fm.get("description", ""),
            }
            if tools:
                claude_fm["tools"] = tools

            fm_str = yaml.dump(claude_fm, default_flow_style=False, sort_keys=False, allow_unicode=True)
            body = interaction_to_natural_language(body)

            out_path = CLAUDE_DIR / "agents" / src_file.name
            generated_files[out_path] = f"---\n{fm_str}---\n{body}"

    # 4. Claude rules — from contextual-rules with paths: key
    rules_src = HAILO_DIR / "contextual-rules"
    if rules_src.is_dir():
        for src_file in sorted(rules_src.glob("*.md")):
            content = read_file(src_file)
            fm, body = strip_yaml_frontmatter(content)
            glob_pattern = fm.get("glob", "")
            out_content = f'---\npaths:\n  - "{glob_pattern}"\n---\n\n{body}'
            out_path = CLAUDE_DIR / "rules" / src_file.name
            generated_files[out_path] = out_content

    # 5. Claude memory — redirect to .hailo/memory/ (Claude reads files directly)
    generated_files[CLAUDE_DIR / "memory" / "MEMORY.md"] = (
        "# Memory Redirect\n\n"
        "Memory files are centralized in `.hailo/memory/`.\n"
        "See `.hailo/memory/MEMORY.md` for the unified index.\n\n"
        "Files:\n"
        "- `.hailo/memory/common_pitfalls.md` — Bugs & anti-patterns (read on every task)\n"
        "- `.hailo/memory/gen_ai_patterns.md` — VLM/LLM architecture patterns\n"
        "- `.hailo/memory/pipeline_optimization.md` — GStreamer bottleneck fixes\n"
        "- `.hailo/memory/camera_and_display.md` — Camera & OpenCV patterns\n"
        "- `.hailo/memory/hailo_platform_api.md` — SDK usage patterns\n"
    )

    # 6. Claude utility skills — thin wrappers pointing to .hailo/
    utility_skills = [
        ("hl-monitoring", "Continuous monitoring patterns for Hailo apps."),
        ("hl-event-detection", "Detect and report events from video streams."),
        ("hl-camera", "Camera setup, USB/RPi configuration, and troubleshooting."),
        ("hl-model-management", "HEF resolution, model download, and config management."),
        ("hl-plan-and-execute", "Plan-and-execute loop pattern for complex builds."),
        ("hl-validate", "Validation at every phase gate."),
    ]
    if skills_src.is_dir():
        for skill_name, desc in utility_skills:
            src_file = skills_src / f"{skill_name}.md"
            if src_file.exists():
                out_content = (
                    f"---\n"
                    f'name: {skill_name}\n'
                    f'description: "{desc}"\n'
                    f"---\n\n"
                    f"<!-- Thin wrapper — canonical doc lives in .hailo/ -->\n\n"
                    f"Read `.hailo/skills/{skill_name}.md` for the complete skill documentation.\n"
                )
                out_path = CLAUDE_DIR / "skills" / skill_name / "SKILL.md"
                generated_files[out_path] = out_content

    # 7. Claude toolsets — thin wrappers pointing to .hailo/
    toolsets_src = HAILO_DIR / "toolsets"
    if toolsets_src.is_dir():
        for src_file in sorted(toolsets_src.glob("*.md")):
            desc = first_paragraph(read_file(src_file))
            out_content = (
                f"# Toolset: {src_file.stem}\n\n"
                f"Read `.hailo/toolsets/{src_file.name}` for the complete reference.\n"
            )
            out_path = CLAUDE_DIR / "toolsets" / src_file.name
            generated_files[out_path] = out_content

    # 8. Claude instructions — thin wrappers pointing to .hailo/
    instructions_src = HAILO_DIR / "instructions"
    if instructions_src.is_dir():
        for src_file in sorted(instructions_src.glob("*.md")):
            out_content = (
                f"# Instruction: {src_file.stem}\n\n"
                f"Read `.hailo/instructions/{src_file.name}` for the complete guide.\n"
            )
            out_path = CLAUDE_DIR / "instructions" / src_file.name
            generated_files[out_path] = out_content

    # 9. Claude prompts — thin wrappers pointing to .hailo/
    prompts_src = HAILO_DIR / "prompts"
    if prompts_src.is_dir():
        for src_file in sorted(prompts_src.glob("*.md")):
            desc = first_paragraph(read_file(src_file))
            out_content = (
                f"# Prompt: {src_file.stem}\n\n"
                f"{desc}\n\n"
                f"Read `.hailo/prompts/{src_file.name}` for the complete prompt template.\n"
            )
            out_path = CLAUDE_DIR / "prompts" / src_file.name
            generated_files[out_path] = out_content

    return generated_files


# ---------------------------------------------------------------------------
# Cursor Generator
# ---------------------------------------------------------------------------


def generate_cursor():
    """Generate .cursor/ from .hailo/."""
    generated_files = {}

    # 1. Global rule — always applied (with routing table)
    routing = ROUTING_TABLE.replace("{pfx}", ".hailo/")
    global_rule = (
        "---\n"
        "alwaysApply: true\n"
        "---\n\n"
        "# Hailo Apps — Global Rules\n\n"
        "All skills, instructions, toolsets, knowledge bases, and memory live in `.hailo/`.\n"
        "Read `.hailo/README.md` for the complete master index.\n\n"
        "## Interactive Workflow (MUST FOLLOW)\n\n"
        "**Always walk through key decisions with the user before building.** Ask 2-3 targeted\n"
        "questions to confirm app type, features, and input source. This creates a collaborative\n"
        "workflow and catches misunderstandings early. Only skip questions if the user explicitly\n"
        "says \"just build it\" or \"use defaults\".\n\n"
        "## Critical Conventions\n\n"
        "1. **Imports are always absolute**: `from hailo_apps.python.core.common.xyz import ...`\n"
        "2. **HEF resolution**: Always use `resolve_hef_path(path, app_name, arch)`\n"
        "3. **Device sharing**: Always use `SHARED_VDEVICE_GROUP_ID` when creating VDevice\n"
        "4. **Logging**: Use `get_logger(__name__)`\n"
        "5. **CLI parsers**: Use `get_pipeline_parser()` or `get_standalone_parser()`\n"
        "6. **Architecture detection**: Use `detect_hailo_arch()` or `--arch` flag\n"
        "7. **USB camera**: Always `--input usb` for auto-detection. Never hardcode `/dev/video0`.\n"
        "8. **SKILL.md is sufficient**: Read SKILL.md + common_pitfalls.md only. Do NOT read source code.\n"
        "9. **Custom background**: Use `background.copy()` — never blend camera feed with background.\n\n"
        + routing + "\n"
        "## Available Skills\n\n"
        "| Skill | Doc |\n"
        "|-------|-----|\n"
        "| Build VLM App | `.hailo/skills/hl-build-vlm-app.md` |\n"
        "| Build Pipeline App | `.hailo/skills/hl-build-pipeline-app.md` |\n"
        "| Build Standalone App | `.hailo/skills/hl-build-standalone-app.md` |\n"
        "| Build Agent App | `.hailo/skills/hl-build-agent-app.md` |\n"
        "| Build LLM App | `.hailo/skills/hl-build-llm-app.md` |\n"
        "| Build Voice App | `.hailo/skills/hl-build-voice-app.md` |\n\n"
        "## Memory\n\n"
        "Persistent knowledge in `.hailo/memory/`. Read at task start, update when learning.\n"
    )
    generated_files[CURSOR_DIR / "rules" / "hailo-global.mdc"] = global_rule

    # 2. Contextual rules — with globs
    rules_src = HAILO_DIR / "contextual-rules"
    if rules_src.is_dir():
        for src_file in sorted(rules_src.glob("*.md")):
            content = read_file(src_file)
            fm, body = strip_yaml_frontmatter(content)
            glob_pattern = fm.get("glob", "")
            body = interaction_to_inline(body)
            out_content = (
                f"---\n"
                f'globs: ["{glob_pattern}"]\n'
                f"alwaysApply: false\n"
                f"---\n\n{body}"
            )
            out_path = CURSOR_DIR / "rules" / (src_file.stem + ".mdc")
            generated_files[out_path] = out_content

    # 3. Skill rules — with description for AI selection
    skills_src = HAILO_DIR / "skills"
    if skills_src.is_dir():
        for src_file in sorted(skills_src.glob("hl-build-*.md")):
            if "-instructions" in src_file.name:
                continue
            content = read_file(src_file)
            # Extract first paragraph as description
            desc = ""
            for line in content.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    desc = line[:200]
                    break

            body = interaction_to_inline(content)
            out_content = (
                f"---\n"
                f'description: "{desc}"\n'
                f"alwaysApply: false\n"
                f"---\n\n{body}"
            )
            out_path = CURSOR_DIR / "rules" / (src_file.stem + ".mdc")
            generated_files[out_path] = out_content

    # 4. Utility skills — description-matched rules
    utility_skills = [
        "hl-monitoring.md", "hl-event-detection.md", "hl-camera.md",
        "hl-model-management.md", "hl-plan-and-execute.md", "hl-validate.md",
    ]
    if skills_src.is_dir():
        for skill_name in utility_skills:
            src_file = skills_src / skill_name
            if src_file.exists():
                content = read_file(src_file)
                desc = first_paragraph(content, prefer_heading=True)
                body = interaction_to_inline(content)
                out_content = (
                    f"---\n"
                    f'description: "{desc}"\n'
                    f"alwaysApply: false\n"
                    f"---\n\n{body}"
                )
                out_path = CURSOR_DIR / "rules" / (src_file.stem + ".mdc")
                generated_files[out_path] = out_content

    # 5. Toolsets — thin redirect rules (Cursor reads .hailo/ directly)
    toolsets_src = HAILO_DIR / "toolsets"
    if toolsets_src.is_dir():
        for src_file in sorted(toolsets_src.glob("*.md")):
            desc = first_paragraph(read_file(src_file), prefer_heading=True)
            # Strip redundant prefix if heading already starts with "Toolset:"
            if desc.lower().startswith("toolset:"):
                desc = desc[len("toolset:"):].strip()
            out_content = (
                f"---\n"
                f'description: "Toolset: {desc}"\n'
                f"alwaysApply: false\n"
                f"---\n\n"
                f"# Toolset: {src_file.stem}\n\n"
                f"Read `.hailo/toolsets/{src_file.name}` for the complete API reference.\n"
            )
            out_path = CURSOR_DIR / "rules" / ("toolset-" + src_file.stem + ".mdc")
            generated_files[out_path] = out_content

    # 6. Memory — thin redirect rules (Cursor reads .hailo/ directly)
    memory_src = HAILO_DIR / "memory"
    if memory_src.is_dir():
        for src_file in sorted(memory_src.glob("*.md")):
            desc = first_paragraph(read_file(src_file), prefer_heading=True)
            # Strip redundant suffix/prefix patterns from memory headings

            desc = re.sub(r'\s*—\s*Memory$', '', desc)
            if desc.lower().startswith("memory:"):
                desc = desc[len("memory:"):].strip()
            out_content = (
                f"---\n"
                f'description: "Memory: {desc}"\n'
                f"alwaysApply: false\n"
                f"---\n\n"
                f"# Memory: {src_file.stem}\n\n"
                f"Read `.hailo/memory/{src_file.name}` for the complete content.\n"
            )
            out_path = CURSOR_DIR / "rules" / ("memory-" + src_file.stem + ".mdc")
            generated_files[out_path] = out_content

    # 7. Instructions — thin redirect rules (Cursor reads .hailo/ directly)
    instructions_src = HAILO_DIR / "instructions"
    if instructions_src.is_dir():
        for src_file in sorted(instructions_src.glob("*.md")):
            desc = first_paragraph(read_file(src_file), prefer_heading=True)
            # Strip redundant prefix if heading already starts with category
            for prefix in ("instruction:", "instructions:"):
                if desc.lower().startswith(prefix):
                    desc = desc[len(prefix):].strip()
                    break
            out_content = (
                f"---\n"
                f'description: "Instruction: {desc}"\n'
                f"alwaysApply: false\n"
                f"---\n\n"
                f"# Instruction: {src_file.stem}\n\n"
                f"Read `.hailo/instructions/{src_file.name}` for the complete guide.\n"
            )
            out_path = CURSOR_DIR / "rules" / ("inst-" + src_file.stem + ".mdc")
            generated_files[out_path] = out_content

    # 8. Prompts — thin redirect rules (Cursor reads .hailo/ directly)
    prompts_src = HAILO_DIR / "prompts"
    if prompts_src.is_dir():
        for src_file in sorted(prompts_src.glob("*.md")):
            desc = first_paragraph(read_file(src_file), prefer_heading=True)
            # Strip redundant prefix if heading already starts with category
            for prefix in ("prompt:", "meta-prompt:"):
                if desc.lower().startswith(prefix):
                    desc = desc[len(prefix):].strip()
                    break
            out_content = (
                f"---\n"
                f'description: "Prompt template: {desc}"\n'
                f"alwaysApply: false\n"
                f"---\n\n"
                f"# Prompt: {src_file.stem}\n\n"
                f"{desc}\n\n"
                f"Read `.hailo/prompts/{src_file.name}` for the complete prompt template.\n"
            )
            out_path = CURSOR_DIR / "rules" / ("prompt-" + src_file.stem + ".mdc")
            generated_files[out_path] = out_content

    return generated_files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GENERATORS = {
    "copilot": generate_copilot,
    "claude": generate_claude,
    "cursor": generate_cursor,
}


def generate(platforms: list) -> dict:
    """Generate platform configs and return {path: content} dict."""
    all_files = {}
    for platform in platforms:
        gen_fn = GENERATORS.get(platform)
        if gen_fn:
            all_files.update(gen_fn())
    return all_files


def check(platforms: list) -> bool:
    """Check if generated output matches committed files. Return True if clean."""
    files = generate(platforms)
    dirty = []
    missing = []

    for path, expected_content in sorted(files.items()):
        if not path.exists():
            missing.append(str(path.relative_to(REPO_ROOT)))
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != expected_content:
            dirty.append(str(path.relative_to(REPO_ROOT)))

    if dirty or missing:
        print(f"Platform configs are STALE ({len(dirty)} changed, {len(missing)} missing):")
        for f in missing:
            print(f"  MISSING: {f}")
        for f in dirty:
            print(f"  CHANGED: {f}")
        print("\nRun: python3 .hailo/scripts/generate_platforms.py --generate")
        return False

    print(f"Platform configs are up-to-date ({len(files)} files checked)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate platform configs from .hailo/")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true", help="Generate platform configs")
    group.add_argument("--check", action="store_true", help="Check if configs are up-to-date")
    parser.add_argument(
        "--platform",
        choices=["copilot", "claude", "cursor", "all"],
        default="all",
        help="Which platform to generate/check (default: all)",
    )
    args = parser.parse_args()

    platforms = list(GENERATORS.keys()) if args.platform == "all" else [args.platform]

    if args.generate:
        files = generate(platforms)
        for path, content in sorted(files.items()):
            write_file(path, content)
            rel = path.relative_to(REPO_ROOT)
            print(f"  Generated: {rel}")
        print(f"\n{len(files)} files generated for: {', '.join(platforms)}")

    elif args.check:
        clean = check(platforms)

        # Also run cross-reference validation if validate_framework is available
        try:
            from validate_framework import (
                ValidationResult,
                validate_routing_table,
                validate_file_tree,
                validate_no_hailo_leaks,
                validate_agent_handoffs,
                validate_skill_sections,
                validate_hailo_source_files,
            )
            print("\n--- Cross-reference validation ---")
            result = ValidationResult(verbose=False)
            validate_hailo_source_files(result)
            if "copilot" in platforms or args.platform == "all":
                validate_routing_table(result)
                validate_file_tree(result)
                validate_no_hailo_leaks(result)
                validate_agent_handoffs(result)
                validate_skill_sections(result)
            print(f"\nCross-ref: {result.summary()}")
            if not result.clean:
                clean = False
        except ImportError:
            pass  # validate_framework.py not available — skip

        sys.exit(0 if clean else 1)


if __name__ == "__main__":
    main()
