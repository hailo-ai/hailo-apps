#!/usr/bin/env python3

import sys
import json
import yaml
from collections import defaultdict
from pathlib import Path

# -----------------------------
# Helpers
# -----------------------------

def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def dump_json(path: Path, data: dict):
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] wrote {path}")


def s3_url(metadata: dict, subpath: str) -> str:
    base = metadata["s3_endpoints"]["cs_bucket"].rstrip("/")
    return f"{base}/{subpath.lstrip('/')}"


def hef_name_from_model(model_name: str) -> str:
    return f"{model_name}.hef"


# -----------------------------
# resources.json
# -----------------------------

def build_resources_json(cfg: dict) -> dict:
    metadata = cfg["metadata"]

    resources = defaultdict(lambda: defaultdict(list))

    def add_resource(app, kind, name, url):
        resources[app][kind].append({
            "name": Path(name).stem,
            "description": name,
            "path": url,
        })

    # -------- images --------
    for img in cfg.get("images", []):
        name = img["name"]
        if "url" in img:
            url = img["url"]
        else:
            url = s3_url(metadata, f"resources/images/{name}")

        for tag in img.get("tag", []):
            add_resource(tag, "images", name, url)

    # -------- videos --------
    for vid in cfg.get("videos", []):
        name = vid["name"]
        url = s3_url(metadata, f"resources/video/{name}")

        for tag in vid.get("tag", []):
            add_resource(tag, "videos", name, url)

    # -------- json --------
    for js in cfg.get("json", []):
        name = js["name"]
        url = s3_url(metadata, f"resources/json/{name}")

        for tag in js.get("tag", []):
            add_resource(tag, "json", name, url)

    return {"resources": dict(resources)}


# -----------------------------
# apps.json
# -----------------------------

SOURCE_MAP = {
    "mz": "model_zoo",
    "s3": "cs",
    "gen-ai-mz": "gen-ai-mz",
}

def build_apps_json(cfg: dict) -> dict:
    apps = {}

    for app_name, app_cfg in cfg.items():
        if app_name in {
            "images", "videos", "json", "metadata"
        }:
            continue

        models_cfg = app_cfg.get("models")
        if not models_cfg:
            continue

        app_models = {}

        for arch, arch_cfg in models_cfg.items():
            if not arch_cfg:
                continue

            for section in ("default", "extra"):
                entries = arch_cfg.get(section)
                if not entries:
                    continue

                for model in entries:
                    model_name = model["name"]

                    entry = app_models.setdefault(model_name, {
                        "arch": set(),
                        "source": None,
                        "hefs": [],
                    })

                    entry["arch"].add(arch)

                    src = model["source"]
                    entry["source"] = SOURCE_MAP.get(src, src)

                    if "url" in model:
                        hef = Path(model["url"]).name
                        if hef not in entry["hefs"]:
                            entry["hefs"].append(hef)
                    else:
                        hef = hef_name_from_model(model_name)
                        if hef not in entry["hefs"]:
                            entry["hefs"].append(hef)

        if app_models:
            apps[app_name] = {
                name: {
                    **data,
                    "arch": sorted(data["arch"]),
                }
                for name, data in app_models.items()
            }

    return {"apps": apps}


# -----------------------------
# Main
# -----------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: generate_jsons.py <resources_config.yaml>")
        sys.exit(1)

    yaml_path = Path(sys.argv[1])
    cfg = load_yaml(yaml_path)

    resources_json = build_resources_json(cfg)
    apps_json = build_apps_json(cfg)

    dump_json(Path("resources.json"), resources_json)
    dump_json(Path("apps.json"), apps_json)


if __name__ == "__main__":
    main()
