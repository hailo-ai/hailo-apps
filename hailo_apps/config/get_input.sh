#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCES_JSON="${RESOURCES_JSON:-${SCRIPT_DIR}/inputs.json}"

# These apps will use the "default" resources section
SPECIAL_DEFAULT_APPS=("object_detection" "instance_segmentation" "pose_estimation" "classifier" "semantic_segmentation" "depth_estimation_mono" "onnxrt_hailo_pipeline")

usage() {
  cat <<EOF
Usage:
  $0 list --app "<app name>"

  $0 get --app "<app name>" --target-dir "<dir>" --i "<ID>"

Where:
  --app        Application key, e.g. "object_detection", "lane_detection", "paddle_ocr"
  --target-dir Directory to download the file into (default: current dir)
  --i          Input ID (NOT a path), e.g.:
                 bus
                 street_drive
               Special value:
                 default  -> use the first image for this app,
                             or the first video if no images exist.

Behavior:
  - Apps: object_detection, instance_segmentation, pose_estimation
      -> use resources from "default".
  - Other apps (lane_detection, paddle_ocr, ...)
      -> must exist in inputs.json and use their own images/videos.
  - No automatic fallback to "default" for unknown apps.
  - --i must be a simple string (no '/' or '\\').
  - Lookup logic:
      If ID == "default":
        1) resources[app].images[0].path (if any)
        2) otherwise resources[app].videos[0].path
      Otherwise:
        1) images[].name == ID
        2) videos[].name == ID
      If not found -> error + print supported inputs for this app.
  - On success, the script prints the downloaded file path to stdout
    (so it can be captured in a variable).

Environment:
  RESOURCES_JSON   Override path to inputs.json (default: <script_dir>/inputs.json)
EOF
}


require_jq() {
  if ! command -v jq >/dev/null 2>&1; then
    echo "Error: jq is required but not installed." >&2
    exit 1
  fi
}

# Still useful if you ever want to normalize app names
normalize_app() {
  echo "$1" | tr '[:upper:]' '[:lower:]'
}

app_exists() {
  local app="$1"
  jq -e --arg app "$app" '.resources[$app] // empty' "$RESOURCES_JSON" >/dev/null 2>&1
}

use_app_for() {
  # Decide which section to use in inputs.json for a given app name
  local app="$1"
  local special

  # Check if this app is one of the special apps that must use "default"
  for special in "${SPECIAL_DEFAULT_APPS[@]}"; do
    if [[ "$special" == "$app" ]]; then
      # Must use "default"
      if ! app_exists "default"; then
        echo "Error: 'default' resources section not found in $RESOURCES_JSON" >&2
        return 1
      fi
      echo "default"
      return 0
    fi
  done

  # Not special: must exist as its own app
  if ! app_exists "$app"; then
    echo "Error: app \"$app\" not found in $RESOURCES_JSON and is not one of the default-mapped apps." >&2
    return 1
  fi

  echo "$app"
  return 0
}

list_app() {
  local app="$1"

  if [ ! -f "$RESOURCES_JSON" ]; then
    echo "Error: inputs.json not found at: $RESOURCES_JSON" >&2
    exit 1
  fi

  require_jq

  local use_app
  if ! use_app="$(use_app_for "$app")"; then
    # use_app_for already printed an error
    exit 1
  fi

  local NAME_COL_WIDTH=20  # visible width after closing quote

  # Bold red "images:"
  printf '\e[1;31mimages:\e[0m\n'
  jq -r --arg app "$use_app" '
    .resources[$app].images // [] |
    .[] |
    [ .name, .description ] | @tsv
  ' "$RESOURCES_JSON" | while IFS=$'\t' read -r name desc; do
    local len padding
    len=${#name}
    if (( len < NAME_COL_WIDTH )); then
      padding=$((NAME_COL_WIDTH - len))
    else
      padding=1
    fi
    printf 'name="%s"%*s | description="%s"\n' "$name" "$padding" "" "$desc"
  done

  echo
  # Bold red "videos:"
  printf '\e[1;31mvideos:\e[0m\n'
  jq -r --arg app "$use_app" '
    .resources[$app].videos // [] |
    .[] |
    [ .name, .description ] | @tsv
  ' "$RESOURCES_JSON" | while IFS=$'\t' read -r name desc; do
    local len padding
    len=${#name}
    if (( len < NAME_COL_WIDTH )); then
      padding=$((NAME_COL_WIDTH - len))
    else
      padding=1
    fi
    printf 'name="%s"%*s | description="%s"\n' "$name" "$padding" "" "$desc"
  done
}

download() {
  local path="$1"
  local target_dir="$2"

  mkdir -p "$target_dir"

  if [[ ! "$path" =~ ^https?:// ]]; then
    echo "Error: resource path is not a URL. Got: $path" >&2
    exit 1
  fi

  local filename
  filename="$(basename "$path")"
  local dest="${target_dir%/}/$filename"

  echo "Downloading $path -> $dest" >&2

  if command -v curl >/dev/null 2>&1; then
    curl -L -o "$dest" "$path"
  else
    wget -O "$dest" "$path"
  fi

  echo "Downloaded to: $dest" >&2
  echo "$dest"   # ← return only the path on stdout
}

get_resource() {
  local app="$1"
  local id="$2"
  local target_dir="$3"

  if [ ! -f "$RESOURCES_JSON" ]; then
    echo "Error: inputs.json not found at: $RESOURCES_JSON" >&2
    exit 1
  fi

  require_jq

  # Validate ID is not a path (unless it is the special "default")
  if [[ "$id" != "default" && ( "$id" == *"/"* || "$id" == *"\\"* ) ]]; then
    echo "Error: --i should be an ID (like bus), not a path. Got: $id" >&2
    exit 1
  fi

  local use_app
  if ! use_app="$(use_app_for "$app")"; then
    exit 1
  fi

  local path=""

  if [ "$id" = "default" ]; then
    # 🟦 Default: pick first image
    path="$(jq -r --arg app "$use_app" '
      (.resources[$app].images // [] | .[0].path) // empty
    ' "$RESOURCES_JSON")"

    # If no images → first video
    if [ -z "$path" ]; then
      path="$(jq -r --arg app "$use_app" '
        (.resources[$app].videos // [] | .[0].path) // empty
      ' "$RESOURCES_JSON")"
    fi

  else
    # 🟨 Normal: find by ID in images
    path="$(jq -r --arg app "$use_app" --arg name "$id" '
      .resources[$app].images // [] |
      map(select(.name == $name)) |
      .[0].path // empty
    ' "$RESOURCES_JSON")"

    # If not found → videos
    if [ -z "$path" ]; then
      path="$(jq -r --arg app "$use_app" --arg name "$id" '
        .resources[$app].videos // [] |
        map(select(.name == $name)) |
        .[0].path // empty
      ' "$RESOURCES_JSON")"
    fi
  fi

  if [ -z "$path" ]; then
    echo "Error: no input found for app \"$app\" (using \"$use_app\") with ID \"$id\"." >&2
    echo
    echo "Supported inputs for this app:" >&2
    list_app "$app" >&2
    exit 1
  fi

  # 🟩 download() now returns the final path on stdout
  download "$path" "$target_dir"
}


# -------- main --------

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

cmd="$1"
shift

APP=""
TARGET_DIR="."
ID=""

while [ $# -gt 0 ]; do
  case "$1" in
    --app)
      APP="$2"
      shift 2
      ;;
    --target-dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    --i)
      ID="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "$cmd" in
  list)
    if [ -z "$APP" ]; then
      echo "Error: --app is required for 'list'." >&2
      usage
      exit 1
    fi
    list_app "$APP"
    ;;
  get)
    if [ -z "$APP" ] || [ -z "$ID" ]; then
      echo "Error: --app and --i are required for 'get'." >&2
      usage
      exit 1
    fi
    get_resource "$APP" "$ID" "$TARGET_DIR"
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac
