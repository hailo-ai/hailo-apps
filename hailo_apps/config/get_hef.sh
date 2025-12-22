#!/usr/bin/env bash
set -euo pipefail

# -------------------------------
# Config / defaults
# -------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NETWORKS_JSON="${NETWORKS_JSON:-"$SCRIPT_DIR/networks.json"}"  # override via env if needed

# -------------------------------
# Helpers
# -------------------------------
err()   { echo "Error: $*" >&2; }
info()  { echo "[INFO] $*"; }

command_exists() { command -v "$1" >/dev/null 2>&1; }

require_jq() {
    command_exists jq || { err "jq is required. Install jq (e.g., sudo apt-get install -y jq)."; exit 1; }
}

get_device_architecture() {
    local out
    out="$(hailortcli fw-control identify 2>/dev/null | tr -d '\0' || true)"
    # Only take the first device architecture found
    local arch
    arch=$(echo "$out" | awk -F: '/Device Architecture/ {print tolower($2)}' | head -n1 | xargs)
    [[ "$arch" == "hailo15h" ]] && arch="hailo10h"  # normalize
    echo "$arch"
}

get_hailort_version() {
    hailortcli -v 2>/dev/null | grep -oE '[0-9]+(\.[0-9]+){1,2}' | head -1
}

validate_arch() {
    local a lc
    lc="$(echo -n "$1" | tr '[:upper:]' '[:lower:]')"
    case "$lc" in
    hailo8|hailo8l|hailo10h) echo "$lc"; return 0 ;;
    *) err "Invalid architecture '$1'. Supported: hailo8, hailo8l, hailo10h"; exit 1 ;;
    esac
}

suggest_name() {
  # Best-effort suggestion using Python's difflib if available
  local bad="$1"; shift
  local candidates=("$@")
  if command_exists python3; then
    python3 - "$bad" "${candidates[@]}" << 'PY'
import sys, difflib
bad = sys.argv[1]
cands = sys.argv[2:]
m = difflib.get_close_matches(bad, cands, n=1, cutoff=0.5)
print(m[0] if m else "")
PY
  else
    echo ""
  fi
}

# -------------------------------
# JSON accessors (via jq)
# -------------------------------
get_app_nets() {
    # $1 app_key (e.g., object_detection)
    require_jq
    jq -r --arg app "$1" '.apps[$app] | keys[]?' "$NETWORKS_JSON"
}

get_net_meta() {
    # $1 app_key, $2 net_name, $3 field
    require_jq
    jq -r --arg app "$1" --arg net "$2" --arg f "$3" '.apps[$app][$net][$f] // empty' "$NETWORKS_JSON"
}

get_net_arch_list() {
    # $1 app_key, $2 net_name
    require_jq
    jq -r --arg app "$1" --arg net "$2" '.apps[$app][$net].arch[]?' "$NETWORKS_JSON"
}

filter_nets_by_arch() {
    # $1 app_key, $2 arch
    require_jq
    jq -r --arg app "$1" --arg arch "$2" '
    .apps[$app] | to_entries[] | select(.value.arch[]? == $arch) | .key
    ' "$NETWORKS_JSON"
}

get_hef_arch() {
    local hef_path="$1"
    if ! command -v hailortcli >/dev/null 2>&1; then
        echo "ERROR: hailortcli not found in PATH" >&2
        return 1
    fi
    # Example line: "Architecture HEF was compiled for: HAILO15H"
    local line
    line="$(hailortcli parse-hef "$hef_path" 2>/dev/null | grep -i 'Architecture HEF was compiled for')" || return 2
    echo "$line" | sed -E 's/.*:\s*([A-Za-z0-9]+).*/\1/'
}

# verify_local_hef <hef_path> <requested_arch> [--auto-rename]
# - If <hef_path> exists:
#     * Parse its arch; if mismatched → either error out (default) or auto-rename and continue.
#     * On success, prints the real path to stdout and returns 0.
# - If parse fails (UNKNOWN), we allow it (best-effort) and return 0.
verify_local_hef() {
    local hef_path="$1"
    local requested_arch="$(echo "$2" | tr '[:lower:]' '[:upper:]')"  # HAILO8 / HAILO8L / HAILO10H

    if [[ ! -f "$hef_path" ]]; then
        return 3
    fi

    local existing_arch
    existing_arch="$(get_hef_arch "$hef_path")" || existing_arch="UNKNOWN"
    local existing_upper="${existing_arch^^}"

    # If architecture parsed successfully:
    if [[ "$existing_upper" != "UNKNOWN" ]]; then

        # ✅ Device hailo8 + HEF HAILO8L → allowed
        if [[ "$requested_arch" == "HAILO8" && "$existing_upper" == "HAILO8L" ]]; then
            echo "Note: Using HAILO8L HEF on HAILO8 device (compatible)." >&2

        # ✅ Device hailo10h + HEF HAILO15H → allowed
        elif [[ "$requested_arch" == "HAILO10H" && "$existing_upper" == "HAILO15H" ]]; then
            echo "Note: Using HAILO15H HEF on HAILO10H device (compatible)." >&2

        # ❌ All other mismatches
        elif [[ "$existing_upper" != "$requested_arch" ]]; then
            echo "ERROR: HEF mismatch for '$hef_path' — requested=$requested_arch, found=$existing_upper" >&2
            return 2
        fi
    fi

    # ✅ Only the resolved path is printed to stdout
    realpath "$hef_path"
    return 0
}


get_key_value() {
    # $1 app_key (e.g., object_detection, classifier, instance_seg)
    # $2 net_name (e.g., hardnet68, yolov5m_seg)
    # $3 key      (e.g., hefs, description, source)
    # $4 sub_key  (optional; e.g., to get first hef: hefs[0])

    require_jq

    local app_key_raw="$1"
    local net_name="$2"
    local key="$3"
    local sub_key="${4:-}"

    # Normalize app name to the JSON key used under .apps[]
    # (adjust if you use different names in networks.json)
    local app_key
    case "$app_key_raw" in
        classifer|classifier)
            app_key="classifier"
            ;;
        instance_seg|instance_segmenation|instance_segmentation)
            app_key="instance_segmentation"
            ;;
        *)
            app_key="$app_key_raw"
            ;;
    esac

    if [[ -n "$sub_key" ]]; then
        jq -r \
           --arg app "$app_key" \
           --arg net "$net_name" \
           --arg key "$key" \
           --arg sk "$sub_key" \
           '.apps[$app][$net][$key][$sk] // empty' \
           "$NETWORKS_JSON"
    else
        jq -r \
           --arg app "$app_key" \
           --arg net "$net_name" \
           --arg key "$key" \
           '.apps[$app][$net][$key] // empty' \
           "$NETWORKS_JSON"
    fi
}


# ---------------------------------------------------------------
# download_hef()
# ---------------------------------------------------------------
# Downloads a compiled .hef model file from either the Model Zoo
# or Customer Success (CS) S3 bucket.
#
# Usage:
#   download_hef <hw_arch> <hailort_version> <source> <network> [dest_dir]
#
# Arguments:
#   hw_arch         Hardware architecture: hailo8 | hailo8l | hailo10h
#   hailort_version HailoRT SDK version (e.g., 5.0.1)
#   source          Source of model: modelzoo | cs
#   network         Model name (e.g., yolov8n)
#   dest_dir        Optional output directory (default: current dir)
download_hef() {
    local hw_arch="$1"           # hailo8 | hailo8l | hailo10h
    local hailort_version="$2"   # e.g. 5.0.1
    local source="$3"            # modelzoo | cs
    local network="$4"           # e.g. yolov8n
    local dest_dir="${5:-.}"

    # These define which Model Zoo version (value) corresponds
    # to a given HailoRT SDK version (key) for each HW architecture.
    declare -A compat_10h=(
    ["5.0.1"]="v5.0.0"
    ["5.0.0"]="v5.0.0"
    ["5.1.0"]="v5.1.0"
    ["5.1.1"]="v5.1.0"
    ["5.1.2"]="v5.1.0"
    )
    declare -A compat_8=(
    ["4.23.0"]="v2.17.0"
    ["4.22.0"]="v2.16.0"
    ["4.21.0"]="v2.15.0"
    ["4.20.0"]="v2.14.0"
    )

    # These are fallback SDK versions to use when a specific
    # HailoRT version doesn’t exist in the mapping above.
    local DEFAULT_MZ_10H="5.1.0"
    local DEFAULT_MZ_8="4.23.0"

    local mz_version url

    # -----------------------------------------------------------
    # Determine Model Zoo version from HailoRT + architecture
    # -----------------------------------------------------------
    # For each architecture, map the HailoRT version to its
    # corresponding Model Zoo release. If unsupported, exit
    # with a helpful message.
    case "$hw_arch" in
    hailo10h)
        mz_version="${compat_10h[$hailort_version]:-}"
        if [[ -z "$mz_version" ]]; then
        err "HailoRT ${hailort_version} is not compatible with ${hw_arch}."
        err "Action: use HailoRT 5.x.x for hailo10h."
        exit 1
        fi
        ;;
    hailo8|hailo8l)
        mz_version="${compat_8[$hailort_version]:-}"
        if [[ -z "$mz_version" ]]; then
        err "HailoRT ${hailort_version} is not compatible with ${hw_arch}."
        err "Action: use HailoRT 4.x.x for ${hw_arch}."
        exit 1
        fi
        ;;
    *)
        err "Unknown hw-arch '${hw_arch}'"
        exit 1
        ;;
    esac

    # -----------------------------------------------------------
    # Build the S3 download URL
    # -----------------------------------------------------------
    # Both Model Zoo and CS follow similar folder structures.
    # modelzoo  → s3://hailo-model-zoo/ModelZoo/Compiled/<mz>/<arch>/<network>.hef
    # cs        → s3://hailo-csdata/resources/hefs/<mz>/<arch>/<network>.hef
    case "${source,,}" in
    modelzoo|model_zoo)
        url="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/${mz_version}/${hw_arch}/${network}.hef"
        ;;
    cs)
        url="https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/${mz_version}/${hw_arch}/${network}.hef"
        ;;
    *)
        err "Unknown source '$source' (expected: modelzoo | cs)"; exit 1;;
    esac

    # -----------------------------------------------------------
    # Verify URL exists; if not, attempt fallback (CS only)
    # -----------------------------------------------------------
    # Performs a HEAD request (-I) to avoid downloading the file.
    # If the request fails and the source is CS, retry using
    # the default Model Zoo version for that architecture.
    if ! curl -sfI "$url" >/dev/null; then
        if [[ "${source,,}" == "cs" ]]; then
            # Select the fallback version based on architecture
            local fallback_mz
            if [[ "$hw_arch" == "hailo10h" ]]; then
                fallback_mz="${compat_10h[$DEFAULT_MZ_10H]:-}"
            else
                fallback_mz="${compat_8[$DEFAULT_MZ_8]:-}"
            fi
            # Retry only if the fallback differs from the original
            if [[ "$fallback_mz" != "$mz_version" ]]; then
                local fallback_url="https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/${fallback_mz}/${hw_arch}/${network}.hef"
                info "Primary CS path not found; retrying with default mz_version '${fallback_mz}' ..."
                if curl -sfI "$fallback_url" >/dev/null; then
                    url="$fallback_url"
                else
                    err "Model not found at: $url"
                    err "Also not found at the default fallback: $fallback_url"
                    return 2
                fi
            else
                err "Model not found at: $url"
                return 2
            fi
        else
            err "Model not found at: $url"
            return 2
        fi
    fi

    mkdir -p "$dest_dir"
    local filepath="${dest_dir}/${network}.hef"
    curl -L --fail --progress-bar -o "$filepath" "$url"
    echo "$filepath"
}


# -------------------------------
# Actions
# -------------------------------
cmd_get_key_value() {
  local app_key="" net_name="" key="" sub_key=""

  while (( $# )); do
    case "$1" in
      --app)
        app_key="$2"; shift 2;;
      --name)
        net_name="$2"; shift 2;;
      --key)
        key="$2"; shift 2;;
      --sub_key|--sub-key)
        sub_key="$2"; shift 2;;
      --)
        shift; break;;
      -h|--help)
        usage; exit 0;;
      *)
        err "Unknown arg for get_key_value: $1"; exit 1;;
    esac
  done

  if [[ -z "$app_key" || -z "$net_name" || -z "$key" ]]; then
    err "Usage: $0 get_key_value --app <app> --name <model> --key <key> [--sub_key <sub>]"
    exit 1
  fi

  get_key_value "$app_key" "$net_name" "$key" "$sub_key"
}


cmd_list() {
  local app_key="" hw_arch=""

  # Parse flags for 'list'
  while (( $# )); do
    case "$1" in
      --app)      app_key="$2"; shift 2;;
      --hw-arch)  hw_arch="$2"; shift 2;;
      --)         shift; break;;
      -h|--help)  usage; exit 0;;
      *)          # allow positional fallback: list <app> [<arch>]
                  if [[ -z "$app_key" ]]; then
                    app_key="$1"
                  elif [[ -z "$hw_arch" ]]; then
                    hw_arch="$1"
                  else
                    err "Unknown arg: $1"; exit 1
                  fi
                  shift;;
    esac
  done

  [[ -n "$app_key" ]] || { err "Missing --app <app_key>"; exit 1; }
  require_jq

  # Collect nets (optionally filtered by arch)
  local -a nets=()
  if [[ -n "$hw_arch" ]]; then
    hw_arch="$(validate_arch "$hw_arch")"
    mapfile -t nets < <(filter_nets_by_arch "$app_key" "$hw_arch")
  else
    mapfile -t nets < <(get_app_nets "$app_key")
  fi

  echo -e "Supported networks for ${app_key}${hw_arch:+ (hw-arch=$hw_arch)}:"

  # ----- pretty printing helpers -----
  _trunc() {  # _trunc <string> <maxlen>
    local s="$1" max="$2"
    (( ${#s} <= max )) && { printf "%s" "$s"; return; }
    printf "%s…" "${s:0:max-1}"
  }

_print_net_line() {  # _print_net_line <name> <desc> <archs> <source> <hefs>
  local name="$1" desc="$2" archs="$3" src="$4" hefs="$5"
  local NAME_W=16 ARCH_W=23 SRC_W=11 HEF_W=12 DESC_W=40
  printf " - %-*s | arch=[%s] | src=%-*s | hefs=[%-*s] | desc=%-*s\n" \
    "$NAME_W" "$name" \
    "$archs" \
    "$SRC_W"  "$(_trunc "$src" "$SRC_W")" \
    "$HEF_W"  "$(_trunc "$hefs" "$HEF_W")" \
    "$DESC_W" "$(_trunc "$desc" "$DESC_W")"
}

  # ----- group by .group (fallback to .bundle_id) -----
  declare -A group_map=()    # group_key -> "net1 net2 ..."
  declare -a group_order=()  # first-seen group keys
  declare -a singles=()      # nets without any group

  local net grp
  for net in "${nets[@]}"; do
    grp="$(get_net_meta "$app_key" "$net" "group")"
    [[ "$grp" == "null" ]] && grp=""
    [[ -z "$grp" ]] && grp="$(get_net_meta "$app_key" "$net" "bundle_id")"  # backward compat
    [[ "$grp" == "null" ]] && grp=""

    if [[ -n "$grp" ]]; then
      if [[ -z "${group_map[$grp]+x}" ]]; then
        group_map[$grp]="$net"
        group_order+=("$grp")
      else
        group_map[$grp]="${group_map[$grp]} $net"
      fi
    else
      singles+=("$net")
    fi
  done

  # ----- 1) print each group with only separators -----
  local _grp n desc src archs hefs
  for _grp in "${group_order[@]}"; do
    echo "---------"
    for n in ${group_map[$_grp]}; do
      desc="$(get_net_meta "$app_key" "$n" "description")"
      src="$(get_net_meta "$app_key" "$n" "source")"
      archs="$(get_net_arch_list "$app_key" "$n" | paste -sd, -)"
      hefs="$(get_net_meta "$app_key" "$n" "hefs" | jq -r -c 'join(",")' 2>/dev/null)"
      _print_net_line "$n" "${desc:-}" "${archs:-}" "${src:-}" "${hefs:-}"
    done
    echo "---------"
  done

  # ----- 2) print singles (no headers, no separators) -----
  for n in "${singles[@]}"; do
    desc="$(get_net_meta "$app_key" "$n" "description")"
    src="$(get_net_meta "$app_key" "$n" "source")"
    archs="$(get_net_arch_list "$app_key" "$n" | paste -sd, -)"
    hefs="$(get_net_meta "$app_key" "$n" "hefs" | jq -r -c 'join(",")' 2>/dev/null)"
    _print_net_line "$n" "${desc:-}" "${archs:-}" "${src:-}" "${hefs:-}"
  done

  # nothing matched
  if (( ${#group_order[@]} == 0 && ${#singles[@]} == 0 )); then
    echo "(No supported networks${hw_arch:+ for hw-arch=$hw_arch})"
  fi
}


cmd_verify_arch() {
  # Inputs:
  #   --hef <path_to_hef>        (required)
  #   --hw-arch <arch>           (optional; if missing, detect from device)
  local hef_path="" hw_arch_in=""
  while (( $# )); do
    case "$1" in
      --hef)      hef_path="$2"; shift 2;;
      --hw-arch)  hw_arch_in="$2"; shift 2;;
      --)         shift; break;;
      -h|--help)  usage; exit 0;;
      *)          err "Unknown arg for verify-arch: $1"; exit 1;;
    esac
  done

  # Required hef path
  if [[ -z "$hef_path" ]]; then
    err "Usage: $0 verify-arch --hef <path_to_hef> [--hw-arch <hailo8|hailo8L|hailo10h>]"
    exit 1
  fi

  if [[ ! -f "$hef_path" ]]; then
    err "HEF file not found: $hef_path"
    exit 1
  fi

  # Determine and validate hw-arch
  local hw_arch="$hw_arch_in"
  if [[ -z "$hw_arch" ]]; then
    hw_arch="$(get_device_architecture || true)"
    if [[ -z "$hw_arch" ]]; then
      err "No device detected. Pass --hw-arch explicitly (e.g., hailo8)."
      exit 1
    fi
    # Only print detection message when user did NOT pass --hw-arch
    echo "Detected device: $hw_arch"
    echo "Verifying HEF compatibility..."
  fi
  hw_arch="$(validate_arch "$hw_arch")"

  # Use existing helper to validate the HEF against the architecture.
  # Third argument '0' = non-interactive / no extra prompts.
  if out_path="$(verify_local_hef "$hef_path" "$hw_arch")"; then
    # On success, always return the resolved real path on stdout.
    realpath "$out_path"
    exit 0
  else
    rc=$?
    # verify_local_hef should have already printed a reason to stderr.
    exit "$rc"
  fi
}


cmd_get() {
  # Inputs:
  #   --app <app_key>            (required)
  #   --net <name|path>          (required)
  #   --hw-arch <arch>           (optional; if missing, detect from device)
  #   --dest <dir>               (optional; default .)
  local app_key="" net_in="" hw_arch_in="" dest_dir="."
  while (( $# )); do
    case "$1" in
      --app)     app_key="$2"; shift 2;;
      --net)     net_in="$2"; shift 2;;
      --hw-arch) hw_arch_in="$2"; shift 2;;
      --dest)    dest_dir="$2"; shift 2;;
      --)        shift; break;;
      -h|--help) usage; exit 0;;
      *) err "Unknown arg: $1"; exit 1;;
    esac
  done
  [[ -n "$app_key" && -n "$net_in" ]] || {
    err "Usage: $0 get --app <app_key> --net <name|path> [--hw-arch <hailo8|hailo8L|hailo10h>] [--dest DIR]"
    exit 1
  }

  # Determine and validate hw-arch
  local hw_arch="$hw_arch_in"
  if [[ -z "$hw_arch" ]]; then
    hw_arch="$(get_device_architecture || true)"
    [[ -n "$hw_arch" ]] || { err "No device detected. Pass --hw-arch explicitly (e.g., hailo8)."; exit 1; }
  fi
  hw_arch="$(validate_arch "$hw_arch")"


  # strip .hef for lookup
  local net_key="$net_in"
  [[ "$net_key" == *.hef ]] && net_key="${net_key%.hef}"

  require_jq

  # Verify net exists in JSON
  if ! get_app_nets "$app_key" | grep -qx "$net_key"; then
    err "Unknown net '$net_key' for app='$app_key'."
    echo -e "\e[33m" >&2
    cmd_list "$app_key" >&2
    echo -e "\e[0m" >&2
    mapfile -t all_nets < <(get_app_nets "$app_key")
    suggestion="$(suggest_name "$net_key" "${all_nets[@]}")"
    [[ -n "$suggestion" ]] && echo -e "\n\e[93mDid you mean: $suggestion?\e\n[0m" >&2
    exit 1
  fi

  # Validate net supports the hw-arch
  if ! get_net_arch_list "$app_key" "$net_key" | grep -qx "$hw_arch"; then
    err "Net '$net_key' does not support hw-arch=$hw_arch."
    echo -e "\e[33m" >&2
    cmd_list "$app_key" "$hw_arch" >&2
    echo -e "\e[0m" >&2
    exit 1
  fi

  # Source (default modelzoo if missing)
  local source
  source="$(get_net_meta "$app_key" "$net_key" "source")"
  [[ -z "$source" ]] && source="modelzoo"

  # HailoRT version
  local hv
  hv="$(get_hailort_version || true)"
  [[ -n "$hv" ]] || { err "Cannot parse HailoRT version. Is hailortcli installed?"; exit 1; }

  info "Resolving net='${net_key}' (hw-arch=${hw_arch}, source=${source}, HailoRT=${hv})"
  set +e
  filepath="$(download_hef "$hw_arch" "$hv" "$source" "$net_key" "$dest_dir")"
  status=$?
  set -e
  if (( status == 0 )); then
    realpath "$filepath"; exit 0
  elif (( status == 2 )); then
    exit 1
  else
    err "Failed to obtain HEF for '${net_key}' (hw-arch=${hw_arch}, source=${source})."
    exit 1
  fi
}


usage() {
  cat <<EOF
Usage:
  $0 list \
        --app <app_key> \
        [--hw-arch <hailo8|hailo8L|hailo10h>]

  $0 get \
        --app <app_key> \
        --net <name|path> \
        [--hw-arch <hailo8|hailo8L|hailo10h>] \
        [--dest DIR]

  $0 verify-arch \
        --hef <path_to_hef> \
        [--hw-arch <hailo8|hailo8L|hailo10h>]

  $0 get_key_value \
        --app <app_key> \
        --name <model_name> \
        --key <key> \
        [--sub_key <sub_key>]

ENV:
  NETWORKS_JSON=<path>
      Override networks.json location (default: $NETWORKS_JSON)
EOF
}

case "${1:-}" in
  list)         shift; cmd_list "$@";;
  get)          shift; cmd_get "$@";;
  verify-arch)  shift; cmd_verify_arch "$@";;
  get_key_value)shift; cmd_get_key_value "$@";;
  -h|--help|"") usage;;
  *)            err "Unknown command: $1"; usage; exit 1;;
esac