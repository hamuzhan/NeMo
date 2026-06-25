#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_nemo.sh — Make NeMo Speech (ASR) ready inside an NGC container.
#
# Safe by design for NGC images:
#   * NEVER touches the container's torch / CUDA / numpy / transformers.
#     Every pip install uses --no-deps, so pinned NGC packages stay put.
#   * Idempotent: re-running only installs what is actually missing.
#   * Does NOT install NeMo itself (this repo is used in-place via PYTHONPATH).
#
# On first run it offers a menu to fetch the Nemotron 3.5 ASR streaming model:
#   1) Download from Hugging Face       2) Use a local .nemo path
#   3) Reuse existing HF cache          4) Skip model setup
#
# Usage:
#   source setup_nemo.sh         # recommended: keeps PYTHONPATH in your shell
#   ./setup_nemo.sh              # also works; prints the export to run
#
# Handy flags:
#   ./setup_nemo.sh --check      # only report environment status, install nothing
#   ./setup_nemo.sh --no-model   # set up deps only, skip the model menu
#   ./setup_nemo.sh --smoke      # after setup, run a tiny load + transcribe test
# ---------------------------------------------------------------------------

# Resolve the NeMo repo root = directory containing this script.
_SELF="${BASH_SOURCE[0]:-$0}"
NEMO_ROOT="$(cd "$(dirname "$_SELF")" && pwd)"

# --- config -----------------------------------------------------------------
MODEL_ID="nvidia/nemotron-3.5-asr-streaming-0.6b"
STATE_DIR="${NEMO_ROOT}/.nemo_setup"
STATE_FILE="${STATE_DIR}/state.env"        # remembers chosen model path
PY="${PYTHON:-python}"

# pip packages NeMo[asr] needs that NGC base images typically lack.
# All installed with --no-deps to protect the container's torch/CUDA stack.
ASR_PKGS=(
  hydra-core lhotse librosa soundfile editdistance kaldialign sacrebleu
  whisper-normalizer text-unidecode "ruamel.yaml" pyloudnorm jiwer
  wrapt smart-open marshmallow nv_one_logger_pytorch_lightning_integration
  scikit-learn
)
# transitive deps that the above need but --no-deps will not pull in.
TRANSITIVE_PKGS=(
  lazy_loader soxr audioread pooch msgpack cytoolz toolz intervaltree joblib
  portalocker colorama lxml rapidfuzz future sortedcontainers narwhals
  threadpoolctl
)

# import-name : pip-name  (import name is what we actually probe)
declare -A IMPORT_TO_PIP=(
  [hydra]=hydra-core [lhotse]=lhotse [librosa]=librosa [soundfile]=soundfile
  [editdistance]=editdistance [kaldialign]=kaldialign [sacrebleu]=sacrebleu
  [whisper_normalizer]=whisper-normalizer [text_unidecode]=text-unidecode
  [ruamel.yaml]=ruamel.yaml [pyloudnorm]=pyloudnorm [jiwer]=jiwer
  [wrapt]=wrapt [smart_open]=smart-open [marshmallow]=marshmallow
  [sklearn]=scikit-learn
  [lazy_loader]=lazy_loader [soxr]=soxr [audioread]=audioread [pooch]=pooch
  [msgpack]=msgpack [cytoolz]=cytoolz [toolz]=toolz [intervaltree]=intervaltree
  [joblib]=joblib [portalocker]=portalocker [colorama]=colorama [lxml]=lxml
  [rapidfuzz]=rapidfuzz [future]=future [sortedcontainers]=sortedcontainers
  [narwhals]=narwhals [threadpoolctl]=threadpoolctl
)

# --- pretty printing --------------------------------------------------------
if [ -t 1 ]; then
  C_G=$'\033[0;32m'; C_Y=$'\033[1;33m'; C_R=$'\033[0;31m'; C_B=$'\033[1;34m'; C_0=$'\033[0m'
else
  C_G=""; C_Y=""; C_R=""; C_B=""; C_0=""
fi
info() { printf "%s[setup]%s %s\n" "$C_B" "$C_0" "$*"; }
ok()   { printf "%s[ ok ]%s %s\n" "$C_G" "$C_0" "$*"; }
warn() { printf "%s[warn]%s %s\n" "$C_Y" "$C_0" "$*"; }
err()  { printf "%s[fail]%s %s\n" "$C_R" "$C_0" "$*"; }

# A script that is sourced must `return`, not `exit`, to avoid killing the shell.
_sourced=0
if [ "${BASH_SOURCE[0]}" != "$0" ]; then _sourced=1; fi
_die() { err "$*"; if [ "$_sourced" = "1" ]; then return 1; else exit 1; fi; }

# --- arg parsing ------------------------------------------------------------
DO_MODEL=1; DO_SMOKE=0; CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --check)    CHECK_ONLY=1 ;;
    --no-model) DO_MODEL=0 ;;
    --smoke)    DO_SMOKE=1 ;;
    -h|--help)
      sed -n '2,40p' "$_SELF" | sed 's/^# \{0,1\}//'
      { [ "$_sourced" = "1" ] && return 0; } || exit 0 ;;
    *) warn "unknown arg: $arg (ignored)" ;;
  esac
done

# ---------------------------------------------------------------------------
# 1. Environment report
# ---------------------------------------------------------------------------
info "NeMo repo root: ${NEMO_ROOT}"
"$PY" - <<'PYEOF'
import sys
print(f"[setup] Python  : {sys.version.split()[0]}")
try:
    import torch
    print(f"[setup] torch   : {torch.__version__}")
    print(f"[setup] CUDA    : {torch.version.cuda}  (available={torch.cuda.is_available()})")
    if torch.cuda.is_available():
        print(f"[setup] GPU     : {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"[setup] torch   : NOT IMPORTABLE ({e})")
PYEOF

# ---------------------------------------------------------------------------
# 2. Probe missing python deps
# ---------------------------------------------------------------------------
probe_missing() {
  # echoes space-separated list of missing pip package names
  local missing=()
  local imp pip
  for imp in "${!IMPORT_TO_PIP[@]}"; do
    pip="${IMPORT_TO_PIP[$imp]}"
    if ! "$PY" -c "import ${imp}" >/dev/null 2>&1; then
      missing+=("$pip")
    fi
  done
  # de-duplicate (sklearn maps twice) and emit
  printf "%s\n" "${missing[@]}" | sort -u | tr '\n' ' '
}

info "Probing NeMo[asr] python dependencies..."
MISSING="$(probe_missing)"
MISSING="$(echo "$MISSING" | xargs)"   # trim

if [ -z "$MISSING" ]; then
  ok "All required python dependencies are present."
else
  warn "Missing: ${MISSING}"
fi

if [ "$CHECK_ONLY" = "1" ]; then
  # also probe libsndfile / ffmpeg for info
  if ldconfig -p 2>/dev/null | grep -qi sndfile; then ok "libsndfile present"; else warn "libsndfile missing (apt-get install -y libsndfile1)"; fi
  command -v ffmpeg >/dev/null 2>&1 && ok "ffmpeg present" || warn "ffmpeg missing (only needed for non-wav/flac decoding)"
  info "--check only: nothing installed."
  { [ "$_sourced" = "1" ] && return 0; } || exit 0
fi

# ---------------------------------------------------------------------------
# 3. Install missing deps (always --no-deps to protect torch/CUDA)
# ---------------------------------------------------------------------------
if [ -n "$MISSING" ]; then
  # Install the full curated set with --no-deps so transitive bits come too,
  # but pip still skips anything already satisfied.
  info "Installing missing deps with --no-deps (torch/CUDA untouched)..."
  if ! "$PY" -m pip install --no-deps --root-user-action=ignore \
        "${ASR_PKGS[@]}" "${TRANSITIVE_PKGS[@]}"; then
    _die "pip install failed. Re-run, or inspect the output above."
  fi

  # Re-probe; if still missing something, surface it explicitly.
  STILL="$(probe_missing | xargs)"
  if [ -n "$STILL" ]; then
    warn "Still missing after install: ${STILL}"
    info "Attempting targeted --no-deps install for the stragglers..."
    "$PY" -m pip install --no-deps --root-user-action=ignore $STILL || true
    STILL="$(probe_missing | xargs)"
  fi
  if [ -n "$STILL" ]; then
    _die "Could not satisfy: ${STILL}. Install manually with: pip install --no-deps ${STILL}"
  fi
  ok "Dependencies installed."
fi

# ---------------------------------------------------------------------------
# 4. Make NeMo importable from this repo (no editable install needed)
# ---------------------------------------------------------------------------
case ":${PYTHONPATH:-}:" in
  *":${NEMO_ROOT}:"*) : ;;                       # already present
  *) export PYTHONPATH="${NEMO_ROOT}${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
ok "PYTHONPATH includes repo root."

info "Verifying ASR collection import..."
if "$PY" -c "import nemo.collections.asr" >/dev/null 2>&1; then
  ok "import nemo.collections.asr  -> OK"
else
  err "ASR import still failing. Full traceback:"
  "$PY" -c "import nemo.collections.asr" || true
  _die "ASR import failed."
fi

# ---------------------------------------------------------------------------
# 5. Model setup menu (first run only, unless --no-model)
# ---------------------------------------------------------------------------
mkdir -p "$STATE_DIR"
# shellcheck disable=SC1090
[ -f "$STATE_FILE" ] && . "$STATE_FILE"

model_ready() { [ -n "${NEMO_MODEL_PATH:-}" ] && [ -f "${NEMO_MODEL_PATH:-/nonexistent}" ]; }

find_hf_cache_nemo() {
  local g
  g=$(ls -1 "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--nvidia--nemotron-3.5-asr-streaming-0.6b/snapshots/*/*.nemo 2>/dev/null | head -1)
  [ -z "$g" ] && g=$(ls -1 "$HOME"/.cache/huggingface/hub/models--nvidia--nemotron-3.5-asr-streaming-0.6b/snapshots/*/*.nemo 2>/dev/null | head -1)
  echo "$g"
}

save_state() {
  printf 'NEMO_MODEL_PATH=%q\n' "$1" > "$STATE_FILE"
  export NEMO_MODEL_PATH="$1"
}

if [ "$DO_MODEL" = "1" ]; then
  if model_ready; then
    ok "Model already configured: ${NEMO_MODEL_PATH}"
  else
    # If it's sitting in the HF cache, offer that as the default fast path.
    CACHED="$(find_hf_cache_nemo)"
    echo
    info "Model '${MODEL_ID}' is not configured yet. Choose how to provide it:"
    echo "    1) Download from Hugging Face  (${MODEL_ID})"
    echo "    2) Use a local .nemo path      (enter path manually)"
    if [ -n "$CACHED" ]; then
      echo "    3) Reuse HF cache              (${CACHED})"
    else
      echo "    3) Reuse HF cache              (none found)"
    fi
    echo "    4) Skip model setup"
    echo
    if [ "$_sourced" = "1" ] || [ -t 0 ]; then
      read -r -p "Selection [1-4] (default 3 if cache exists, else 1): " CHOICE
    else
      CHOICE=""   # non-interactive: pick a sensible default below
    fi
    [ -z "$CHOICE" ] && { [ -n "$CACHED" ] && CHOICE=3 || CHOICE=1; }

    case "$CHOICE" in
      1)
        info "Downloading ${MODEL_ID} from Hugging Face..."
        DL="$("$PY" - "$MODEL_ID" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
import glob, os
mid = sys.argv[1]
d = snapshot_download(repo_id=mid, allow_patterns=["*.nemo"])
files = glob.glob(os.path.join(d, "*.nemo"))
print(files[0] if files else "")
PYEOF
)"
        if [ -n "$DL" ] && [ -f "$DL" ]; then
          save_state "$DL"; ok "Downloaded: $DL"
        else
          _die "Download failed (check network / HF auth)."
        fi
        ;;
      2)
        read -r -p "Enter absolute path to .nemo file: " MP
        MP="${MP/#\~/$HOME}"
        if [ -f "$MP" ]; then save_state "$MP"; ok "Using local model: $MP"
        else _die "File not found: $MP"; fi
        ;;
      3)
        if [ -n "$CACHED" ]; then save_state "$CACHED"; ok "Using cached model: $CACHED"
        else _die "No cached model found. Re-run and pick 1 or 2."; fi
        ;;
      4|*)
        warn "Skipping model setup. Set NEMO_MODEL_PATH yourself later."
        ;;
    esac
  fi
fi

# ---------------------------------------------------------------------------
# 6. Optional smoke test
# ---------------------------------------------------------------------------
if [ "$DO_SMOKE" = "1" ]; then
  if model_ready; then
    info "Running smoke test (load model + 1s silent transcribe)..."
    HF_HUB_OFFLINE=1 "$PY" - "$NEMO_MODEL_PATH" <<'PYEOF'
import sys, numpy as np, tempfile, os, soundfile as sf
from nemo.collections.asr.models import ASRModel
mp = sys.argv[1]
m = ASRModel.restore_from(mp, map_location="cuda")
print("[smoke] loaded:", type(m).__name__)
# 1 second of silence @ 16k just to exercise the transcribe path
wav = os.path.join(tempfile.mkdtemp(), "silence.wav")
sf.write(wav, np.zeros(16000, dtype="float32"), 16000)
out = m.transcribe([wav], target_lang="en-US")
print("[smoke] transcribe ran, output:", out)
PYEOF
    [ $? -eq 0 ] && ok "Smoke test passed." || warn "Smoke test reported an error (see above)."
  else
    warn "No model configured; skipping smoke test."
  fi
fi

# ---------------------------------------------------------------------------
# 7. Final summary
# ---------------------------------------------------------------------------
echo
ok "NeMo environment is ready."
info "Repo root      : ${NEMO_ROOT}"
info "PYTHONPATH     : ${PYTHONPATH}"
[ -n "${NEMO_MODEL_PATH:-}" ] && info "Model path     : ${NEMO_MODEL_PATH}"
if [ "$_sourced" != "1" ]; then
  echo
  warn "You ran this with ./ — PYTHONPATH was set only for this script."
  warn "To keep it in your shell, run:  source setup_nemo.sh"
  echo "    export PYTHONPATH=\"${NEMO_ROOT}:\$PYTHONPATH\""
  [ -n "${NEMO_MODEL_PATH:-}" ] && echo "    export NEMO_MODEL_PATH=\"${NEMO_MODEL_PATH}\""
fi
echo
info "Streaming inference example:"
cat <<EXAMPLE
    python examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \\
        model_path="\${NEMO_MODEL_PATH}" \\
        dataset_manifest=<manifest.json> \\
        batch_size=1 target_lang=en-US \\
        att_context_size="[56,13]" strip_lang_tags=true \\
        output_path=./streaming_out
EXAMPLE
