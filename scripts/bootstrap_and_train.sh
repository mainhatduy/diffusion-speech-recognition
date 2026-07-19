#!/bin/bash
# =============================================================================
#  bootstrap_and_train.sh
#  Clone → install deps → load checkpoint → move to output dir → resume training
#
#  Usage (from any directory):
#    bash scripts/bootstrap_and_train.sh [OPTIONS]
#
#  Options:
#    --repo         Git repo URL  (default: derived from git remote "origin")
#    --branch       Git branch    (default: main)
#    --clone-dir    Where to clone (default: ./diffusion-speech-recognition)
#    --hf-repo      HuggingFace checkpoint repo ID
#                   (default: aiai-laboratory/discrete-diffusion-vi-multitask-checkpoint)
#    --output-dir   Local output dir to place the checkpoint into
#                   (default: outputs/vi_en_deep_fusion)
#    --config       Training config JSON
#                   (default: configs/vi_en_deep_fusion.json)
#    --gpu          CUDA_VISIBLE_DEVICES value (default: 0)
#    --skip-clone   Skip git clone (useful when already inside the repo)
#    --skip-install Skip uv sync (useful when .venv already exists)
#    --skip-download  Skip HF checkpoint download
#    --force-download Force re-download even if output dir is non-empty
#    -h, --help     Show this help
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_step()  { echo -e "\n${BOLD}${GREEN}══════════════════════════════════════════════${NC}"; \
              echo -e "${BOLD}${GREEN}  $*${NC}"; \
              echo -e "${BOLD}${GREEN}══════════════════════════════════════════════${NC}"; }

# ── Environment Tokens ──────────────────────────────────────────────────────
# Tokens are set here by default; override by exporting before calling the script.
export HF_TOKEN="${HF_TOKEN:-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}"
export WANDB_API_KEY="${WANDB_API_KEY:-1ce0793819a037f2b3729996816b5732ac107e84}"

# ── Default values ────────────────────────────────────────────────────────────
GIT_REPO=""
GIT_BRANCH="main"
CLONE_DIR="diffusion-speech-recognition"
HF_CHECKPOINT_REPO="aiai-laboratory/discrete-diffusion-vi-multitask-checkpoint"
OUTPUT_DIR="outputs/vi_en_deep_fusion"
TRAINING_CONFIG="configs/vi_en_deep_fusion.json"
GPU="0"
SKIP_CLONE=false
SKIP_INSTALL=false
SKIP_DOWNLOAD=false
FORCE_DOWNLOAD=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)            GIT_REPO="$2";           shift 2 ;;
        --branch)          GIT_BRANCH="$2";         shift 2 ;;
        --clone-dir)       CLONE_DIR="$2";          shift 2 ;;
        --hf-repo)         HF_CHECKPOINT_REPO="$2"; shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2";         shift 2 ;;
        --config)          TRAINING_CONFIG="$2";    shift 2 ;;
        --gpu)             GPU="$2";                shift 2 ;;
        --skip-clone)      SKIP_CLONE=true;         shift ;;
        --skip-install)    SKIP_INSTALL=true;       shift ;;
        --skip-download)   SKIP_DOWNLOAD=true;      shift ;;
        --force-download)  FORCE_DOWNLOAD=true;     shift ;;
        -h|--help)
            grep '^#' "$0" | head -n 25 | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   Diffusion-Speech Bootstrap & Resume Training       ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
log_info "HF checkpoint repo : ${HF_CHECKPOINT_REPO}"
log_info "Output dir         : ${OUTPUT_DIR}"
log_info "Training config    : ${TRAINING_CONFIG}"
log_info "GPU(s)             : ${GPU}"
echo ""

# ── STEP 1: Clone ─────────────────────────────────────────────────────────────
if [ "$SKIP_CLONE" = false ]; then
    log_step "STEP 1 / 4 — Git Clone"

    if [ -z "$GIT_REPO" ]; then
        # If already inside a git repo, derive the remote URL
        if git rev-parse --is-inside-work-tree &>/dev/null 2>&1; then
            GIT_REPO=$(git remote get-url origin 2>/dev/null || true)
        fi
    fi

    if [ -z "$GIT_REPO" ]; then
        log_error "No git repo URL found. Pass --repo <url> or run from inside the repo."
        exit 1
    fi

    log_info "Cloning ${GIT_REPO} (branch: ${GIT_BRANCH}) → ${CLONE_DIR}"
    git clone --branch "$GIT_BRANCH" "$GIT_REPO" "$CLONE_DIR"
    cd "$CLONE_DIR"
    log_ok "Clone complete."
else
    log_warn "Skipping clone (--skip-clone). Assuming CWD is the project root."
    if ! [ -f "pyproject.toml" ]; then
        log_error "pyproject.toml not found. Are you inside the project directory?"
        exit 1
    fi
fi

PROJECT_ROOT="$(pwd)"
log_info "Project root: ${PROJECT_ROOT}"

# ── STEP 2: Install dependencies ──────────────────────────────────────────────
if [ "$SKIP_INSTALL" = false ]; then
    log_step "STEP 2 / 4 — Install Dependencies (uv sync)"

    if ! command -v uv &>/dev/null; then
        log_error "'uv' is not installed. Install it first: https://docs.astral.sh/uv/"
        exit 1
    fi

    uv sync
    log_ok "uv sync complete."
else
    log_warn "Skipping uv sync (--skip-install)."
fi

# Activate venv so that subsequent uv run calls always use it
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    log_ok "Activated .venv"
fi

# ── STEP 3: Download checkpoint from HuggingFace ──────────────────────────────
if [ "$SKIP_DOWNLOAD" = false ]; then
    log_step "STEP 3 / 4 — Download Checkpoint from HuggingFace"
    log_info "Source repo : ${HF_CHECKPOINT_REPO}"
    log_info "Staging dir : ${OUTPUT_DIR}/_hf_staging"

    FORCE_FLAG=""
    if [ "$FORCE_DOWNLOAD" = true ]; then
        FORCE_FLAG="--force"
    fi

    # Download to a temporary staging sub-directory so we don't clobber
    # existing checkpoints in OUTPUT_DIR directly.
    STAGING_DIR="${OUTPUT_DIR}/_hf_staging"
    mkdir -p "$STAGING_DIR"

    uv run python scripts/model-manager/load_checkpoint.py \
        --repo_id   "$HF_CHECKPOINT_REPO" \
        --target_dir "$STAGING_DIR" \
        $FORCE_FLAG

    log_ok "Download complete. Staging dir: ${STAGING_DIR}"

    # ── Find the latest checkpoint (by step number) ───────────────────────────
    log_step "STEP 3b / 4 — Identify & Move Latest Checkpoint"

    LATEST_CKPT=""
    MAX_STEP=-1

    for ckpt_path in "${STAGING_DIR}"/checkpoint-*; do
        [ -d "$ckpt_path" ] || continue
        ckpt_name=$(basename "$ckpt_path")
        step_num="${ckpt_name#checkpoint-}"
        if [[ "$step_num" =~ ^[0-9]+$ ]] && [ "$step_num" -gt "$MAX_STEP" ]; then
            MAX_STEP="$step_num"
            LATEST_CKPT="$ckpt_path"
        fi
    done

    if [ -z "$LATEST_CKPT" ]; then
        log_error "No checkpoint-XXXXX directories found in ${STAGING_DIR}"
        exit 1
    fi

    CKPT_NAME=$(basename "$LATEST_CKPT")
    DEST_CKPT="${OUTPUT_DIR}/${CKPT_NAME}"

    log_info "Latest checkpoint : ${CKPT_NAME}  (step ${MAX_STEP})"
    log_info "Moving ${LATEST_CKPT}  →  ${DEST_CKPT}"

    mkdir -p "$OUTPUT_DIR"
    mv "$LATEST_CKPT" "$DEST_CKPT"

    # Move any top-level metadata files (args.json, tokenizer files, etc.)
    for meta_file in "${STAGING_DIR}"/*.json "${STAGING_DIR}"/*.model "${STAGING_DIR}"/*.txt; do
        [ -f "$meta_file" ] || continue
        meta_dest="${OUTPUT_DIR}/$(basename "$meta_file")"
        if [ ! -f "$meta_dest" ]; then
            mv "$meta_file" "$meta_dest"
            log_info "Moved metadata: $(basename "$meta_file")"
        fi
    done

    # Clean up staging dir if now empty
    rmdir "$STAGING_DIR" 2>/dev/null || true

    log_ok "Checkpoint ready at: ${DEST_CKPT}"

else
    log_warn "Skipping download (--skip-download). Searching for existing checkpoint in ${OUTPUT_DIR}…"

    DEST_CKPT=""
    MAX_STEP=-1
    for ckpt_path in "${OUTPUT_DIR}"/checkpoint-*; do
        [ -d "$ckpt_path" ] || continue
        ckpt_name=$(basename "$ckpt_path")
        step_num="${ckpt_name#checkpoint-}"
        if [[ "$step_num" =~ ^[0-9]+$ ]] && [ "$step_num" -gt "$MAX_STEP" ]; then
            MAX_STEP="$step_num"
            DEST_CKPT="$ckpt_path"
        fi
    done

    if [ -z "$DEST_CKPT" ]; then
        log_error "No checkpoint-XXXXX found in ${OUTPUT_DIR}. Run without --skip-download first."
        exit 1
    fi
    log_ok "Found existing checkpoint: $(basename "$DEST_CKPT")"
fi

# ── STEP 4: Resume Training ────────────────────────────────────────────────────
log_step "STEP 4 / 4 — Resume Training"

if [ ! -f "$TRAINING_CONFIG" ]; then
    log_error "Training config not found: ${TRAINING_CONFIG}"
    exit 1
fi

log_info "Config            : ${TRAINING_CONFIG}"
log_info "Resume checkpoint : ${DEST_CKPT}"
log_info "GPU(s)            : ${GPU}"
echo ""
log_info "Launching training…"
echo ""

CUDA_VISIBLE_DEVICES="$GPU" uv run python src/train.py \
    "$TRAINING_CONFIG" \
    --resume_from_checkpoint "$DEST_CKPT"

echo ""
log_ok "Training session finished (or exited cleanly)."
echo -e "${BOLD}${GREEN}╔══════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║   Pipeline completed. ✓          ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════╝${NC}"
