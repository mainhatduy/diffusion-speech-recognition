#!/bin/bash
# =============================================================================
#  run.sh — One-shot entry point (run this on a fresh machine)
#
#  Paste this single command into any terminal:
#
#    bash <(curl -fsSL https://raw.githubusercontent.com/aiai-laboratory/diffusion-speech-recognition/main/scripts/run.sh)
#
#  Or if you already have the repo:
#    bash scripts/run.sh
#
#  Optional env overrides (export before running):
#    GIT_REPO=<url>  GIT_BRANCH=<branch>  GPU=<0|0,1|...>
# =============================================================================

set -euo pipefail

# ── Tokens ────────────────────────────────────────────────────────────────────
export HF_TOKEN="${HF_TOKEN:-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}"
export WANDB_API_KEY="${WANDB_API_KEY:-1ce0793819a037f2b3729996816b5732ac107e84}"

# ── Config ────────────────────────────────────────────────────────────────────
GIT_REPO="${GIT_REPO:-https://github.com/aiai-laboratory/diffusion-speech-recognition}"
GIT_BRANCH="${GIT_BRANCH:-main}"
CLONE_DIR="${CLONE_DIR:-diffusion-speech-recognition}"
GPU="${GPU:-0}"

# ── Check if already inside the repo ─────────────────────────────────────────
if [ -f "pyproject.toml" ] && [ -f "scripts/bootstrap_and_train.sh" ]; then
    echo "[run.sh] Already inside the project. Running bootstrap directly..."
    chmod +x scripts/bootstrap_and_train.sh
    exec bash scripts/bootstrap_and_train.sh \
        --skip-clone \
        --gpu "$GPU"
fi

# ── Fresh machine: clone first, then bootstrap ────────────────────────────────
echo "[run.sh] Fresh machine detected. Cloning ${GIT_REPO} (branch: ${GIT_BRANCH})..."
git clone --branch "$GIT_BRANCH" "$GIT_REPO" "$CLONE_DIR"
cd "$CLONE_DIR"

chmod +x scripts/bootstrap_and_train.sh
exec bash scripts/bootstrap_and_train.sh \
    --skip-clone \
    --gpu "$GPU"
