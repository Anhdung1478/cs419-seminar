#!/usr/bin/env bash
# One-time environment setup inside WSL2 Ubuntu.
# Usage (from repo root, inside WSL):  bash scripts/00_setup.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_DIR="$HOME/miniforge3"
ENV=rlt

# 1. Miniforge (conda-forge only; no Anaconda ToS channel)
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  curl -fsSL -o /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
  rm /tmp/miniforge.sh
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

# 2. Env: Python 3.10 + JDK 11 (pyserini 0.22 / Anserini needs Java 11)
if ! conda env list | grep -q "^$ENV "; then
  conda create -y -n "$ENV" python=3.10 openjdk=11 c-compiler cxx-compiler
fi
conda activate "$ENV"
command -v gcc >/dev/null || conda install -y c-compiler cxx-compiler   # for any sdist-only wheels

# 3. Python packages
pip install --upgrade pip
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r "$REPO/requirements.txt"

# 4. Upstream patches (idempotent)
bash "$REPO/scripts/apply_patches.sh"

# 5. Sanity checks
python "$REPO/src/check_env.py"
