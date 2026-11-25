#!/usr/bin/env bash
# Latest Kaggle CLI in its own env (kag, Python 3.12): the rlt env (3.10) is stuck on 1.7.x,
# which cannot choose the GPU type (T4 x2) when pushing kernels.
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda env list | grep -q '^kag ' || conda create -q -y -n kag python=3.12 >/dev/null
conda activate kag
pip install -q --root-user-action=ignore -U kaggle
kaggle --version
kaggle kernels push --help | grep -iA4 accel || echo "no --accelerator flag"
