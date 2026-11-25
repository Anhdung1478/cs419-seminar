# Source me:  source scripts/env.sh   (sets paths, activates conda env, cd's into work/)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO
export UP="$REPO/third_party/RLT4Reranking"
export WORK="$REPO/work"
export D="datasets/msmarco-v1-passage"          # relative to $WORK; upstream scripts parse these paths
export PYSERINI_CACHE="$WORK/cache/pyserini"
export HF_HOME="$WORK/cache/hf"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$WORK"/{logs,cache} "$WORK/$D"/{queries,qrels,collection,runs,features,labels} "$WORK"/{output,checkpoint}
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then   # PC (WSL); on Kaggle the system python is used
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate rlt
fi
cd "$WORK"
