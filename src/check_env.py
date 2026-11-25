"""Quick environment check for the `rlt` conda env."""
import gensim
import numpy
import peft
import pytrec_eval  # noqa: F401
import torch
import transformers
from pyserini.search.lucene import LuceneImpactSearcher, LuceneSearcher  # noqa: F401

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__, "| peft", peft.__version__,
      "| gensim", gensim.__version__, "| numpy", numpy.__version__)
print("gensim corpus_file support:", gensim.models.doc2vec_corpusfile is not None
      if hasattr(gensim.models, "doc2vec_corpusfile") else "unknown")
print("pyserini OK")
