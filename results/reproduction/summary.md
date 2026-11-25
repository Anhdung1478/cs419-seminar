# Reproduction vs paper: summary

Per method family over all 4 tables x 2 years (nDCG@10 and avg. k differences; exact = share of rows equal to the paper at 3 decimals).

| family   |   n |   mean_abs_d_ndcg |   max_abs_d_ndcg |   exact_ndcg |   mean_abs_d_k |
|:---------|----:|------------------:|-----------------:|-------------:|---------------:|
| attncut  |  24 |             0.002 |            0.005 |        0.333 |         54.738 |
| bicut    |  24 |             0.003 |            0.013 |        0.417 |         22.725 |
| choppy   |  24 |             0.007 |            0.044 |        0.25  |         60.217 |
| fixed    |  56 |             0     |            0     |        1     |          0     |
| greedy   |  24 |             0     |            0     |        1     |          0     |
| mmoecut  |  24 |             0.005 |            0.022 |        0.208 |         48.387 |
| oracle   |   8 |             0     |            0     |        1     |          0.012 |
| surprise |   8 |             0     |            0     |        1     |          0     |

## Largest nDCG@10 gaps

| table                    | year   | name                                                                                                                 |     k |   ndcg |   paper_k |   paper_ndcg |   d_ndcg |
|:-------------------------|:-------|:---------------------------------------------------------------------------------------------------------------------|------:|-------:|----------:|-------------:|---------:|
| table2_splade_rankllama  | DL20   | choppy-ckpt-dl-19-passage.original-splade-pp-ed-pytorch-1000.choppy.rankllama-1000-ndcg@10-eet-alpha-0.001-beta2-100 |   1   |  0.72  |      20.6 |        0.764 |   -0.044 |
| table2_splade_rankllama  | DL19   | choppy-ckpt-dl-20-passage.original-splade-pp-ed-pytorch-1000.choppy.rankllama-1000-ndcg@10-eet-alpha-0.001-beta2-100 |  42.1 |  0.762 |       1   |        0.731 |    0.031 |
| table1_bm25_rankllama    | DL20   | mmoecut-ckpt-dl-19-passage.original-bm25-1000.mmoecut.rankllama-1000-ndcg@10-eet-alpha-0.001-beta1-100               | 105   |  0.674 |     131   |        0.696 |   -0.022 |
| table4_bm25_monot5       | DL19   | choppy-ckpt-dl-20-passage.original-bm25-1000.choppy.monot5-1000-ndcg@10-eet-alpha-0.001-beta0-100                    | 521.5 |  0.707 |     783.6 |        0.727 |   -0.02  |
| table4_bm25_monot5       | DL19   | choppy-ckpt-dl-20-passage.original-bm25-1000.choppy.monot5-1000-ndcg@10-eet-alpha-0.001-beta2-100                    | 136.9 |  0.693 |     116.7 |        0.711 |   -0.018 |
| table3_repllama_rankllma | DL20   | mmoecut-ckpt-dl-19-passage.original-repllama-1000.mmoecut.rankllama-1000-ndcg@10-eet-alpha-0.001-beta1-100           | 243.4 |  0.746 |     291.9 |        0.762 |   -0.016 |
| table1_bm25_rankllama    | DL20   | mmoecut-ckpt-dl-19-passage.original-bm25-1000.mmoecut.rankllama-1000-ndcg@10-eet-alpha-0.001-beta2-100               |  41.1 |  0.652 |      52.5 |        0.665 |   -0.013 |
| table4_bm25_monot5       | DL20   | choppy-ckpt-dl-19-passage.original-bm25-1000.choppy.monot5-1000-ndcg@10-eet-alpha-0.001-beta2-100                    |  82.8 |  0.656 |     110.6 |        0.669 |   -0.013 |
