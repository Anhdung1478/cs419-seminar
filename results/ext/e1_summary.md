# E1 - seed variance of supervised RLT (RankLLaMA, nDCG@10)

| test_ret   | test_ds       | config        |   seeds |   k_mean |   k_std |   ndcg_mean |   ndcg_std |   ndcg_min |   ndcg_max |
|:-----------|:--------------|:--------------|--------:|---------:|--------:|------------:|-----------:|-----------:|-----------:|
| BM25       | dl-19-passage | attncut beta0 |       5 |  936.698 |  36.819 |       0.746 |      0.001 |      0.745 |      0.747 |
| BM25       | dl-19-passage | attncut beta1 |       5 |  148.522 |  44.951 |       0.722 |      0.004 |      0.717 |      0.726 |
| BM25       | dl-19-passage | attncut beta2 |       5 |   64.788 |   4.05  |       0.699 |      0.005 |      0.695 |      0.708 |
| BM25       | dl-19-passage | choppy beta0  |       5 |  861.58  | 186.589 |       0.744 |      0.002 |      0.741 |      0.747 |
| BM25       | dl-19-passage | choppy beta1  |       5 |  216.076 |  13.977 |       0.739 |      0.002 |      0.737 |      0.741 |
| BM25       | dl-19-passage | choppy beta2  |       5 |  140.8   |  20.652 |       0.728 |      0.009 |      0.717 |      0.74  |
| BM25       | dl-19-passage | mmoecut beta0 |       5 |  869.366 |  68.027 |       0.745 |      0.002 |      0.742 |      0.747 |
| BM25       | dl-19-passage | mmoecut beta1 |       5 |  156.234 |  34.327 |       0.719 |      0.01  |      0.708 |      0.729 |
| BM25       | dl-19-passage | mmoecut beta2 |       5 |   60.788 |   9.901 |       0.693 |      0.01  |      0.682 |      0.708 |
| BM25       | dl-20-passage | attncut beta0 |       5 |  822.328 |  34.006 |       0.757 |      0.005 |      0.751 |      0.762 |
| BM25       | dl-20-passage | attncut beta1 |       5 |  112.862 |  19.094 |       0.694 |      0.006 |      0.686 |      0.701 |
| BM25       | dl-20-passage | attncut beta2 |       5 |   75.78  |  16.355 |       0.681 |      0.009 |      0.666 |      0.691 |
| BM25       | dl-20-passage | choppy beta0  |       5 |  749.12  |  88.165 |       0.748 |      0.005 |      0.742 |      0.753 |
| BM25       | dl-20-passage | choppy beta1  |       5 |  169.47  |  26.742 |       0.693 |      0.01  |      0.681 |      0.702 |
| BM25       | dl-20-passage | choppy beta2  |       5 |   80.766 |  23.02  |       0.677 |      0.011 |      0.667 |      0.696 |
| BM25       | dl-20-passage | mmoecut beta0 |       5 |  721.482 |  37.387 |       0.748 |      0.009 |      0.739 |      0.762 |
| BM25       | dl-20-passage | mmoecut beta1 |       5 |  165.216 |  66.819 |       0.692 |      0.013 |      0.674 |      0.709 |
| BM25       | dl-20-passage | mmoecut beta2 |       5 |   69.266 |  22.237 |       0.673 |      0.014 |      0.654 |      0.687 |
| SPLADE++   | dl-19-passage | attncut beta0 |       5 |  995.72  |   7.368 |       0.768 |      0     |      0.768 |      0.768 |
| SPLADE++   | dl-19-passage | attncut beta1 |       5 |  613.806 | 518.213 |       0.769 |      0.001 |      0.768 |      0.771 |
| SPLADE++   | dl-19-passage | attncut beta2 |       5 |  422.87  | 513.981 |       0.767 |      0.003 |      0.762 |      0.771 |
| SPLADE++   | dl-19-passage | choppy beta0  |       5 |  531.99  | 440.896 |       0.767 |      0.002 |      0.763 |      0.768 |
| SPLADE++   | dl-19-passage | choppy beta1  |       5 |  232.148 | 431.095 |       0.752 |      0.019 |      0.731 |      0.768 |
| SPLADE++   | dl-19-passage | choppy beta2  |       5 |   26.028 |  35.308 |       0.746 |      0.02  |      0.731 |      0.773 |
| SPLADE++   | dl-19-passage | mmoecut beta0 |       5 |  904.568 | 210.039 |       0.768 |      0     |      0.768 |      0.768 |
| SPLADE++   | dl-19-passage | mmoecut beta1 |       5 |  807.348 | 426.87  |       0.769 |      0.003 |      0.768 |      0.775 |
| SPLADE++   | dl-19-passage | mmoecut beta2 |       5 |  800.94  | 437.848 |       0.764 |      0.009 |      0.748 |      0.768 |
| SPLADE++   | dl-20-passage | attncut beta0 |       5 |  999.2   |   0.837 |       0.768 |      0     |      0.768 |      0.768 |
| SPLADE++   | dl-20-passage | attncut beta1 |       5 |  799.8   | 434.853 |       0.77  |      0.004 |      0.768 |      0.776 |
| SPLADE++   | dl-20-passage | attncut beta2 |       5 |  448.93  | 496.354 |       0.772 |      0.004 |      0.768 |      0.778 |
| SPLADE++   | dl-20-passage | choppy beta0  |       5 |  717.816 | 388.734 |       0.765 |      0.007 |      0.753 |      0.77  |
| SPLADE++   | dl-20-passage | choppy beta1  |       5 |   24.074 |  29.564 |       0.746 |      0.024 |      0.72  |      0.77  |
| SPLADE++   | dl-20-passage | choppy beta2  |       5 |   29.736 |  39.755 |       0.743 |      0.022 |      0.72  |      0.77  |
| SPLADE++   | dl-20-passage | mmoecut beta0 |       5 |  999     |   0.707 |       0.768 |      0     |      0.768 |      0.768 |
| SPLADE++   | dl-20-passage | mmoecut beta1 |       5 |  998     |   0.707 |       0.768 |      0     |      0.768 |      0.768 |
| SPLADE++   | dl-20-passage | mmoecut beta2 |       5 |  997.2   |   0.837 |       0.768 |      0     |      0.768 |      0.768 |

## Spread summary
- mean nDCG@10 std across seeds: 0.007
- max nDCG@10 range (max-min) across seeds: 0.050
- mean avg-k std across seeds: 143.5

- BM25 dl-19-passage beta0: best method per seed -> {'attncut': 4, 'choppy': 1}
- BM25 dl-19-passage beta1: best method per seed -> {'choppy': 5}
- BM25 dl-19-passage beta2: best method per seed -> {'choppy': 5}
- BM25 dl-20-passage beta0: best method per seed -> {'attncut': 3, 'mmoecut': 1, 'choppy': 1}
- BM25 dl-20-passage beta1: best method per seed -> {'mmoecut': 2, 'choppy': 2, 'attncut': 1}
- BM25 dl-20-passage beta2: best method per seed -> {'attncut': 3, 'choppy': 2}
- SPLADE++ dl-19-passage beta0: best method per seed -> {'attncut': 5}
- SPLADE++ dl-19-passage beta1: best method per seed -> {'attncut': 4, 'mmoecut': 1}
- SPLADE++ dl-19-passage beta2: best method per seed -> {'attncut': 3, 'mmoecut': 1, 'choppy': 1}
- SPLADE++ dl-20-passage beta0: best method per seed -> {'attncut': 4, 'choppy': 1}
- SPLADE++ dl-20-passage beta1: best method per seed -> {'attncut': 5}
- SPLADE++ dl-20-passage beta2: best method per seed -> {'attncut': 5}
