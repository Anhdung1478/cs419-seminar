# E4 - MonoT5 score as an extra RLT feature (RankLLaMA pipelines)

|                                                |   ('avg_k', 'base') |   ('avg_k', 'monot5') |   ('ndcg10', 'base') |   ('ndcg10', 'monot5') |   ('delta', 'ndcg10') |   ('delta', 'avg_k') |
|:-----------------------------------------------|--------------------:|----------------------:|---------------------:|-----------------------:|----------------------:|---------------------:|
| ('BM25', 'dl-19-passage', 'attncut beta0')     |              924.49 |                912.58 |                0.747 |                  0.745 |                -0.002 |                -11.9 |
| ('BM25', 'dl-19-passage', 'attncut beta1')     |              213.63 |                143.79 |                0.719 |                  0.726 |                 0.007 |                -69.8 |
| ('BM25', 'dl-19-passage', 'attncut beta2')     |               65.58 |                 61.56 |                0.695 |                  0.705 |                 0.01  |                 -4   |
| ('BM25', 'dl-19-passage', 'choppy beta0')      |              671.51 |                721.81 |                0.744 |                  0.743 |                -0.001 |                 50.3 |
| ('BM25', 'dl-19-passage', 'choppy beta1')      |              198.3  |                148.26 |                0.738 |                  0.723 |                -0.015 |                -50   |
| ('BM25', 'dl-19-passage', 'choppy beta2')      |              127.56 |                 70.28 |                0.726 |                  0.697 |                -0.029 |                -57.3 |
| ('BM25', 'dl-19-passage', 'mmoecut beta0')     |              843.44 |                903.44 |                0.743 |                  0.745 |                 0.002 |                 60   |
| ('BM25', 'dl-19-passage', 'mmoecut beta1')     |              155.67 |                 95.14 |                0.708 |                  0.723 |                 0.015 |                -60.5 |
| ('BM25', 'dl-19-passage', 'mmoecut beta2')     |               59.26 |                 46.67 |                0.687 |                  0.677 |                -0.01  |                -12.6 |
| ('BM25', 'dl-20-passage', 'attncut beta0')     |              821.65 |                813.57 |                0.76  |                  0.753 |                -0.007 |                 -8.1 |
| ('BM25', 'dl-20-passage', 'attncut beta1')     |              103.7  |                100.3  |                0.694 |                  0.677 |                -0.017 |                 -3.4 |
| ('BM25', 'dl-20-passage', 'attncut beta2')     |               65.7  |                 62.98 |                0.682 |                  0.66  |                -0.022 |                 -2.7 |
| ('BM25', 'dl-20-passage', 'choppy beta0')      |              669.2  |                392.2  |                0.749 |                  0.722 |                -0.027 |               -277   |
| ('BM25', 'dl-20-passage', 'choppy beta1')      |              153.2  |                116.04 |                0.691 |                  0.699 |                 0.008 |                -37.2 |
| ('BM25', 'dl-20-passage', 'choppy beta2')      |               58.85 |                 71.85 |                0.667 |                  0.688 |                 0.021 |                 13   |
| ('BM25', 'dl-20-passage', 'mmoecut beta0')     |              670.33 |                727.41 |                0.739 |                  0.755 |                 0.016 |                 57.1 |
| ('BM25', 'dl-20-passage', 'mmoecut beta1')     |              105    |                 97.52 |                0.674 |                  0.694 |                 0.02  |                 -7.5 |
| ('BM25', 'dl-20-passage', 'mmoecut beta2')     |               42.13 |                 52.07 |                0.652 |                  0.649 |                -0.003 |                  9.9 |
| ('RepLLaMA', 'dl-19-passage', 'attncut beta0') |              896.86 |                593.33 |                0.763 |                  0.766 |                 0.003 |               -303.5 |
| ('RepLLaMA', 'dl-19-passage', 'attncut beta1') |              524.65 |                232.47 |                0.767 |                  0.766 |                -0.001 |               -292.2 |
| ('RepLLaMA', 'dl-19-passage', 'attncut beta2') |              215.33 |                 94.23 |                0.766 |                  0.768 |                 0.002 |               -121.1 |
| ('RepLLaMA', 'dl-19-passage', 'choppy beta0')  |              124.35 |                362.44 |                0.766 |                  0.762 |                -0.004 |                238.1 |
| ('RepLLaMA', 'dl-19-passage', 'choppy beta1')  |               52.05 |                 82.63 |                0.766 |                  0.77  |                 0.004 |                 30.6 |
| ('RepLLaMA', 'dl-19-passage', 'choppy beta2')  |               56.28 |                 52.26 |                0.767 |                  0.768 |                 0.001 |                 -4   |
| ('RepLLaMA', 'dl-19-passage', 'mmoecut beta0') |              454.42 |                452.4  |                0.758 |                  0.767 |                 0.009 |                 -2   |
| ('RepLLaMA', 'dl-19-passage', 'mmoecut beta1') |              608.6  |                133.21 |                0.764 |                  0.77  |                 0.006 |               -475.4 |
| ('RepLLaMA', 'dl-19-passage', 'mmoecut beta2') |              531.95 |                 74.98 |                0.761 |                  0.767 |                 0.006 |               -457   |
| ('RepLLaMA', 'dl-20-passage', 'attncut beta0') |              669.26 |                248.8  |                0.77  |                  0.765 |                -0.005 |               -420.5 |
| ('RepLLaMA', 'dl-20-passage', 'attncut beta1') |              362.15 |                 88.67 |                0.769 |                  0.765 |                -0.004 |               -273.5 |
| ('RepLLaMA', 'dl-20-passage', 'attncut beta2') |              263.07 |                 66.54 |                0.768 |                  0.761 |                -0.007 |               -196.5 |
| ('RepLLaMA', 'dl-20-passage', 'choppy beta0')  |              144.81 |                 72.74 |                0.765 |                  0.765 |                 0     |                -72.1 |
| ('RepLLaMA', 'dl-20-passage', 'choppy beta1')  |               72.09 |                 61.57 |                0.765 |                  0.762 |                -0.003 |                -10.5 |
| ('RepLLaMA', 'dl-20-passage', 'choppy beta2')  |               97.02 |                 61.7  |                0.765 |                  0.762 |                -0.003 |                -35.3 |
| ('RepLLaMA', 'dl-20-passage', 'mmoecut beta0') |              123.02 |                243.59 |                0.725 |                  0.765 |                 0.04  |                120.6 |
| ('RepLLaMA', 'dl-20-passage', 'mmoecut beta1') |              243.37 |                 90.28 |                0.746 |                  0.767 |                 0.021 |               -153.1 |
| ('RepLLaMA', 'dl-20-passage', 'mmoecut beta2') |              212.35 |                 85.26 |                0.757 |                  0.765 |                 0.008 |               -127.1 |
| ('SPLADE++', 'dl-19-passage', 'attncut beta0') |              999    |                805.91 |                0.768 |                  0.768 |                 0     |               -193.1 |
| ('SPLADE++', 'dl-19-passage', 'attncut beta1') |               68.91 |                701.56 |                0.768 |                  0.766 |                -0.002 |                632.6 |
| ('SPLADE++', 'dl-19-passage', 'attncut beta2') |               24.12 |                231.7  |                0.762 |                  0.768 |                 0.006 |                207.6 |
| ('SPLADE++', 'dl-19-passage', 'choppy beta0')  |              124.53 |                289.35 |                0.771 |                  0.764 |                -0.007 |                164.8 |
| ('SPLADE++', 'dl-19-passage', 'choppy beta1')  |               46.91 |                 53.77 |                0.762 |                  0.77  |                 0.008 |                  6.9 |
| ('SPLADE++', 'dl-19-passage', 'choppy beta2')  |              255.14 |                 49.74 |                0.769 |                  0.775 |                 0.006 |               -205.4 |
| ('SPLADE++', 'dl-19-passage', 'mmoecut beta0') |              530.67 |                998    |                0.768 |                  0.768 |                 0     |                467.3 |
| ('SPLADE++', 'dl-19-passage', 'mmoecut beta1') |               34.23 |                998    |                0.776 |                  0.768 |                -0.008 |                963.8 |
| ('SPLADE++', 'dl-19-passage', 'mmoecut beta2') |               18.09 |                 77.51 |                0.754 |                  0.754 |                 0     |                 59.4 |
| ('SPLADE++', 'dl-20-passage', 'attncut beta0') |              999    |                999.94 |                0.768 |                  0.768 |                 0     |                  0.9 |
| ('SPLADE++', 'dl-20-passage', 'attncut beta1') |              998    |                730.5  |                0.768 |                  0.768 |                 0     |               -267.5 |
| ('SPLADE++', 'dl-20-passage', 'attncut beta2') |              101.43 |                341.8  |                0.772 |                  0.77  |                -0.002 |                240.4 |
| ('SPLADE++', 'dl-20-passage', 'choppy beta0')  |              108.76 |                999.52 |                0.767 |                  0.768 |                 0.001 |                890.8 |
| ('SPLADE++', 'dl-20-passage', 'choppy beta1')  |               28.44 |                 13.81 |                0.764 |                  0.757 |                -0.007 |                -14.6 |
| ('SPLADE++', 'dl-20-passage', 'choppy beta2')  |                1    |                 88    |                0.72  |                  0.77  |                 0.05  |                 87   |
| ('SPLADE++', 'dl-20-passage', 'mmoecut beta0') |             1000    |                  7.3  |                0.768 |                  0.736 |                -0.032 |               -992.7 |
| ('SPLADE++', 'dl-20-passage', 'mmoecut beta1') |              997    |                  8.5  |                0.768 |                  0.739 |                -0.029 |               -988.5 |
| ('SPLADE++', 'dl-20-passage', 'mmoecut beta2') |               26.83 |                 10    |                0.783 |                  0.741 |                -0.042 |                -16.8 |

## Mean change (monot5 - base) by method and retriever

|                         |   ndcg10 |    avg_k |
|:------------------------|---------:|---------:|
| ('BM25', 'attncut')     |   -0.005 |  -16.65  |
| ('BM25', 'choppy')      |   -0.007 |  -59.7   |
| ('BM25', 'mmoecut')     |    0.007 |    7.733 |
| ('RepLLaMA', 'attncut') |   -0.002 | -267.883 |
| ('RepLLaMA', 'choppy')  |   -0.001 |   24.467 |
| ('RepLLaMA', 'mmoecut') |    0.015 | -182.333 |
| ('SPLADE++', 'attncut') |    0     |  103.483 |
| ('SPLADE++', 'choppy')  |    0.008 |  154.917 |
| ('SPLADE++', 'mmoecut') |   -0.018 |  -84.583 |

