# E3 - cross-retriever transfer, scores=raw (nDCG@10 on test retriever)

|                                                |   BM25 |   RepLLaMA |   SPLADE++ |
|:-----------------------------------------------|-------:|-----------:|-----------:|
| ('BM25', 'dl-19-passage', 'attncut beta0')     |  0.747 |      0.768 |      0.768 |
| ('BM25', 'dl-19-passage', 'attncut beta1')     |  0.719 |      0.768 |      0.768 |
| ('BM25', 'dl-19-passage', 'attncut beta2')     |  0.695 |      0.768 |      0.768 |
| ('BM25', 'dl-19-passage', 'choppy beta0')      |  0.741 |      0.766 |      0.768 |
| ('BM25', 'dl-19-passage', 'choppy beta1')      |  0.738 |      0.77  |      0.731 |
| ('BM25', 'dl-19-passage', 'choppy beta2')      |  0.713 |      0.768 |      0.731 |
| ('BM25', 'dl-19-passage', 'mmoecut beta0')     |  0.743 |      0.769 |      0.731 |
| ('BM25', 'dl-19-passage', 'mmoecut beta1')     |  0.708 |      0.769 |      0.731 |
| ('BM25', 'dl-19-passage', 'mmoecut beta2')     |  0.687 |      0.766 |      0.768 |
| ('BM25', 'dl-20-passage', 'attncut beta0')     |  0.76  |      0.769 |      0.768 |
| ('BM25', 'dl-20-passage', 'attncut beta1')     |  0.694 |      0.77  |      0.768 |
| ('BM25', 'dl-20-passage', 'attncut beta2')     |  0.682 |      0.769 |      0.768 |
| ('BM25', 'dl-20-passage', 'choppy beta0')      |  0.753 |      0.768 |      0.72  |
| ('BM25', 'dl-20-passage', 'choppy beta1')      |  0.701 |      0.766 |      0.72  |
| ('BM25', 'dl-20-passage', 'choppy beta2')      |  0.667 |      0.767 |      0.72  |
| ('BM25', 'dl-20-passage', 'mmoecut beta0')     |  0.739 |      0.769 |      0.733 |
| ('BM25', 'dl-20-passage', 'mmoecut beta1')     |  0.674 |      0.77  |      0.728 |
| ('BM25', 'dl-20-passage', 'mmoecut beta2')     |  0.648 |      0.768 |      0.732 |
| ('SPLADE++', 'dl-19-passage', 'attncut beta0') |  0.657 |      0.755 |      0.768 |
| ('SPLADE++', 'dl-19-passage', 'attncut beta1') |  0.65  |      0.755 |      0.768 |
| ('SPLADE++', 'dl-19-passage', 'attncut beta2') |  0.678 |      0.76  |      0.762 |
| ('SPLADE++', 'dl-19-passage', 'choppy beta0')  |  0.744 |      0.762 |      0.768 |
| ('SPLADE++', 'dl-19-passage', 'choppy beta1')  |  0.747 |      0.765 |      0.767 |
| ('SPLADE++', 'dl-19-passage', 'choppy beta2')  |  0.729 |      0.764 |      0.77  |
| ('SPLADE++', 'dl-19-passage', 'mmoecut beta0') |  0.747 |      0.763 |      0.768 |
| ('SPLADE++', 'dl-19-passage', 'mmoecut beta1') |  0.747 |      0.763 |      0.775 |
| ('SPLADE++', 'dl-19-passage', 'mmoecut beta2') |  0.747 |      0.763 |      0.747 |
| ('SPLADE++', 'dl-20-passage', 'attncut beta0') |  0.642 |      0.756 |      0.768 |
| ('SPLADE++', 'dl-20-passage', 'attncut beta1') |  0.643 |      0.756 |      0.768 |
| ('SPLADE++', 'dl-20-passage', 'attncut beta2') |  0.632 |      0.757 |      0.77  |
| ('SPLADE++', 'dl-20-passage', 'choppy beta0')  |  0.745 |      0.768 |      0.768 |
| ('SPLADE++', 'dl-20-passage', 'choppy beta1')  |  0.736 |      0.763 |      0.769 |
| ('SPLADE++', 'dl-20-passage', 'choppy beta2')  |  0.673 |      0.769 |      0.773 |
| ('SPLADE++', 'dl-20-passage', 'mmoecut beta0') |  0.762 |      0.768 |      0.768 |
| ('SPLADE++', 'dl-20-passage', 'mmoecut beta1') |  0.762 |      0.768 |      0.768 |
| ('SPLADE++', 'dl-20-passage', 'mmoecut beta2') |  0.762 |      0.768 |      0.784 |

# E3 - cross-retriever transfer, scores=zscore (nDCG@10 on test retriever)

|                                                |   BM25 |   RepLLaMA |   SPLADE++ |
|:-----------------------------------------------|-------:|-----------:|-----------:|
| ('BM25', 'dl-19-passage', 'attncut beta0')     |  0.747 |      0.763 |      0.768 |
| ('BM25', 'dl-19-passage', 'attncut beta1')     |  0.741 |      0.766 |      0.771 |
| ('BM25', 'dl-19-passage', 'attncut beta2')     |  0.728 |      0.768 |      0.766 |
| ('BM25', 'dl-19-passage', 'choppy beta0')      |  0.745 |      0.765 |      0.768 |
| ('BM25', 'dl-19-passage', 'choppy beta1')      |  0.739 |      0.769 |      0.768 |
| ('BM25', 'dl-19-passage', 'choppy beta2')      |  0.731 |      0.767 |      0.769 |
| ('BM25', 'dl-19-passage', 'mmoecut beta0')     |  0.744 |      0.763 |      0.766 |
| ('BM25', 'dl-19-passage', 'mmoecut beta1')     |  0.736 |      0.769 |      0.769 |
| ('BM25', 'dl-19-passage', 'mmoecut beta2')     |  0.712 |      0.768 |      0.766 |
| ('BM25', 'dl-20-passage', 'attncut beta0')     |  0.761 |      0.768 |      0.768 |
| ('BM25', 'dl-20-passage', 'attncut beta1')     |  0.7   |      0.765 |      0.768 |
| ('BM25', 'dl-20-passage', 'attncut beta2')     |  0.676 |      0.764 |      0.768 |
| ('BM25', 'dl-20-passage', 'choppy beta0')      |  0.754 |      0.768 |      0.767 |
| ('BM25', 'dl-20-passage', 'choppy beta1')      |  0.677 |      0.769 |      0.772 |
| ('BM25', 'dl-20-passage', 'choppy beta2')      |  0.675 |      0.768 |      0.775 |
| ('BM25', 'dl-20-passage', 'mmoecut beta0')     |  0.756 |      0.767 |      0.768 |
| ('BM25', 'dl-20-passage', 'mmoecut beta1')     |  0.652 |      0.756 |      0.77  |
| ('BM25', 'dl-20-passage', 'mmoecut beta2')     |  0.638 |      0.757 |      0.769 |
| ('SPLADE++', 'dl-19-passage', 'attncut beta0') |  0.725 |      0.763 |      0.768 |
| ('SPLADE++', 'dl-19-passage', 'attncut beta1') |  0.715 |      0.768 |      0.769 |
| ('SPLADE++', 'dl-19-passage', 'attncut beta2') |  0.71  |      0.768 |      0.77  |
| ('SPLADE++', 'dl-19-passage', 'choppy beta0')  |  0.697 |      0.758 |      0.76  |
| ('SPLADE++', 'dl-19-passage', 'choppy beta1')  |  0.704 |      0.765 |      0.771 |
| ('SPLADE++', 'dl-19-passage', 'choppy beta2')  |  0.676 |      0.765 |      0.768 |
| ('SPLADE++', 'dl-19-passage', 'mmoecut beta0') |  0.68  |      0.752 |      0.747 |
| ('SPLADE++', 'dl-19-passage', 'mmoecut beta1') |  0.67  |      0.754 |      0.755 |
| ('SPLADE++', 'dl-19-passage', 'mmoecut beta2') |  0.665 |      0.76  |      0.764 |
| ('SPLADE++', 'dl-20-passage', 'attncut beta0') |  0.744 |      0.771 |      0.768 |
| ('SPLADE++', 'dl-20-passage', 'attncut beta1') |  0.688 |      0.762 |      0.773 |
| ('SPLADE++', 'dl-20-passage', 'attncut beta2') |  0.677 |      0.763 |      0.774 |
| ('SPLADE++', 'dl-20-passage', 'choppy beta0')  |  0.629 |      0.756 |      0.777 |
| ('SPLADE++', 'dl-20-passage', 'choppy beta1')  |  0.626 |      0.76  |      0.776 |
| ('SPLADE++', 'dl-20-passage', 'choppy beta2')  |  0.627 |      0.761 |      0.779 |
| ('SPLADE++', 'dl-20-passage', 'mmoecut beta0') |  0.64  |      0.745 |      0.76  |
| ('SPLADE++', 'dl-20-passage', 'mmoecut beta1') |  0.609 |      0.752 |      0.758 |
| ('SPLADE++', 'dl-20-passage', 'mmoecut beta2') |  0.596 |      0.754 |      0.77  |

## Transfer gap = nDCG@10(trained on other retriever) - nDCG@10(trained on test retriever)

|                                |   mean |    min |   count |
|:-------------------------------|-------:|-------:|--------:|
| ('raw', 'BM25', 'SPLADE++')    | -0.023 | -0.053 |      18 |
| ('raw', 'SPLADE++', 'BM25')    | -0     | -0.118 |      18 |
| ('zscore', 'BM25', 'SPLADE++') |  0.002 | -0.01  |      18 |
| ('zscore', 'SPLADE++', 'BM25') | -0.046 | -0.125 |      18 |

## Mean avg. k by train/test retriever

|                        |   BM25 |   RepLLaMA |   SPLADE++ |
|:-----------------------|-------:|-----------:|-----------:|
| ('raw', 'BM25')        |  334.3 |      456.3 |      445.6 |
| ('raw', 'SPLADE++')    |  561.9 |      592.9 |      442.4 |
| ('zscore', 'BM25')     |  374.6 |      366.3 |      373.5 |
| ('zscore', 'SPLADE++') |  138.4 |      125.5 |      123   |

