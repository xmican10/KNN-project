# Evaluate scripts

## eval.py

Ours evaluation script for DAVIS2016 dataset.

## metrics.py

Implementation of Jaccard index and F-score for DAVIS2016 dataset.

## eval_res.py

Script generates image outputs of input model into desired directory. This directory can be submitted to the `eval.py` script.

## eval_compare.py

Script generates comparrison of expected and predicted mask.

## eval_core.py

Script implements functions for running `eval_res.py` and `eval_compare.py` scripts.
