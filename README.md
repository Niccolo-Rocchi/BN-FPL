# How to use

Two independent experiment pipelines, both evaluating MOSAIC's local learning & update phase, sharing the same core simulation logic (`exp1.py`'s `exp()`).

## Pipeline 1: full hyperparameter grid

Compares MOSAIC against IDM and MLE across a grid of `n_clients`/`ess`/`prob_shift`/`alpha`/sample size. First, modify the config file `conf1.yaml` to set the grid. Then run code:

1. without Docker

```
python exp1.py
```
2. with Docker

```
docker build . -t bn-fpl
docker run [-d] [--rm] -v ./results:/workspace/results bn-fpl python exp1.py
```
Results can be found under the `res_path` set in `conf1.yaml` (default `./results`). These can be plotted by running the `plot1.ipynb` notebook.

## Pipeline 2: weighting-schema comparison across distribution shift

Fixes `n_clients`/`ess`/`alpha`/sample size to a single value each (set in `conf2.yaml`) and sweeps only `prob_shift`, to compare the three weighting schemas as heterogeneity grows. Run as above with `exp2.py` instead of `exp1.py`. Plot with `plot2.ipynb`.
