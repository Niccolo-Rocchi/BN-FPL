# How to use

## Pipeline for the local learning & update phase: full hyperparameter grid

Compares MOSAIC against IDM and MLE across a grid of `n_clients`/`ess`/`prob_shift`/`alpha`/sample size. First, modify the config file `conf1.yaml` to set the grid. Then run code:

1. without Docker

```
python exp.py
```
2. with Docker

```
docker build . -t bn-fpl
docker run [-d] [--rm] -v ./results:/workspace/results bn-fpl python exp.py
```
Results can be found under the `res_path` set in `conf.yaml` (default `./results`). These can be plotted by running the `plot.ipynb` notebook.
