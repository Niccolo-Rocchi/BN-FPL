# How to use

First, modify the config file `conf.yaml` to set the grid of hyperparameters. Then run code:

1. without Docker

```
python exp.py
```
2. with Docker

```
docker build . -t bn-fpl
docker run [-d] [--rm] -v ./results:/workspace/results bn-fpl python exp.py
```
Results can be found under `./results`. These can be plotted by running the `Plot_JS.ipynb` notebook.