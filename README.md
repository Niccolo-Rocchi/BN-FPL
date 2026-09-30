# How to use

`MOSAIC` is a framework for federated parameter learning of Bayesian networks under distribution shift. It has two phases: clients first learn and update local credal networks using each other's credal sets as an informative prior (**local learning & update**), then a server clusters clients per mechanism to disentangle who shares a distribution with whom (**global optimization**).

Each phase has its own script, config file, and plotting notebook:

| Phase | Script | Config | Notebook | Results |
|---|---|---|---|---|
| Local learning & update | `exp_local.py` | `conf_local.yaml` | `plot_local.ipynb` | `results_local/` |
| Global optimization | `exp_global.py` | `conf_global.yaml` | `plot_global.ipynb` | `results_global/` |

The following pipeline describes how to run the code for either phase.

## Pipeline: full hyperparameter grid

Compares MOSAIC against baselines across a grid of `n_clients`/`ess`/`prob_shift`/`alpha` for varying sample size. First, modify the relevant config file to set the grid. Then run code:

1. without Docker

```
python <exp_name>.py
```
2. with Docker

```
docker build . -t bn-fpl
docker run [-d] [--rm] -v ./results:/workspace/results bn-fpl python <exp_name>.py
```
Results can be found under the `res_path` set in the config file (see table above). These can be plotted by running the matching notebook.
