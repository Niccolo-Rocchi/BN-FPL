import gc
import itertools
import multiprocessing as mp
import pickle
import sys
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import pyagrum as gum
from tqdm import tqdm

from src.utils import jsd_bn, jsd_credal_stats, perturb_bn_params, snapshot_cpts

sys.path.insert(0, str(Path().resolve().parents[1]))
from src.config import create_clean_dir, load_config, set_seed
from src.mosaic import Client

# Set number of threads for parallel computation
import os
n_jobs = max(1, len(os.sched_getaffinity(0)) - 1)

# All implemented weighting schemas (see PriorCPT.compute) are evaluated in
# every run, so their JSD curves can be compared on the same plot -- the MLE
# and IDM (no update) curves are identical across schemas and computed once.
WEIGHTING_SCHEMES = (1, 2, 3)

# Hyperparameters swept as a grid (cartesian product): each is a list in
# conf.yaml, even when it holds a single value. One full (s_sizes x
# n_repetitions x WEIGHTING_SCHEMES) sweep is run per combination.
GRID_KEYS = ("n_clients", "ess", "prob_shift", "alpha")


def init_clients(config, verbose=False) -> list:
    E = config["n_clients"]
    alpha = config["alpha"]
    bn_base = gum.loadBN(config["bn_base_path"])

    clients = {}
    for e in range(E):

        # Copy `bn_base` for the first client
        p = config["prob_shift"] if e != 0 else 0

        # Init the client
        gt, mask = perturb_bn_params(
            bn_base, alpha=alpha, prob=p
        )  # If p=0 then just copy `bn_base`
        client = Client(gt, mask)

        # Collect
        clients[e] = client

        if verbose and e != 0:
            d = jsd_bn(clients[e].gt, clients[0].gt, "joint")
            print("Dist. from client 0: ", d)

    return clients


def exp(config, n, rep) -> tuple:
    """
    Fully self-contained: everything needed (including every hyperparameter
    combination's own config) is passed in as an argument, nothing is read
    from worker-global state. This is what lets main() freely interleave
    tasks from DIFFERENT hyperparameter combinations across all workers in
    a single flat pool, instead of processing one combination at a time.
    """
    # Each (n, rep) task needs its own independent random stream. The worker
    # pool uses "fork", so sibling worker processes inherit an IDENTICAL RNG
    # state at fork time; without reseeding here, several "repetitions"
    # processed by distinct, freshly-forked workers would silently generate
    # IDENTICAL data (confirmed empirically: client 0's data was byte-for-
    # byte identical across repetitions for several sample sizes). The seed
    # is still fully reproducible given (n, rep) -- deliberately NOT mixed
    # with the hyperparameter combination, so different combinations use
    # "matched" randomness for the same (n, rep) (common random numbers),
    # making cross-combination comparisons a bit less noisy; this has no
    # bearing on correctness, since each task still reseeds independently
    # before generating its own data.
    task_seed = hash((n, rep)) % (2**32)
    np.random.seed(task_seed)
    gum.initRandom(task_seed)

    client_num = config["client_num"]
    n_bns = config["n_bns"]
    bn_base = gum.loadBN(config["bn_base_path"])

    # Generate a fresh data-generating process for this repetition (i.e. new
    # client perturbations too, not just new sampled data from a DGP fixed
    # once for the whole sweep) -- cap6_extract.tex's "Reiterations" repeats
    # the whole "above steps", which includes "Data Generation".
    clients = init_clients(config)

    # Generate clients' data and learn models
    for e in clients:
        c = clients[e]
        c.generate_base_info(n, config["ess"])

    # Choose client
    client_exp = clients[client_num]

    # Set prior(s) clients: client_exp must be first (see PriorCPT.set_clients),
    # followed by all other clients, used as prior candidates.
    prior_clients = [client_exp] + [c for e, c in clients.items() if e != client_num]

    task_id = tuple(config[k] for k in GRID_KEYS) + (n, rep)
    row = {k: config[k] for k in GRID_KEYS}
    row["size"] = n
    row["rep"] = rep

    # Archived models for this task (see snapshot_cpts): the client's exact
    # MLE, its local IDM credal set (no update), and -- per weighting schema
    # -- the prior and the resulting MOSAIC-updated credal set. Needed later
    # for the global optimization phase (cap6_extract.tex), which reads
    # theta^i in K^{i+}_{X|pi_X} directly from the updated credal sets.
    models = {}

    # MLE and IDM (no update): identical across weighting schemas, computed once.
    row["mle"] = jsd_bn(bn_base, client_exp.bn_mle, target="joint")
    models["bn_mle"] = snapshot_cpts(client_exp.bn_mle)

    idm_stats = jsd_credal_stats(
        bn_base, client_exp.cn.bn_min, client_exp.cn.bn_max, n_bns
    )
    row["idm_min"], row["idm_mean"], row["idm_max"] = (
        idm_stats["min"],
        idm_stats["mean"],
        idm_stats["max"],
    )
    models["idm_min"] = snapshot_cpts(client_exp.cn.bn_min)
    models["idm_max"] = snapshot_cpts(client_exp.cn.bn_max)

    # MOSAIC, once per weighting schema. The network-wide median fraction of
    # clients intersecting the prior only depends on the target/candidates'
    # own credal sets (not on the weighting formula), so it's identical
    # across schemas -- kept from weighting=2, where it's most directly
    # interpretable (hard intersection cutoff).
    intersection_frac = None
    for w in WEIGHTING_SCHEMES:
        client_exp.reset_prior()
        assert client_exp.prior_cn.is_vacuous_all()
        median_intersection = client_exp.prior_cn.compute(prior_clients, weighting=w)
        assert not client_exp.prior_cn.is_vacuous_any()
        if w == 2:
            intersection_frac = median_intersection

        models[f"prior_w{w}_min"] = snapshot_cpts(client_exp.prior_cn.bn_min)
        models[f"prior_w{w}_max"] = snapshot_cpts(client_exp.prior_cn.bn_max)

        client_exp.mosaic_cn()

        mos_stats = jsd_credal_stats(
            bn_base, client_exp.cn_mosaic.bn_min, client_exp.cn_mosaic.bn_max, n_bns
        )
        row[f"mos_w{w}_min"] = mos_stats["min"]
        row[f"mos_w{w}_mean"] = mos_stats["mean"]
        row[f"mos_w{w}_max"] = mos_stats["max"]
        models[f"mos_w{w}_min"] = snapshot_cpts(client_exp.cn_mosaic.bn_min)
        models[f"mos_w{w}_max"] = snapshot_cpts(client_exp.cn_mosaic.bn_max)

    row["intersection_frac"] = intersection_frac

    return row, models, task_id


def _exp_star(args):
    try:
        return exp(*args)
    except Exception:
        tb = traceback.format_exc()
        print("ERROR", args, tb, flush=True)
        return None, None, None


def build_tasks(config, sizes) -> list:
    """
    Flatten every (hyperparameter combination x size x repetition) into a
    single task list, `[(combo_config, n, rep), ...]`, so all n_jobs workers
    stay busy for the WHOLE grid in one pool (see main()). Submitting only
    n_repetitions tasks at a time (one size, one combination at a time) --
    the previous design -- left most cores idle whenever n_repetitions <
    n_jobs (e.g. 2 repetitions on a 15-worker pool: 13 idle, every batch).
    Each task carries its own fully-resolved hyperparameter combination
    (`combo_config`), so tasks from different combinations can be freely
    interleaved across workers, in any order, without cross-contamination:
    exp() reads everything from its own `config` argument, never from
    shared/global state.
    """
    grid_values = [config[k] for k in GRID_KEYS]
    return [
        (dict(config, **dict(zip(GRID_KEYS, combo))), n, rep)
        for combo in itertools.product(*grid_values)
        for n in sizes
        for rep in range(config["n_repetitions"])
    ]


def main():

    # Set seed
    set_seed()

    # Choose configurationc file
    config = load_config("conf.yaml")

    # Create the (single) empty results folder for the whole grid
    res_path = Path(config["res_path"])
    create_clean_dir(res_path)

    sizes_dict = config["s_sizes"]
    sizes = [
        int(x)
        for x in np.arange(sizes_dict["min"], sizes_dict["max"], sizes_dict["step"])
    ]

    tasks = build_tasks(config, sizes)
    n_combos = 1
    for v in (config[k] for k in GRID_KEYS):
        n_combos *= len(v)
    print(
        f"# {n_combos} hyperparameter combinations x {len(sizes)} sizes x "
        f"{config['n_repetitions']} repetitions = {len(tasks)} total tasks "
        f"on {n_jobs} workers",
        flush=True,
    )

    # Workers only ever RETURN (row, models, task_id) tuples through the
    # pool -- they never touch the filesystem. df_tot.csv/models.pkl are
    # written exactly once, here, after every task has completed, so there
    # is no concurrent-write risk regardless of how many workers run.
    rows = []
    models_by_task = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_jobs) as pool:
        for row, models, task_id in tqdm(
            pool.imap_unordered(_exp_star, tasks), total=len(tasks)
        ):
            if row is None:
                continue
            rows.append(row)
            models_by_task[task_id] = models

    df_tot = pd.DataFrame(rows)
    df_tot.to_csv(res_path / "df_tot.csv", index=False)

    # task_id keys are (n_clients, ess, prob_shift, alpha, size, rep).
    with open(res_path / "models.pkl", "wb") as f:
        pickle.dump(models_by_task, f)

    gc.collect()


if __name__ == "__main__":
    main()
