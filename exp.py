import csv
import gc
import itertools
import multiprocessing as mp
import pickle
import time
from pathlib import Path
import traceback

import numpy as np
import psutil
import pyagrum as gum
from tqdm import tqdm

from src.utils import jsd_bn, jsd_credal_stats, perturb_bn_params, snapshot_cpts

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

# Exact set (and order) of keys every `exp()` call's `row` carries -- fixed
# by WEIGHTING_SCHEMES/GRID_KEYS above, identical for every task regardless
# of hyperparameters. Used both as the CSV header (main()) and as a
# self-consistency check on every row before it is written.
ROW_FIELDNAMES = (
    list(GRID_KEYS)
    + ["size", "rep", "mle", "idm_min", "idm_mean", "idm_max"]
    + [f"mos_w{w}_{stat}" for w in WEIGHTING_SCHEMES for stat in ("min", "mean", "max")]
    + ["intersection_frac"]
)


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
    # Whether to archive CPT snapshots for this task (see `models` below).
    # Needed only for the future global optimization phase; the models
    # dominate a task's memory footprint (~90KB vs ~1KB for `row` alone,
    # measured), so this is off by default for a plain local-learning-and-
    # update sweep. Read from `config` (not a separate argument) to keep
    # `exp()`'s signature self-contained, matching every other hyperparameter.
    save_models = config.get("save_models", True)
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
    # Skipped entirely (not just left unused) when save_models is False, so
    # disabling it also saves the snapshot_cpts() compute, not just the RAM.
    models = {}

    # MLE and IDM (no update): identical across weighting schemas, computed once.
    row["mle"] = jsd_bn(bn_base, client_exp.bn_mle, target="joint")
    if save_models:
        models["bn_mle"] = snapshot_cpts(client_exp.bn_mle)

    idm_stats = jsd_credal_stats(
        bn_base, client_exp.cn.bn_min, client_exp.cn.bn_max, n_bns
    )
    row["idm_min"], row["idm_mean"], row["idm_max"] = (
        idm_stats["min"],
        idm_stats["mean"],
        idm_stats["max"],
    )
    if save_models:
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

        if save_models:
            models[f"prior_w{w}_min"] = snapshot_cpts(client_exp.prior_cn.bn_min)
            models[f"prior_w{w}_max"] = snapshot_cpts(client_exp.prior_cn.bn_max)

        client_exp.mosaic_cn()

        mos_stats = jsd_credal_stats(
            bn_base, client_exp.cn_mosaic.bn_min, client_exp.cn_mosaic.bn_max, n_bns
        )
        row[f"mos_w{w}_min"] = mos_stats["min"]
        row[f"mos_w{w}_mean"] = mos_stats["mean"]
        row[f"mos_w{w}_max"] = mos_stats["max"]
        if save_models:
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


# Filename for one task's archived models, encoding its own task_id (see
# `exp()`'s return value) so results are stored one-file-per-task instead of
# a single end-of-run pickle -- see main() for why.
def _model_filename(task_id: tuple) -> str:
    return "_".join(str(x) for x in task_id) + ".pkl"


def _check_unique_task_ids(tasks: list) -> None:
    """
    task_id = (n_clients, ess, prob_shift, alpha, size, rep) is what names
    every CSV row and every model file (see main()), so it must be unique
    across the whole task list. It is unique BY CONSTRUCTION as long as
    every grid list in conf.yaml holds distinct values (build_tasks's
    Cartesian product can't otherwise produce the same combination twice)
    -- this catches the one way that invariant can break (e.g. an
    accidental duplicate like `ess: [1, 1]`), which would otherwise
    silently make two different tasks overwrite the same CSV row / model
    file.
    """
    task_ids = [tuple(cfg[k] for k in GRID_KEYS) + (n, rep) for cfg, n, rep in tasks]
    if len(task_ids) != len(set(task_ids)):
        seen, dupes = set(), set()
        for t in task_ids:
            (dupes if t in seen else seen).add(t)
        raise AssertionError(
            "Duplicate task_id(s) in the task list -- check conf.yaml for "
            f"duplicate values within a single grid hyperparameter list: {dupes}"
        )


def _log_memory(proc: psutil.Process, n_done: int, n_total: int, n_failed: int, t_start: float) -> None:
    """
    Print current memory usage (parent + all live worker children) and
    throughput. See main()'s docstring-comment for how to read this.
    """
    parent_rss = proc.memory_info().rss

    children_rss = 0
    n_children = 0
    for c in proc.children(recursive=True):
        try:
            children_rss += c.memory_info().rss
            n_children += 1
        except psutil.NoSuchProcess:
            # Child exited between listing and sampling -- harmless, skip it.
            pass

    elapsed = time.time() - t_start
    rate = n_done / elapsed if elapsed > 0 else 0.0
    eta_min = (n_total - n_done) / rate / 60 if rate > 0 else float("nan")

    tqdm.write(
        f"[mem] {n_done}/{n_total} done ({n_failed} failed) | "
        f"parent RSS: {parent_rss / 1e9:.2f} GB | "
        f"{n_children} workers RSS (sum): {children_rss / 1e9:.2f} GB | "
        f"{rate:.2f} tasks/sec | ETA: {eta_min:.1f} min"
    )


def main():

    # Set seed
    set_seed()

    # Choose configurationc file
    config = load_config("conf.yaml")
    save_models = config.get("save_models", True)

    # Create the (single) empty results folder for the whole grid
    res_path = Path(config["res_path"])
    create_clean_dir(res_path)
    models_dir = res_path / "models"
    if save_models:
        models_dir.mkdir(parents=True, exist_ok=True)

    sizes_dict = config["s_sizes"]
    sizes = [
        int(x)
        for x in np.arange(sizes_dict["min"], sizes_dict["max"], sizes_dict["step"])
    ]

    tasks = build_tasks(config, sizes)
    _check_unique_task_ids(tasks)

    n_combos = 1
    for v in (config[k] for k in GRID_KEYS):
        n_combos *= len(v)
    print(
        f"# {n_combos} hyperparameter combinations x {len(sizes)} sizes x "
        f"{config['n_repetitions']} repetitions = {len(tasks)} total tasks "
        f"on {n_jobs} workers (save_models={save_models})",
        flush=True,
    )
    # Log memory roughly 200 times over the whole run, regardless of grid size.
    mem_log_every = max(1, len(tasks) // 200)

    # Workers only ever RETURN (row, models, task_id) tuples through the pool
    # -- they never touch the filesystem (see `exp`/`_exp_star`). Every
    # result is written to disk THE MOMENT it arrives here, in this single
    # parent process/thread, instead of being buffered in memory for the
    # whole run: `imap_unordered` yields one result at a time, so there is
    # no concurrent-write risk regardless of how many workers run, and no
    # possibility of two tasks racing on the same file. This replaces the
    # previous design, which held every row AND every task's models in RAM
    # (`rows`/`models_by_task` lists) until the very end -- measured at
    # ~90KB/task in-memory just for `models`, i.e. several GB for a large
    # grid, growing for the run's entire multi-hour duration and never
    # freed. Each row is self-identified by its own hyperparameter/size/rep
    # columns (checked against task_id below), so results always land at the
    # correct place in df_tot.csv independently of completion order.
    csv_path = res_path / "df_tot.csv"
    n_written = 0
    n_failed = 0
    proc = psutil.Process()
    t_start = time.time()

    with open(csv_path, "w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=ROW_FIELDNAMES)
        writer.writeheader()
        csv_f.flush()

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            for row, models, task_id in tqdm(
                pool.imap_unordered(_exp_star, tasks), total=len(tasks)
            ):
                if row is None:
                    n_failed += 1
                    continue

                # The row must describe exactly the task it was computed
                # for -- both in WHICH hyperparameters (task_id) and in
                # HAVING every expected column (no more, no less).
                row_task_id = tuple(row[k] for k in GRID_KEYS) + (row["size"], row["rep"])
                assert row_task_id == task_id, (row_task_id, task_id)
                assert set(row.keys()) == set(ROW_FIELDNAMES), (
                    set(row.keys()) ^ set(ROW_FIELDNAMES)
                )

                writer.writerow(row)
                csv_f.flush()  # visible to any reader / survives a killed process
                n_written += 1
                if n_written % mem_log_every == 0:
                    os.fsync(csv_f.fileno())  # survives an OS-level crash too

                if save_models and models:
                    model_path = models_dir / _model_filename(task_id)
                    # Append (not replace-suffix) -- filenames already
                    # contain dots from float hyperparameters (e.g.
                    # "..._0.5_...pkl"), and with_suffix() would need to
                    # correctly single out the trailing ".pkl" among those.
                    tmp_path = model_path.with_name(model_path.name + ".tmp")
                    with open(tmp_path, "wb") as mf:
                        pickle.dump(models, mf)
                    tmp_path.rename(model_path)  # atomic on POSIX: no reader
                    # ever observes a partially-written model file.

                if n_written % mem_log_every == 0:
                    _log_memory(proc, n_written, len(tasks), n_failed, t_start)

        os.fsync(csv_f.fileno())

    _log_memory(proc, n_written, len(tasks), n_failed, t_start)
    print(
        f"Done: {n_written} succeeded, {n_failed} failed, out of {len(tasks)} total tasks.",
        flush=True,
    )
    gc.collect()


if __name__ == "__main__":
    main()
