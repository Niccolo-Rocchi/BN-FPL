import csv
import gc
import itertools
import multiprocessing as mp
# Set number of threads for parallel computation
import os
import pickle
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import pyagrum as gum

from src.config import create_clean_dir, load_config, set_seed
from src.mosaic import Client
from src.utils import (gt_containment_frac, jsd_bn, jsd_credal_stats,
                       perturb_bn_params, snapshot_cpts)

n_jobs = max(1, len(os.sched_getaffinity(0)) - 1)

# All implemented weighting schemas (see PriorCPT.compute) are evaluated
# every run, so their JSD curves can be compared on the same plot.
WEIGHTING_SCHEMES = (1, 2, 3)

# Hyperparameters swept as a grid (cartesian product): each is a list in
# conf.yaml, even when it holds a single value. One full (s_sizes x
# n_repetitions x WEIGHTING_SCHEMES) sweep is run per combination.
GRID_KEYS = ("n_clients", "ess", "prob_shift", "alpha")

# Exact set of keys every `exp()` call's `row` carries, fixed by
# WEIGHTING_SCHEMES/GRID_KEYS above. Used as the CSV header and to check
# each row's schema before it is written.
ROW_FIELDNAMES = (
    list(GRID_KEYS)
    + ["size", "rep", "mle", "idm_min", "idm_mean", "idm_max"]
    + [f"mos_w{w}_{stat}" for w in WEIGHTING_SCHEMES for stat in ("min", "mean", "max")]
    + ["intersection_frac"]
    + ["idm_gt_contained"]
    + [f"mos_w{w}_gt_contained" for w in WEIGHTING_SCHEMES]
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
    Fully self-contained: everything is read from `config`, nothing from
    worker-global state, so tasks from different hyperparameter
    combinations can be freely interleaved across workers.
    """
    # Each (n, rep) task needs its own independent random stream: forked
    # worker processes inherit identical RNG state, so without reseeding
    # here different repetitions could silently produce identical data.
    task_seed = hash((n, rep)) % (2**32)
    np.random.seed(task_seed)
    gum.initRandom(task_seed)

    client_num = config["client_num"]
    n_bns = config["n_bns"]
    # Whether to archive CPT snapshots for this task (see `models` below).
    # Off by default, since the models dominate a task's memory footprint.
    save_models = config.get("save_models", True)
    bn_base = gum.loadBN(config["bn_base_path"])

    # Regenerates the whole data-generating process for this repetition,
    # not just new sampled data (see cap6_extract.tex's "Reiterations").
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

    # Archived models for this task (see snapshot_cpts): MLE, local IDM
    # credal set, and per-schema prior and MOSAIC-updated credal set.
    # Skipped entirely (not just unused) when save_models is False.
    models = {}

    # MLE and IDM (no update): identical across weighting schemas, computed once.
    row["mle"] = jsd_bn(bn_base, client_exp.bn_mle, target="joint")
    if save_models:
        models["bn_mle"] = snapshot_cpts(client_exp.bn_mle)

    idm_stats = jsd_credal_stats(
        bn_base, client_exp.cn.bn_min, client_exp.cn.bn_max, n_bns
    )
    row["idm_min"], row["idm_mean"], row["idm_max"] = (
        min(idm_stats["min"], row["mle"]),
        idm_stats["mean"],
        idm_stats["max"],
    )
    if save_models:
        models["idm_min"] = snapshot_cpts(client_exp.cn.bn_min)
        models["idm_max"] = snapshot_cpts(client_exp.cn.bn_max)

    # Empirical check of "Reliability of credal sets" (cap6_extract.tex,
    # `as:credal`): how often bn_base's own value is contained in this
    # credal set. Computed from the live bn_min/bn_max, not through `models`.
    row["idm_gt_contained"] = gt_containment_frac(
        bn_base, client_exp.cn.bn_min, client_exp.cn.bn_max
    )

    # MOSAIC, once per weighting schema. The intersection fraction is the
    # same across schemas (it doesn't depend on the weighting formula);
    # kept from weighting=2, where it's most directly interpretable.
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
        row[f"mos_w{w}_gt_contained"] = gt_containment_frac(
            bn_base, client_exp.cn_mosaic.bn_min, client_exp.cn_mosaic.bn_max
        )
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


def hyperparameter_combos(config) -> list:
    """
    Cartesian product over GRID_KEYS, except `alpha` is deduplicated to a
    single value whenever `prob_shift == 0.0`: at prob=0 no perturbation
    happens, so alpha has no effect and sweeping it would only waste compute.
    """
    combos = []
    for n_clients, ess, prob_shift in itertools.product(
        config["n_clients"], config["ess"], config["prob_shift"]
    ):
        alpha_values = [config["alpha"][0]] if prob_shift == 0.0 else config["alpha"]
        for alpha in alpha_values:
            combos.append(
                {"n_clients": n_clients, "ess": ess, "prob_shift": prob_shift, "alpha": alpha}
            )
    return combos


def build_tasks(config, sizes) -> list:
    """
    Flattens every (hyperparameter combination x size x repetition) into a
    single task list, so all n_jobs workers stay busy across the whole
    grid in one pool. Each task carries its own resolved combination.
    """
    return [
        (dict(config, **combo), n, rep)
        for combo in hyperparameter_combos(config)
        for n in sizes
        for rep in range(config["n_repetitions"])
    ]


# Filename for one task's archived models, derived from its own task_id.
# Results are stored one file per task, not a single end-of-run pickle.
def _model_filename(task_id: tuple) -> str:
    return "_".join(str(x) for x in task_id) + ".pkl"


def _check_unique_task_ids(tasks: list) -> None:
    """
    task_id = (n_clients, ess, prob_shift, alpha, size, rep) names every
    CSV row and model file, so it must be unique across `tasks`. Catches,
    e.g., an accidental duplicate value in a conf.yaml grid list.
    """
    task_ids = [tuple(cfg[k] for k in GRID_KEYS) + (n, rep) for cfg, n, rep in tasks]
    if len(task_ids) != len(set(task_ids)):
        seen, dupes = set(), set()
        for t in task_ids:
            (dupes if t in seen else seen).add(t)
        raise AssertionError(
            "Duplicate task_id(s) in the task list; check the config file "
            f"for duplicate values within a single grid hyperparameter list: {dupes}"
        )


def _read_pss_kb(pid: int) -> int:
    """
    PSS (Proportional Set Size) for one process: unlike RSS, it divides
    each shared page's cost among the processes sharing it, so summing PSS
    across workers doesn't overcount shared memory. Returns 0 if unavailable.
    """
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return 0


def _log_memory(
    proc: psutil.Process, n_done: int, n_total: int, n_failed: int, t_start: float
) -> None:
    """
    Print current memory usage (parent + all live worker children) and
    throughput. See main()'s docstring-comment for how to read this.
    """
    parent_rss = proc.memory_info().rss

    children_rss = 0
    children_pss = 0
    n_children = 0
    for c in proc.children(recursive=True):
        try:
            children_rss += c.memory_info().rss
            children_pss += _read_pss_kb(c.pid) * 1024
            n_children += 1
        except psutil.NoSuchProcess:
            # Child exited between listing and sampling; harmless, skip it.
            pass

    elapsed = time.time() - t_start
    rate = n_done / elapsed if elapsed > 0 else 0.0
    eta_min = (n_total - n_done) / rate / 60 if rate > 0 else float("nan")

    print(
        f"{n_done}/{n_total} done | "
        f"parent RSS: {parent_rss / 1e9:.2f} GB | "
        f"{n_children} workers RSS (sum): {children_rss / 1e9:.2f} GB | "
        # f"/ PSS sum {children_pss / 1e9:.2f} GB "
        f"{rate:.2f} tasks/s | ETA: {eta_min:.1f} min",
        flush=True,
    )


def run_grid(
    tasks: list,
    res_path: Path,
    save_models: bool,
    max_tasks_per_child: int,
    row_fieldnames: tuple = ROW_FIELDNAMES,
    task_fn=_exp_star,
) -> None:
    """
    Shared, memory-safe execution engine for a flattened task list. Writes
    each result to disk immediately and retires workers periodically
    (`maxtasksperchild`) to bound a native memory leak in `hopsy`.

    `row_fieldnames`/`task_fn` default to this module's own (local
    learning & update phase); exp_global.py (global optimization phase)
    reuses this engine unchanged by passing its own CSV schema and
    per-task worker instead.
    """
    _check_unique_task_ids(tasks)

    res_path = Path(res_path)
    create_clean_dir(res_path)
    models_dir = res_path / "models"
    if save_models:
        models_dir.mkdir(parents=True, exist_ok=True)

    # Log memory roughly 200 times over the whole run, regardless of grid size.
    mem_log_every = max(1, len(tasks) // 200)

    csv_path = res_path / "df_tot.csv"
    n_written = 0
    n_failed = 0
    proc = psutil.Process()
    t_start = time.time()

    with open(csv_path, "w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=row_fieldnames)
        writer.writeheader()
        csv_f.flush()

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_jobs, maxtasksperchild=max_tasks_per_child) as pool:
            for row, models, task_id in pool.imap_unordered(task_fn, tasks):
                if row is None:
                    n_failed += 1
                    continue

                # The row must describe exactly the task it was computed
                # for: matching task_id, with every expected column.
                row_task_id = tuple(row[k] for k in GRID_KEYS) + (
                    row["size"],
                    row["rep"],
                )
                assert row_task_id == task_id, (row_task_id, task_id)
                assert set(row.keys()) == set(row_fieldnames), set(row.keys()) ^ set(
                    row_fieldnames
                )

                writer.writerow(row)
                csv_f.flush()  # visible to any reader / survives a killed process
                n_written += 1
                if n_written % mem_log_every == 0:
                    os.fsync(csv_f.fileno())  # survives an OS-level crash too

                if save_models and models:
                    model_path = models_dir / _model_filename(task_id)
                    # Append, not replace-suffix: filenames already contain
                    # dots from float hyperparameters, which with_suffix()
                    # would misparse.
                    tmp_path = model_path.with_name(model_path.name + ".tmp")
                    with open(tmp_path, "wb") as mf:
                        pickle.dump(models, mf)
                    tmp_path.rename(model_path)  # atomic on POSIX

                if n_written % mem_log_every == 0:
                    _log_memory(proc, n_written, len(tasks), n_failed, t_start)

        os.fsync(csv_f.fileno())

    _log_memory(proc, n_written, len(tasks), n_failed, t_start)
    print(
        f"Done: {n_written} succeeded, {n_failed} failed, out of {len(tasks)} total tasks.",
        flush=True,
    )
    gc.collect()


def main():

    # Set seed
    set_seed()

    # Choose configuration file
    config = load_config("conf.yaml")
    save_models = config.get("save_models", True)
    max_tasks_per_child = config.get("max_tasks_per_child", 100)

    sizes_dict = config["s_sizes"]
    sizes = [
        int(x)
        for x in np.arange(sizes_dict["min"], sizes_dict["max"], sizes_dict["step"])
    ]

    tasks = build_tasks(config, sizes)

    # NOT the naive n_clients x ess x prob_shift x alpha product: alpha is
    # deduplicated away for prob_shift == 0.0 (see hyperparameter_combos).
    n_combos = len(hyperparameter_combos(config))
    print(
        f"# {n_combos} hyperparameter combinations x {len(sizes)} sizes x "
        f"{config['n_repetitions']} repetitions = {len(tasks)} total tasks "
        f"on {n_jobs} workers (save_models={save_models})",
        flush=True,
    )

    run_grid(tasks, config["res_path"], save_models, max_tasks_per_child)


if __name__ == "__main__":
    main()
