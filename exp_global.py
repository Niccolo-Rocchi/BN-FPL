import os
import traceback

import numpy as np
import pyagrum as gum

import exp_local
from src.config import load_config, set_seed
from src.global_opt import (all_mechanisms, ari_ami, evaluate_mechanism,
                            evaluate_mechanism_mle, pairwise_confusion,
                            precision_recall_f1)
from src.mosaic import Client
from src.utils import perturb_bn_params, snapshot_cpts

n_jobs = exp_local.n_jobs
GRID_KEYS = exp_local.GRID_KEYS

# W-1 excluded: it behaves too similarly to W-2 to be worth a third curve.
WEIGHTING_SCHEMES = (2, 3)

# One method per client source: MOSAIC-updated (per weighting schema), the
# no-update ablation, and the naive MLE baseline (oracle k).
METHODS = tuple(f"mos_w{w}" for w in WEIGHTING_SCHEMES) + ("noupdate", "mle")

METRIC_STATS = ("precision", "recall", "f1", "ari", "ami", "n_clusters_mean")

ROW_FIELDNAMES = (
    list(GRID_KEYS)
    + ["size", "rep"]
    + [f"{m}_{stat}" for m in METHODS for stat in METRIC_STATS]
    + ["n_clusters_true_mean"]
)


def init_clients(config) -> dict:
    """
    Every client, including client 0, is subject to `prob_shift`: the
    ground truth is pairwise, so no client needs to stay a fixed anchor.
    """
    E = config["n_clients"]
    alpha = config["alpha"]
    p = config["prob_shift"]
    bn_base = gum.loadBN(config["bn_base_path"])

    clients = {}
    for e in range(E):
        gt, mask = perturb_bn_params(bn_base, alpha=alpha, prob=p)
        clients[e] = Client(gt, mask)

    return clients


def _network_metrics(mechanism_results: list) -> dict:
    """
    Precision/recall/F1 are pooled (micro-averaged) over every client pair
    of every mechanism; ARI/AMI are computed per mechanism, then macro-averaged.
    """
    tp = fp = fn = 0
    aris, amis, n_clusters = [], [], []
    for r in mechanism_results:
        t, p = r["true_labels"], r["pred_labels"]
        tp_i, fp_i, fn_i, _ = pairwise_confusion(t, p)
        tp += tp_i
        fp += fp_i
        fn += fn_i

        ari, ami = ari_ami(t, p)
        aris.append(ari)
        amis.append(ami)
        n_clusters.append(len(np.unique(p)))

    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ari": float(np.mean(aris)),
        "ami": float(np.mean(amis)),
        "n_clusters_mean": float(np.mean(n_clusters)),
    }


def exp_global(config, n, rep) -> tuple:
    """
    Fully self-contained: everything is read from `config`, nothing from
    worker-global state.
    """
    task_seed = hash((n, rep)) % (2**32)
    np.random.seed(task_seed)
    gum.initRandom(task_seed)

    E = config["n_clients"]
    ess = config["ess"]
    save_models = config.get("save_models", True)
    bn_base = gum.loadBN(config["bn_base_path"])

    clients = init_clients(config)
    for e in clients:
        clients[e].generate_base_info(n, ess)

    mechanisms = all_mechanisms(bn_base)

    task_id = tuple(config[k] for k in GRID_KEYS) + (n, rep)
    row = {k: config[k] for k in GRID_KEYS}
    row["size"] = n
    row["rep"] = rep

    models = {}

    # No-update ablation: local IDM credal sets, before any cross-client update.
    noupdate_results = [
        evaluate_mechanism(clients, var, r, cn_attr="cn") for var, r in mechanisms
    ]
    row.update(
        {f"noupdate_{k}": v for k, v in _network_metrics(noupdate_results).items()}
    )
    # True cluster count, read straight from the perturbation masks.
    row["n_clusters_true_mean"] = float(
        np.mean([len(np.unique(res["true_labels"])) for res in noupdate_results])
    )
    if save_models:
        for e in range(E):
            models[f"client{e}_bn_mle"] = snapshot_cpts(clients[e].bn_mle)
            models[f"client{e}_idm_min"] = snapshot_cpts(clients[e].cn.bn_min)
            models[f"client{e}_idm_max"] = snapshot_cpts(clients[e].cn.bn_max)

    # Naive point-estimate baseline: independent of weighting -> computed once.
    mle_results = [evaluate_mechanism_mle(clients, var, r) for var, r in mechanisms]
    row.update({f"mle_{k}": v for k, v in _network_metrics(mle_results).items()})

    # MOSAIC, once per weighting schema: every client is, in turn, the
    # update's target, overwriting its own cn_mosaic in place.
    for w in WEIGHTING_SCHEMES:
        for e in range(E):
            target = clients[e]
            target.reset_prior()
            prior_clients = [target] + [c for k, c in clients.items() if k != e]
            assert target.prior_cn.is_vacuous_all()
            target.prior_cn.compute(prior_clients, weighting=w)
            assert not target.prior_cn.is_vacuous_any()
            target.mosaic_cn()

        mos_results = [
            evaluate_mechanism(clients, var, r, cn_attr="cn_mosaic")
            for var, r in mechanisms
        ]
        row.update(
            {f"mos_w{w}_{k}": v for k, v in _network_metrics(mos_results).items()}
        )

        if save_models:
            for e in range(E):
                models[f"client{e}_mos_w{w}_min"] = snapshot_cpts(
                    clients[e].cn_mosaic.bn_min
                )
                models[f"client{e}_mos_w{w}_max"] = snapshot_cpts(
                    clients[e].cn_mosaic.bn_max
                )

    return row, models, task_id


def _exp_global_star(args):
    try:
        return exp_global(*args)
    except Exception:
        tb = traceback.format_exc()
        print("ERROR", args, tb, flush=True)
        return None, None, None


def main():
    set_seed()

    config = load_config("conf_global.yaml")
    save_models = config.get("save_models", True)
    max_tasks_per_child = config.get("max_tasks_per_child", 100)
    # Every client is a target here, so peak memory per task is higher; cap
    # it via conf_global.yaml's `max_workers` if cores exceed available RAM.
    n_workers = (
        min(n_jobs, config["max_workers"]) if config.get("max_workers") else n_jobs
    )

    sizes_dict = config["s_sizes"]
    sizes = [
        int(x)
        for x in np.arange(sizes_dict["min"], sizes_dict["max"], sizes_dict["step"])
    ]

    tasks = exp_local.build_tasks(config, sizes)

    n_combos = len(exp_local.hyperparameter_combos(config))
    print(
        f"# {n_combos} hyperparameter combinations x {len(sizes)} size(s) x "
        f"{config['n_repetitions']} repetitions = {len(tasks)} total tasks "
        f"on {n_workers} workers (save_models={save_models})",
        flush=True,
    )

    exp_local.run_grid(
        tasks,
        config["res_path"],
        save_models,
        max_tasks_per_child,
        row_fieldnames=ROW_FIELDNAMES,
        task_fn=_exp_global_star,
        n_workers=n_workers,
    )


if __name__ == "__main__":
    main()
