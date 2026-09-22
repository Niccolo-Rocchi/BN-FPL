import gc
import multiprocessing as mp
import sys
from pathlib import Path
import traceback

import numpy as np
import pyagrum as gum

from src.utils import get_confs, get_kl, get_kl_cset, jsd, jsd_bn, perturb_bn_params

sys.path.insert(0, str(Path().resolve().parents[1]))
from src.config import create_clean_dir, load_config, set_seed
from src.mosaic import Client

# Set number of threads for parallel computation
import os
n_jobs = max(1, len(os.sched_getaffinity(0)) - 1)

# No pickling
_client_num = None
_config = None


def _init_worker(client_num, config):
    global _client_num, _config
    _client_num = client_num
    _config = config


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


def save_results(client, ss_path, rep, median_intersection=None):
    # Save the client's exact MLE (used as the "mle" reference in JSD.py and
    # by Plot_KL.ipynb; NOT client.bn, which is learned with a small
    # smoothing prior -- see mle_bn_from_counts / learn_bn_params).
    gum.saveBN(client.bn_mle, f"{ss_path}/{rep}-bn.bif")

    # Save the client's CN
    gum.saveBN(client.cn.bn_min, f"{ss_path}/{rep}-idm_bn_min.bif")
    gum.saveBN(client.cn.bn_max, f"{ss_path}/{rep}-idm_bn_max.bif")

    # Save the client's prior
    gum.saveBN(client.prior_cn.bn_min, f"{ss_path}/{rep}-prior_bn_min.bif")
    gum.saveBN(client.prior_cn.bn_max, f"{ss_path}/{rep}-prior_bn_max.bif")

    # Save mosaic results for client
    gum.saveBN(client.cn_mosaic.bn_min, f"{ss_path}/{rep}-mos_bn_min.bif")
    gum.saveBN(client.cn_mosaic.bn_max, f"{ss_path}/{rep}-mos_bn_max.bif")

    # Save the network-wide median fraction of clients intersecting the prior
    if median_intersection is not None:
        with open(f"{ss_path}/{rep}-intersection.txt", "w") as f:
            f.write(str(median_intersection))


def exp(n, ss_path, rep):
    # print("## Repetition: ", rep, flush=True)

    # Each (n, rep) task needs its own independent random stream. The worker
    # pool uses "fork", so sibling worker processes inherit an IDENTICAL RNG
    # state at fork time; without reseeding here, several "repetitions"
    # processed by distinct, freshly-forked workers would silently generate
    # IDENTICAL data (confirmed empirically: client 0's data was byte-for-
    # byte identical across repetitions for several sample sizes). The seed
    # is still fully reproducible given (n, rep).
    task_seed = hash((n, rep)) % (2**32)
    np.random.seed(task_seed)
    gum.initRandom(task_seed)

    config = _config
    client_num = _client_num

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

    # (Re-)compute the prior for the client
    client_exp.reset_prior()
    assert client_exp.prior_cn.is_vacuous_all()
    median_intersection = client_exp.prior_cn.compute(prior_clients, **config["prior_args"])
    assert not client_exp.prior_cn.is_vacuous_any()

    # Run mosaic
    client_exp.mosaic_cn()

    # Save results
    save_results(client_exp, ss_path, rep, median_intersection)


def _exp_star(args):
    try:
        return exp(*args)
    except Exception:
        tb = traceback.format_exc()
        print("ERROR", args, tb, flush=True)
        return


def main():

    # Set seed
    set_seed()

    # Choose configurationc file
    config = load_config("conf.yaml")

    # Create empty folders
    int_str = "Int" if config["prior_args"]["intersection"] else "NoInt"
    res_str = config["res_path"] + "_" + str(config["prob_shift"]) + "_" + int_str
    res_path = Path(res_str) 
    create_clean_dir(res_path)

    # Choose client
    client_num = config["client_num"]

    # For each sample size ...
    sizes_dict = config["s_sizes"]
    sizes = [
        int(x)
        for x in np.arange(sizes_dict["min"], sizes_dict["max"], sizes_dict["step"])
    ]
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=n_jobs,
        initializer=_init_worker,
        initargs=(client_num, config),
    ) as pool:
        for n in sizes:
            print("# Sample size: ", n, flush=True)
            ss_path = res_path / f"ss{n}"
            create_clean_dir(ss_path)

            tasks = [(n, ss_path, rep) for rep in range(config["n_repetitions"])]
            pool.map(_exp_star, tasks)

    gc.collect()


if __name__ == "__main__":
    main()