import copy
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
_clients_template = None
_client_num = None
_config = None


def _init_worker(clients_template, client_num, config):
    global _clients_template, _client_num, _config
    _clients_template = clients_template
    _client_num = client_num
    _config = config


def init_clients(config, verbose=False) -> list:
    E = config["n_clients"]
    eps = config["eps"]
    bn_base = gum.loadBN(config["bn_base_path"])

    clients = {}
    for e in range(E):

        # Copy `bn_base` for the first client
        p = config["prob_shift"] if e != 0 else 0

        # Init the client
        gt, mask = perturb_bn_params(
            bn_base, eps=eps, prob=p
        )  # If p=0 then just copy `bn_base`
        client = Client(gt, mask)

        # Collect
        clients[e] = client

        if verbose and e != 0:
            d = jsd_bn(clients[e].gt, clients[0].gt, "joint")
            print("Dist. from client 0: ", d)

    return clients


def save_results(client, ss_path, rep):
    # Save the client's BN
    gum.saveBN(client.bn, f"{ss_path}/{rep}-bn.bif")

    # Save the client's CN
    gum.saveBN(client.cn.bn_min, f"{ss_path}/{rep}-idm_bn_min.bif")
    gum.saveBN(client.cn.bn_max, f"{ss_path}/{rep}-idm_bn_max.bif")

    # Save the client's prior
    gum.saveBN(client.prior_cn.bn_min, f"{ss_path}/{rep}-prior_bn_min.bif")
    gum.saveBN(client.prior_cn.bn_max, f"{ss_path}/{rep}-prior_bn_max.bif")

    # Save mosaic results for client
    gum.saveBN(client.cn_mosaic.bn_min, f"{ss_path}/{rep}-mos_bn_min.bif")
    gum.saveBN(client.cn_mosaic.bn_max, f"{ss_path}/{rep}-mos_bn_max.bif")


def exp(n, ss_path, rep):
    # print("## Repetition: ", rep, flush=True)

    clients = copy.deepcopy(_clients_template)
    config = _config
    client_num = _client_num

    # Generate clients' data and learn models
    for e in clients:
        c = clients[e]
        c.generate_base_info(n, config["ess"])

    # Choose client
    client_exp = clients[client_num]

    # Set prior(s) clients
    prior_clients = list(clients.copy().values())

    # (Re-)compute the prior for the client
    client_exp.reset_prior()
    assert client_exp.prior_cn.is_vacuous_all()
    client_exp.prior_cn.compute(prior_clients, **config["prior_args"])
    assert not client_exp.prior_cn.is_vacuous_any()

    # Run mosaic
    client_exp.mosaic_cn()

    # Save results
    save_results(client_exp, ss_path, rep)


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

    # Initialize clients
    clients = init_clients(config, verbose=True)

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
        initargs=(clients, client_num, config),
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