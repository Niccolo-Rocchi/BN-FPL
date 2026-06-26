import ast
import sys
from pathlib import Path
import gc
import copy
import numpy as np
import pyagrum as gum
import multiprocessing  # noqa: F401 # pylint: disable=unused-import
from joblib import Parallel, delayed

from src.utils import get_confs, get_kl, get_kl_cset, perturb_bn_params


sys.path.insert(0, str(Path().resolve().parents[1]))
from src.config import create_clean_dir, load_config, set_seed
from src.mosaic import Client

def init_clients(config) -> list:

    clients = {}
    E = config["n_clients"]
    p = 0
    bn_base = gum.loadBN(config["bn_base_path"])
    for e in range(E):

        # Init the client
        gt, mask = perturb_bn_params(bn_base, eps=0.1, prob=p)  # If p=0 then it just copies `bn_base`
        client = Client(gt, mask)

        # Collect
        clients[e] = client
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

def exp(client_num, clients, n: int, ss_path, rep, config:dict):

    print("## Repetition: ", rep, flush=True)

    # Generate clients' data and learn models
    for e in clients:
        c = clients[e]
        c.generate_base_info(n, config["ess"])
    
    # Choose client
    client_exp = clients[client_num]
    
    # Set prior(s) clients
    prior_clients_dict = clients
    prior_clients_dict.pop(client_num)
    prior_clients = list(prior_clients_dict.values())

    # (Re-)compute the prior for the client
    client_exp.reset_prior()
    assert client_exp.prior_cn.is_vacuous_all()
    client_exp.prior_cn.compute(prior_clients, *ast.literal_eval(config["prior_args"]))
    assert not client_exp.prior_cn.is_vacuous_any()

    # Run mosaic
    client_exp.mosaic_cn()

    # Save results
    save_results(client_exp, ss_path, rep)

def main():

    # Set seed
    set_seed()

    # Read configurations
    config = load_config("conf_iid.yaml")

    # Create empty folders
    base_path = Path("results")
    create_clean_dir(base_path)
    
    # Initialize clients
    clients = init_clients(config)
    
    # Choose client
    client_num = config["client_num"]

    # For each sample size ...
    sizes_dict = config["s_sizes"]
    sizes = [int(x) for x in np.arange(sizes_dict["min"], sizes_dict["max"], sizes_dict["step"])]
    for n in sizes:
        
        print("# Sample size: ", n, flush=True)

        # Create empty folder for results
        ss_path = base_path / f"ss{n}"
        create_clean_dir(ss_path) 

    #     # Run experiment, parallelized on repetitions
    #     _ = Parallel(n_jobs=2)(
    #     delayed(exp)(client_num, copy.deepcopy(clients), n, ss_path, rep, config) for rep in range(config["n_repetitions"])
    # )
        
        # Single-core
        for rep in range(config["n_repetitions"]):
            exp(client_num, copy.deepcopy(clients), n, ss_path, rep, config)

    # Clean
    gc.collect()

            
if __name__ == "__main__":
    main()