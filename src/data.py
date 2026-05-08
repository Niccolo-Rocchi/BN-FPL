import ast
from itertools import product

import numpy as np
import pandas as pd
import pyagrum as gum
from numpy.random import randint

from src.config import get_cur_dir, safe_assert, set_seed


def generate_naivebayes(config):

    # Set paths
    cur_dir = get_cur_dir(config)
    bns_path = cur_dir / config["bns_path"]
    data_path = cur_dir / config["data_path"]

    # Set seed
    set_seed()

    # Retrieve hyperparameters
    n_modmax = config["n_modmax"]
    gpop_ss = config["gpop_ss"]
    pool_ss = int(gpop_ss * config["pool_prop"])
    rpop_ss = int(gpop_ss * config["rpop_prop"])

    # Set BN (naive Bayes) structure
    bn_str_gen = (
        f'{config["target_var"]}->X{i}[{randint(2, n_modmax+1)}]'
        for i in range(config["n_nodes"] - 1)
    )
    bn_str = "; ".join(bn_str_gen)

    # For each model ...
    for i in range(config["n_models"]):

        # ... generate BN, ...
        bn = gum.fastBN(bn_str)
        gum.saveBN(bn, f'{bns_path / "gt"}/{f"exp{i}"}.bif')

        with open(f'{cur_dir}/{config["exp_meta"]}', "a") as m:
            m.write(
                f'- exp{i}. Naive Bayes: {config["n_nodes"]} nodes. Complexity: {bn.dim()} Max categories: {n_modmax}\n'
            )

        # ... and generate gpop from BN
        gpop = generate_unique(bn, config["gpop_ss"])

        # For any data sample ...
        for sample in range(config["samples"]):

            # ... sample pool and rpop
            shuffled_idx = np.random.permutation(gpop.index)

            pool_idx = shuffled_idx[:pool_ss]
            rpop_idx = shuffled_idx[pool_ss : pool_ss + rpop_ss]

            gpop[f"in-pool-{sample}"] = gpop.index.isin(pool_idx)
            gpop[f"in-rpop-{sample}"] = gpop.index.isin(rpop_idx)

            # Debug
            safe_assert(pool_ss == len(pool_idx))
            safe_assert(rpop_ss == len(rpop_idx))
            safe_assert(sum(gpop[f"in-pool-{sample}"]) == pool_ss)
            safe_assert(sum(gpop[f"in-rpop-{sample}"]) == rpop_ss)
            safe_assert(sum(gpop[f"in-pool-{sample}"] & gpop[f"in-rpop-{sample}"]) == 0)
            safe_assert(
                sum(~gpop[f"in-pool-{sample}"] & ~gpop[f"in-rpop-{sample}"])
                == gpop_ss - pool_ss - rpop_ss
            )

        # Save gpop
        gpop.to_csv(f"{data_path}/exp{i}.csv", index=False)


def generate_randombn(config):

    # Set paths
    cur_dir = get_cur_dir(config)
    bns_path = cur_dir / config["bns_path"]
    data_path = cur_dir / config["data_path"]

    # Retrieve hyperparameters
    n_nodes_vec = ast.literal_eval(config["n_nodes_vec"])
    edge_ratio_vec = ast.literal_eval(config["edge_ratio_vec"])
    gpop_ss = config["gpop_ss"]
    pool_ss = int(gpop_ss * config["pool_prop"])
    rpop_ss = int(gpop_ss * config["rpop_prop"])

    # Set seed
    set_seed()

    # For each configuration ...
    for i, (n, r) in enumerate(product(n_nodes_vec, edge_ratio_vec)):

        # ... generate BN, ...
        bn_gen = gum.BNGenerator()
        bn = bn_gen.generate(n_nodes=n, n_arcs=int(n * r), n_modmax=config["n_modmax"])
        gum.saveBN(bn, f'{bns_path / "gt"}/{f"exp{i}"}.bif')

        with open(f'{cur_dir}/{config["exp_meta"]}', "a") as m:
            m.write(
                f"- exp{i}. Nodes: {n} Edges: {int(n * r)} Complexity: {bn.dim()}\n"
            )

        # ... and generate gpop from BN
        gpop = generate_unique(bn, config["gpop_ss"])

        # For any data sample ...
        for sample in range(config["samples"]):

            # ... sample pool and rpop
            shuffled_idx = np.random.permutation(gpop.index)

            pool_idx = shuffled_idx[:pool_ss]
            rpop_idx = shuffled_idx[pool_ss : pool_ss + rpop_ss]

            gpop[f"in-pool-{sample}"] = gpop.index.isin(pool_idx)
            gpop[f"in-rpop-{sample}"] = gpop.index.isin(rpop_idx)

            # Debug
            safe_assert(pool_ss == len(pool_idx))
            safe_assert(rpop_ss == len(rpop_idx))
            safe_assert(sum(gpop[f"in-pool-{sample}"]) == pool_ss)
            safe_assert(sum(gpop[f"in-rpop-{sample}"]) == rpop_ss)
            safe_assert(sum(gpop[f"in-pool-{sample}"] & gpop[f"in-rpop-{sample}"]) == 0)
            safe_assert(
                sum(~gpop[f"in-pool-{sample}"] & ~gpop[f"in-rpop-{sample}"])
                == gpop_ss - pool_ss - rpop_ss
            )

        # Save gpop
        gpop.to_csv(f"{data_path}/exp{i}.csv", index=False)


# Generate unique data points from a given BN
def generate_unique(bn: gum.BayesNet, n_samples: int) -> pd.DataFrame:

    # Generate data
    data_gen = gum.BNDatabaseGenerator(bn)
    data_gen.drawSamples(n_samples * 2)
    data = data_gen.to_pandas()

    # Ensure data items are unique
    data_unique = data.drop_duplicates()
    check = 0
    while len(data_unique) < n_samples:
        data_gen.drawSamples((n_samples - len(data_unique)) * 5)
        data = data_gen.to_pandas()
        data_unique = pd.concat([data_unique, data], axis=0).drop_duplicates()
        check += 1
        if check >= 1e6:
            raise ValueError(
                "Too many iterations, please check the data hyperparameters."
            )
    data_unique = data_unique.sample(n=n_samples, ignore_index=True)

    # Debug
    safe_assert(all(data_unique == data_unique.drop_duplicates()))
    safe_assert(len(data_unique) == n_samples)

    return data_unique
