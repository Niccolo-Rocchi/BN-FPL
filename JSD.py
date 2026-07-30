import os
import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pyagrum as gum
from tqdm import tqdm

from src.config import load_config
from src.utils import jsd, jsd_bn, jsd_bounds_from_samples, sample_from_cn, vertices_cn


def _limit_threads():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

def process_rep(rep, sizes, res_path, bn_base_path, n_bns, targ):
    _limit_threads()

    res_path = Path(res_path)
    bn_base = gum.loadBN(bn_base_path)

    bn_list = []
    idm_list = []
    mos_list = []
    for n in sizes:
        base_path = res_path / f"ss{n}"

        # BN
        bn = gum.loadBN(f"{base_path}/{rep}-bn.bif")
        bn_list.append(bn)

        # CN
        idm_bn_min = gum.loadBN(f"{base_path}/{rep}-idm_bn_min.bif")
        idm_bn_max = gum.loadBN(f"{base_path}/{rep}-idm_bn_max.bif")
        idm_list.append((idm_bn_min, idm_bn_max))

        # Mosaic
        mos_bn_min = gum.loadBN(f"{base_path}/{rep}-mos_bn_min.bif")
        mos_bn_max = gum.loadBN(f"{base_path}/{rep}-mos_bn_max.bif")
        mos_list.append((mos_bn_min, mos_bn_max))

    res = pd.DataFrame({"size": sizes})

    for label in ["idm", "mos"]:

        min_list, max_list, mean_list = [], [], []      
        list_str = f"{label}_list"
        
        for i in range(len(res)):

            # Choose the CN
            bn_min, bn_max = eval(list_str)[i]

            # Vertices (for max JSD)
            vertices_bns = vertices_cn(bn_min, bn_max, n_bns=None)

            # Inner points (for min JSD)
            sampled_bns = sample_from_cn(bn_min, bn_max, n_bns=n_bns)

            # All BNs
            all_bns = vertices_bns + sampled_bns

            # Results
            results = jsd_bounds_from_samples(bn_base, all_bns, target=targ)
            min_list.append(results["min"])
            max_list.append(results["max"])
            mean_list.append(results["mean"])

        res[f"{label}_min"] = min_list
        res[f"{label}_max"] = max_list
        res[f"{label}_mean"] = mean_list

    mle_list = []
    for i in range(len(res)):
        bn = bn_list[i]
        mle_list.append(jsd_bn(bn_base, bn, target=targ))

    res["mle"] = mle_list
    return res


if __name__ == "__main__":

    # Read configurations
    config = load_config("conf.yaml")
    sizes_dict = config["s_sizes"]
    sizes = [
        int(x) for x in np.arange(sizes_dict["min"], sizes_dict["max"], sizes_dict["step"])
    ]
    n_reps = config["n_repetitions"]
    int_str = "Int" if config["prior_args"]["intersection"] else "NoInt"
    res_str = config["res_path"] + "_" + str(config["prob_shift"]) + "_" + int_str
    res_path = Path(res_str) 

    n_workers = min(n_reps, os.cpu_count())

    ctx = mp.get_context("spawn")
    worker = partial(
        process_rep,
        sizes=sizes,
        res_path=res_path,
        n_bns=config["n_bns"],
        bn_base_path=config["bn_base_path"],  # path al file .bif di bn_base
        targ="joint"
    )

    with ctx.Pool(processes=n_workers) as pool:
        df_list = list(tqdm(pool.imap(worker, range(n_reps)), total=n_reps))

    df_tot = pd.concat(df_list, axis=0)
    df_tot.to_csv(res_path / "df_tot.csv")