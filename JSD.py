import os
import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pyagrum as gum
from tqdm import tqdm

from src.config import get_res_path, load_config
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
    intersection_list = []
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

        # Network-wide median fraction of clients intersecting the prior
        try:
            with open(f"{base_path}/{rep}-intersection.txt") as f:
                intersection_list.append(float(f.read()))
        except FileNotFoundError:
            intersection_list.append(np.nan)

    res = pd.DataFrame({"size": sizes})
    res["intersection_frac"] = intersection_list

    for label in ["idm", "mos"]:

        min_list, max_list, mean_list = [], [], []      
        list_str = f"{label}_list"
        
        for i in range(len(res)):

            # Choose the CN
            bn_min, bn_max = eval(list_str)[i]

            # Max JSD: JSD(bn_base, .) is convex in its second argument, so
            # its maximum over the credal set's strong extension is attained
            # exactly at one of the (exhaustively enumerated) vertices --
            # this is not an approximation.
            vertices_bns = vertices_cn(bn_min, bn_max, n_bns=None)
            vertex_results = jsd_bounds_from_samples(bn_base, vertices_bns, target=targ)

            # Min and mean JSD: both approximated by sampling `n_bns` BNs
            # from the credal set's interior (cap6_extract.tex, Evaluation).
            # These must NOT be mixed with `vertices_bns`: there can be
            # orders of magnitude more vertices than samples (e.g. up to
            # 2^10 = 1024 for the Cancer network vs. n_bns=50), which would
            # make "mean" an average dominated by extreme points instead of
            # a genuine sampling-based estimate.
            sampled_bns = sample_from_cn(bn_min, bn_max, n_bns=n_bns)
            sample_results = jsd_bounds_from_samples(bn_base, sampled_bns, target=targ)

            min_list.append(sample_results["min"])
            max_list.append(vertex_results["max"])
            mean_list.append(sample_results["mean"])

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
    res_path = Path(get_res_path(config))

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

    print("Computing JSD...")
    with ctx.Pool(processes=n_workers) as pool:
        df_list = list(tqdm(pool.imap(worker, range(n_reps)), total=n_reps))

    df_tot = pd.concat(df_list, axis=0)
    df_tot.to_csv(res_path / "df_tot.csv")