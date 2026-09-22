from fractions import Fraction
from tempfile import TemporaryDirectory

import cdd
import cdd.gmp
import cvxpy as cp
import hopsy
import numpy as np
import pandas as pd
import pyagrum as gum
from scipy.optimize import linprog
import itertools

from src.config import safe_assert


# Check whether two csets intersect
def check_intersection(V1, V2) -> bool:
    """
    V1: (M, D), V2: (N, D)
    Returns (bool intersection) between two credal sets,
    represented as sets of extreme vertices
    """

    m, d = V1.shape
    n, _ = V2.shape

    c = np.zeros(m + n)

    A_eq_coords = np.hstack((V1.T, -V2.T))  # (D, m+n)
    A_eq_sum1 = np.hstack((np.ones(m), np.zeros(n)))
    A_eq_sum2 = np.hstack((np.zeros(m), np.ones(n)))
    A_eq = np.vstack((A_eq_coords, A_eq_sum1, A_eq_sum2))
    b_eq = np.hstack((np.zeros(d), [1.0, 1.0]))

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")

    if res.success:
        # point = V1.T @ res.x[:m]   # == V2.T @ res.x[m:]
        return True
    return False


# Define a vacuous CN based on a given BN
def vac_cn(bn: gum.BayesNet):

    bn_min = gum.BayesNet(bn)
    bn_max = gum.BayesNet(bn)

    for n in bn.nodes():
        bn_min.cpt(n).fillWith(0)
        bn_max.cpt(n).fillWith(1)

    cn = gum.CredalNet(bn_min, bn_max)
    cn.intervalToCredal()

    return cn


# Perturb BN parameters with probability `prob` and concentration `alpha`
def perturb_bn_params(bn: gum.BayesNet, alpha: float, prob: float = 1.0) -> gum.BayesNet:
    """
    Each conditional X|\pi_X is perturbed with probability `prob`.
    The perturbed row is drawn from Dirichlet(alpha * row), which is
    centered exactly on the original row (E[p_new] = row) with
    Var[p_new_i] = row_i*(1-row_i)/(alpha+1): larger `alpha` means a
    tighter (smaller) shift, and alpha -> infinity recovers the original
    row exactly. Unlike additive Gaussian noise followed by clipping and
    renormalization, every draw is automatically a valid probability
    vector, with no ad hoc floor/renormalization needed.
    """
    bn_new = gum.BayesNet(bn)
    bn_mask = gum.BayesNet(bn)

    for node_id in bn_new.nodes():
        cpt_new = bn_new.cpt(node_id)
        cpt_resh = get_tabular_cpt(cpt_new)

        mask = np.ones(cpt_resh.shape)

        for idx in range(cpt_resh.shape[0]):
            if np.random.choice([True, False], p=[prob, 1.0 - prob]):
                mask[idx, :] = 0

                row = cpt_resh[idx, :]
                # Guard against an exact-zero entry, which would make
                # np.random.dirichlet's concentration parameter <= 0.
                concentration = alpha * np.clip(row, 1e-9, None)
                cpt_resh[idx, :] = np.random.dirichlet(concentration)

        cpt_new.fillWith(cpt_resh.flatten())
        bn_mask.cpt(node_id).fillWith(mask.flatten())

        # Debug
        safe_assert(np.allclose(np.sum(cpt_resh, axis=1), 1))

    return bn_new, bn_mask


# Resample BN parameters with probability `prob`
def resample_bn_params(
    bn: gum.BayesNet, alpha: float = 1.0, prob: float = 1.0
) -> gum.BayesNet:
    """
    Copy the `bn` structure and resample its parameters from a Dirichlet distribution.
    Here, `prob` is the probability that a conditional X|\pi_X is resampled.
    The output is the resampled BN (`bn_new`) and a mask BN (`bn_mask`).
    The latter indicates the differences between `bn` and `bn_new` (1 = equal, 0 = different).
    """
    bn_new = gum.BayesNet(bn)
    bn_mask = gum.BayesNet(bn)

    for node_id in bn_new.nodes():
        cpt = bn_new.cpt(node_id)
        var_size = bn_new.variable(node_id).domainSize()
        n_rows = cpt.domainSize() // var_size

        samples = np.random.dirichlet(alpha=[alpha] * var_size, size=n_rows)

        mask = np.ones(samples.shape)

        for row in range(n_rows):
            if np.random.choice([True, False], p=[prob, 1.0 - prob]):
                mask[row, :] = 0

        cpt_resh = cpt[:].reshape(n_rows, var_size)

        cpt_new = mask * cpt_resh[:] + (1 - mask) * samples

        cpt.fillWith(cpt_new.flatten().tolist())
        bn_mask.cpt(node_id).fillWith(mask.flatten().tolist())

        # Debug
        safe_assert(np.allclose(np.sum(cpt_resh, axis=1), 1))

    return bn_new, bn_mask

# Compute the Jensen–Shannon divergence (JSD) between two distributions
def jsd(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

# Compute the Jensen–Shannon divergence (JSD) between two BNs
def jsd_bn(B1, B2, target="marginals"):

    if target == "joint":
        names = sorted(list(B1.names()))


        p_B1 = get_joint(B1, names)
        p_B2 = get_joint(B2, names)

        return jsd(p_B1, p_B2)
    
    ieB1 = gum.LazyPropagation(B1)
    ieB1.makeInference()

    ieB2 = gum.LazyPropagation(B2)
    ieB2.makeInference()
    
    results = {node: [] for node in B1.names()}
    for node in B1.names():
        p_B1 = ieB1.posterior(node).tolist()
        p_B2 = ieB2.posterior(node).tolist()
        results[node].append(jsd(p_B1, p_B2))

    return results

# Compute the JSD bounds between a BN `B` and a set of BNs `sampled_bns`
def jsd_bounds_from_samples(B, sampled_bns, target="marginals"):

    names = sorted(list(B.names()))

    if target == "joint":
        p_B = get_joint(B, names)

        vals = []
        for bn in sampled_bns:
            p_C = get_joint(bn, names)
            vals.append(jsd(p_B, p_C))
        return {"min": min(vals), "max": max(vals), "mean": float(np.mean(vals)), "all": vals}

    ieB = gum.LazyPropagation(B)
    ieB.makeInference()

    results = {node: [] for node in B.names()}
    for bn in sampled_bns:
        ie = gum.LazyPropagation(bn)
        ie.makeInference()
        for node in B.names():
            p_B = ieB.posterior(node).tolist()
            p_C = ie.posterior(node).tolist()
            results[node].append(jsd(p_B, p_C))

    return {
        node: {"min": min(v), "max": max(v), "mean": float(np.mean(v)), "std": np.std(v)}
        for node, v in results.items()
    }


# Get the joint distribution of a BN, ordered by `names`
def get_joint(bn:gum.BayesNet, names:list):

    ie = gum.LazyPropagation(bn)
    ie.eraseAllEvidence()
    ie.addJointTarget(set(names))
    ie.makeInference()
    p = ie.jointPosterior(set(names))
    p = p.reorganize(names)
    p = p.toarray().tolist()

    return p

# Compute the KL between a cset and the ground-truth distribution
def get_kl_cset(cn: tuple, gt: gum.BayesNet, var, parents):
    """
    Let X|\pa_X be a conditional distribution.
    The function returns the KL between a credal set in `cn` and the ground-truth (in `gt`).
    The KL is computed as the maximum KL over the vertices of the cset.
    `cn` is a tuple of (bn_min, bn_max).
    """

    # Get the ground-truth distribution
    var_obj = gt.variable(var)
    parents_idx = get_cpt_index(gt, var, parents)
    gt_distr = get_tabular_cpt(gt.cpt(var))[parents_idx, :]
    gt_distr_smoothed = np.clip(gt_distr, 1e-12, None)
    t_gt = gum.Tensor(var_obj)
    t_gt.fillWith(gt_distr_smoothed)

    # Get the vertices of the cset
    bn_min, bn_max = cn
    cpt_min, cpt_max = get_tabular_cpt(bn_min.cpt(var)), get_tabular_cpt(
        bn_max.cpt(var)
    )
    cset_min, cset_max = cpt_min[parents_idx, :], cpt_max[parents_idx, :]
    vertices = vertices_cset(cset_min, cset_max).tolist()

    # For each vertex ...
    kl_list = []
    for v in vertices:

        # ... compute the KL against the ground-truth
        t = gum.Tensor(var_obj)
        v_smoothed = np.clip(v, 1e-12, None)
        t.fillWith(v_smoothed)
        kl_sym = (t.KL(t_gt) + t_gt.KL(t)) / 2
        kl_list.append(kl_sym)

    # Get the maximum KL
    return max(kl_list)


# Compute the KL between the learned distribution and the ground-truth one
def get_kl(bn: gum.BayesNet, gt: gum.BayesNet, var, parents):
    """
    Let X|\pa_X be a conditional distribution.
    The function returns the KL between a given distribution X|\pa_X
    of a BN (`bn`) and its ground-truth (`gt`).
    """

    # Get the ground-truth distribution
    var_obj = bn.variable(var)
    parents_idx = get_cpt_index(gt, var, parents)
    gt_distr = get_tabular_cpt(gt.cpt(var))[parents_idx, :]
    gt_distr_smoothed = np.clip(gt_distr, 1e-12, None)
    t_gt = gum.Tensor(var_obj)
    t_gt.fillWith(gt_distr_smoothed)

    # Get the compared distribution
    distr = get_tabular_cpt(bn.cpt(var))[parents_idx, :]
    distr_smoothed = np.clip(distr, 1e-12, None)
    t = gum.Tensor(var_obj)
    t.fillWith(distr_smoothed)

    # Get the KL
    kl_sym = (t.KL(t_gt) + t_gt.KL(t)) / 2
    return kl_sym


# Create the BN storing the counts of events. `bn` is only used for its
# structure (node names, parents, variable labels and the CPTs' row/column
# order); its CPT *values* are never read, so the result only depends on
# `data`, unlike a previous version of this function which reconstructed
# joint counts as (bn's CPT value) * (parent-marginal count) -- correct only
# when `bn`'s CPT happens to already hold the exact empirical MLE of `data`
# (true for `Client.learn_bn`'s output, but silently wrong whenever called
# with any other CPT, e.g. `mle_cn`/`mne_cn`/`ran_cn`/`centroid_cn`/
# `maxent_cn`, which pass a CN's bn_min).
def get_bn_counts(bn, data):

    # Init the BN
    bn_counts = gum.BayesNet(bn)

    for node in bn.names():

        var_size = bn.variable(node).domainSize()
        node_labels = list(bn.variable(node).labels())  # CPT's own column order

        parent_confs = get_parent_confs(bn, node)  # native row order, not topandas'

        if parent_confs != [None]:
            n_rows = len(parent_confs)
            index_df = pd.DataFrame(parent_confs)
            parent_cols = index_df.columns.tolist()

            # Full (parent configuration x node value) grid, in the CPT's own
            # row/column order: `node_labels * n_rows` tiles the node's
            # labels, aligning with `index_df`'s rows repeated `var_size`
            # times each.
            full_grid = index_df.loc[index_df.index.repeat(var_size)].reset_index(
                drop=True
            )
            full_grid[node] = node_labels * n_rows

            subset = parent_cols + [node]
            data_counts = data.value_counts(subset=subset).reset_index(name="counts")
            res = pd.merge(full_grid, data_counts, on=subset, how="left")
            res["counts"] = res["counts"].fillna(0).astype(int)

            cpt_new = res["counts"].to_numpy().reshape(n_rows, var_size)

        else:
            value_counts = data[node].value_counts()
            cpt_new = np.array([[value_counts.get(lab, 0) for lab in node_labels]])

        bn_counts.cpt(node).fillWith(cpt_new.flatten().tolist())

        # Debug
        safe_assert(np.sum(cpt_new) == len(data))

    return bn_counts


# Get the exact (unsmoothed) MLE BN from a BN of counts (see get_bn_counts):
# P(X=x|pi_X) = N[x,pi_X] / N[pi_X], row by row. This is the correct theta_hat
# to use whenever the true MLE is needed (e.g. as a reference for evaluation,
# or in the MOSAIC update rule): `learn_bn_params`'s output should NOT be used
# for this, since its smoothing prior (needed only when the learnt BN is
# itself used to generate further data, to avoid zero probabilities) biases
# it away from the true MLE -- a bias that is small in absolute terms but can
# be large relative to the tiny sample sizes typical of this pipeline.
def mle_bn_from_counts(bn_counts: gum.BayesNet) -> gum.BayesNet:

    bn = gum.BayesNet(bn_counts)

    for var in bn.names():
        counts = get_tabular_cpt(bn_counts.cpt(var))
        var_size = counts.shape[1]
        row_sums = counts.sum(axis=1, keepdims=True)

        # Where a parent configuration was never observed (row_sums == 0),
        # the MLE is undefined; default to a uniform distribution there.
        mle = np.divide(
            counts,
            row_sums,
            out=np.full_like(counts, 1.0 / var_size, dtype=float),
            where=row_sums != 0,
        )
        bn.cpt(var).fillWith(mle.flatten().tolist())

        # Debug
        safe_assert(np.allclose(mle.sum(axis=1), 1.0))

    return bn


# Get a list of (var, parents) configurations from a BN
def get_confs(bn: gum.BayesNet) -> list:
    confs = []
    for var in bn.names():

        # For every parent configuration ...
        for parents in get_parent_confs(bn, var):

            confs.append([var, parents])
    return confs


# Get a bidimensional CPT
def get_tabular_cpt(cpt) -> np.array:

    cpt = np.atleast_2d(cpt[:])
    if cpt.ndim == 2:
        return cpt

    n_rows, var_size = get_cpt_shape(cpt)

    return cpt.reshape(n_rows, var_size)


# Get the shape of a BN's CPT
def get_cpt_shape(cpt) -> tuple:

    cpt_arr = np.atleast_2d(cpt[:])
    var_size = cpt_arr.shape[-1]
    n_rows = np.prod(cpt_arr.shape[:-1])

    return n_rows, var_size


# List the parent configurations of `var` in `bn`, in the same row order as
# get_tabular_cpt(bn.cpt(var)) -- i.e. pyagrum's native order (as used by
# cpt[:] and .fillWith()), NOT cpt.topandas()'s alphabetically-sorted order.
# The two differ (silently) for any variable whose labels are not already in
# alphabetical order -- e.g. every variable in cancer.bif (True/False,
# low/high, positive/negative). Returns [None] for a root variable.
def get_parent_confs(bn: gum.BayesNet, var: str) -> list:

    cpt = bn.cpt(var)
    parent_vars = [v for v in cpt.variablesSequence() if v.name() != var]
    if not parent_vars:
        return [None]

    inst = gum.Instantiation()
    for v in parent_vars:
        inst.add(v)

    confs = []
    inst.setFirst()
    while not inst.end():
        confs.append(
            {
                inst.variable(k).name(): inst.variable(k).label(inst.val(k))
                for k in range(inst.nbrDim())
            }
        )
        inst.inc()

    return confs


# Get the index in a CPT corresponding to a specific configuration of the parents
def get_cpt_index(bn: gum.BayesNet, var: str, parents: dict):
    """
    Notice: the CPT is thought as a bidimensional matrix, with rows ordered
    as in get_parent_confs (i.e. get_tabular_cpt's native order).
    """
    if parents is None:
        return 0

    parents = {k: str(v) for k, v in parents.items()}
    for idx, conf in enumerate(get_parent_confs(bn, var)):
        if conf is not None and all(str(conf.get(k)) == v for k, v in parents.items()):
            return idx

    raise RuntimeError(f"Wrong parent configuration for variable '{var}'.")


# Get the BN inside a CN with max entropy distribution
def maxent_cn(bn_min, bn_max) -> gum.BayesNet:

    # Init an empty BN
    bn = gum.BayesNet(bn_min)

    # For each variable ...
    for var in bn.names():

        # ... get the maxent CPT, ...
        cpt = maxent_cpt(bn_min.cpt(var), bn_max.cpt(var))

        # ... and fill the BN
        bn.cpt(var).fillWith(cpt.flatten())

    # Debug
    safe_assert(check_consistency(bn, bn_min, bn_max) == 0)

    return bn


# Get the BN CPT inside a CN CPT with max entropy distribution
def maxent_cpt(cpt_min, cpt_max) -> np.array:

    # Transform CPTs into pandas dataframes
    cpt_min = get_tabular_cpt(cpt_min)
    cpt_max = get_tabular_cpt(cpt_max)

    # For each row in the CPT ...
    cpt = []
    for row in range(cpt_min.shape[0]):

        # ... get the maxent credal set, ...
        c = maxent_cset(cpt_min[row, :], cpt_max[row, :])
        cpt.append(c)

    # Reshape the CPT
    cpt = np.array(cpt)

    # Debug
    safe_assert(cpt_min.shape == cpt_max.shape)
    safe_assert(cpt.shape == cpt_min.shape)

    return cpt


# Get the max-entropy distribution inside a credal set
def maxent_cset(vec_min, vec_max) -> np.array:

    rank = {v: k for k, v in enumerate(sorted(set(vec_min)))}
    vec_order = np.array([rank[val] for val in vec_min])
    s = 1 - np.sum(vec_min)
    out = vec_min

    while s > 0:
        idx0 = np.where(vec_order == 0)[0]
        idx1 = np.where(vec_order == 1)[0]
        idx0_len = len(idx0)
        idx1_len = len(idx1)

        if idx1_len != 0:
            diff = out[idx1[0]] - out[idx0[0]]
            s_cond = s / idx0_len < diff
            mat = np.stack(
                [
                    (
                        (s / idx0_len) * np.ones(len(idx0))
                        if s_cond
                        else diff * np.ones(len(idx0))
                    ),
                    vec_max[idx0] - out[idx0],
                ]
            )
        else:
            s_cond = True
            mat = np.stack(
                [(s / idx0_len) * np.ones(len(idx0)), vec_max[idx0] - out[idx0]]
            )

        mat_min = np.min(mat)
        q = np.argwhere(mat == mat_min)

        if np.any(q[:, 0] == 1):
            if len(idx0) > len(q):
                vec_order[~np.isin(np.arange(len(out)), idx0[q[:, 1]])] += 1
        elif not s_cond:
            vec_order[idx0[q[:, 1]]] += 1

        out[idx0] += mat_min
        s -= mat_min * len(idx0)
        vec_order -= 1

    return out


# Get the max likelihood BN inside a CN
def mle_cn(bn_min, bn_max, data) -> gum.BayesNet:

    # Init an empty BN
    bn = gum.BayesNet(bn_min)

    # Store counts
    bn_counts = get_bn_counts(bn, data)

    # For each variable ...
    for var in bn.names():

        # ... get the MLE CPT, ...
        cpt = mle_cpt(bn_min.cpt(var), bn_max.cpt(var), bn_counts.cpt(var))

        # ... and fill the BN
        bn.cpt(var).fillWith(cpt.flatten())

    # Debug
    safe_assert(check_consistency(bn, bn_min, bn_max) == 0)

    return bn


# Get the max likelihood BN CPT inside a CN CPT
def mle_cpt(cpt_min, cpt_max, cpt_counts) -> np.array:

    # Transform CPTs into pandas dataframes
    cpt_min = get_tabular_cpt(cpt_min)
    cpt_max = get_tabular_cpt(cpt_max)
    cpt_counts = get_tabular_cpt(cpt_counts)

    # For each row in the CPT ...
    cpt = []
    for row in range(cpt_min.shape[0]):

        # ... get the MLE credal set, ...
        c = mle_cset(cpt_min[row, :], cpt_max[row, :], cpt_counts[row, :])
        cpt.append(c)

    # Reshape the CPT
    cpt = np.array(cpt)

    # Debug
    safe_assert(cpt_min.shape == cpt_max.shape)
    safe_assert(cpt.shape == cpt_min.shape)

    return cpt


# Get the max likelihood distribution inside a credal set
def mle_cset(vec_min, vec_max, counts) -> np.array:

    # Number of variables to optimize
    n_par = len(vec_min)
    p = cp.Variable(n_par)

    # Log-likelihood to maximize
    objective = cp.Maximize(counts @ cp.log(p))

    # Constraints
    constraints = [cp.sum(p) == 1, p >= np.maximum(vec_min, 10e-9), p <= vec_max]

    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False)

    mle_vec = np.array(p.value)

    # Debug
    safe_assert(len(vec_min) == len(vec_max))

    return mle_vec


# Get the max (-likelihood) BN (MNE) within a CN, i.e., the min likelihood BN.
def mne_cn(bn_min, bn_max, data) -> gum.BayesNet:

    # Init an empty BN
    bn = gum.BayesNet(bn_min)

    # Store counts
    bn_counts = get_bn_counts(bn, data)

    # For each variable ...
    for var in bn.names():

        # ... get the MNE CPT, ...
        cpt = mne_cpt(bn_min.cpt(var), bn_max.cpt(var), bn_counts.cpt(var))

        # ... and fill the BN
        bn.cpt(var).fillWith(cpt.flatten())

    # Debug
    safe_assert(check_consistency(bn, bn_min, bn_max) == 0)

    return bn


# Get the MNE BN CPT inside a CN CPT
def mne_cpt(cpt_min, cpt_max, cpt_counts) -> np.array:

    # Transform CPTs into pandas dataframes
    cpt_min = get_tabular_cpt(cpt_min)
    cpt_max = get_tabular_cpt(cpt_max)
    cpt_counts = get_tabular_cpt(cpt_counts)

    # For each row in the CPT ...
    cpt = []
    for row in range(cpt_min.shape[0]):

        # ... get the MNE credal set, ...
        c = mne_cset(cpt_min[row, :], cpt_max[row, :], cpt_counts[row, :])
        cpt.append(c)

    # Reshape the CPT
    cpt = np.array(cpt)

    # Debug
    safe_assert(cpt_min.shape == cpt_max.shape)
    safe_assert(cpt.shape == cpt_min.shape)

    return cpt


def _neg_loglik(vec, counts) -> float:
    """
    -sum(counts * log(vec)), with the convention 0*log(0)=0: a category with
    zero probability contributes 0 if it was never observed (counts=0), but
    +inf if it was observed (counts>0) -- i.e. that vector is infinitely
    unlikely, and must never be silently treated as a neutral (0) penalty.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_vec = np.full_like(vec, -np.inf, dtype=float)
        np.log(vec, out=log_vec, where=vec > 0)
        contrib = counts * -log_vec
    return float(np.sum(np.where(counts == 0, 0.0, contrib)))


# Get the MNE distribution inside a credal set
def mne_cset(vec_min, vec_max, counts) -> np.array:

    # Get the credal set vertices
    vertices = vertices_cset(vec_min, vec_max)

    # Get the vertex that has the maximum (-likelihood), i.e. the minimum
    # likelihood (worst fit to `counts`) within the credal set
    vec_best = vertices[0, :]
    mne_best = _neg_loglik(vec_best, counts)

    for row in range(vertices.shape[0]):

        vec = vertices[row, :]
        mne = _neg_loglik(vec, counts)

        if mne > mne_best:
            mne_best = mne
            vec_best = vec

    # Debug
    safe_assert(len(vec_min) == len(vec_max))

    return vec_best


# Get a random BN inside a CN
def ran_cn(bn_min, bn_max) -> gum.BayesNet:

    # Init an empty BN
    bn = gum.BayesNet(bn_min)

    # For each variable ...
    for var in bn.names():

        # ... get a random CPT, ...
        cpt = ran_cpt(bn_min.cpt(var), bn_max.cpt(var))

        # ... and fill the BN
        bn.cpt(var).fillWith(cpt.flatten())

    # Debug
    safe_assert(check_consistency(bn, bn_min, bn_max) == 0)

    return bn


# Get a random BN CPT inside a CN CPT
def ran_cpt(cpt_min, cpt_max) -> np.array:

    # Transform CPTs into pandas dataframes
    cpt_min = get_tabular_cpt(cpt_min)
    cpt_max = get_tabular_cpt(cpt_max)

    # For each row in the CPT ...
    cpt = []
    for row in range(cpt_min.shape[0]):

        # ... sample randomly from the credal set, ...
        c = ran_cset(cpt_min[row, :], cpt_max[row, :])
        cpt.append(c)

    # Reshape the CPT
    cpt = np.array(cpt)

    # Debug
    safe_assert(cpt_min.shape == cpt_max.shape)
    safe_assert(cpt.shape == cpt_min.shape)

    return cpt


# Get a random distribution inside a credal set
def ran_cset(vec_min, vec_max) -> np.array:

    # Get the credal set vertices
    vertices = vertices_cset(vec_min, vec_max)

    # Sample weights for vertices
    n = len(vertices)
    w = np.random.dirichlet(np.ones(n))

    # Draw the linear combination of vertices
    ran_vec = w @ vertices

    # Debug
    safe_assert(len(vec_min) == len(vec_max))

    return ran_vec


# Get the centroid of a CN
def centroid_cn(bn_min, bn_max) -> gum.BayesNet:

    # Init an empty BN
    bn = gum.BayesNet(bn_min)

    # For each variable ...
    for var in bn.names():

        # ... get the centroid CPT, ...
        cpt = centroid_cpt(bn_min.cpt(var), bn_max.cpt(var))

        # ... and fill the BN
        bn.cpt(var).fillWith(cpt.flatten())

    # Debug
    safe_assert(check_consistency(bn, bn_min, bn_max) == 0)

    return bn


# Get the centroid of a CN CPT
def centroid_cpt(cpt_min, cpt_max) -> np.array:

    # Transform CPTs into pandas dataframes
    cpt_min = get_tabular_cpt(cpt_min)
    cpt_max = get_tabular_cpt(cpt_max)

    # For each row in the CPT ...
    cpt = []
    for row in range(cpt_min.shape[0]):

        # ... get the centroid credal set, ...
        c = centroid_cset(cpt_min[row, :], cpt_max[row, :])
        cpt.append(c)

    # Reshape the CPT
    cpt = np.array(cpt)

    # Debug
    safe_assert(cpt_min.shape == cpt_max.shape)
    safe_assert(cpt.shape == cpt_min.shape)

    return cpt


# Get the centroid of a credal set as the average of its extreme points
def centroid_cset(vec_min, vec_max) -> np.array:

    # Get the credal set vertices
    vertices = vertices_cset(vec_min, vec_max)

    # Compute the centroid as the average across extreme points
    centroid = np.sum(vertices, axis=0) / len(vertices)

    # Debug
    safe_assert(len(vec_min) == len(vec_max))

    return centroid


# Get the credal set vertices
def vertices_cset(vec_min, vec_max) -> np.array:

    # Degenerate case. Shape is kept consistent with the general case (a 2D
    # array of shape (n_vertices, n_par), here n_vertices=1) since callers
    # (e.g. `check_intersection`, `ran_cset`, `centroid_cset`, `mne_cset`)
    # index vertices by row.
    if np.all(vec_min == vec_max):
        return np.atleast_2d(vec_min)

    # Define the (in)equalities (i.e., get the H-representation of the credal set)
    n_par = len(vec_min)
    A = np.concatenate(
        (-np.eye(n_par), np.eye(n_par), np.atleast_2d(np.ones(n_par))), axis=0
    )
    b = np.concatenate((vec_max, -vec_min, np.atleast_1d(-1))).reshape(len(A), 1)
    bA = np.concatenate((b, A), axis=1)
    bA_frac = np.array(
        [[Fraction(x).limit_denominator() for x in row] for row in bA], dtype=object
    )  # Needed for numerical stability
    mat_frac = cdd.gmp.matrix_from_array(
        array=bA_frac, rep_type=cdd.RepType.INEQUALITY, lin_set=set([len(A) - 1])
    )

    # Get the polytope and extreme points. Each point is a row of the matrix `vertices`
    poly_frac = cdd.gmp.polyhedron_from_matrix(mat_frac)
    ext_frac = cdd.gmp.copy_generators(poly_frac)
    vertices_frac = np.array(ext_frac.array)[:, 1:]
    vertices = np.array([[float(x) for x in row] for row in vertices_frac], dtype=float)

    # Debug
    safe_assert(len(b) == 2 * len(vec_min) + 1)
    safe_assert(A.shape == (len(b), len(vec_min)))
    safe_assert(bA.shape == (2 * len(vec_min) + 1, len(vec_min) + 1))
    safe_assert(vertices.shape[1] == n_par)

    return vertices

# Get CPT vertices by combining all the local ones
def vertices_cpt(cpt_min, cpt_max):
    """
    For a single CPT (with min/max bounds), this function calculates the local vertices
    row by row (one row = one parent configuration) and performs
    the Cartesian product to obtain all combinations of complete CPTs
    compatible with that variable.

    Returns: 1D array generator (flattened CPT, ready for `fillWith`)
    """
    cpt_min_arr = get_tabular_cpt(cpt_min[:])
    cpt_max_arr = get_tabular_cpt(cpt_max[:])

    row_vertices = []
    for row in range(cpt_min_arr.shape[0]):
        v = vertices_cset(cpt_min_arr[row, :], cpt_max_arr[row, :])
        v = np.atleast_2d(v)
        row_vertices.append(v)

    n_combos = 1
    for v in row_vertices:
        n_combos *= v.shape[0]

    for combo in itertools.product(*row_vertices):
        yield np.array(combo).flatten(), n_combos

# Get (a subset of) all vertices of a CN's strong extension
def vertices_cn(bn_min, bn_max, n_bns=None, seed=42, verbose=False):
    """
    Samples (or generates ALL) the exact BNs obtained by combining the
    local vertices of each CPT.

    n_bns : int  -> samples `n_bns` random combinations (no guarantee of no repetition
                    between them, independent sampling index by index)
            None -> generates the EXHAUSTIVE enumeration of all combinations
                    (exact superset of the vertices of the strong extension)
    """
    dag = gum.BayesNet(bn_min)

    # bn.names() returns a Python set, whose iteration order is randomized
    # per-process (via PYTHONHASHSEED) -- sort for reproducibility, since
    # the `seed` argument below is otherwise silently ineffective across
    # process boundaries (e.g. a multiprocessing worker).
    names = sorted(dag.names())

    if n_bns is None:

        # --- All combinations ---
        var_cpt_combos = {
        var: [c[0] for c in vertices_cpt(bn_min.cpt(var), bn_max.cpt(var))]
        for var in dag.names()
    }
        total_combos = 1
        for var in names:
            total_combos *= len(var_cpt_combos[var])

        if verbose:
            print(f"[sample_extreme_bns] n_bns=None: Generating all "
                  f"{total_combos} combinations.", flush=True)

        bns = []
        for selection in itertools.product(*[range(len(var_cpt_combos[v])) for v in names]):
            bn = gum.BayesNet(dag)
            for var, idx in zip(names, selection):
                bn.cpt(var).fillWith(var_cpt_combos[var][idx])
            bns.append(bn)
        return bns

    else:
        # --- Random combinations ---
        rng = np.random.default_rng(seed)

        var_row_vertices = {}
        for var in names:
            cpt_min_arr = get_tabular_cpt(bn_min.cpt(var)[:])
            cpt_max_arr = get_tabular_cpt(bn_max.cpt(var)[:])
            rows = []
            for row in range(cpt_min_arr.shape[0]):
                v = np.atleast_2d(vertices_cset(cpt_min_arr[row, :], cpt_max_arr[row, :]))
                rows.append(v)
            var_row_vertices[var] = rows

        bns = []
        for _ in range(n_bns):
            bn = gum.BayesNet(dag)
            for var in names:
                rows = var_row_vertices[var]
                cpt_vec = np.concatenate([
                    rows[row][rng.integers(rows[row].shape[0])]
                    for row in range(len(rows))
                ])
                bn.cpt(var).fillWith(cpt_vec)
            bns.append(bn)
        return bns
    
# BNs sampler from a CN
def sample_from_cn(bn_min, bn_max, n_bns: int) -> list:

    # Get the DAG and extreme BNs
    dag = gum.BayesNet(bn_min)

    # bn.names() returns a Python set, whose iteration order is randomized
    # per-process (via PYTHONHASHSEED); sort so each variable consistently
    # gets the SAME seed_offset (and hence the same hopsy sampling seed,
    # via sample_from_cpts's `hash((seed_offset, row))`) regardless of which
    # process runs this -- otherwise two runs of the identical (bn_min,
    # bn_max) credal set silently produce different samples whenever run in
    # separate processes (e.g. multiprocessing workers), even though every
    # seed involved is nominally deterministic.
    names = sorted(dag.names())

    # For each variable ...
    cpts_dict = {}
    for i, var in enumerate(names):

        # ... sample `n_bns` CPTs from the CN
        cpts_dict[var] = sample_from_cpts(bn_min.cpt(var), bn_max.cpt(var), n_bns, seed_offset=i)

    # For each sample ...
    bns = []
    for i in range(n_bns):

        # ... init an empty BN ...
        bn = gum.BayesNet(dag)

        # ... and fill its CPTs
        for var in names:
            bn.cpt(var).fillWith(cpts_dict[var][i])

        bns.append(bn)

        # Debug
        safe_assert(check_consistency(bn, bn_min, bn_max) == 0)

    # Debug
    safe_assert(len(cpts_dict) == len(names))
    safe_assert(len(bns) == n_bns)

    return bns


# Sample from two extreme CPTs
def sample_from_cpts(cpt_min, cpt_max, n_bns, seed_offset = 0) -> list:

    # Transform CPTs into pandas dataframes
    cpt_min = get_tabular_cpt(cpt_min)
    cpt_max = get_tabular_cpt(cpt_max)

    # For each row in the CPT ...
    credal_dict = {}
    for row in range(cpt_min.shape[0]):

        # ... sample `n_bns` points from the credal set
        credal_dict[row] = sample_from_cset(cpt_min[row, :], cpt_max[row, :], n_bns, seed=hash((seed_offset, row)) % (2**32))

    # For each sample ...
    cpt_samples = []
    for i in range(n_bns):

        # ... build the CPT
        cpt = []
        for row in range(cpt_min.shape[0]):
            cpt.append(credal_dict[row][i])

        cpt = np.array(cpt).flatten()
        cpt_samples.append(cpt)

    # Debug
    safe_assert(cpt_min.shape == cpt_max.shape)
    safe_assert(len(credal_dict) == cpt_min.shape[0])
    safe_assert(len(cpt_samples) == n_bns)

    return cpt_samples


# Sample from a credal set K(x | pi_x), i.e., a constrained polytope.
def sample_from_cset(vec_min, vec_max, n_bns, seed = 42) -> list:
    """
    We assume a credal set is a polytope in a space of #X parameters, defined by a:
     - Multi-dimensional rectangle, i.e., inequality constraint Ax <= b, and
     - Hyperplane (provided all the variables sum up to 1), i.e., equality constraint A_eq x = b_eq.
    This is true if the CN has been learnt by local IDM, for instance.
    """

    # Degenerate case
    if np.all(vec_min == vec_max): 
        return [vec_min]*n_bns

    # Define the rectangle
    n_par = len(vec_min)
    A = np.concatenate((np.eye(n_par), -np.eye(n_par)), axis=0)
    b = np.concatenate((vec_max, -vec_min))
    rectangle = hopsy.Problem(A=A, b=b)

    # Define the hyperplane
    A_eq = np.array([np.ones(n_par)])
    b_eq = np.array([1.0])

    # Define the polytope as a constrained rectangle (i.e., get the H-representation of the credal set)
    constrained_rectangle = hopsy.add_equality_constraints(
        rectangle, A_eq=A_eq, b_eq=b_eq
    )

    # Sample from the polytope
    mc = hopsy.MarkovChain(constrained_rectangle)
    rng = hopsy.RandomNumberGenerator(seed)
    _, constrained_samples = hopsy.sample(mc, rng, n_bns, thinning=10)
    constrained_samples = constrained_samples[0]

    # Debug
    safe_assert(np.all(vec_min <= vec_max))
    safe_assert(n_par == len(vec_max))
    safe_assert(n_par == A.shape[1])
    safe_assert(n_par == A_eq.shape[1])
    safe_assert(len(constrained_samples) == n_bns)
    for i in constrained_samples:
        safe_assert(len(i) == n_par)

    return constrained_samples


# Check the consistency of a BN as sampled from a CN. Returns the number of inconsistent CPTs.
def check_consistency(bn, bn_min, bn_max, verbose=False) -> int:

    n_issues = 0

    for var in bn.names():
        bn_cpt = get_tabular_cpt(bn.cpt(var))
        bn_min_cpt = get_tabular_cpt(bn_min.cpt(var))
        bn_max_cpt = get_tabular_cpt(bn_max.cpt(var))

        # Check if probabilities sum to 1
        sum_vec = np.sum(bn_cpt, axis=1)
        probability_consistency = np.all(np.abs(sum_vec - 1) < 1e-5)

        # Check if the BN CPT is >= min CPT
        min_consistency = np.all(bn_cpt - bn_min_cpt >= -1e-5)

        # Check if the BN CPT is <= max CPT
        max_consistency = np.all(bn_cpt - bn_max_cpt <= 1e-5)

        consistency = probability_consistency and min_consistency and max_consistency

        if consistency:
            continue
        else:
            n_issues += 1
            if verbose:
                print("Variable: ", var)
                print("probability_consistency: ", probability_consistency)
                print("min_consistency: ", min_consistency)
                print("max_consistency: ", max_consistency)
                print("BN CPT: ")
                print(bn_cpt)
                print("BN min CPT: ")
                print(bn_min_cpt)
                print("BN max CPT: ")
                print(bn_max_cpt)

    return n_issues


# Extract BN min and BN max from a CN
def get_min_max_bns(cn, exp: str = ""):

    with TemporaryDirectory() as tmp_path:
        cn.saveBNsMinMax(f"{tmp_path}/bn_min_{exp}.bif", f"{tmp_path}/bn_max_{exp}.bif")
        bn_min = gum.loadBN(f"{tmp_path}/bn_min_{exp}.bif")
        bn_max = gum.loadBN(f"{tmp_path}/bn_max_{exp}.bif")

    return bn_min, bn_max


# Generate a random CN with local IDM
def generate_random_cn(n_nodes, edge_density, n_modmax, ess, s_size) -> tuple:

    # Generate a BN
    bn_gen = gum.BNGenerator()
    bn = bn_gen.generate(
        n_nodes=n_nodes, n_arcs=int(n_nodes * edge_density), n_modmax=n_modmax
    )

    # Generate data
    data_gen = gum.BNDatabaseGenerator(bn)
    data_gen.drawSamples(s_size)
    data = data_gen.to_pandas()

    # Learn the CN by local IDM
    bn_counts = get_bn_counts(bn, data)
    cn = gum.CredalNet(bn_counts)
    cn.idmLearning(ess)

    return cn, data


# Get a value from a BN's CPT
def cpt_value(
    bn: gum.BayesNet, x_var: str, x_value: float, parents: dict = None
) -> float:
    """
    Get P(X=x | parents) from the BN's CPT of X.
    `x_var` is the X name, while `x_value` is x.
    """

    cpt = bn.cpt(x_var)
    inst = gum.Instantiation(cpt)
    inst[x_var] = x_value

    if parents:
        for var in parents.keys():
            inst[var] = parents[var]
        safe_assert(bn.parents(x_var) == set(bn.ids(parents.keys())))
    else:
        safe_assert(len(bn.parents(x_var)) == 0)

    return max(cpt.get(inst), 1e-10)  # Smoothing


# Learn BN parameters from a given BN and data
def learn_bn_params(bn, data):

    bn_copy = gum.BayesNet(bn)

    learner = gum.BNLearner(data, bn_copy)
    learner.useSmoothingPrior(1e-6)
    bn_learnt = learner.learnParameters(bn_copy)

    return bn_learnt

# Extract a subgraph from a given BN
def get_subgraph(bn, vars_to_keep: set):

    sub = gum.BayesNet(bn)
    for var in set(bn.names()) - vars_to_keep:
        sub.erase(var)

    return sub
