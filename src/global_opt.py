"""
Server-side global optimization phase (cap6_extract.tex, Sec. "Global
Optimization", Eq. eq:global_opt), solved independently per mechanism X|pi_X
(the credal network is locally and separately specified).

For a binary variable, a CPT row's credal set is a 1-D interval, so Helly's
theorem applies: a set of intervals shares a common point iff every pair of
them does. This makes the alpha term ("maximize sum(delta_ij) subject to
delta=1 implying a shared theta") solvable as an exact correlation-clustering
MILP (cluster_milp): the transitivity constraints plus pairwise-intersection
bounds force every cluster to be a clique, which by Helly has a non-empty
joint intersection. Plain connected components lack this guarantee (three
intervals can be pairwise-chained without a common point), so are not used.

The full objective (alpha and beta both positive) is solved as the
LEXICOGRAPHIC limit alpha >> beta, not a literal weighted sum: first
maximize the alpha term alone, then, only among tied-optimal partitions,
break ties by the beta term (summed entropy of each cluster's
representative, see cluster_milp_lexicographic). This matches
cap6_extract.tex's own prose ("the entropy term selects, among all possible
parameter sets, the least committal ones") and avoids having to justify a
numeric alpha/beta ratio. Ties are not a vacuous edge case: they occur in
7-16% of mechanisms on real Cancer-network data, driven by the discrete
graph structure (e.g. a "hub" client whose credal set intersects two
mutually non-intersecting others), not by floating-point coincidence.
"""

import itertools

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from src.utils import credal_sets_intersect, get_tabular_cpt, jsd, maxent_cset


# Per-row ground-truth cluster labels from the clients' perturbation masks
# (see perturb_bn_params): mask=1 clients all share bn_base's theta (one
# cluster); each mask=0 client got its own independent draw (singleton).
def ground_truth_labels(mask_row: np.array) -> np.array:
    mask_row = np.asarray(mask_row, dtype=bool)
    labels = np.arange(1, len(mask_row) + 1)
    labels[mask_row] = 0
    return labels


# Pairwise credal-set intersection graph for one mechanism row, across E
# clients' (cpt_min, cpt_max) rows (already indexed to the row of interest).
def intersection_graph(rows_min: list, rows_max: list) -> np.array:
    E = len(rows_min)
    edge = np.zeros((E, E), dtype=bool)
    for i, j in itertools.combinations(range(E), 2):
        does_intersect = credal_sets_intersect(
            rows_min[i], rows_max[i], rows_min[j], rows_max[j]
        )
        edge[i, j] = edge[j, i] = does_intersect
    return edge


# Soft alternative to the hard intersection test: the minimum JS divergence
# between any point of one credal set and any point of the other (0 exactly
# when they intersect). Closed-form for binary X, since JSD between two
# Bernoulli distributions is monotonic in |p-q|.
def credal_jsd_distance(
    cpt_min_1: np.array, cpt_max_1: np.array, cpt_min_2: np.array, cpt_max_2: np.array
) -> float:
    if credal_sets_intersect(cpt_min_1, cpt_max_1, cpt_min_2, cpt_max_2):
        return 0.0
    if cpt_max_1[0] < cpt_min_2[0]:
        p, q = cpt_max_1[0], cpt_min_2[0]
    else:
        p, q = cpt_max_2[0], cpt_min_1[0]
    return float(jsd([p, 1 - p], [q, 1 - q]))


# Pairwise JSD-distance matrix for one mechanism row, across E clients.
def jsd_distance_matrix(rows_min: list, rows_max: list) -> np.array:
    E = len(rows_min)
    dist = np.zeros((E, E))
    for i, j in itertools.combinations(range(E), 2):
        d = credal_jsd_distance(rows_min[i], rows_max[i], rows_min[j], rows_max[j])
        dist[i, j] = dist[j, i] = d
    return dist


# Clusters E clients from a JSD-distance matrix given the true cluster
# count k as an oracle. Complete linkage keeps every cluster mutually
# close, avoiding the "chain" pathology single linkage would allow.
def cluster_jsd_hierarchical(dist: np.array, k: int) -> np.array:
    E = dist.shape[0]
    if k <= 1 or E <= 1:
        return np.zeros(E, dtype=int)
    k = min(k, E)
    model = AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="complete"
    )
    return model.fit_predict(dist)


# Builds the correlation-clustering MILP shared by cluster_milp and
# is_clustering_optimum_unique: binary z_ij (one per client pair, "same
# cluster"), upper-bounded by the intersection graph (z_ij<=1 only where
# credal sets actually intersect) and constrained to be transitive.
def _build_correlation_milp(edge: np.array):
    E = edge.shape[0]
    pairs = list(itertools.combinations(range(E), 2))
    P = len(pairs)
    pair_idx = {p: i for i, p in enumerate(pairs)}

    c = -np.ones(P)  # maximize sum(z) == minimize -sum(z)
    ub = np.array([1.0 if edge[i, j] else 0.0 for (i, j) in pairs])
    bounds = Bounds(lb=np.zeros(P), ub=ub)
    integrality = np.ones(P)

    rows = []
    for i, j, k in itertools.combinations(range(E), 3):
        zij, zjk, zik = pair_idx[(i, j)], pair_idx[(j, k)], pair_idx[(i, k)]
        for a, b, out in ((zij, zjk, zik), (zij, zik, zjk), (zjk, zik, zij)):
            row = np.zeros(P)
            row[a], row[b], row[out] = 1.0, 1.0, -1.0
            rows.append(row)

    constraints = [LinearConstraint(np.array(rows), -np.inf, 1.0)] if rows else []
    return c, bounds, integrality, constraints, pairs


# Exact MILP solution for "maximize sum(delta_ij) s.t. delta=1 implies a
# common theta": correlation clustering restricted to pairs that intersect,
# with transitivity constraints. Every cluster is then a genuine Helly clique.
def cluster_milp(edge: np.array) -> np.array:
    E = edge.shape[0]
    if E < 2:
        return np.zeros(E, dtype=int)

    c, bounds, integrality, constraints, pairs = _build_correlation_milp(edge)
    res = milp(c, integrality=integrality, bounds=bounds, constraints=constraints)
    if not res.success:
        raise RuntimeError(f"cluster_milp: MILP solve failed ({res.message}).")
    z = np.round(res.x).astype(bool)

    return _labels_from_pairs(E, pairs, z)


# Full two-stage lexicographic procedure (see module docstring): matches
# cluster_milp's optimum, then picks the tied partition (enumerated exactly
# via no-good cuts, capped at max_ties) maximizing total_entropy.
def cluster_milp_lexicographic(
    edge: np.array, rows_min: list, rows_max: list, max_ties: int = 200
) -> np.array:
    E = edge.shape[0]
    if E < 3:
        return cluster_milp(edge)  # 0 or 1 pairs: no possible tie

    c, bounds, integrality, constraints, pairs = _build_correlation_milp(edge)
    cur_constraints = list(constraints)
    candidates = []

    res = milp(c, integrality=integrality, bounds=bounds, constraints=cur_constraints)
    if not res.success:
        raise RuntimeError(f"cluster_milp_lexicographic: MILP solve failed ({res.message}).")
    obj_star = -res.fun

    P = len(pairs)
    while res.success and -res.fun >= obj_star - 1e-9 and len(candidates) < max_ties:
        z = np.round(res.x).astype(bool)
        candidates.append(z)

        cut_row = np.where(z, 1.0, -1.0)
        cut_rhs = float(np.sum(z)) - 1.0
        cur_constraints = cur_constraints + [
            LinearConstraint(cut_row.reshape(1, P), -np.inf, cut_rhs)
        ]
        res = milp(c, integrality=integrality, bounds=bounds, constraints=cur_constraints)

    if len(candidates) == 1:
        return _labels_from_pairs(E, pairs, candidates[0])

    best_labels, best_entropy = None, -np.inf
    for z in candidates:
        labels = _labels_from_pairs(E, pairs, z)
        entropy = total_entropy(rows_min, rows_max, labels)
        if entropy > best_entropy:
            best_entropy, best_labels = entropy, labels
    return best_labels


# Whether cluster_milp's solution is the unique maximizer of sum(delta_ij),
# checked by re-solving with a no-good cut excluding it and comparing
# objectives. Standalone diagnostic; cluster_milp_lexicographic does this
# itself rather than calling it.
def is_clustering_optimum_unique(edge: np.array) -> bool:
    E = edge.shape[0]
    if E < 3:
        return True  # 0 or 1 pairs: nothing to tie on

    c, bounds, integrality, constraints, pairs = _build_correlation_milp(edge)
    first = milp(c, integrality=integrality, bounds=bounds, constraints=constraints)
    if not first.success:
        raise RuntimeError(f"is_clustering_optimum_unique: MILP solve failed ({first.message}).")
    z_first = np.round(first.x)
    obj_first = -first.fun

    # No-good cut: forbid exactly reproducing z_first (excludes only that
    # single vertex, not every solution with the same objective value).
    P = len(pairs)
    cut_row = np.where(z_first > 0.5, 1.0, -1.0)
    cut_rhs = float(np.sum(z_first > 0.5)) - 1.0
    constraints = list(constraints) + [LinearConstraint(cut_row.reshape(1, P), -np.inf, cut_rhs)]

    second = milp(c, integrality=integrality, bounds=bounds, constraints=constraints)
    if not second.success:
        return True  # no other feasible solution at all -> unique
    obj_second = -second.fun
    return obj_second < obj_first - 1e-9


def _binary_entropy(p: float, eps: float = 1e-12) -> float:
    p = min(max(p, eps), 1 - eps)
    return float(-p * np.log(p) - (1 - p) * np.log(1 - p))


# Per-cluster max-entropy representative (beta term's theta^i): the point
# closest to 0.5 inside the cluster's joint intersection [L,R], guaranteed
# non-empty by Helly whenever `labels` came from cluster_milp.
def cluster_representative_thetas(rows_min: list, rows_max: list, labels: np.array) -> dict:
    reps = {}
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        L = max(rows_min[i][0] for i in idx)
        R = min(rows_max[i][0] for i in idx)
        reps[lab] = maxent_cset(np.array([L, 1 - R]), np.array([R, 1 - L]))
    return reps


# sum_i H(theta^i), the beta term: every CLIENT contributes its own
# cluster's representative entropy once, so a cluster of size k contributes
# k times that value (the sum is over clients, not over distinct clusters).
def total_entropy(rows_min: list, rows_max: list, labels: np.array) -> float:
    reps = cluster_representative_thetas(rows_min, rows_max, labels)
    return sum(_binary_entropy(reps[lab][0]) for lab in labels)


# Connected components of the "z=1" edges give cluster labels directly: the
# transitivity constraints in cluster_milp make same-label membership a
# genuine equivalence relation (no risk of a non-clique component here,
# unlike running connected components on the raw intersection graph).
def _labels_from_pairs(E: int, pairs: list, same: np.array) -> np.array:
    parent = list(range(E))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for (i, j), s in zip(pairs, same):
        if s:
            union(i, j)

    roots = np.array([find(i) for i in range(E)])
    # Relabel roots to consecutive integers for a tidy, deterministic output.
    _, labels = np.unique(roots, return_inverse=True)
    return labels


# Naive point-estimate baseline: exact 1-D k-means (WCSS-optimal) over the
# clients' scalar MLEs, given the true cluster count k as an oracle (a
# conservative advantage MOSAIC itself does not get). Plain O(n^2 k) DP,
# not the "cut the k-1 largest gaps" heuristic (see test_global_opt.py).
def cluster_1d_wcss_optimal(values: np.array, k: int) -> np.array:
    values = np.asarray(values, dtype=float)
    n = len(values)

    if k <= 1 or n <= 1:
        return np.zeros(n, dtype=int)
    k = min(k, n)

    order = np.argsort(values)
    v = values[order]

    # Prefix sums for O(1) segment WCSS: WCSS(v[i:j]) = SS[j]-SS[i] -
    # (S[j]-S[i])^2 / (j-i).
    S = np.concatenate(([0.0], np.cumsum(v)))
    SS = np.concatenate(([0.0], np.cumsum(v**2)))

    def seg_wcss(i, j):
        s, ss = S[j] - S[i], SS[j] - SS[i]
        return ss - s * s / (j - i)

    # dp[j] = optimal WCSS partitioning v[:j] into `c` clusters (rolling over
    # c=1..k); back[c][j] = the last cluster's start index, for backtracking
    # the cut points once c reaches k.
    dp = [0.0] + [seg_wcss(0, j) for j in range(1, n + 1)]  # c=1
    back = [[0] * (n + 1)]
    for c in range(2, k + 1):
        new_dp = [np.inf] * (n + 1)
        new_back = [0] * (n + 1)
        for j in range(c, n + 1):
            best_i, best_val = None, np.inf
            for i in range(c - 1, j):
                val = dp[i] + seg_wcss(i, j)
                if val < best_val:
                    best_val, best_i = val, i
            new_dp[j], new_back[j] = best_val, best_i
        dp, back = new_dp, back + [new_back]

    # Backtrack the k cluster boundaries from n down to 0.
    bounds = [n]
    j = n
    for c in range(k, 0, -1):
        i = back[c - 1][j] if c > 1 else 0
        bounds.append(i)
        j = i
    bounds = sorted(set(bounds))

    labels_sorted = np.zeros(n, dtype=int)
    for lab, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        labels_sorted[a:b] = lab

    labels = np.empty(n, dtype=int)
    labels[order] = labels_sorted
    return labels


# Pairwise same/different confusion counts (TP/FP/FN/TN) for one mechanism,
# over all client pairs. Summed (not averaged) across mechanisms before
# computing P/R/F1: a network-wide MICRO average over individual pairs.
def pairwise_confusion(true_labels: np.array, pred_labels: np.array) -> tuple:
    E = len(true_labels)
    iu = np.triu_indices(E, k=1)

    t = true_labels[:, None] == true_labels[None, :]
    p = pred_labels[:, None] == pred_labels[None, :]
    t, p = t[iu], p[iu]

    tp = int(np.sum(t & p))
    fp = int(np.sum(~t & p))
    fn = int(np.sum(t & ~p))
    tn = int(np.sum(~t & ~p))
    return tp, fp, fn, tn


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple:
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else float("nan")
    )
    return precision, recall, f1


# ARI/AMI are properties of a pair of partitions of the SAME item set, so
# (unlike pairwise P/R/F1) they cannot be pooled across mechanisms: each
# mechanism has its own true partition. Computed per mechanism, then meant
# to be MACRO-averaged (plain mean) across mechanisms by the caller.
def ari_ami(true_labels: np.array, pred_labels: np.array) -> tuple:
    return (
        adjusted_rand_score(true_labels, pred_labels),
        adjusted_mutual_info_score(true_labels, pred_labels),
    )


# Convenience: the full per-mechanism-row picture (ground truth, predicted
# clusters via the full lexicographic procedure, edge graph) for one
# (var, row) from a given credal-set source ("cn" for local IDM,
# "cn_mosaic" for the MOSAIC-updated set), across all E clients.
def evaluate_mechanism(clients: dict, var: str, row: int, cn_attr: str) -> dict:
    E = len(clients)
    rows_min, rows_max = [], []
    for e in range(E):
        cpt_min, cpt_max = getattr(clients[e], cn_attr).cpt(var)
        rows_min.append(cpt_min[row, :])
        rows_max.append(cpt_max[row, :])

    edge = intersection_graph(rows_min, rows_max)
    pred_labels = cluster_milp_lexicographic(edge, rows_min, rows_max)

    mask_row = np.array(
        [get_tabular_cpt(clients[e].mask.cpt(var))[row, 0] for e in range(E)]
    )
    true_labels = ground_truth_labels(mask_row)

    return {"true_labels": true_labels, "pred_labels": pred_labels, "edge": edge}


# Same as evaluate_mechanism, but for the naive MLE point-estimate baseline
# (see cluster_1d_wcss_optimal): no credal sets involved, just each client's
# own exact MLE for this row's category 0.
def evaluate_mechanism_mle(clients: dict, var: str, row: int) -> dict:
    E = len(clients)
    values = np.array(
        [get_tabular_cpt(clients[e].bn_mle.cpt(var))[row, 0] for e in range(E)]
    )
    mask_row = np.array(
        [get_tabular_cpt(clients[e].mask.cpt(var))[row, 0] for e in range(E)]
    )
    true_labels = ground_truth_labels(mask_row)
    k_true = len(np.unique(true_labels))
    pred_labels = cluster_1d_wcss_optimal(values, k_true)

    return {"true_labels": true_labels, "pred_labels": pred_labels}


# Iterates every (var, row) mechanism of the network once; the caller
# supplies which client and which row-count-per-var (both cheaply derived
# from any one client's BN, since the graph structure is shared).
def all_mechanisms(bn) -> list:
    from src.utils import get_parent_confs

    mechanisms = []
    for var in bn.names():
        n_rows = len(get_parent_confs(bn, var))
        for row in range(n_rows):
            mechanisms.append((var, row))
    return mechanisms
