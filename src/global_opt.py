"""
Server-side global optimization phase (cap6_extract.tex, Sec. "Global
Optimization", Eq. eq:global_opt), restricted to alpha=1, beta=0: the
entropy term drops out and the problem becomes, for each mechanism X|pi_X
independently, finding the delta^{ij} that maximize the number of
same-cluster client pairs subject to feasibility (delta^{ij}=1 requires a
common theta^i=theta^j inside both clients' credal sets).

For a binary variable (every variable in the Cancer network), a single
CPT row's credal set is a scalar interval on category 0, so pairwise
"do these two credal sets intersect" checks are genuine 1-D interval
overlaps. Helly's theorem for 1-D intervals then says: a set of intervals
has a common point iff every pair in it does. This is exactly what makes
"maximize sum(delta_ij) subject to delta=1 implying a shared theta" solvable
as an exact correlation-clustering MILP (cluster_milp below): the
transitivity constraints, combined with the pairwise-intersection upper
bounds, force every resulting cluster to be a clique in the intersection
graph, which by Helly is exactly a group with a non-empty joint
intersection. Connected components of the raw intersection graph do NOT
have this guarantee (three intervals can be pairwise-chained without a
common point: A-B and B-C intersect, A-C does not), so are not used here.
"""

import itertools

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from src.utils import credal_sets_intersect, get_tabular_cpt, jsd


# Per-row (X|pi_X) ground-truth cluster labels from the E clients' own
# perturbation masks (see perturb_bn_params/init_clients): mask[e]=1 means
# client e's row was left identical to bn_base, mask[e]=0 means it was
# redrawn from an independent, continuous Dirichlet. Two clients therefore
# share the exact same theta iff both kept the row (mask=1); any client
# with mask=0 is (almost surely, by construction) different from every
# other client, including other mask=0 ones, since each was perturbed with
# its own independent draw. This gives a genuine equivalence relation with
# no clustering needed: one cluster for all mask=1 clients, one singleton
# cluster per mask=0 client.
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


# Soft alternative to the hard 0/1 intersection test: the MINIMUM JS
# divergence between any point of one credal set and any point of the
# other (a Hausdorff-style "closest approach" distance between the two
# sets). Exactly 0 when the sets intersect -- so this is a strict
# generalization of credal_sets_intersect's boolean, not a different
# criterion -- and, for disjoint sets, grows continuously with how far
# apart they are, instead of collapsing every non-overlap to the same "no".
# Only implemented for binary X (as elsewhere in this module): JSD between
# two Bernoulli distributions is monotonic in |p-q|, so the minimizing pair
# is always the two sets' closest boundary points, giving an O(1) closed
# form (no optimization needed).
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


# Clusters E clients from a pairwise JSD-distance matrix, given the
# ground-truth cluster count k as an oracle (same convention as
# cluster_1d_wcss_optimal: an advantage MOSAIC does NOT get, so the
# comparison stays conservative). COMPLETE linkage, not average/single: a
# cluster should only ever contain clients that are ALL mutually close
# (bounded worst-case pairwise distance), which is the natural
# distance-based analogue of cluster_milp's clique requirement -- single
# linkage would reintroduce exactly the "chain" pathology that
# cluster_connected_components has (see its own docstring).
def cluster_jsd_hierarchical(dist: np.array, k: int) -> np.array:
    E = dist.shape[0]
    if k <= 1 or E <= 1:
        return np.zeros(E, dtype=int)
    k = min(k, E)
    model = AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="complete"
    )
    return model.fit_predict(dist)


# Exact solution to "maximize sum(delta_ij) s.t. delta=1 implies a common
# theta", for a binary-variable mechanism: correlation clustering restricted
# to only merge pairs that intersect (z_ij <= edge_ij), with the standard
# transitivity constraints. Combined with Helly's theorem (see module
# docstring), every resulting cluster is guaranteed to have a genuinely
# common point, unlike a plain connected-components heuristic.
def cluster_milp(edge: np.array) -> np.array:
    E = edge.shape[0]
    pairs = list(itertools.combinations(range(E), 2))
    P = len(pairs)
    pair_idx = {p: i for i, p in enumerate(pairs)}

    if P == 0:
        return np.zeros(0, dtype=int)

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

    constraints = None
    if rows:
        A = np.array(rows)
        constraints = LinearConstraint(A, -np.inf, 1.0)

    res = milp(c, integrality=integrality, bounds=bounds, constraints=constraints)
    if not res.success:
        raise RuntimeError(f"cluster_milp: MILP solve failed ({res.message}).")
    z = np.round(res.x).astype(bool)

    return _labels_from_pairs(E, pairs, z)


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


# Naive point-estimate baseline: exact 1-D k-means clustering of the E
# clients' scalar MLEs, given the ground-truth cluster count k as an oracle
# (an advantage MOSAIC does NOT get, so the comparison stays conservative,
# i.e. never biased in MOSAIC's favor).
#
# For 1-D points, SOME optimal k-way partition under within-cluster sum of
# squares (WCSS) is always contiguous once points are sorted (a classical
# fact underlying "Ckmeans.1d.dp"/Fisher's algorithm) -- but WHICH
# contiguous split is optimal is not simply "cut the k-1 largest gaps": an
# earlier version of this function used that heuristic and a brute-force
# check (comparing it against exhaustively enumerated contiguous splits on
# thousands of random small inputs) found it strictly WCSS-suboptimal in
# ~20% of cases (largest-gap cutting exactly solves a DIFFERENT criterion,
# maximizing the smallest inter-cluster gap). This is a plain O(n^2 k) DP
# instead: exact, and more than fast enough for the handful of clients used
# here.
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

    # dp[j] = optimal WCSS partitioning v[:j] into `c` clusters (rolling
    # over c=1..k); back[c][j] = the last cluster's start index, for
    # backtracking the cut points once c reaches k. dp[0] (the empty
    # prefix) is unused by any later c>=2 lookup (those only ever read
    # dp[i] for i>=c-1>=1) but is set to 0 rather than seg_wcss(0, 0)
    # (0/0) to avoid a spurious divide-by-zero warning.
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
# vectorized over all C(E,2) client pairs. Meant to be accumulated (summed)
# across every mechanism in the network before computing precision/recall/F1,
# i.e. a network-wide MICRO average: pairwise P/R/F1 are properties of
# individual client pairs, and every mechanism contributes the same number
# of pairs (E is fixed), so pooling raw counts first is the natural choice.
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
# clusters, edge graph) for one (var, row) from a given credal-set source
# ("cn" for local IDM, "cn_mosaic" for the MOSAIC-updated set), across all
# E clients.
def evaluate_mechanism(clients: dict, var: str, row: int, cn_attr: str) -> dict:
    E = len(clients)
    rows_min, rows_max = [], []
    for e in range(E):
        cpt_min, cpt_max = getattr(clients[e], cn_attr).cpt(var)
        rows_min.append(cpt_min[row, :])
        rows_max.append(cpt_max[row, :])

    edge = intersection_graph(rows_min, rows_max)
    pred_labels = cluster_milp(edge)

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
