"""
Tests for src/global_opt.py: the global optimization phase restricted to
alpha=1, beta=0 (pure clustering of clients per mechanism), see the
module's own docstring for the Helly's-theorem argument behind cluster_milp.
"""

import numpy as np
import pyagrum as gum
import pytest

from src.global_opt import (all_mechanisms, ari_ami, cluster_1d_wcss_optimal,
                            cluster_jsd_hierarchical, cluster_milp,
                            credal_jsd_distance, evaluate_mechanism,
                            evaluate_mechanism_mle, ground_truth_labels,
                            intersection_graph, jsd_distance_matrix,
                            pairwise_confusion, precision_recall_f1)
from src.mosaic import CN, CN_CPT, Client
from src.utils import get_cpt_shape, jsd

# ground_truth_labels ---------------------------------------------------


def test_ground_truth_labels_all_kept():
    mask = np.array([1, 1, 1, 1])
    labels = ground_truth_labels(mask)
    assert np.all(labels == 0)


def test_ground_truth_labels_all_perturbed_are_singletons():
    mask = np.array([0, 0, 0])
    labels = ground_truth_labels(mask)
    assert len(set(labels)) == 3  # every client its own cluster


def test_ground_truth_labels_mixed():
    mask = np.array([1, 1, 0, 1, 0])
    labels = ground_truth_labels(mask)
    # Clients 0,1,3 (mask=1) share one cluster; clients 2,4 are singletons,
    # each distinct from everyone else (including each other).
    assert labels[0] == labels[1] == labels[3]
    assert labels[2] != labels[0]
    assert labels[4] != labels[0]
    assert labels[2] != labels[4]


# intersection_graph ------------------------------------------------------


def test_intersection_graph_matches_pairwise_overlap():
    # Category-0 segments [0.1,0.3], [0.2,0.4], [0.6,0.8]; category 1 is
    # each one's mirror image (1 - category 0), as a real binary CPT row.
    rows_min = [np.array([0.1, 0.7]), np.array([0.2, 0.6]), np.array([0.6, 0.2])]
    rows_max = [np.array([0.3, 0.9]), np.array([0.4, 0.8]), np.array([0.8, 0.4])]
    edge = intersection_graph(rows_min, rows_max)

    assert edge[0, 1] and edge[1, 0]  # [0.1,0.3] and [0.2,0.4] overlap
    assert not edge[0, 2]  # [0.1,0.3] and [0.6,0.8] don't
    assert not edge[1, 2]  # [0.2,0.4] and [0.6,0.8] don't
    assert not np.any(np.diag(edge))


# cluster_milp: the central correctness property (Helly-consistent
# clustering, see module docstring) --------------------------------------


def test_cluster_milp_full_triangle_merges_all():
    # All 3 pairwise intersect -> by Helly, a common point exists -> merge.
    edge = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
    labels = cluster_milp(edge)
    assert labels[0] == labels[1] == labels[2]


def test_cluster_milp_chain_does_not_over_merge():
    # A-B and B-C intersect, A-C does not: no common point for all three
    # (Helly), so the optimal delta merges only ONE of the two edges, never
    # all three clients together.
    edge = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    labels = cluster_milp(edge)
    assert len(set(labels.tolist())) == 2  # exactly one merge, one singleton
    # Whichever pair merged, A and C must NOT end up together (infeasible).
    assert not (labels[0] == labels[1] == labels[2])


def test_cluster_milp_no_edges_all_singletons():
    edge = np.zeros((4, 4), dtype=bool)
    labels = cluster_milp(edge)
    assert len(set(labels.tolist())) == 4


def test_cluster_milp_two_separate_cliques():
    # {0,1,2} pairwise intersect, {3,4} pairwise intersect, nothing across.
    edge = np.zeros((5, 5), dtype=bool)
    for i, j in [(0, 1), (0, 2), (1, 2), (3, 4)]:
        edge[i, j] = edge[j, i] = True
    labels = cluster_milp(edge)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]


def test_cluster_milp_maximizes_pair_count_over_greedy_alternative():
    # A "star": center 0 intersects 1,2,3, but 1,2,3 pairwise don't. Naive
    # connected components would merge all 4 (infeasible, no common point
    # for {0,1,2,3} let alone {1,2,3}); the MILP must instead pick exactly
    # one spoke to merge with the center (objective=1), since merging any
    # two spokes together (without the center) is not allowed (no edge) and
    # merging 3+ clients requires ALL pairs among them to intersect.
    edge = np.zeros((4, 4), dtype=bool)
    for j in (1, 2, 3):
        edge[0, j] = edge[j, 0] = True
    labels = cluster_milp(edge)
    counts = np.bincount(labels)
    assert sorted(counts.tolist()) == [1, 1, 2]  # one pair merged, two singletons


# credal_jsd_distance / jsd_distance_matrix / cluster_jsd_hierarchical -----


def test_credal_jsd_distance_zero_when_intersecting():
    # Category-0 segments [0.2, 0.5] and [0.4, 0.7] overlap on [0.4, 0.5].
    d = credal_jsd_distance(
        np.array([0.2, 0.5]), np.array([0.5, 0.8]), np.array([0.4, 0.3]), np.array([0.7, 0.6])
    )
    assert d == 0.0


def test_credal_jsd_distance_matches_closest_boundary_points_when_disjoint():
    min1, max1 = np.array([0.1, 0.85]), np.array([0.15, 0.9])
    min2, max2 = np.array([0.5, 0.3]), np.array([0.7, 0.5])
    d = credal_jsd_distance(min1, max1, min2, max2)
    expected = jsd([0.15, 0.85], [0.5, 0.5])  # closest points: 0.15 (max1) vs 0.5 (min2)
    assert d == pytest.approx(expected)


def test_credal_jsd_distance_monotonic_in_gap_size():
    min1, max1 = np.array([0.1, 0.85]), np.array([0.15, 0.9])
    min2_near, max2_near = np.array([0.5, 0.3]), np.array([0.7, 0.5])
    min2_far, max2_far = np.array([0.9, 0.05]), np.array([0.95, 0.1])

    d_near = credal_jsd_distance(min1, max1, min2_near, max2_near)
    d_far = credal_jsd_distance(min1, max1, min2_far, max2_far)
    assert d_far > d_near


def test_credal_jsd_distance_symmetric():
    min1, max1 = np.array([0.1, 0.85]), np.array([0.15, 0.9])
    min2, max2 = np.array([0.5, 0.3]), np.array([0.7, 0.5])
    assert credal_jsd_distance(min1, max1, min2, max2) == pytest.approx(
        credal_jsd_distance(min2, max2, min1, max1)
    )


def test_jsd_distance_matrix_diagonal_is_zero_and_symmetric():
    rows_min = [np.array([0.1, 0.85]), np.array([0.5, 0.3]), np.array([0.9, 0.05])]
    rows_max = [np.array([0.15, 0.9]), np.array([0.7, 0.5]), np.array([0.95, 0.1])]
    dist = jsd_distance_matrix(rows_min, rows_max)
    assert np.all(np.diag(dist) == 0.0)
    assert np.allclose(dist, dist.T)


def test_cluster_jsd_hierarchical_recovers_two_well_separated_pairs():
    rows_min = [
        np.array([0.1, 0.85]),
        np.array([0.12, 0.83]),
        np.array([0.6, 0.35]),
        np.array([0.62, 0.33]),
    ]
    rows_max = [
        np.array([0.15, 0.9]),
        np.array([0.17, 0.88]),
        np.array([0.65, 0.4]),
        np.array([0.67, 0.38]),
    ]
    dist = jsd_distance_matrix(rows_min, rows_max)
    labels = cluster_jsd_hierarchical(dist, k=2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_cluster_jsd_hierarchical_k1_everyone_together():
    dist = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    labels = cluster_jsd_hierarchical(dist, k=1)
    assert np.all(labels == labels[0])


# cluster_1d_wcss_optimal ---------------------------------------------------


def _wcss(values, labels):
    total = 0.0
    for lab in set(labels.tolist()):
        v = values[labels == lab]
        total += np.sum((v - v.mean()) ** 2)
    return total


def _brute_force_optimal_wcss(values, k):
    import itertools

    order = np.argsort(values)
    v = values[order]
    n = len(v)
    best = None
    for cuts in itertools.combinations(range(1, n), k - 1):
        bounds = (0,) + cuts + (n,)
        labels = np.zeros(n, dtype=int)
        for lab, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            labels[a:b] = lab
        w = _wcss(v, labels)
        if best is None or w < best:
            best = w
    return best


def test_cluster_1d_wcss_optimal_k1_everyone_together():
    values = np.array([0.1, 0.9, 0.5, 0.3])
    labels = cluster_1d_wcss_optimal(values, k=1)
    assert np.all(labels == labels[0])


def test_cluster_1d_wcss_optimal_k_equals_e_all_singletons():
    values = np.array([0.1, 0.9, 0.5, 0.3])
    labels = cluster_1d_wcss_optimal(values, k=len(values))
    assert len(set(labels.tolist())) == len(values)


def test_cluster_1d_wcss_optimal_obvious_two_pairs():
    # Two tight pairs far apart: {0.10, 0.12} and {0.80, 0.83}.
    values = np.array([0.80, 0.10, 0.83, 0.12])
    labels = cluster_1d_wcss_optimal(values, k=2)
    assert labels[0] == labels[2]  # 0.80, 0.83
    assert labels[1] == labels[3]  # 0.10, 0.12
    assert labels[0] != labels[1]


def test_cluster_1d_wcss_optimal_invariant_to_input_order():
    rng = np.random.default_rng(0)
    values = rng.uniform(size=8)
    perm = rng.permutation(8)

    labels = cluster_1d_wcss_optimal(values, k=3)
    labels_perm = cluster_1d_wcss_optimal(values[perm], k=3)

    # Same clustering, up to the permutation and up to a relabeling.
    pairs_orig = {
        (i, j) for i in range(8) for j in range(8) if labels[i] == labels[j]
    }
    pairs_perm = {
        (perm[i], perm[j])
        for i in range(8)
        for j in range(8)
        if labels_perm[i] == labels_perm[j]
    }
    assert pairs_orig == pairs_perm


def test_cluster_1d_wcss_optimal_matches_brute_force_on_random_inputs():
    # Regression test: an earlier version of this function cut the k-1
    # LARGEST GAPS between sorted values instead of solving for the true
    # WCSS optimum -- a different criterion (it exactly maximizes the
    # smallest inter-cluster gap instead), found to be strictly
    # WCSS-suboptimal in ~20% of random small inputs by this exact check.
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(4, 8))
        k = int(rng.integers(2, min(4, n)))
        values = np.round(rng.uniform(0, 10, size=n), 2)

        labels = cluster_1d_wcss_optimal(values, k)
        got = _wcss(values, labels)
        optimal = _brute_force_optimal_wcss(values, k)

        assert got <= optimal + 1e-6
        assert len(set(labels.tolist())) == k


def test_cluster_1d_wcss_optimal_beats_largest_gap_heuristic_on_known_case():
    # A concrete instance where the old largest-gap heuristic (see the
    # regression test above) was strictly worse than the true optimum.
    values = np.array([0.28, 1.24, 3.0, 4.23, 5.41, 6.71, 8.63])
    labels = cluster_1d_wcss_optimal(values, k=2)
    got = _wcss(values, labels)
    optimal = _brute_force_optimal_wcss(values, k=2)
    assert got == pytest.approx(optimal, abs=1e-6)


# pairwise_confusion / precision_recall_f1 / ari_ami -----------------------


def test_pairwise_confusion_hand_computed():
    t = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, 1])
    # Pairs (0,1)FN (0,2)TN (0,3)TN (1,2)FP (1,3)FP (2,3)TP
    tp, fp, fn, tn = pairwise_confusion(t, p)
    assert (tp, fp, fn, tn) == (1, 2, 1, 2)


def test_precision_recall_f1_hand_computed():
    precision, recall, f1 = precision_recall_f1(tp=1, fp=2, fn=1)
    assert precision == pytest.approx(1 / 3)
    assert recall == pytest.approx(1 / 2)
    assert f1 == pytest.approx(0.4)


def test_precision_recall_f1_nan_on_no_positives():
    precision, recall, f1 = precision_recall_f1(tp=0, fp=0, fn=0)
    assert np.isnan(precision) and np.isnan(recall) and np.isnan(f1)


def test_ari_ami_identical_partitions_is_one():
    labels = np.array([0, 0, 1, 1, 2])
    ari, ami = ari_ami(labels, labels)
    assert ari == pytest.approx(1.0)
    assert ami == pytest.approx(1.0)


def test_ari_ami_bounded_above_by_one():
    t = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 1, 1, 2, 2, 0])
    ari, ami = ari_ami(t, p)
    assert ari <= 1.0 + 1e-9
    assert ami <= 1.0 + 1e-9


# all_mechanisms ------------------------------------------------------------


def test_all_mechanisms_covers_every_row_of_cancer_bn():
    bn = gum.loadBN("cancer.bif")
    mechanisms = all_mechanisms(bn)

    # Cancer: P, S (roots, 1 row each), C (2 binary parents -> 4 rows),
    # X, D (1 binary parent each -> 2 rows each) = 1+1+4+2+2 = 10.
    assert len(mechanisms) == 10
    assert len(set(mechanisms)) == 10  # no duplicates
    vars_seen = {var for var, _ in mechanisms}
    assert vars_seen == set(bn.names())


# evaluate_mechanism / evaluate_mechanism_mle: small integration test ------


def _binary_client(p0_min, p0_max, mask_row_value, mle=None):
    """A single-variable ("X") client with an exact credal set [p0_min,
    p0_max] for category 0 (mirrors test_mosaic.py's _make_client_with_cset/
    _seg pattern), and a mask row fixed to `mask_row_value`.
    """
    bn = gum.fastBN("X[2]")
    bn.cpt("X").fillWith([0.5, 0.5])

    mask = gum.BayesNet(bn)
    mask.cpt("X").fillWith([mask_row_value, mask_row_value])

    c = Client(gum.BayesNet(bn), mask)

    bn_min, bn_max = gum.BayesNet(bn), gum.BayesNet(bn)
    bn_min.cpt("X").fillWith([p0_min, 1 - p0_max])
    bn_max.cpt("X").fillWith([p0_max, 1 - p0_min])

    cn = CN.__new__(CN)
    cn.cpts = {"X": CN_CPT("X", get_cpt_shape(bn.cpt("X")), (bn_min.cpt("X"), bn_max.cpt("X")))}
    cn.bn_min, cn.bn_max, cn.cn = bn_min, bn_max, None
    cn.bn_min.setProperty("name", "bn_min")
    cn.bn_max.setProperty("name", "bn_max")

    c.cn = cn
    c.cn_mosaic = cn  # same set for both sources in this simplified fixture

    if mle is None:
        mle = (p0_min + p0_max) / 2
    c.bn_mle = gum.BayesNet(bn)
    c.bn_mle.cpt("X").fillWith([mle, 1 - mle])

    return c


def test_evaluate_mechanism_matches_hand_crafted_ground_truth_and_clusters():
    # Clients 0,1 kept the baseline row (mask=1) -> ground truth: same
    # cluster. Client 2 was perturbed (mask=0) -> its own singleton,
    # EVEN THOUGH its credal set still happens to overlap client 0's.
    clients = {
        0: _binary_client(0.2, 0.3, mask_row_value=1),
        1: _binary_client(0.2, 0.3, mask_row_value=1),
        2: _binary_client(0.25, 0.4, mask_row_value=0),
    }

    result = evaluate_mechanism(clients, "X", row=0, cn_attr="cn")

    assert result["true_labels"][0] == result["true_labels"][1]
    assert result["true_labels"][2] != result["true_labels"][0]
    # Predicted: all three pairwise intersect (0-1 identical, 0-2 and 1-2
    # overlap via [0.25,0.3]), so MILP correctly merges all three too.
    assert result["pred_labels"][0] == result["pred_labels"][1] == result["pred_labels"][2]


def test_evaluate_mechanism_mle_uses_oracle_k_from_ground_truth():
    clients = {
        0: _binary_client(0.2, 0.3, mask_row_value=1, mle=0.21),
        1: _binary_client(0.2, 0.3, mask_row_value=1, mle=0.22),
        2: _binary_client(0.7, 0.8, mask_row_value=0, mle=0.75),
    }

    result = evaluate_mechanism_mle(clients, "X", row=0)

    assert result["true_labels"][0] == result["true_labels"][1]
    assert result["true_labels"][2] != result["true_labels"][0]
    # k_true=2 -> the gap-cut baseline should isolate the far-away client 2.
    assert result["pred_labels"][0] == result["pred_labels"][1]
    assert result["pred_labels"][2] != result["pred_labels"][0]
