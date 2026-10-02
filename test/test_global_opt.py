"""
Tests for src/global_opt.py: clustering clients per mechanism via the
lexicographic procedure (Eq. eq:global_opt's lambda_1 term first, lambda_2
only to break ties). See the module's own docstring for the Helly's-theorem
argument behind cluster_milp and for why ties are common, not negligible.
"""

import numpy as np
import pyagrum as gum
import pytest

from src.global_opt import (all_mechanisms, ari_ami, cluster_1d_wcss_optimal,
                            cluster_jsd_hierarchical, cluster_milp,
                            cluster_milp_lexicographic,
                            cluster_representative_thetas, credal_jsd_distance,
                            evaluate_mechanism, evaluate_mechanism_mle,
                            ground_truth_labels, intersection_graph,
                            is_clustering_optimum_unique, jsd_distance_matrix,
                            pairwise_confusion, precision_recall_f1,
                            total_entropy)
from src.mosaic import CN, CN_CPT, Client
from src.utils import get_cpt_shape, jsd

# ground_truth_labels


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


# intersection_graph


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
# clustering, see module docstring)


def test_cluster_milp_full_triangle_merges_all():
    # All 3 pairwise intersect -> by Helly, a common point exists -> merge.
    edge = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
    labels = cluster_milp(edge)
    assert labels[0] == labels[1] == labels[2]


def test_cluster_milp_chain_does_not_over_merge():
    # A-B and B-C intersect, A-C does not: by Helly, no common point exists
    # for all three, so the optimal delta merges only one of the two edges.
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
    # A "star": center 0 intersects 1,2,3, but 1,2,3 pairwise don't. The MILP
    # must pick exactly one spoke to merge, not fall back to connected components.
    edge = np.zeros((4, 4), dtype=bool)
    for j in (1, 2, 3):
        edge[0, j] = edge[j, 0] = True
    labels = cluster_milp(edge)
    counts = np.bincount(labels)
    assert sorted(counts.tolist()) == [1, 1, 2]  # one pair merged, two singletons


# is_clustering_optimum_unique / cluster_representative_thetas /
# total_entropy / cluster_milp_lexicographic: the two-stage procedure.


def test_is_clustering_optimum_unique_triangle_is_unique():
    # All 3 pairwise intersect: the only way to reach objective=3 is
    # merging all three, no alternative achieves the same value.
    edge = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
    assert is_clustering_optimum_unique(edge) is True


def test_is_clustering_optimum_unique_chain_is_not_unique():
    # A-B and B-C intersect, A-C does not: merging {A,B} or {B,C} both
    # reach objective=1, a genuine tie from graph structure.
    edge = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    assert is_clustering_optimum_unique(edge) is False


def test_is_clustering_optimum_unique_star_is_not_unique():
    edge = np.zeros((4, 4), dtype=bool)
    for j in (1, 2, 3):
        edge[0, j] = edge[j, 0] = True
    assert is_clustering_optimum_unique(edge) is False


def test_is_clustering_optimum_unique_no_edges_is_unique():
    edge = np.zeros((4, 4), dtype=bool)
    assert is_clustering_optimum_unique(edge) is True


def test_cluster_representative_thetas_picks_half_when_inside_interval():
    # A single cluster {0,1} whose joint intersection is [0.4,0.6],
    # containing 0.5 -> max-entropy representative is exactly 0.5.
    rows_min = [np.array([0.3, 0.3]), np.array([0.4, 0.2])]
    rows_max = [np.array([0.6, 0.7]), np.array([0.8, 0.6])]
    labels = np.array([0, 0])
    reps = cluster_representative_thetas(rows_min, rows_max, labels)
    assert reps[0][0] == pytest.approx(0.5)


def test_total_entropy_weights_by_cluster_size():
    # A cluster of size 3 (all sharing the maxent point 0.5) must count
    # its entropy 3 times (sum over CLIENTS i, not over distinct clusters).
    rows_min = [np.array([0.3, 0.3])] * 3
    rows_max = [np.array([0.7, 0.7])] * 3
    labels = np.array([0, 0, 0])
    from src.global_opt import _binary_entropy

    assert total_entropy(rows_min, rows_max, labels) == pytest.approx(
        3 * _binary_entropy(0.5)
    )


def _star_with_known_best_leaf():
    """
    Center 0 intersects leaves 1,2,3 (which pairwise don't intersect, so
    exactly one merge is optimal). Leaf 1's overlap with the center
    contains exactly 0.5 (max possible entropy); leaves 2 and 3's overlaps
    do not. So the entropy-maximizing tie-break must pick {0,1}, never
    {0,2} or {0,3}: verified against total_entropy directly in
    test_cluster_milp_lexicographic_breaks_tie_by_entropy below.
    """
    rows_min = [
        np.array([0.30, 0.30]),  # center: [0.30, 0.70]
        np.array(
            [0.45, 0.45]
        ),  # leaf1:  [0.45, 0.55] -> overlap w/ center contains 0.5
        np.array(
            [0.05, 0.68]
        ),  # leaf2:  [0.05, 0.32] -> overlap w/ center = [0.30,0.32]
        np.array(
            [0.66, 0.05]
        ),  # leaf3:  [0.66, 0.95] -> overlap w/ center = [0.66,0.70]
    ]
    rows_max = [
        np.array([0.70, 0.70]),
        np.array([0.55, 0.55]),
        np.array([0.32, 0.95]),
        np.array([0.95, 0.34]),
    ]
    return rows_min, rows_max


def test_cluster_milp_lexicographic_star_is_a_genuine_tie():
    rows_min, rows_max = _star_with_known_best_leaf()
    edge = intersection_graph(rows_min, rows_max)
    assert is_clustering_optimum_unique(edge) is False


def test_cluster_milp_lexicographic_breaks_tie_by_entropy():
    rows_min, rows_max = _star_with_known_best_leaf()
    edge = intersection_graph(rows_min, rows_max)

    # Independently confirm leaf1 is really the entropy-maximizing choice,
    # not just trusting cluster_milp_lexicographic's own internal logic.
    entropy_leaf1 = total_entropy(rows_min, rows_max, np.array([0, 0, 1, 2]))
    entropy_leaf2 = total_entropy(rows_min, rows_max, np.array([0, 1, 0, 2]))
    entropy_leaf3 = total_entropy(rows_min, rows_max, np.array([0, 1, 2, 0]))
    assert entropy_leaf1 > entropy_leaf2
    assert entropy_leaf1 > entropy_leaf3

    labels = cluster_milp_lexicographic(edge, rows_min, rows_max)
    assert labels[0] == labels[1]  # center merged with leaf1 ...
    assert labels[2] != labels[0]  # ... not leaf2 ...
    assert labels[3] != labels[0]  # ... nor leaf3
    assert labels[2] != labels[3]  # leaves stay pairwise distinct too


def test_cluster_milp_lexicographic_matches_cluster_milp_when_unique():
    # No tie possible here (see test_cluster_milp_full_triangle_merges_all):
    # the lexicographic wrapper must agree with the plain MILP exactly.
    edge = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
    rows_min = [np.array([0.3, 0.3])] * 3
    rows_max = [np.array([0.7, 0.7])] * 3
    plain = cluster_milp(edge)
    lexi = cluster_milp_lexicographic(edge, rows_min, rows_max)
    assert (plain[0] == plain[1] == plain[2]) == (lexi[0] == lexi[1] == lexi[2])


def test_cluster_milp_lexicographic_tie_rate_on_real_data_is_substantial():
    # Fast smoke check that the tie rate is substantial (not near-zero)
    # at low ESS, grounding the module docstring's claim.
    import exp_global as exp_g

    bn_base = gum.loadBN("cancer.bif")
    mechanisms = all_mechanisms(bn_base)

    n_ties, n_total = 0, 0
    config = {
        "n_clients": 10,
        "ess": 2,
        "alpha": 0.1,
        "prob_shift": 0.5,
        "bn_base_path": "cancer.bif",
    }
    for rep in range(5):
        seed = hash((rep, "tie_rate_smoke")) % (2**32)
        np.random.seed(seed)
        gum.initRandom(seed)
        clients = exp_g.init_clients(config)
        for c in clients.values():
            c.generate_base_info(200, config["ess"])
        for var, row in mechanisms:
            rows_min, rows_max = [], []
            for e in range(10):
                cpt_min, cpt_max = clients[e].cn.cpt(var)
                rows_min.append(cpt_min[row, :])
                rows_max.append(cpt_max[row, :])
            edge = intersection_graph(rows_min, rows_max)
            n_total += 1
            if not is_clustering_optimum_unique(edge):
                n_ties += 1
                # Must not raise, and must return valid labels for every
                # client even when a tie is present.
                labels = cluster_milp_lexicographic(edge, rows_min, rows_max)
                assert len(labels) == 10

    assert n_total == 50
    assert n_ties > 0  # ties must actually occur in this smoke sample


# credal_jsd_distance / jsd_distance_matrix / cluster_jsd_hierarchical


def test_credal_jsd_distance_zero_when_intersecting():
    # Category-0 segments [0.2, 0.5] and [0.4, 0.7] overlap on [0.4, 0.5].
    d = credal_jsd_distance(
        np.array([0.2, 0.5]),
        np.array([0.5, 0.8]),
        np.array([0.4, 0.3]),
        np.array([0.7, 0.6]),
    )
    assert d == 0.0


def test_credal_jsd_distance_matches_closest_boundary_points_when_disjoint():
    min1, max1 = np.array([0.1, 0.85]), np.array([0.15, 0.9])
    min2, max2 = np.array([0.5, 0.3]), np.array([0.7, 0.5])
    d = credal_jsd_distance(min1, max1, min2, max2)
    expected = jsd(
        [0.15, 0.85], [0.5, 0.5]
    )  # closest points: 0.15 (max1) vs 0.5 (min2)
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


# cluster_1d_wcss_optimal


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
    pairs_orig = {(i, j) for i in range(8) for j in range(8) if labels[i] == labels[j]}
    pairs_perm = {
        (perm[i], perm[j])
        for i in range(8)
        for j in range(8)
        if labels_perm[i] == labels_perm[j]
    }
    assert pairs_orig == pairs_perm


def test_cluster_1d_wcss_optimal_matches_brute_force_on_random_inputs():
    # Regression test: an earlier version cut the k-1 largest gaps between
    # sorted values instead of solving for the true WCSS optimum.
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


# pairwise_confusion / precision_recall_f1 / ari_ami


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


# all_mechanisms


def test_all_mechanisms_covers_every_row_of_cancer_bn():
    bn = gum.loadBN("cancer.bif")
    mechanisms = all_mechanisms(bn)

    # Cancer: P, S (roots, 1 row each), C (2 binary parents -> 4 rows),
    # X, D (1 binary parent each -> 2 rows each) = 1+1+4+2+2 = 10.
    assert len(mechanisms) == 10
    assert len(set(mechanisms)) == 10  # no duplicates
    vars_seen = {var for var, _ in mechanisms}
    assert vars_seen == set(bn.names())


# evaluate_mechanism / evaluate_mechanism_mle: small integration test


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
    cn.cpts = {
        "X": CN_CPT("X", get_cpt_shape(bn.cpt("X")), (bn_min.cpt("X"), bn_max.cpt("X")))
    }
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
    # Clients 0,1 kept the baseline row -> same cluster. Client 2 was
    # perturbed -> its own singleton, even though its set overlaps client 0's.
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
    assert (
        result["pred_labels"][0] == result["pred_labels"][1] == result["pred_labels"][2]
    )


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
