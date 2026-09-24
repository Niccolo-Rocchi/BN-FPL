"""
Tests for src/utils.py.

Covers both the functions exercised by the current `exp.py` local
learning & update pipeline (vertices_cset, check_intersection, get_bn_counts,
perturb_bn_params, get_cpt_index/shape/tabular, get_min_max_bns,
jsd_credal_stats) and the broader utility library (mle_*, mne_*, ran_*,
centroid_*, maxent_*, resample_bn_params) that isn't currently wired into
exp.py but is kept for other uses (notebooks, future work).
"""

import pickle

import cvxpy as cp
import numpy as np
import pyagrum as gum
import pytest

from src.config import set_seed
from src.utils import (centroid_cn, centroid_cset, check_consistency,
                       check_intersection, get_bn_counts, get_cpt_index,
                       get_cpt_shape, get_min_max_bns, get_parent_confs,
                       get_tabular_cpt, jsd, learn_bn_params, lookup_cpt_row,
                       maxent_cn, maxent_cset, mle_bn_from_counts, mle_cn,
                       mle_cset, mne_cn, mne_cset, perturb_bn_params, ran_cn,
                       ran_cset, resample_bn_params, snapshot_cpts, vac_cn,
                       vertices_cset)

# BIF-file round trips (used internally by get_min_max_bns / pyagrum's
# CredalNet, see cap6_extract.tex investigation) truncate to ~6 significant
# digits; this tolerance is used throughout wherever such a round trip is
# in the path being tested.
BIF_ATOL = 1e-5


@pytest.fixture(autouse=True)
def _seed():
    set_seed()


@pytest.fixture
def bn_with_parent():
    # A (root, 3 categories) -> B (2 categories)
    return gum.fastBN("A[3]->B[2]")


@pytest.fixture
def bn_nonalpha():
    # The real network used by exp.py. Its labels are deliberately NOT in
    # alphabetical order for any variable (True/False, low/high,
    # positive/negative) -- this is the regression fixture for the
    # topandas()-vs-raw-array-order mismatch: cpt.topandas() sorts rows and
    # columns alphabetically by label, while cpt[:] (get_tabular_cpt) and
    # .fillWith() follow pyagrum's native declaration order. bn_with_parent
    # above (fastBN's auto-generated "0","1",... labels) can NOT catch this
    # class of bug, since declaration order there already happens to be
    # alphabetical.
    return gum.loadBN("cancer.bif")


# --------------------------------------------------------------------------
# CPT shape / indexing helpers
# --------------------------------------------------------------------------


def test_get_tabular_cpt_shapes(bn_with_parent):
    bn = bn_with_parent
    cpt_a = get_tabular_cpt(bn.cpt("A"))  # root: 1 row x 3 categories
    cpt_b = get_tabular_cpt(bn.cpt("B"))  # 1 parent (3 states) x 2 categories

    assert cpt_a.shape == (1, 3)
    assert cpt_b.shape == (3, 2)
    assert np.allclose(cpt_a.sum(axis=1), 1)
    assert np.allclose(cpt_b.sum(axis=1), 1)


def test_get_cpt_shape_matches_tabular(bn_with_parent):
    bn = bn_with_parent
    assert get_cpt_shape(bn.cpt("A")) == (1, 3)
    assert get_cpt_shape(bn.cpt("B")) == (3, 2)


def test_get_cpt_index_root_is_zero(bn_with_parent):
    assert get_cpt_index(bn_with_parent, "A", None) == 0


def test_get_cpt_index_matches_row_order(bn_with_parent):
    bn = bn_with_parent
    for i, parents in enumerate(get_parent_confs(bn, "B")):
        assert get_cpt_index(bn, "B", parents) == i


def test_get_cpt_index_matches_raw_array_nonalpha_labels(bn_nonalpha):
    # Regression test: for a variable whose labels are NOT alphabetically
    # ordered (P: low,high; S/C/D/X: True,False -- every variable in
    # cancer.bif), get_cpt_index's row number must index correctly into
    # get_tabular_cpt's array, which follows pyagrum's native (declaration)
    # order -- NOT cpt.topandas()'s alphabetically-sorted order.
    bn = bn_nonalpha
    cpt_min = get_tabular_cpt(bn.cpt("C"))  # C | P, S -- 2 parents, 4 rows
    for i, parents in enumerate(get_parent_confs(bn, "C")):
        idx = get_cpt_index(bn, "C", parents)
        assert idx == i
        # Cross-check against the ground-truth cancer.bif values directly.
        expected = {
            ("low", "True"): [0.03, 0.97],
            ("high", "True"): [0.05, 0.95],
            ("low", "False"): [0.001, 0.999],
            ("high", "False"): [0.02, 0.98],
        }[(parents["P"], parents["S"])]
        assert np.allclose(cpt_min[idx], expected)


def test_get_cpt_index_invalid_parent_raises(bn_with_parent):
    with pytest.raises(RuntimeError):
        get_cpt_index(bn_with_parent, "B", {"A": 99})


# --------------------------------------------------------------------------
# vertices_cset / check_intersection
# --------------------------------------------------------------------------


def test_vertices_cset_within_bounds_and_valid_distributions():
    vec_min = np.array([0.1, 0.2, 0.0])
    vec_max = np.array([0.5, 0.6, 0.3])
    vertices = vertices_cset(vec_min, vec_max)

    assert vertices.ndim == 2
    assert vertices.shape[1] == 3
    assert np.all(vertices >= vec_min - 1e-9)
    assert np.all(vertices <= vec_max + 1e-9)
    assert np.allclose(vertices.sum(axis=1), 1.0)


def test_vertices_cset_binary_known_case():
    # For 2 categories the credal set [min,max] x [min,max] (Delta-restricted)
    # collapses to exactly two vertices: (min0, 1-min0) and (1-min1, min1),
    # i.e. (vec_min[0], vec_max[1]) and (vec_max[0], vec_min[1]).
    vec_min = np.array([0.2, 0.3])
    vec_max = np.array([0.5, 0.8])
    vertices = vertices_cset(vec_min, vec_max)

    expected = {(0.2, 0.8), (0.5, 0.5)}
    got = {tuple(np.round(v, 6)) for v in vertices}
    assert got == expected


def test_vertices_cset_degenerate_is_2d_and_matches_point():
    vec = np.array([0.2, 0.5, 0.3])
    vertices = vertices_cset(vec, vec.copy())

    # Regression test: this used to return a 1D array, breaking any caller
    # that indexes vertices by row (check_intersection, ran_cset,
    # centroid_cset, mne_cset).
    assert vertices.shape == (1, 3)
    assert np.allclose(vertices[0], vec)


def test_check_intersection_overlapping_segments():
    v1 = vertices_cset(np.array([0.2, 0.4]), np.array([0.5, 0.8]))
    v2 = vertices_cset(np.array([0.4, 0.3]), np.array([0.7, 0.6]))
    assert check_intersection(v1, v2) is True


def test_check_intersection_disjoint_segments():
    v1 = vertices_cset(np.array([0.0, 0.8]), np.array([0.1, 1.0]))
    v2 = vertices_cset(np.array([0.8, 0.0]), np.array([1.0, 0.2]))
    assert check_intersection(v1, v2) is False


def test_check_intersection_degenerate_credal_sets():
    # Regression: degenerate (single-point) credal sets must not crash
    # check_intersection now that vertices_cset returns 2D arrays for them.
    v_point = vertices_cset(np.array([0.3, 0.7]), np.array([0.3, 0.7]))
    v_touching = vertices_cset(np.array([0.1, 0.5]), np.array([0.5, 0.9]))
    v_far = vertices_cset(np.array([0.8, 0.0]), np.array([1.0, 0.2]))

    assert check_intersection(v_point, v_touching) is True
    assert check_intersection(v_point, v_far) is False


# --------------------------------------------------------------------------
# vac_cn
# --------------------------------------------------------------------------


def test_vac_cn_is_min0_max1(bn_with_parent):
    cn = vac_cn(bn_with_parent)
    bn_min, bn_max = get_min_max_bns(cn)
    for var in bn_with_parent.names():
        assert np.allclose(get_tabular_cpt(bn_min.cpt(var)), 0.0)
        assert np.allclose(get_tabular_cpt(bn_max.cpt(var)), 1.0)


# --------------------------------------------------------------------------
# perturb_bn_params
# --------------------------------------------------------------------------


def test_perturb_bn_params_prob0_leaves_bn_unchanged(bn_with_parent):
    bn_new, mask = perturb_bn_params(bn_with_parent, alpha=20, prob=0.0)
    for var in bn_with_parent.names():
        assert np.allclose(
            get_tabular_cpt(bn_new.cpt(var)), get_tabular_cpt(bn_with_parent.cpt(var))
        )
        assert np.all(get_tabular_cpt(mask.cpt(var)) == 1)


def test_perturb_bn_params_prob1_perturbs_every_row(bn_with_parent):
    bn_new, mask = perturb_bn_params(bn_with_parent, alpha=5, prob=1.0)
    for var in bn_with_parent.names():
        assert np.all(get_tabular_cpt(mask.cpt(var)) == 0)
        cpt_new = get_tabular_cpt(bn_new.cpt(var))
        assert np.allclose(cpt_new.sum(axis=1), 1.0)
        assert np.all(cpt_new >= 0)


def test_perturb_bn_params_mask_matches_changed_rows(bn_with_parent):
    bn_new, mask = perturb_bn_params(bn_with_parent, alpha=5, prob=0.5)
    for var in bn_with_parent.names():
        cpt_orig = get_tabular_cpt(bn_with_parent.cpt(var))
        cpt_new = get_tabular_cpt(bn_new.cpt(var))
        mask_arr = get_tabular_cpt(mask.cpt(var))
        row_unchanged = np.all(np.isclose(cpt_orig, cpt_new), axis=1)
        row_mask = np.all(mask_arr == 1, axis=1)
        assert np.array_equal(row_unchanged, row_mask)


def test_perturb_bn_params_actually_changes_perturbed_rows(bn_with_parent):
    # With prob=1 and a small alpha (i.e. a large shift), rows must differ
    # from the original (statistically certain, not run-dependent given the
    # fixed seed).
    bn_new, _ = perturb_bn_params(bn_with_parent, alpha=2, prob=1.0)
    for var in bn_with_parent.names():
        cpt_orig = get_tabular_cpt(bn_with_parent.cpt(var))
        cpt_new = get_tabular_cpt(bn_new.cpt(var))
        assert not np.allclose(cpt_orig, cpt_new, atol=1e-3)


def test_perturb_bn_params_mean_is_centered_on_original_row():
    # E[p_new] = p_old (Dirichlet(alpha * p_old) is centered exactly on
    # p_old); check empirically over many draws for a single row.
    bn = gum.fastBN("A[2]")
    bn.cpt("A").fillWith([0.3, 0.7])

    alpha = 10.0
    draws = []
    for _ in range(3000):
        bn_new, _ = perturb_bn_params(bn, alpha=alpha, prob=1.0)
        draws.append(get_tabular_cpt(bn_new.cpt("A"))[0])
    mean = np.mean(draws, axis=0)
    assert np.allclose(mean, [0.3, 0.7], atol=0.02)


def test_perturb_bn_params_variance_matches_dirichlet_formula():
    # Var[p_new_i] = p_i*(1-p_i) / (alpha+1).
    bn = gum.fastBN("A[2]")
    bn.cpt("A").fillWith([0.3, 0.7])

    alpha = 10.0
    draws = []
    for _ in range(3000):
        bn_new, _ = perturb_bn_params(bn, alpha=alpha, prob=1.0)
        draws.append(get_tabular_cpt(bn_new.cpt("A"))[0, 0])
    empirical_var = np.var(draws)
    expected_var = 0.3 * 0.7 / (alpha + 1)
    assert abs(empirical_var - expected_var) < 0.002


def test_perturb_bn_params_handles_exact_zero_entry():
    # Regression: alpha * 0 is an invalid Dirichlet concentration; must not
    # raise (see the np.clip guard in perturb_bn_params).
    bn = gum.fastBN("A[2]")
    bn.cpt("A").fillWith([0.0, 1.0])
    bn_new, _ = perturb_bn_params(bn, alpha=20, prob=1.0)
    cpt_new = get_tabular_cpt(bn_new.cpt("A"))
    assert np.isclose(cpt_new.sum(), 1.0)
    assert np.all(cpt_new >= 0)


# --------------------------------------------------------------------------
# resample_bn_params (not wired into exp.py, kept as an alternative shift
# generator -- tested per explicit request)
# --------------------------------------------------------------------------


def test_resample_bn_params_mask_and_validity(bn_with_parent):
    bn_new, mask = resample_bn_params(bn_with_parent, alpha=1.0, prob=0.5)
    for var in bn_with_parent.names():
        cpt_orig = get_tabular_cpt(bn_with_parent.cpt(var))
        cpt_new = get_tabular_cpt(bn_new.cpt(var))
        mask_arr = get_tabular_cpt(mask.cpt(var))

        assert np.allclose(cpt_new.sum(axis=1), 1.0)
        row_unchanged = np.all(np.isclose(cpt_orig, cpt_new), axis=1)
        row_mask = np.all(mask_arr == 1, axis=1)
        assert np.array_equal(row_unchanged, row_mask)


# --------------------------------------------------------------------------
# get_bn_counts: verify the indirect (MLE-times-parent-marginal) count
# recovery exactly matches direct joint counting from the raw data.
# --------------------------------------------------------------------------


def test_get_bn_counts_matches_direct_counting(bn_with_parent):
    bn = bn_with_parent
    gen = gum.BNDatabaseGenerator(bn)
    gen.drawSamples(500)
    data = gen.to_pandas()

    learnt = learn_bn_params(bn, data)
    bn_counts = get_bn_counts(learnt, data)

    for var in bn.names():
        parents = sorted(bn.parents(var), key=lambda i: bn.variable(i).name())
        parent_names = [bn.variable(i).name() for i in parents]
        var_labels = list(bn.variable(var).labels())  # CPT's own column order

        index_df = learnt.cpt(var).topandas().index.to_frame(index=False)
        n_rows = get_cpt_shape(learnt.cpt(var))[0]
        var_size = bn.variable(var).domainSize()

        direct_counts = np.zeros((n_rows, var_size))
        for row in range(n_rows):
            row_conf = dict(index_df.iloc[row]) if len(parent_names) else None
            for j, lab in enumerate(var_labels):
                query = data
                if row_conf is not None:
                    for p, v in row_conf.items():
                        query = query[query[p] == v]
                direct_counts[row, j] = (query[var] == lab).sum()

        got_counts = get_tabular_cpt(bn_counts.cpt(var))
        assert np.allclose(got_counts, direct_counts), var

    assert sum(get_tabular_cpt(bn_counts.cpt(v)).sum() for v in ["A"]) == len(data)


def test_get_bn_counts_total_equals_sample_size(bn_with_parent):
    bn = bn_with_parent
    gen = gum.BNDatabaseGenerator(bn)
    gen.drawSamples(123)
    data = gen.to_pandas()

    learnt = learn_bn_params(bn, data)
    bn_counts = get_bn_counts(learnt, data)
    # Every node's CPT counts must sum to the total sample size (each row is
    # counted once, regardless of how many parents/categories it has).
    for var in bn.names():
        assert get_tabular_cpt(bn_counts.cpt(var)).sum() == len(data)


def test_get_bn_counts_matches_direct_counting_nonalpha_labels(bn_nonalpha):
    # Regression test on the real cancer.bif network (non-alphabetical
    # labels for every variable) -- get_bn_counts must assign counts to the
    # correct physical row/column, not the topandas()-sorted one.
    bn = bn_nonalpha
    gen = gum.BNDatabaseGenerator(bn)
    gen.drawSamples(500)
    data = gen.to_pandas()

    learnt = learn_bn_params(bn, data)
    bn_counts = get_bn_counts(learnt, data)

    for var in bn.names():
        var_labels = list(bn.variable(var).labels())
        got_counts = get_tabular_cpt(bn_counts.cpt(var))
        for row, parents in enumerate(get_parent_confs(bn, var)):
            sub = data
            if parents is not None:
                for p, v in parents.items():
                    sub = sub[sub[p] == v]
            for col, lab in enumerate(var_labels):
                direct = (sub[var] == lab).sum()
                assert got_counts[row, col] == direct, (var, parents, lab)


# --------------------------------------------------------------------------
# mle_bn_from_counts: the exact (unsmoothed) MLE, as opposed to
# learn_bn_params's smoothed output.
# --------------------------------------------------------------------------


def test_mle_bn_from_counts_matches_exact_ratio(bn_with_parent):
    bn = bn_with_parent
    gen = gum.BNDatabaseGenerator(bn)
    gen.drawSamples(500)
    data = gen.to_pandas()

    learnt = learn_bn_params(bn, data)
    bn_counts = get_bn_counts(learnt, data)
    mle = mle_bn_from_counts(bn_counts)

    for var in bn.names():
        counts = get_tabular_cpt(bn_counts.cpt(var))
        expected = counts / counts.sum(axis=1, keepdims=True)
        got = get_tabular_cpt(mle.cpt(var))
        assert np.allclose(got, expected)
        # Unlike learn_bn_params's output, this must be the EXACT ratio, not
        # merely close to it.
        assert np.array_equal(got, expected)


def test_mle_bn_from_counts_differs_from_smoothed_learn_bn_params(bn_with_parent):
    # Regression guard: the two must NOT coincide (else the smoothing bias
    # this function exists to avoid would go undetected).
    bn = bn_with_parent
    gen = gum.BNDatabaseGenerator(bn)
    gen.drawSamples(30)  # small N: the smoothing bias is more visible
    data = gen.to_pandas()

    learnt = learn_bn_params(bn, data)
    bn_counts = get_bn_counts(learnt, data)
    mle = mle_bn_from_counts(bn_counts)

    any_differs = any(
        not np.array_equal(
            get_tabular_cpt(learnt.cpt(var)), get_tabular_cpt(mle.cpt(var))
        )
        for var in bn.names()
    )
    assert any_differs


def test_mle_bn_from_counts_defaults_to_uniform_when_unobserved():
    bn = gum.fastBN("X[2]->Y[3]")
    bn_counts = gum.BayesNet(bn)
    # X: 7 vs 3. Y|X=0: fully unobserved: Y|X=1: observed.
    bn_counts.cpt("X").fillWith([7.0, 3.0])
    bn_counts.cpt("Y").fillWith([0.0, 0.0, 0.0, 2.0, 0.0, 1.0])

    mle = mle_bn_from_counts(bn_counts)
    got_x = get_tabular_cpt(mle.cpt("X"))
    got_y = get_tabular_cpt(mle.cpt("Y"))

    assert np.allclose(got_x, [[0.7, 0.3]])
    assert np.allclose(got_y[0], [1 / 3, 1 / 3, 1 / 3])  # unobserved -> uniform
    assert np.allclose(got_y[1], [2 / 3, 0.0, 1 / 3])


# --------------------------------------------------------------------------
# learn_bn_params: sanity (converges to the generating distribution, given
# enough data; with only the 1e-6 smoothing prior the deviation must be
# tiny).
# --------------------------------------------------------------------------


def test_learn_bn_params_close_to_ground_truth_at_large_n():
    bn = gum.fastBN("A[2]")
    bn.cpt("A").fillWith([0.3, 0.7])
    gen = gum.BNDatabaseGenerator(bn)
    gen.drawSamples(20000)
    data = gen.to_pandas()

    learnt = learn_bn_params(bn, data)
    got = get_tabular_cpt(learnt.cpt("A"))[0]
    assert np.allclose(got, [0.3, 0.7], atol=0.02)


# --------------------------------------------------------------------------
# mle_cset / mle_cpt / mle_cn
# --------------------------------------------------------------------------


def _reference_mle(vec_min, vec_max, counts):
    n = len(vec_min)
    p = cp.Variable(n)
    objective = cp.Maximize(counts @ cp.log(p))
    constraints = [cp.sum(p) == 1, p >= np.maximum(vec_min, 1e-9), p <= vec_max]
    cp.Problem(objective, constraints).solve()
    return np.array(p.value)


def test_mle_cset_unconstrained_matches_empirical_ratio():
    # Box doesn't bind: MLE should equal counts / sum(counts) exactly.
    vec_min, vec_max = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    counts = np.array([3.0, 7.0])
    got = mle_cset(vec_min, vec_max, counts)
    assert np.allclose(got, [0.3, 0.7], atol=1e-4)


def test_mle_cset_constrained_matches_reference_solver():
    vec_min, vec_max = np.array([0.5, 0.0]), np.array([1.0, 0.5])
    counts = np.array([3.0, 7.0])  # unconstrained optimum (0.3, 0.7) is infeasible
    got = mle_cset(vec_min, vec_max, counts)
    ref = _reference_mle(vec_min, vec_max, counts)
    assert np.allclose(got, ref, atol=1e-3)
    assert got[0] >= vec_min[0] - 1e-6
    assert got[1] <= vec_max[1] + 1e-6


def test_mle_cn_is_consistent_with_credal_net(bn_with_parent):
    bn = bn_with_parent
    gen = gum.BNDatabaseGenerator(bn)
    gen.drawSamples(200)
    data = gen.to_pandas()

    learnt = learn_bn_params(bn, data)
    bn_counts = get_bn_counts(learnt, data)
    cn = gum.CredalNet(bn_counts)
    cn.idmLearning(1)
    bn_min, bn_max = get_min_max_bns(cn)

    mle_bn = mle_cn(bn_min, bn_max, data)
    assert check_consistency(mle_bn, bn_min, bn_max) == 0


def test_mle_cpt_row_alignment_nonalpha_labels(bn_nonalpha):
    # Regression test (deterministic): mle_cpt used to read cpt_min/
    # cpt_max/cpt_counts via cpt.topandas() (alphabetically-sorted rows),
    # so its *returned* row order followed that sort -- silently wrong once
    # mle_cn later writes it back via .fillWith() (native-order rows).
    # Counts are chosen to be clearly row-distinguishable (each row favors
    # a different category), so any row permutation is caught exactly.
    from src.utils import mle_cpt

    bn = bn_nonalpha  # cancer.bif
    parent_confs = get_parent_confs(bn, "C")  # native row order: P inner, S outer
    assert parent_confs == [
        {"P": "low", "S": "True"},
        {"P": "high", "S": "True"},
        {"P": "low", "S": "False"},
        {"P": "high", "S": "False"},
    ]

    bn_counts = gum.BayesNet(bn)
    bn_counts.cpt("C").fillWith([90, 10, 10, 90, 50, 50, 1, 99])  # native order
    bn_min = gum.BayesNet(bn)
    bn_min.cpt("C").fillWith([0.0] * 8)
    bn_max = gum.BayesNet(bn)
    bn_max.cpt("C").fillWith([1.0] * 8)

    result = mle_cpt(bn_min.cpt("C"), bn_max.cpt("C"), bn_counts.cpt("C"))

    # C's own labels, native order (see bn_nonalpha / cancer.bif: True, False)
    expected = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.01, 0.99]])
    assert np.allclose(result, expected, atol=1e-3)  # cvxpy solver tolerance


# --------------------------------------------------------------------------
# mne_cset / mne_cpt / mne_cn: worst-case (minimum likelihood) vertex.
# Regression test for the log(0)-handling bug (a vertex assigning zero
# probability to an observed category must be recognized as infinitely
# unlikely, not neutral).
# --------------------------------------------------------------------------


def test_mne_cset_picks_true_worst_case_vertex_with_zero_probability():
    vec_min, vec_max = np.array([0.0, 0.5]), np.array([0.5, 1.0])
    counts = np.array([10.0, 1.0])  # category 0 heavily observed
    got = mne_cset(vec_min, vec_max, counts)
    # The true minimum-likelihood vertex assigns 0 probability to the
    # heavily-observed category 0.
    assert np.isclose(got[0], 0.0, atol=1e-6)


def test_mne_cset_matches_brute_force_when_no_zero_probability():
    # Away from the zero-probability edge case, mne_cset's vertex-search must
    # agree with a brute-force grid search over the (1D, since binary)
    # feasible segment.
    vec_min, vec_max = np.array([0.2, 0.3]), np.array([0.6, 0.8])
    counts = np.array([9.0, 1.0])

    grid = np.linspace(vec_min[0], vec_max[0], 2001)
    grid = grid[(1 - grid >= vec_min[1] - 1e-9) & (1 - grid <= vec_max[1] + 1e-9)]
    neg_loglik = -(counts[0] * np.log(grid) + counts[1] * np.log(1 - grid))
    worst_p0 = grid[np.argmax(neg_loglik)]

    got = mne_cset(vec_min, vec_max, counts)
    assert abs(got[0] - worst_p0) < 1e-2


def test_mne_cn_is_consistent_with_credal_net(bn_with_parent):
    bn = bn_with_parent
    gen = gum.BNDatabaseGenerator(bn)
    gen.drawSamples(200)
    data = gen.to_pandas()

    learnt = learn_bn_params(bn, data)
    bn_counts = get_bn_counts(learnt, data)
    cn = gum.CredalNet(bn_counts)
    cn.idmLearning(1)
    bn_min, bn_max = get_min_max_bns(cn)

    mne_bn = mne_cn(bn_min, bn_max, data)
    assert check_consistency(mne_bn, bn_min, bn_max) == 0


# --------------------------------------------------------------------------
# ran_cset / ran_cn: samples must lie inside the credal set.
# --------------------------------------------------------------------------


def test_ran_cset_samples_are_inside_credal_set():
    vec_min, vec_max = np.array([0.1, 0.2, 0.0]), np.array([0.5, 0.6, 0.3])
    for _ in range(50):
        p = ran_cset(vec_min, vec_max)
        assert np.all(p >= vec_min - 1e-9)
        assert np.all(p <= vec_max + 1e-9)
        assert np.isclose(np.sum(p), 1.0)


def test_ran_cn_is_consistent_with_credal_net(bn_with_parent):
    bn = bn_with_parent
    gen = gum.BNDatabaseGenerator(bn)
    gen.drawSamples(200)
    data = gen.to_pandas()

    learnt = learn_bn_params(bn, data)
    bn_counts = get_bn_counts(learnt, data)
    cn = gum.CredalNet(bn_counts)
    cn.idmLearning(1)
    bn_min, bn_max = get_min_max_bns(cn)

    for _ in range(5):
        ran_bn = ran_cn(bn_min, bn_max)
        assert check_consistency(ran_bn, bn_min, bn_max) == 0


# --------------------------------------------------------------------------
# centroid_cset / centroid_cn
# --------------------------------------------------------------------------


def test_centroid_cset_binary_known_case():
    # Two vertices (0.2, 0.8) and (0.5, 0.5) -> centroid is their average.
    vec_min, vec_max = np.array([0.2, 0.5]), np.array([0.5, 0.8])
    got = centroid_cset(vec_min, vec_max)
    assert np.allclose(sorted(got), sorted([0.35, 0.65]), atol=1e-6)
    assert np.isclose(np.sum(got), 1.0)


def test_centroid_cn_is_consistent_with_credal_net(bn_with_parent):
    bn = bn_with_parent
    gen = gum.BNDatabaseGenerator(bn)
    gen.drawSamples(200)
    data = gen.to_pandas()

    learnt = learn_bn_params(bn, data)
    bn_counts = get_bn_counts(learnt, data)
    cn = gum.CredalNet(bn_counts)
    cn.idmLearning(1)
    bn_min, bn_max = get_min_max_bns(cn)

    centroid_bn = centroid_cn(bn_min, bn_max)
    assert check_consistency(centroid_bn, bn_min, bn_max) == 0


# --------------------------------------------------------------------------
# maxent_cset / maxent_cn: cross-checked against an independent cvxpy
# maximum-entropy solver.
# --------------------------------------------------------------------------


def _reference_maxent(vec_min, vec_max):
    n = len(vec_min)
    p = cp.Variable(n)
    objective = cp.Maximize(cp.sum(cp.entr(p)))
    constraints = [cp.sum(p) == 1, p >= vec_min, p <= vec_max]
    cp.Problem(objective, constraints).solve()
    return np.array(p.value)


@pytest.mark.parametrize(
    "vec_min,vec_max",
    [
        (np.array([0.0, 0.0]), np.array([1.0, 1.0])),  # vacuous -> uniform
        (np.array([0.1, 0.2, 0.0]), np.array([0.5, 0.6, 0.3])),
        (np.array([0.3, 0.3, 0.1]), np.array([0.5, 0.5, 0.2])),
    ],
)
def test_maxent_cset_matches_reference_solver(vec_min, vec_max):
    got = maxent_cset(vec_min.copy(), vec_max.copy())
    ref = _reference_maxent(vec_min, vec_max)
    assert np.allclose(got, ref, atol=1e-3)


def test_maxent_cn_is_consistent_with_credal_net(bn_with_parent):
    bn = bn_with_parent
    gen = gum.BNDatabaseGenerator(bn)
    gen.drawSamples(200)
    data = gen.to_pandas()

    learnt = learn_bn_params(bn, data)
    bn_counts = get_bn_counts(learnt, data)
    cn = gum.CredalNet(bn_counts)
    cn.idmLearning(1)
    bn_min, bn_max = get_min_max_bns(cn)

    maxent_bn = maxent_cn(bn_min, bn_max)
    assert check_consistency(maxent_bn, bn_min, bn_max) == 0


# --------------------------------------------------------------------------
# jsd: symmetry, bounds, and identity.
# --------------------------------------------------------------------------


def test_jsd_identity_is_zero():
    p = np.array([0.3, 0.7])
    assert jsd(p, p) < 1e-9


def test_jsd_is_symmetric():
    p, q = np.array([0.1, 0.9]), np.array([0.6, 0.4])
    assert np.isclose(jsd(p, q), jsd(q, p))


def test_jsd_is_bounded_in_0_1():
    p, q = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    d = jsd(p, q)
    assert 0.0 <= d <= 1.0 + 1e-9


# --------------------------------------------------------------------------
# snapshot_cpts: plain-numpy archival of a BN's CPTs (exp.py's models.pkl).
# --------------------------------------------------------------------------


def test_snapshot_cpts_matches_get_tabular_cpt(bn_with_parent):
    bn = bn_with_parent
    snap = snapshot_cpts(bn)

    assert set(snap.keys()) == set(bn.names())
    for var, entry in snap.items():
        assert set(entry.keys()) == {"cpt", "parents", "labels"}
        assert np.array_equal(entry["cpt"], get_tabular_cpt(bn.cpt(var)))
        assert np.allclose(entry["cpt"].sum(axis=1), 1.0)
        assert len(entry["parents"]) == entry["cpt"].shape[0]
        assert entry["labels"] == list(bn.variable(var).labels())
        assert len(entry["labels"]) == entry["cpt"].shape[1]


def test_snapshot_cpts_is_independent_copy(bn_with_parent):
    bn = bn_with_parent
    snap = snapshot_cpts(bn)
    var = next(iter(snap))
    snap[var]["cpt"][0, 0] = -999.0  # mutate the snapshot

    # The source BN's own CPT must be unaffected.
    assert get_tabular_cpt(bn.cpt(var))[0, 0] != -999.0


def test_snapshot_cpts_row_column_labels_match_ground_truth_nonalpha(bn_nonalpha):
    # Regression test on the real cancer.bif network (non-alphabetical
    # labels for every variable): the saved "parents"/"labels" metadata must
    # let a reader reconstruct P(X=x|pi_X) correctly, WITHOUT needing to
    # separately call get_parent_confs (or, worse, cpt.topandas(), which
    # would silently scramble both rows and columns here -- see the
    # topandas-vs-raw-array investigation).
    bn = bn_nonalpha
    snap = snapshot_cpts(bn)

    ground_truth = {
        ("low", "True"): {"True": 0.03, "False": 0.97},
        ("high", "True"): {"True": 0.05, "False": 0.95},
        ("low", "False"): {"True": 0.001, "False": 0.999},
        ("high", "False"): {"True": 0.02, "False": 0.98},
    }
    entry = snap["C"]
    for row, parents in enumerate(entry["parents"]):
        key = (parents["P"], parents["S"])
        for col, label in enumerate(entry["labels"]):
            assert np.isclose(
                entry["cpt"][row, col], ground_truth[key][label], atol=1e-4
            ), (
                row,
                col,
                parents,
                label,
            )


# --------------------------------------------------------------------------
# lookup_cpt_row: the reference way to read a value back out of a
# snapshot_cpts() entry -- this is the actual "upload"/reload path future
# code (and the Plot_JS.ipynb demo cell) is expected to use, so it gets its
# own tests, on a fixed network, on genuinely random networks (random
# structure, not just a fixed one), and on the real Cancer network with
# hand-verified ground truth.
# --------------------------------------------------------------------------


def test_lookup_cpt_row_root_variable(bn_with_parent):
    bn = bn_with_parent
    snap = snapshot_cpts(bn)
    got = lookup_cpt_row(snap["A"], parents=None)
    assert np.allclose(got, get_tabular_cpt(bn.cpt("A"))[0])


def test_lookup_cpt_row_with_parents_matches_live_bn(bn_with_parent):
    bn = bn_with_parent
    snap = snapshot_cpts(bn)
    for parents in get_parent_confs(bn, "B"):
        got = lookup_cpt_row(snap["B"], parents)
        idx = get_cpt_index(bn, "B", parents)
        expected = get_tabular_cpt(bn.cpt("B"))[idx]
        assert np.allclose(got, expected), parents


def test_lookup_cpt_row_unknown_parents_raises(bn_with_parent):
    bn = bn_with_parent
    snap = snapshot_cpts(bn)
    with pytest.raises(ValueError):
        lookup_cpt_row(snap["B"], {"A": "not_a_real_label"})


def test_lookup_cpt_row_on_random_networks():
    # "Reti randomiche": genuinely random STRUCTURE (not just fixed
    # structure with default values), across several independent draws, so
    # correctness isn't accidentally specific to one topology.
    for trial in range(10):
        bn_gen = gum.BNGenerator()
        bn = bn_gen.generate(n_nodes=6, n_arcs=8, n_modmax=4)
        snap = snapshot_cpts(bn)
        for var in bn.names():
            for parents in get_parent_confs(bn, var):
                got = lookup_cpt_row(snap[var], parents)
                idx = get_cpt_index(bn, var, parents)
                expected = get_tabular_cpt(bn.cpt(var))[idx]
                assert np.allclose(got, expected), (trial, var, parents)


def test_lookup_cpt_row_matches_ground_truth_on_cancer_network(bn_nonalpha):
    # The critical case: real cancer.bif, non-alphabetical labels for every
    # variable -- exactly where a topandas()-based reader would silently
    # scramble rows/columns.
    bn = bn_nonalpha
    snap = snapshot_cpts(bn)

    ground_truth = {
        ("low", "True"): [0.03, 0.97],
        ("high", "True"): [0.05, 0.95],
        ("low", "False"): [0.001, 0.999],
        ("high", "False"): [0.02, 0.98],
    }
    for (p, s), expected in ground_truth.items():
        got = lookup_cpt_row(snap["C"], {"P": p, "S": s})
        assert np.allclose(got, expected, atol=1e-4), (p, s)

    # Root variables too.
    assert np.allclose(lookup_cpt_row(snap["P"]), [0.9, 0.1])
    assert np.allclose(lookup_cpt_row(snap["S"]), [0.3, 0.7])


# --------------------------------------------------------------------------
# Full round trip: snapshot -> pickle.dumps -> pickle.loads -> lookup_cpt_row
# (the exact path exp.py's models.pkl / a future "upload" consumer go
# through), cross-checked against directly querying the SAME live BN --
# both on the real Cancer network (with a genuine, non-trivial parameter
# perturbation, not hand-typed "nice" numbers) and on random networks.
# --------------------------------------------------------------------------


def test_snapshot_pickle_roundtrip_matches_live_model_on_cancer_network():
    bn = gum.loadBN("cancer.bif")
    perturbed, _ = perturb_bn_params(bn, alpha=10, prob=1.0)

    snap = snapshot_cpts(perturbed)
    reloaded = pickle.loads(pickle.dumps(snap))  # the actual serialization step

    for var in perturbed.names():
        for parents in get_parent_confs(perturbed, var):
            got = lookup_cpt_row(reloaded[var], parents)
            idx = get_cpt_index(perturbed, var, parents)
            expected = get_tabular_cpt(perturbed.cpt(var))[idx]
            assert np.allclose(got, expected), (var, parents)


def test_snapshot_pickle_roundtrip_matches_live_model_on_random_networks():
    for trial in range(5):
        bn_gen = gum.BNGenerator()
        bn = bn_gen.generate(n_nodes=7, n_arcs=10, n_modmax=3)

        snap = snapshot_cpts(bn)
        reloaded = pickle.loads(pickle.dumps(snap))

        for var in bn.names():
            for parents in get_parent_confs(bn, var):
                got = lookup_cpt_row(reloaded[var], parents)
                idx = get_cpt_index(bn, var, parents)
                expected = get_tabular_cpt(bn.cpt(var))[idx]
                assert np.allclose(got, expected), (trial, var, parents)
