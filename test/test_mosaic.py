"""
Tests for src/mosaic.py: CN/CN_CPT, PriorCPT/PriorCN (prior computation and
the weighting schemas), and Client (local IDM learning and the MOSAIC
update rule).

Where exact arithmetic matters, clients are built with fully controlled
credal sets/counts (bypassing random data generation and, where possible,
pyagrum's CredalNet/BIF round trip -- see BIF_ATOL below) so expected values
can be hand-computed and checked with tight tolerances.
"""

import warnings

import numpy as np
import pyagrum as gum
import pytest

from src.config import set_seed
from src.mosaic import CN, CN_CPT, Client, PriorCN
from src.utils import get_cpt_shape, jsd

# See test_utils.py: pyagrum's BIF writer (used internally whenever a CN is
# built from a gum.CredalNet, e.g. Client.learn_cn) truncates to ~6
# significant digits.
BIF_ATOL = 1e-5


@pytest.fixture(autouse=True)
def _seed():
    set_seed()


# --------------------------------------------------------------------------
# Helpers: build clients with an exactly-controlled credal set / counts for
# a single binary root variable "X", without going through data generation.
# --------------------------------------------------------------------------


def _base_bn():
    bn = gum.fastBN("X[2]")
    bn.cpt("X").fillWith([0.5, 0.5])
    return bn


def _seg(p0_min, p0_max):
    """
    Feasible bounds for a binary-variable credal set defined by the segment
    p0 in [p0_min, p0_max] (p1 = 1 - p0). This mirrors exactly how IDM's own
    2-category bounds are shaped (cpt_min[1] == 1 - cpt_max[0] and vice
    versa) and is guaranteed feasible, unlike an arbitrary independent box
    [a,a] x [b,b] which is only feasible when a <= 0.5 <= b.
    """
    return [p0_min, 1 - p0_max], [p0_max, 1 - p0_min]


def _cn_from_bounds(bn_min, bn_max):
    """
    Build a CN directly from bn_min/bn_max, bypassing CN's `bn_min_max=`
    constructor path (which calls gum.CredalNet.intervalToCredal). That path
    is not used anywhere in the actual exp1.py/mosaic.py pipeline (every real
    CN is built via the `cn=` path instead) and turns out to be fragile: it
    can raise a pyagrum FatalError (LRSWrapper::_initLrs_) for some bound
    combinations. The resulting gum.CredalNet is, in any case, never read
    downstream (only CN_CPT.vertices / bn_min / bn_max are).
    """
    cn = CN.__new__(CN)
    cn.cpts = {}
    for var in bn_min.names():
        shape = get_cpt_shape(bn_min.cpt(var))
        cn.cpts[var] = CN_CPT(var, shape, (bn_min.cpt(var), bn_max.cpt(var)))
    cn.bn_min, cn.bn_max, cn.cn = bn_min, bn_max, None
    cn.bn_min.setProperty("name", "bn_min")
    cn.bn_max.setProperty("name", "bn_max")
    return cn


def _make_client_with_cset(p_min, p_max, n_data=100, mle=None):
    """
    A client whose credal set for X is exactly [p_min, p_max] (no IDM/BIF).
    Also sets bn_mle (needed when this client is the *target* of weighting=3,
    which reads its own MLE as the JSD reference point), defaulting to the
    credal set's own midpoint if not given explicitly.
    """
    bn = _base_bn()
    mask = gum.BayesNet(bn)
    for n in mask.nodes():
        mask.cpt(n).fillWith(1)
    c = Client(gum.BayesNet(bn), mask)

    bn_min, bn_max = gum.BayesNet(bn), gum.BayesNet(bn)
    bn_min.cpt("X").fillWith(list(p_min))
    bn_max.cpt("X").fillWith(list(p_max))
    c.cn = _cn_from_bounds(bn_min, bn_max)
    c.data = np.zeros((n_data, 1))

    if mle is None:
        mle = ((np.array(p_min) + np.array(p_max)) / 2).tolist()
    c.bn_mle = gum.BayesNet(bn)
    c.bn_mle.cpt("X").fillWith(list(mle))

    return c


def _make_client_exact_counts(counts, ess):
    """A client for X with exact integer `counts` and local-IDM credal set."""
    counts = np.asarray(counts, dtype=float)
    N = counts.sum()
    bn = _base_bn()
    mask = gum.BayesNet(bn)
    for n in mask.nodes():
        mask.cpt(n).fillWith(1)
    c = Client(gum.BayesNet(bn), mask)

    c.bn = gum.BayesNet(bn)
    c.bn.cpt("X").fillWith((counts / N).tolist())
    c.bn_mle = gum.BayesNet(bn)
    c.bn_mle.cpt("X").fillWith((counts / N).tolist())
    c.bn_counts = gum.BayesNet(bn)
    c.bn_counts.cpt("X").fillWith(counts.tolist())
    c.ess = ess

    cpt_min = counts / (N + ess)
    cpt_max = (counts + ess) / (N + ess)
    bn_min, bn_max = gum.BayesNet(bn), gum.BayesNet(bn)
    bn_min.cpt("X").fillWith(cpt_min.tolist())
    bn_max.cpt("X").fillWith(cpt_max.tolist())
    c.cn = _cn_from_bounds(bn_min, bn_max)
    c.data = np.zeros((int(N), 1))
    return c


# --------------------------------------------------------------------------
# CN / CN_CPT
# --------------------------------------------------------------------------


def test_cn_cpt_min_max_roundtrip():
    bn = _base_bn()
    bn_min, bn_max = gum.BayesNet(bn), gum.BayesNet(bn)
    bn_min.cpt("X").fillWith([0.2, 0.3])
    bn_max.cpt("X").fillWith([0.6, 0.7])
    cn = _cn_from_bounds(bn_min, bn_max)

    cpt_min, cpt_max = cn.cpt("X")
    assert np.allclose(cpt_min, [[0.2, 0.3]])
    assert np.allclose(cpt_max, [[0.6, 0.7]])


def test_cn_update_cpt_changes_bn_min_max():
    bn = _base_bn()
    bn_min, bn_max = gum.BayesNet(bn), gum.BayesNet(bn)
    bn_min.cpt("X").fillWith([0.0, 0.0])
    bn_max.cpt("X").fillWith([1.0, 1.0])
    cn = _cn_from_bounds(bn_min, bn_max)

    cn.update_cpt("X", (np.array([[0.3, 0.4]]), np.array([[0.5, 0.6]])))
    cpt_min, cpt_max = cn.cpt("X")
    assert np.allclose(cpt_min, [[0.3, 0.4]])
    assert np.allclose(cpt_max, [[0.5, 0.6]])


def test_cn_cpt_vertices_within_bounds():
    bn = _base_bn()
    bn_min, bn_max = gum.BayesNet(bn), gum.BayesNet(bn)
    bn_min.cpt("X").fillWith([0.2, 0.3])
    bn_max.cpt("X").fillWith([0.6, 0.7])
    cn = _cn_from_bounds(bn_min, bn_max)

    vertices = cn.cpts["X"].vertices[0]
    assert np.all(vertices >= np.array([0.2, 0.3]) - 1e-9)
    assert np.all(vertices <= np.array([0.6, 0.7]) + 1e-9)
    assert np.allclose(vertices.sum(axis=1), 1.0)


# --------------------------------------------------------------------------
# PriorCPT.compute (via PriorCN.compute_cpt): weighting schemas, vacuous
# fallback, and the intersection_frac diagnostic.
# --------------------------------------------------------------------------


def test_prior_weighting1_is_simple_average_of_bounds():
    target = _make_client_with_cset(*_seg(0.35, 0.55))
    c1_min, c1_max = _seg(0.1, 0.3)
    c2_min, c2_max = _seg(0.5, 0.7)
    c1 = _make_client_with_cset(c1_min, c1_max)
    c2 = _make_client_with_cset(c2_min, c2_max)

    target.reset_prior()
    target.prior_cn.compute([target, c1, c2], weighting=1)

    prior_min, prior_max = target.prior_cn.cpt("X")
    assert np.allclose(prior_min, [(np.array(c1_min) + np.array(c2_min)) / 2])
    assert np.allclose(prior_max, [(np.array(c1_max) + np.array(c2_max)) / 2])


def test_prior_weighting2_all_overlap_equals_weighting1():
    # target and both candidates share a common overlap region.
    target = _make_client_with_cset([0.3, 0.3], [0.6, 0.6])
    c1 = _make_client_with_cset([0.2, 0.2], [0.5, 0.5])
    c2 = _make_client_with_cset([0.4, 0.4], [0.7, 0.7])

    target.reset_prior()
    median = target.prior_cn.compute([target, c1, c2], weighting=2)
    prior_min, prior_max = target.prior_cn.cpt("X")
    assert np.allclose(prior_min, [[0.3, 0.3]])
    assert np.allclose(prior_max, [[0.6, 0.6]])
    assert median == 1.0


def test_prior_weighting2_excludes_non_overlapping_candidate():
    target = _make_client_with_cset([0.4, 0.4], [0.6, 0.6])
    c1 = _make_client_with_cset([0.3, 0.3], [0.5, 0.5])  # overlaps
    c2 = _make_client_with_cset(*_seg(0.05, 0.15))  # does not overlap

    target.reset_prior()
    median = target.prior_cn.compute([target, c1, c2], weighting=2)
    prior_min, prior_max = target.prior_cn.cpt("X")
    # Only c1 contributes.
    assert np.allclose(prior_min, [[0.3, 0.3]])
    assert np.allclose(prior_max, [[0.5, 0.5]])
    assert median == 0.5  # 1 of 2 candidates intersects


def test_prior_weighting2_no_overlap_falls_back_to_vacuous_row():
    # Regression test for the vacuous-prior bug (used to produce (0,0)).
    # Whether this case also emits a warning is not checked here -- that's
    # a pure UX/logging choice (currently silenced in PriorCPT.compute to
    # avoid spamming stdout during a large grid run), not part of the
    # behavior this test guards.
    target = _make_client_with_cset(*_seg(0.02, 0.08))
    c1 = _make_client_with_cset(*_seg(0.8, 0.9))
    c2 = _make_client_with_cset(*_seg(0.65, 0.78))

    target.reset_prior()
    median = target.prior_cn.compute([target, c1, c2], weighting=2)

    prior_min, prior_max = target.prior_cn.cpt("X")
    assert np.allclose(prior_min, [[0.0, 0.0]])
    assert np.allclose(prior_max, [[1.0, 1.0]])
    assert median == 0.0


def test_prior_zero_candidates_is_fully_vacuous():
    target = _make_client_with_cset([0.3, 0.3], [0.6, 0.6])
    target.reset_prior()
    target.prior_cn.compute([target], weighting=2)

    prior_min, prior_max = target.prior_cn.cpt("X")
    assert np.allclose(prior_min, [[0.0, 0.0]])
    assert np.allclose(prior_max, [[1.0, 1.0]])


def test_prior_weighting2_single_candidate_respects_intersection_filter():
    # Regression test: the former len(clients)==2 shortcut bypassed the
    # intersection filter entirely. Whether this also emits a warning is
    # not checked here -- see the comment in the no-overlap test above.
    target = _make_client_with_cset(*_seg(0.02, 0.08))
    c1 = _make_client_with_cset(*_seg(0.8, 0.9))  # does not overlap

    target.reset_prior()
    target.prior_cn.compute([target, c1], weighting=2)
    prior_min, prior_max = target.prior_cn.cpt("X")
    assert np.allclose(prior_min, [[0.0, 0.0]])
    assert np.allclose(prior_max, [[1.0, 1.0]])


def test_prior_weighting2_single_candidate_overlapping_returns_its_cset():
    target = _make_client_with_cset([0.3, 0.3], [0.6, 0.6])
    c1 = _make_client_with_cset([0.2, 0.2], [0.5, 0.5])  # overlaps

    target.reset_prior()
    target.prior_cn.compute([target, c1], weighting=2)
    prior_min, prior_max = target.prior_cn.cpt("X")
    assert np.allclose(prior_min, [[0.2, 0.2]])
    assert np.allclose(prior_max, [[0.5, 0.5]])


# --------------------------------------------------------------------------
# weighting=3: soft, JSD-based proximity to the target's own MLE.
# --------------------------------------------------------------------------


def test_prior_weighting3_matches_hand_computed_maxjsd_weights():
    target = _make_client_with_cset([0.4, 0.4], [0.6, 0.6], mle=[0.5, 0.5])
    c1 = _make_client_with_cset([0.48, 0.48], [0.52, 0.52])  # narrow, close
    c2 = _make_client_with_cset([0.1, 0.1], [0.9, 0.9])  # wide, contains 0.5 too

    target.reset_prior()
    target.prior_cn.compute([target, c1, c2], weighting=3)
    prior_min, prior_max = target.prior_cn.cpt("X")

    mle = np.array([0.5, 0.5])
    d1 = max(jsd(mle, v) for v in ([0.48, 0.52], [0.52, 0.48]))
    d2 = max(jsd(mle, v) for v in ([0.1, 0.9], [0.9, 0.1]))
    w1_raw, w2_raw = np.log(2) - d1, np.log(2) - d2
    w1, w2 = w1_raw / (w1_raw + w2_raw), w2_raw / (w1_raw + w2_raw)

    expected_min = w1 * np.array([0.48, 0.48]) + w2 * np.array([0.1, 0.1])
    expected_max = w1 * np.array([0.52, 0.52]) + w2 * np.array([0.9, 0.9])
    assert np.allclose(prior_min, [expected_min])
    assert np.allclose(prior_max, [expected_max])


def test_prior_weighting3_favors_narrow_close_over_wide_containing():
    # Regression test for the "containment" critique: a wide candidate that
    # merely CONTAINS the target's MLE must not automatically outweigh a
    # narrow candidate close to it -- unlike a raw intersection-measure
    # scheme (mu(K^e cap K^i)), which rewards width per se.
    target = _make_client_with_cset([0.4, 0.4], [0.6, 0.6], mle=[0.5, 0.5])
    c1 = _make_client_with_cset([0.48, 0.48], [0.52, 0.52])  # narrow, close
    c2 = _make_client_with_cset([0.1, 0.1], [0.9, 0.9])  # wide, contains target's MLE

    target.reset_prior()
    target.prior_cn.compute([target, c1, c2], weighting=3)
    # The resulting prior must be dominated by c1 (narrow+close): its bounds
    # pull the prior much closer to c1's own cset than to c2's.
    prior_min, prior_max = target.prior_cn.cpt("X")
    assert prior_min[0, 0] > 0.3  # far above c2's min (0.1), close to c1's (0.48)
    assert prior_max[0, 0] < 0.7  # far below c2's max (0.9), close to c1's (0.52)


def test_prior_weighting3_never_zero_even_when_disjoint():
    target = _make_client_with_cset(*_seg(0.02, 0.08), mle=[0.05, 0.95])
    c1 = _make_client_with_cset(*_seg(0.8, 0.9))  # far away, disjoint from target

    target.reset_prior()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # must NOT warn/fall back to vacuous
        target.prior_cn.compute([target, c1], weighting=3)

    prior_min, prior_max = target.prior_cn.cpt("X")
    assert not np.allclose(prior_min, [[0.0, 0.0]])
    assert not np.allclose(prior_max, [[1.0, 1.0]])
    # A single candidate always ends up with full normalized weight.
    c1_min, c1_max = c1.get_cset("X")
    assert np.allclose(prior_min, [c1_min])
    assert np.allclose(prior_max, [c1_max])


# --------------------------------------------------------------------------
# PriorCN.compute: network-wide median intersection fraction.
# --------------------------------------------------------------------------


def test_prior_cn_median_over_two_variable_network():
    bn = gum.fastBN("X[2]->Y[2]")
    mask = gum.BayesNet(bn)
    for n in mask.nodes():
        mask.cpt(n).fillWith(1)

    target = Client(gum.BayesNet(bn), gum.BayesNet(mask))
    c1 = Client(gum.BayesNet(bn), gum.BayesNet(mask))

    for c, x_bounds, y_bounds in [
        (target, ([0.4, 0.4], [0.6, 0.6]), ([0.4, 0.4], [0.6, 0.6])),
        (c1, ([0.5, 0.5], [0.55, 0.55]), _seg(0.85, 0.92)),  # X overlaps, Y doesn't
    ]:
        bn_min, bn_max = gum.BayesNet(bn), gum.BayesNet(bn)
        bn_min.cpt("X").fillWith(x_bounds[0])
        bn_max.cpt("X").fillWith(x_bounds[1])
        bn_min.cpt("Y").fillWith(
            y_bounds[0] * 2
        )  # Y has 1 parent config (X's 2 states)
        bn_max.cpt("Y").fillWith(y_bounds[1] * 2)
        c.cn = _cn_from_bounds(bn_min, bn_max)
        c.data = np.zeros((10, 1))

    target.reset_prior()
    median = target.prior_cn.compute([target, c1], weighting=2)

    # X: 1 row, intersects (frac=1.0). Y: 2 rows (one per X value), neither
    # intersects (frac=0.0 each). Overall: median([1.0, 0.0, 0.0]) == 0.0.
    assert median == 0.0


# --------------------------------------------------------------------------
# Client.get_cset
# --------------------------------------------------------------------------


def test_get_cset_root_variable_shape():
    # For a root variable (no parents) with parents=None, get_cset falls
    # through to a single-row lookup (get_cpt_index returns 0), so the
    # result is the 1D row -- not the 2D "all rows" shape used when `var`
    # has parents (see get_cset's docstring/branching).
    c = _make_client_with_cset([0.2, 0.3], [0.6, 0.7])
    cpt_min, cpt_max = c.get_cset("X")
    assert cpt_min.shape == (2,)
    assert cpt_max.shape == (2,)
    assert np.allclose(cpt_min, [0.2, 0.3])
    assert np.allclose(cpt_max, [0.6, 0.7])


def test_get_cset_with_parent_returns_all_rows_by_default():
    bn = gum.fastBN("X[2]->Y[2]")
    mask = gum.BayesNet(bn)
    for n in mask.nodes():
        mask.cpt(n).fillWith(1)
    c = Client(gum.BayesNet(bn), gum.BayesNet(mask))

    bn_min, bn_max = gum.BayesNet(bn), gum.BayesNet(bn)
    bn_min.cpt("X").fillWith([0.2, 0.3])
    bn_max.cpt("X").fillWith([0.6, 0.7])
    bn_min.cpt("Y").fillWith([0.3, 0.3, 0.35, 0.35])
    bn_max.cpt("Y").fillWith([0.6, 0.6, 0.65, 0.65])
    c.cn = _cn_from_bounds(bn_min, bn_max)

    cpt_min, cpt_max = c.get_cset("Y")
    assert cpt_min.shape == (2, 2)
    assert cpt_max.shape == (2, 2)


def test_get_cset_with_explicit_parent_returns_single_row():
    bn = gum.fastBN("X[2]->Y[2]")
    mask = gum.BayesNet(bn)
    for n in mask.nodes():
        mask.cpt(n).fillWith(1)
    c = Client(gum.BayesNet(bn), gum.BayesNet(mask))

    bn_min, bn_max = gum.BayesNet(bn), gum.BayesNet(bn)
    bn_min.cpt("Y").fillWith([0.35, 0.35, 0.4, 0.4])
    bn_max.cpt("Y").fillWith([0.65, 0.65, 0.6, 0.6])
    bn_min.cpt("X").fillWith([0.0, 0.0])
    bn_max.cpt("X").fillWith([1.0, 1.0])
    c.cn = _cn_from_bounds(bn_min, bn_max)

    row0_min, row0_max = c.get_cset("Y", parents={"X": "0"})
    row1_min, row1_max = c.get_cset("Y", parents={"X": "1"})
    assert np.allclose(row0_min, [0.35, 0.35])
    assert np.allclose(row1_min, [0.4, 0.4])
    assert np.allclose(row0_max, [0.65, 0.65])
    assert np.allclose(row1_max, [0.6, 0.6])


# --------------------------------------------------------------------------
# Client.mosaic_cn_cpt: the MOSAIC update rule.
# --------------------------------------------------------------------------


def test_mosaic_update_matches_hand_computed_formula():
    # counts N[x=0]=3, N[x=1]=7 (N=10), ess=2.
    c = _make_client_exact_counts([3, 7], ess=2)
    c.prior_cn = PriorCN(c.gt)
    c.prior_cn.update_cpt("X", (np.array([[0.2, 0.1]]), np.array([[0.4, 0.3]])))

    c.mosaic_cn_cpt("X")
    got_min, got_max = c.cn_mosaic.cpt("X")

    n_pi, ess = 10, 2
    mle = np.array([0.3, 0.7])
    expected_min = n_pi / (n_pi + ess) * mle + ess / (n_pi + ess) * np.array([0.2, 0.1])
    expected_max = n_pi / (n_pi + ess) * mle + ess / (n_pi + ess) * np.array([0.4, 0.3])

    assert np.allclose(got_min, [expected_min])
    assert np.allclose(got_max, [expected_max])


def test_mosaic_update_with_vacuous_prior_is_identity():
    # K^{e+} must equal K^e exactly when the prior is vacuous ([0,1]).
    for ess in [1, 2, 5]:
        for counts in [[3, 7], [0, 10], [50, 1]]:
            c = _make_client_exact_counts(counts, ess=ess)
            c.prior_cn = PriorCN(c.gt)  # default: vacuous
            c.mosaic_cn_cpt("X")

            got_min, got_max = c.cn_mosaic.cpt("X")
            exp_min, exp_max = c.get_cset("X")
            assert np.allclose(got_min, exp_min, atol=1e-9)
            assert np.allclose(got_max, exp_max, atol=1e-9)


def test_mosaic_update_shrinks_credal_set_when_prior_informative():
    c = _make_client_exact_counts([3, 7], ess=2)
    c.prior_cn = PriorCN(c.gt)
    c.prior_cn.update_cpt("X", (np.array([[0.2, 0.1]]), np.array([[0.4, 0.3]])))

    c.mosaic_cn_cpt("X")
    got_min, got_max = c.cn_mosaic.cpt("X")
    orig_min, orig_max = c.get_cset("X")

    # K^{e+} subseteq K^e (Prop. in cap6_extract.tex).
    assert np.all(got_min >= orig_min - 1e-9)
    assert np.all(got_max <= orig_max + 1e-9)
    # And it's a strict subset (informative, non-vacuous prior).
    assert np.any(got_max - got_min < orig_max - orig_min - 1e-9)


def test_mosaic_soundness_mle_in_update_iff_mle_in_prior():
    # Proposition (soundness of the prior): theta_hat in K^{e+} iff theta_hat in T.
    c = _make_client_exact_counts([3, 7], ess=2)
    mle = np.array([0.3, 0.7])

    # Case 1: prior T contains theta_hat.
    c.prior_cn = PriorCN(c.gt)
    c.prior_cn.update_cpt("X", (np.array([[0.1, 0.5]]), np.array([[0.5, 0.9]])))
    assert np.all(mle >= [0.1, 0.5]) and np.all(mle <= [0.5, 0.9])
    c.mosaic_cn_cpt("X")
    got_min, got_max = c.cn_mosaic.cpt("X")
    assert np.all(mle >= got_min[0] - 1e-9) and np.all(mle <= got_max[0] + 1e-9)

    # Case 2: prior T does NOT contain theta_hat.
    c2 = _make_client_exact_counts([3, 7], ess=2)
    c2.prior_cn = PriorCN(c2.gt)
    c2.prior_cn.update_cpt("X", (np.array([[0.8, 0.0]]), np.array([[0.95, 0.15]])))
    assert not (np.all(mle >= [0.8, 0.0]) and np.all(mle <= [0.95, 0.15]))
    c2.mosaic_cn_cpt("X")
    got_min2, got_max2 = c2.cn_mosaic.cpt("X")
    assert not (np.all(mle >= got_min2[0] - 1e-9) and np.all(mle <= got_max2[0] + 1e-9))


# --------------------------------------------------------------------------
# End-to-end sanity: generate_base_info -> learn_cn -> mosaic_cn, on real
# (randomly generated) data, tying the whole Client-level flow together.
# --------------------------------------------------------------------------


def test_generate_base_info_and_vacuous_mosaic_matches_local_idm():
    bn = gum.fastBN("X[2]->Y[2]")
    mask = gum.BayesNet(bn)
    for n in mask.nodes():
        mask.cpt(n).fillWith(1)
    c = Client(gum.BayesNet(bn), gum.BayesNet(mask))

    c.generate_base_info(n=200, ess=2)
    c.mosaic_cn()  # prior stays vacuous (never computed)

    for var in c.bn.names():
        got = [x.flatten() for x in c.cn_mosaic.cpt(var)]
        exp = [x.flatten() for x in c.get_cset(var)]
        assert np.allclose(got, exp, atol=BIF_ATOL)
