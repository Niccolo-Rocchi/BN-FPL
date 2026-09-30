"""
Tests for exp_global.py: the global optimization phase (lexicographic
alpha >> beta clustering, see src/global_opt.py), the counterpart to
test_exp_local.py for exp_local.py's local learning & update phase.
"""

import numpy as np
import pyagrum as gum
import pytest

import exp_global as exp_g
from src.config import set_seed
from src.utils import get_tabular_cpt

BASE_CONFIG = {
    "n_clients": 5,
    "ess": 2,
    "alpha": 20,
    "prob_shift": 0.5,
    "res_path": "results_global",
    "bn_base_path": "cancer.bif",
}


@pytest.fixture(autouse=True)
def _seed():
    set_seed()


def test_init_clients_perturbs_every_client_symmetrically():
    # Unlike exp_local.py's init_clients (client 0 always == bn_base exactly),
    # prob_shift applies to every client here, including client 0.
    np.random.seed(0)
    gum.initRandom(0)
    clients = exp_g.init_clients(dict(BASE_CONFIG, prob_shift=1.0, alpha=0.01))
    bn_base = gum.loadBN(BASE_CONFIG["bn_base_path"])

    for c in clients.values():
        all_kept = all(
            np.all(get_tabular_cpt(c.mask.cpt(var)) == 1) for var in bn_base.names()
        )
        assert not all_kept  # prob_shift=1.0 -> every row of every client perturbed


def test_init_clients_prob_shift_zero_matches_bn_base_exactly():
    clients = exp_g.init_clients(dict(BASE_CONFIG, prob_shift=0.0))
    bn_base = gum.loadBN(BASE_CONFIG["bn_base_path"])

    for c in clients.values():
        for var in bn_base.names():
            assert np.all(get_tabular_cpt(c.mask.cpt(var)) == 1)
            assert np.allclose(
                get_tabular_cpt(c.gt.cpt(var)), get_tabular_cpt(bn_base.cpt(var))
            )


def test_exp_global_returns_expected_schema_and_sane_values():
    row, models, task_id = exp_g.exp_global(BASE_CONFIG, 50, rep=0)

    assert set(row.keys()) == set(exp_g.ROW_FIELDNAMES)
    assert row["size"] == 50 and row["rep"] == 0
    for k in exp_g.GRID_KEYS:
        assert row[k] == BASE_CONFIG[k]
    assert task_id == (5, 2, 0.5, 20, 50, 0)  # GRID_KEYS order + (size, rep)

    for m in exp_g.METHODS:
        for stat in ("precision", "recall", "f1"):
            v = row[f"{m}_{stat}"]
            assert np.isnan(v) or 0.0 - 1e-9 <= v <= 1.0 + 1e-9
        for stat in ("ari", "ami"):
            v = row[f"{m}_{stat}"]
            assert np.isnan(v) or -1.0 - 1e-9 <= v <= 1.0 + 1e-9
        assert row[f"{m}_n_clusters_mean"] >= 1.0

    assert row["n_clusters_true_mean"] >= 1.0
    assert len(models) > 0  # save_models defaults to True


def test_exp_global_save_models_false_returns_empty_models():
    row, models, _ = exp_g.exp_global(dict(BASE_CONFIG, save_models=False), 50, rep=0)
    assert models == {}
    assert set(row.keys()) == set(exp_g.ROW_FIELDNAMES)


def test_exp_global_prob_shift_zero_gives_one_true_cluster_everywhere():
    # Every client stays byte-identical to bn_base (see the init_clients
    # test above), so the ground-truth partition is trivially "everyone in
    # one cluster" for every mechanism, regardless of how well any
    # clustering METHOD actually recovers it.
    row, _, _ = exp_g.exp_global(dict(BASE_CONFIG, prob_shift=0.0), 50, rep=0)
    assert row["n_clusters_true_mean"] == pytest.approx(1.0)


def test_exp_global_reproducible_given_same_seed_inputs():
    row1, _, _ = exp_g.exp_global(BASE_CONFIG, 50, rep=3)
    row2, _, _ = exp_g.exp_global(BASE_CONFIG, 50, rep=3)
    assert row1 == row2 or all(
        (row1[k] == row2[k]) or (np.isnan(row1[k]) and np.isnan(row2[k]))
        for k in row1
    )
