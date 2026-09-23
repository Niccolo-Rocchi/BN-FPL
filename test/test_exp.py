"""
Tests for exp.py: the end-to-end local learning & update pipeline (one
sample size, a handful of repetitions, no multiprocessing -- `exp()` is
called directly with `_init_worker`'s globals set up by hand).
"""

import numpy as np
import pyagrum as gum
import pytest

import exp as exp_mod
from src.config import create_clean_dir, set_seed


BASE_CONFIG = {
    "n_clients": 5,
    "client_num": 0,
    "ess": 2,
    "alpha": 20,
    "prob_shift": 1.0,
    "weighting": 2,
    "res_path": "results",
    "bn_base_path": "cancer.bif",
}


@pytest.fixture(autouse=True)
def _seed():
    set_seed()


@pytest.fixture
def clients_template():
    return exp_mod.init_clients(BASE_CONFIG)


def test_exp_runs_end_to_end_for_client_zero(tmp_path):
    exp_mod._init_worker(0, BASE_CONFIG)
    ss_path = tmp_path / "ss50"
    create_clean_dir(ss_path)

    exp_mod.exp(50, ss_path, rep=0)

    for suffix in [
        "bn.bif",
        "idm_bn_min.bif",
        "idm_bn_max.bif",
        "prior_bn_min.bif",
        "prior_bn_max.bif",
        "mos_bn_min.bif",
        "mos_bn_max.bif",
        "intersection.txt",
    ]:
        assert (ss_path / f"0-{suffix}").exists(), suffix

    with open(ss_path / "0-intersection.txt") as f:
        median = float(f.read())
    assert 0.0 <= median <= 1.0


@pytest.mark.parametrize("client_num", [0, 2, 4])
def test_exp_works_for_any_client_num(tmp_path, client_num):
    config = dict(BASE_CONFIG, client_num=client_num)
    exp_mod._init_worker(client_num, config)
    ss_path = tmp_path / f"ss50_client{client_num}"
    create_clean_dir(ss_path)

    # Must not raise, regardless of which client is the update target.
    exp_mod.exp(50, ss_path, rep=0)
    assert (ss_path / "0-mos_bn_min.bif").exists()


def test_different_reps_produce_different_data(tmp_path):
    # Regression test: exp() must reseed per (n, rep) task and regenerate the
    # whole DGP (client perturbations included, not just the sampled data)
    # for each repetition. Before the fix, forked worker processes inherited
    # an identical RNG state, so distinct repetitions processed by different
    # (freshly-forked) workers silently produced byte-for-byte identical
    # client-0 data.
    exp_mod._init_worker(0, BASE_CONFIG)

    results = []
    for rep in [0, 1]:
        ss_path = tmp_path / f"ss40_rep{rep}"
        create_clean_dir(ss_path)
        exp_mod.exp(40, ss_path, rep=rep)
        results.append(gum.loadBN(str(ss_path / f"{rep}-bn.bif")))

    any_differs = any(
        not np.allclose(
            results[0].cpt(var)[:].flatten(), results[1].cpt(var)[:].flatten()
        )
        for var in results[0].names()
    )
    assert any_differs, "rep=0 and rep=1 produced identical client-0 data"


def test_same_task_is_reproducible(tmp_path):
    # The reseeding fix must not break reproducibility: the SAME (n, rep)
    # run twice must give identical results.
    exp_mod._init_worker(0, BASE_CONFIG)

    saved = []
    for i in range(2):
        ss_path = tmp_path / f"ss40_run{i}"
        create_clean_dir(ss_path)
        exp_mod.exp(40, ss_path, rep=0)
        saved.append(gum.loadBN(str(ss_path / "0-bn.bif")))

    for var in saved[0].names():
        assert np.allclose(
            saved[0].cpt(var)[:].flatten(), saved[1].cpt(var)[:].flatten()
        )


def test_saved_bn_bif_is_the_exact_mle_not_smoothed(tmp_path, clients_template):
    # save_results must save client.bn_mle (exact) under "-bn.bif", not
    # client.bn (smoothed via learn_bn_params) -- this is what JSD.py's
    # "mle" reference curve and Plot_KL.ipynb both read back.
    client = clients_template[0]
    client.generate_base_info(30, BASE_CONFIG["ess"])

    # Force client.bn (smoothed) and client.bn_mle (exact) to visibly
    # differ, regardless of what generate_base_info's random data happened
    # to produce, by overwriting client.bn with an obviously-smoothed stand
    # -in and keeping bn_mle untouched.
    smoothed_stub = gum.BayesNet(client.bn_mle)
    for var in smoothed_stub.names():
        var_size = smoothed_stub.variable(var).domainSize()
        n_rows = smoothed_stub.cpt(var).domainSize() // var_size
        smoothed_stub.cpt(var).fillWith([1.0 / var_size] * var_size * n_rows)
    client.bn = smoothed_stub

    ss_path = tmp_path / "ss30_mle_check"
    create_clean_dir(ss_path)
    exp_mod.save_results(client, ss_path, rep=0)

    saved_bn = gum.loadBN(str(ss_path / "0-bn.bif"))
    for var in saved_bn.names():
        assert np.allclose(
            saved_bn.cpt(var)[:].flatten(), client.bn_mle.cpt(var)[:].flatten()
        )
    any_differs = any(
        not np.allclose(
            saved_bn.cpt(var)[:].flatten(), client.bn.cpt(var)[:].flatten()
        )
        for var in saved_bn.names()
    )
    assert any_differs


def test_prior_clients_excludes_target_regardless_of_client_num(clients_template):
    # Direct regression test for the client_num generalization: prior_clients
    # must always start with the target client and never include it again.
    import copy

    clients = copy.deepcopy(clients_template)
    for client_num in [0, 1, 3]:
        client_exp = clients[client_num]
        prior_clients = [client_exp] + [
            c for e, c in clients.items() if e != client_num
        ]

        assert prior_clients[0] is client_exp
        assert client_exp not in prior_clients[1:]
        assert len(prior_clients) == len(clients)
        # every other client appears exactly once among the candidates
        other_labels = sorted(c.label for c in prior_clients[1:])
        expected_labels = sorted(
            c.label for e, c in clients.items() if e != client_num
        )
        assert other_labels == expected_labels


def test_init_clients_first_client_is_unperturbed():
    clients = exp_mod.init_clients(BASE_CONFIG)
    bn_base = gum.loadBN(BASE_CONFIG["bn_base_path"])
    for var in bn_base.names():
        assert np.allclose(
            clients[0].gt.cpt(var)[:].flatten(), bn_base.cpt(var)[:].flatten()
        )


def test_init_clients_other_clients_perturbed_with_prob_shift_1(clients_template):
    # With prob_shift=1.0, every mechanism of every non-first client must be
    # marked as perturbed (mask == 0) relative to bn_base.
    for e in range(1, BASE_CONFIG["n_clients"]):
        client = clients_template[e]
        for var in client.mask.names():
            assert np.all(client.mask.cpt(var)[:] == 0)


def test_save_results_writes_intersection_file(tmp_path):
    exp_mod._init_worker(0, BASE_CONFIG)
    ss_path = tmp_path / "ss80"
    create_clean_dir(ss_path)
    exp_mod.exp(80, ss_path, rep=1)

    assert (ss_path / "1-intersection.txt").exists()
