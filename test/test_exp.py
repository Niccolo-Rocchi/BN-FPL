"""
Tests for exp.py: the end-to-end local learning & update pipeline.

exp(config, n, rep) is fully self-contained (no worker-global state, so it
is called directly here with an explicit config -- no pool/initializer
needed) and returns (row, models, task_id): `row` is the JSD summary dict
(destined for one line of df_tot.csv), `models` is a dict of raw CPT
snapshots (destined for its own results/models/<task_id>.pkl file, kept for
the future global optimization phase -- empty when config["save_models"] is
False, which conf.yaml sets for a plain local-learning-and-update sweep
(exp()'s own fallback, if the key is missing entirely, is True), and
`task_id` is the (n_clients, ess, prob_shift, alpha, size, rep) key both are
filed under.
"""

import copy

import numpy as np
import pyagrum as gum
import pytest

import exp as exp_mod
from src.config import set_seed

EXPECTED_ROW_KEYS = (
    set(exp_mod.GRID_KEYS)
    | {"size", "rep", "mle", "idm_min", "idm_mean", "idm_max", "intersection_frac"}
    | {
        f"mos_w{w}_{stat}"
        for w in exp_mod.WEIGHTING_SCHEMES
        for stat in ("min", "mean", "max")
    }
)

EXPECTED_MODEL_KEYS = (
    {"bn_mle", "idm_min", "idm_max"}
    | {
        f"{kind}_w{w}_{bound}"
        for kind in ("prior", "mos")
        for w in exp_mod.WEIGHTING_SCHEMES
        for bound in ("min", "max")
    }
)


BASE_CONFIG = {
    "n_clients": 5,
    "client_num": 0,
    "ess": 2,
    "alpha": 20,
    "prob_shift": 1.0,
    "res_path": "results",
    "bn_base_path": "cancer.bif",
    "n_bns": 10,
}


@pytest.fixture(autouse=True)
def _seed():
    set_seed()


@pytest.fixture
def clients_template():
    return exp_mod.init_clients(BASE_CONFIG)


def test_exp_returns_expected_schema_and_sane_values():
    row, models, task_id = exp_mod.exp(BASE_CONFIG, 50, rep=0)

    assert set(row.keys()) == EXPECTED_ROW_KEYS
    assert row["size"] == 50 and row["rep"] == 0
    for k in exp_mod.GRID_KEYS:
        assert row[k] == BASE_CONFIG[k]

    assert row["idm_min"] <= row["idm_mean"] + 1e-9 <= row["idm_max"] + 1e-9
    for w in exp_mod.WEIGHTING_SCHEMES:
        assert row[f"mos_w{w}_min"] <= row[f"mos_w{w}_mean"] + 1e-9
        assert row[f"mos_w{w}_mean"] <= row[f"mos_w{w}_max"] + 1e-9

    assert 0.0 <= row["intersection_frac"] <= 1.0
    assert row["mle"] >= 0.0

    assert task_id == (5, 2, 1.0, 20, 50, 0)  # GRID_KEYS order + (size, rep)


def test_exp_models_snapshot_is_well_formed():
    # Each variable's snapshot is self-describing: {"cpt", "parents",
    # "labels"} (see snapshot_cpts) -- not a bare array -- specifically so a
    # future reader never has to separately reconstruct a gum.BayesNet (and
    # risk using cpt.topandas()'s row order by mistake) to know what each
    # row/column means.
    row, models, task_id = exp_mod.exp(BASE_CONFIG, 50, rep=0)

    assert set(models.keys()) == EXPECTED_MODEL_KEYS
    bn_base = gum.loadBN(BASE_CONFIG["bn_base_path"])
    for key, cpts in models.items():
        assert set(cpts.keys()) == set(bn_base.names()), key
        for var, entry in cpts.items():
            assert set(entry.keys()) == {"cpt", "parents", "labels"}, (key, var)
            assert entry["cpt"].ndim == 2, (key, var)
            assert len(entry["parents"]) == entry["cpt"].shape[0], (key, var)
            assert entry["labels"] == list(bn_base.variable(var).labels()), (key, var)

    # every "min" snapshot must be componentwise <= its "max" counterpart
    for w in exp_mod.WEIGHTING_SCHEMES:
        for var in bn_base.names():
            assert np.all(
                models[f"mos_w{w}_min"][var]["cpt"]
                <= models[f"mos_w{w}_max"][var]["cpt"] + 1e-9
            )
    # bn_mle's rows must be valid probability distributions.
    for var, entry in models["bn_mle"].items():
        assert np.allclose(entry["cpt"].sum(axis=1), 1.0)


def test_row_fieldnames_matches_row_schema():
    # main() writes df_tot.csv incrementally with csv.DictWriter(fieldnames=
    # ROW_FIELDNAMES), which raises if a row's keys don't match exactly --
    # this locks that invariant as a fast, direct test (rather than only
    # discovering a mismatch mid-run, potentially hours in).
    assert set(exp_mod.ROW_FIELDNAMES) == EXPECTED_ROW_KEYS
    assert len(exp_mod.ROW_FIELDNAMES) == len(set(exp_mod.ROW_FIELDNAMES))


def test_exp_save_models_false_skips_model_computation():
    # save_models=False (the default for a plain local-learning-and-update
    # sweep, see conf.yaml) must skip snapshot_cpts() entirely -- not just
    # leave `models` unused -- so it also saves the compute, not only the
    # eventual pickle/RAM. `row` must be entirely unaffected.
    config = dict(BASE_CONFIG, save_models=False)
    row, models, task_id = exp_mod.exp(config, 50, rep=0)

    assert models == {}
    assert set(row.keys()) == EXPECTED_ROW_KEYS
    assert task_id == (5, 2, 1.0, 20, 50, 0)


def test_exp_save_models_defaults_to_true():
    # BASE_CONFIG has no "save_models" key -- omitting it must preserve the
    # old (models-always-saved) behavior, so no other test needs updating.
    row, models, task_id = exp_mod.exp(BASE_CONFIG, 50, rep=0)
    assert set(models.keys()) == EXPECTED_MODEL_KEYS


def test_model_filename_is_unique_across_a_grid():
    config = dict(
        BASE_CONFIG,
        n_clients=[5, 9],
        ess=[1, 2],
        prob_shift=[0.0, 0.5],
        alpha=[10, 20],
        n_repetitions=2,
    )
    tasks = exp_mod.build_tasks(config, sizes=[10, 20])
    task_ids = [
        tuple(cfg[k] for k in exp_mod.GRID_KEYS) + (n, rep) for cfg, n, rep in tasks
    ]
    filenames = [exp_mod._model_filename(t) for t in task_ids]

    assert len(filenames) == len(set(filenames))
    # The filename must be exactly recoverable as the task_id's own fields,
    # in order -- not e.g. missing a field, which could silently collide.
    for t, fname in zip(task_ids, filenames):
        assert fname == "_".join(str(x) for x in t) + ".pkl"


def test_check_unique_task_ids_passes_for_a_normal_grid():
    config = dict(
        BASE_CONFIG,
        n_clients=[5, 9],
        ess=[1, 2],
        prob_shift=[0.0],
        alpha=[10],
        n_repetitions=2,
    )
    tasks = exp_mod.build_tasks(config, sizes=[10, 20])
    exp_mod._check_unique_task_ids(tasks)  # must not raise


def test_check_unique_task_ids_raises_on_duplicate_grid_value():
    # Regression test for the exact failure mode this check exists to catch:
    # a duplicate value inside one grid hyperparameter list (e.g. a typo'd
    # `ess: [1, 1]` in conf.yaml) makes build_tasks silently emit the same
    # task_id twice, which would otherwise make the second task's CSV row /
    # model file silently overwrite the first's.
    config = dict(
        BASE_CONFIG,
        n_clients=[5],
        ess=[1, 1],
        prob_shift=[0.0],
        alpha=[10],
        n_repetitions=1,
    )
    tasks = exp_mod.build_tasks(config, sizes=[10])
    with pytest.raises(AssertionError):
        exp_mod._check_unique_task_ids(tasks)


@pytest.mark.parametrize("client_num", [0, 2, 4])
def test_exp_works_for_any_client_num(client_num):
    config = dict(BASE_CONFIG, client_num=client_num)

    # Must not raise, regardless of which client is the update target.
    row, models, task_id = exp_mod.exp(config, 50, rep=0)
    assert set(row.keys()) == EXPECTED_ROW_KEYS
    assert set(models.keys()) == EXPECTED_MODEL_KEYS


def test_different_reps_produce_different_data():
    # Regression test: exp() must reseed per (n, rep) task and regenerate the
    # whole DGP (client perturbations included, not just the sampled data)
    # for each repetition. Before the fix, forked worker processes inherited
    # an identical RNG state, so distinct repetitions processed by different
    # (freshly-forked) workers silently produced byte-for-byte identical
    # client-0 data -- visible here as an identical "mle" JSD value.
    row0, _, _ = exp_mod.exp(BASE_CONFIG, 40, rep=0)
    row1, _, _ = exp_mod.exp(BASE_CONFIG, 40, rep=1)

    assert row0["mle"] != row1["mle"]


def test_same_task_is_reproducible():
    # The reseeding fix must not break reproducibility: the SAME (config, n,
    # rep) run twice must give identical results.
    row_a, _, task_id_a = exp_mod.exp(BASE_CONFIG, 40, rep=0)
    row_b, _, task_id_b = exp_mod.exp(BASE_CONFIG, 40, rep=0)

    assert task_id_a == task_id_b
    for key in EXPECTED_ROW_KEYS:
        # atol handles residual float non-associativity (e.g. multi-threaded
        # BLAS summation order) between two otherwise identical runs.
        assert np.isclose(row_a[key], row_b[key], atol=1e-9), key


def test_different_hyperparameter_combinations_do_not_cross_contaminate():
    # exp() is fully stateless (no worker-global config): this is what makes
    # it safe for main()'s flattened pool to freely interleave tasks from
    # DIFFERENT hyperparameter combinations across all workers at once,
    # instead of processing one combination at a time. Simulated here by
    # interleaving two configs that differ in n_clients (so they produce
    # differently-sized `clients` dicts) for the SAME (n, rep): each call
    # must reflect only its own config, regardless of call order, and must
    # be independently reproducible.
    config_a = dict(BASE_CONFIG, n_clients=5)
    config_b = dict(BASE_CONFIG, n_clients=9)

    row_a1, _, id_a1 = exp_mod.exp(config_a, 40, rep=0)
    row_b1, _, id_b1 = exp_mod.exp(config_b, 40, rep=0)
    row_a2, _, id_a2 = exp_mod.exp(config_a, 40, rep=0)
    row_b2, _, id_b2 = exp_mod.exp(config_b, 40, rep=0)

    assert row_a1["n_clients"] == row_a2["n_clients"] == 5
    assert row_b1["n_clients"] == row_b2["n_clients"] == 9
    assert id_a1 == id_a2 and id_b1 == id_b2
    assert id_a1 != id_b1

    assert np.isclose(row_a1["mle"], row_a2["mle"], atol=1e-9)
    assert np.isclose(row_b1["mle"], row_b2["mle"], atol=1e-9)


def test_build_tasks_covers_every_combination_exactly_once():
    # Direct test for the CSV-attribution concern: every (hyperparameter
    # combination, size, rep) must appear in the flattened task list exactly
    # once, each carrying its OWN correctly-resolved hyperparameter values
    # (not e.g. always the last combination's, a classic late-binding-
    # closure bug this dict-per-task design avoids).
    config = dict(
        BASE_CONFIG,
        n_clients=[5, 9],
        ess=[1, 2],
        prob_shift=[0.0],
        alpha=[10],
        n_repetitions=2,
    )
    sizes = [10, 20]

    tasks = exp_mod.build_tasks(config, sizes)

    assert len(tasks) == 2 * 2 * 1 * 1 * len(sizes) * config["n_repetitions"]

    seen = set()
    for combo_config, n, rep in tasks:
        key = (combo_config["n_clients"], combo_config["ess"], n, rep)
        assert key not in seen, f"duplicate task: {key}"
        seen.add(key)

        # every non-grid key is carried over unchanged from the base config
        assert combo_config["alpha"] == 10
        assert combo_config["prob_shift"] == 0.0
        assert combo_config["bn_base_path"] == BASE_CONFIG["bn_base_path"]

    expected_keys = {
        (nc, ess, n, rep)
        for nc in (5, 9)
        for ess in (1, 2)
        for n in sizes
        for rep in range(2)
    }
    assert seen == expected_keys


def test_prior_clients_excludes_target_regardless_of_client_num(clients_template):
    # Direct regression test for the client_num generalization: prior_clients
    # must always start with the target client and never include it again.
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
