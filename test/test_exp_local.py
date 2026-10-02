"""
Tests for exp_local.py: the full-grid local learning & update pipeline.

exp(config, n, rep) is fully self-contained, so it is called directly
here with an explicit config. It returns (row, models, task_id): `row` is
one df_tot.csv line, `models` is a dict of CPT snapshots (empty unless
save_models is True), and `task_id` identifies both.
"""

import copy

import numpy as np
import pyagrum as gum
import pytest

import exp_local as exp_mod
from src.config import set_seed
from src.utils import gt_containment_frac

EXPECTED_ROW_KEYS = (
    set(exp_mod.GRID_KEYS)
    | {"size", "rep", "mle", "idm_min", "idm_mean", "idm_max", "intersection_frac"}
    | {
        f"mos_w{w}_{stat}"
        for w in exp_mod.WEIGHTING_SCHEMES
        for stat in ("min", "mean", "max")
    }
    | {"idm_gt_contained"}
    | {f"mos_w{w}_gt_contained" for w in exp_mod.WEIGHTING_SCHEMES}
)

EXPECTED_MODEL_KEYS = {"bn_mle", "idm_min", "idm_max"} | {
    f"{kind}_w{w}_{bound}"
    for kind in ("prior", "mos")
    for w in exp_mod.WEIGHTING_SCHEMES
    for bound in ("min", "max")
}


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
    # theta_hat^e is always a member of the local IDM credal set, for any
    # ess; see test_idm_min_never_exceeds_mle below.
    assert row["idm_min"] <= row["mle"] + 1e-9

    assert 0.0 <= row["idm_gt_contained"] <= 1.0
    for w in exp_mod.WEIGHTING_SCHEMES:
        assert 0.0 <= row[f"mos_w{w}_gt_contained"] <= 1.0

    assert task_id == (5, 2, 1.0, 20, 50, 0)  # GRID_KEYS order + (size, rep)


def test_gt_contained_matches_direct_computation():
    # Cross-check exp()'s own gt_contained values against gt_containment_frac
    # called directly on the same, independently rebuilt credal sets.
    config = BASE_CONFIG
    np.random.seed(hash((50, 0)) % (2**32))
    gum.initRandom(hash((50, 0)) % (2**32))
    clients = exp_mod.init_clients(config)
    for c in clients.values():
        c.generate_base_info(50, config["ess"])
    client_exp = clients[config["client_num"]]
    prior_clients = [client_exp] + [
        c for e, c in clients.items() if e != config["client_num"]
    ]
    bn_base = gum.loadBN(config["bn_base_path"])

    row, _, _ = exp_mod.exp(config, 50, rep=0)

    expected_idm = gt_containment_frac(
        bn_base, client_exp.cn.bn_min, client_exp.cn.bn_max
    )
    assert row["idm_gt_contained"] == pytest.approx(expected_idm)

    for w in exp_mod.WEIGHTING_SCHEMES:
        client_exp.reset_prior()
        client_exp.prior_cn.compute(prior_clients, weighting=w)
        client_exp.mosaic_cn()
        expected_mos = gt_containment_frac(
            bn_base, client_exp.cn_mosaic.bn_min, client_exp.cn_mosaic.bn_max
        )
        assert row[f"mos_w{w}_gt_contained"] == pytest.approx(expected_mos)


@pytest.mark.parametrize("ess", [1, 2, 5, 10, 20])
def test_idm_min_never_exceeds_mle(ess):
    # Regression test: IDM's credal interval always contains the MLE, so
    # idm_min must never exceed row["mle"].
    config = dict(BASE_CONFIG, n_clients=2, prob_shift=0.0, ess=ess, n_bns=50)
    for n in (20, 100, 300):
        row, _, _ = exp_mod.exp(config, n, rep=0)
        assert row["idm_min"] <= row["mle"] + 1e-9, (ess, n, row["idm_min"], row["mle"])


def test_exp_models_snapshot_is_well_formed():
    # Each variable's snapshot is self-describing: {"cpt", "parents",
    # "labels"}, not a bare array.
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
    # main()'s csv.DictWriter raises if a row's keys don't match
    # ROW_FIELDNAMES exactly; this locks that invariant directly.
    assert set(exp_mod.ROW_FIELDNAMES) == EXPECTED_ROW_KEYS
    assert len(exp_mod.ROW_FIELDNAMES) == len(set(exp_mod.ROW_FIELDNAMES))


def test_exp_save_models_false_skips_model_computation():
    # save_models=False must skip snapshot_cpts() entirely, not just leave
    # `models` unused, so it also saves the compute. `row` is unaffected.
    config = dict(BASE_CONFIG, save_models=False)
    row, models, task_id = exp_mod.exp(config, 50, rep=0)

    assert models == {}
    assert set(row.keys()) == EXPECTED_ROW_KEYS
    assert task_id == (5, 2, 1.0, 20, 50, 0)


def test_exp_save_models_defaults_to_true():
    # BASE_CONFIG has no "save_models" key; omitting it must preserve the
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
    # in order, not e.g. missing a field, which could silently collide.
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
    # Regression test: a duplicate value in one grid list makes build_tasks
    # silently emit the same task_id twice, overwriting the first's output.
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
    # Regression test: exp() must reseed per (n, rep) task, not inherit the
    # same RNG state across forked workers.
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
    # exp() is fully stateless: interleaved calls with different configs
    # for the same (n, rep) must each reflect only their own config.
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
    # Every (combination, size, rep) must appear exactly once in the
    # flattened task list, each carrying its own resolved hyperparameters.
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


def test_hyperparameter_combos_deduplicates_alpha_only_for_prob_shift_zero():
    # alpha is irrelevant at prob_shift=0.0, so sweeping it there would only
    # produce redundant, byte-identical tasks.
    config = dict(
        BASE_CONFIG,
        n_clients=[5],
        ess=[1],
        prob_shift=[0.0, 0.5],
        alpha=[10, 20, 30],
    )
    combos = exp_mod.hyperparameter_combos(config)

    zero_combos = [c for c in combos if c["prob_shift"] == 0.0]
    nonzero_combos = [c for c in combos if c["prob_shift"] == 0.5]

    assert len(zero_combos) == 1  # not len(alpha) == 3
    assert zero_combos[0]["alpha"] == config["alpha"][0]
    assert len(nonzero_combos) == len(config["alpha"]) == 3
    assert sorted(c["alpha"] for c in nonzero_combos) == sorted(config["alpha"])

    # build_tasks must reflect the same deduplicated total, not the naive
    # full product.
    tasks = exp_mod.build_tasks(dict(config, n_repetitions=1), sizes=[10])
    assert len(tasks) == len(combos) == 4  # 1 (prob_shift=0) + 3 (prob_shift=0.5)


def test_prob_shift_zero_ignores_alpha():
    # At prob_shift=0.0, every client is an unperturbed copy of bn_base
    # regardless of alpha, so exp()'s result must not depend on it.
    config_a = dict(BASE_CONFIG, prob_shift=0.0, alpha=5)
    config_b = dict(BASE_CONFIG, prob_shift=0.0, alpha=500)

    row_a, _, id_a = exp_mod.exp(config_a, 40, rep=0)
    row_b, _, id_b = exp_mod.exp(config_b, 40, rep=0)

    for key in EXPECTED_ROW_KEYS - {"alpha"}:
        assert np.isclose(row_a[key], row_b[key], atol=1e-12), key


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
        expected_labels = sorted(c.label for e, c in clients.items() if e != client_num)
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
