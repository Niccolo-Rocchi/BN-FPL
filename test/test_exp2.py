"""
Tests for exp2.py: the prob_shift-only sweep. exp2.py reuses exp1.exp()/
run_grid() as-is (see exp1.py) -- only build_tasks() is specific to this
pipeline, so that's what's tested here; the per-task computation itself is
already covered by test_exp1.py.
"""
import pytest

import exp1
import exp2
from src.config import set_seed

BASE_CONFIG = {
    "n_clients": 6,
    "client_num": 0,
    "ess": 3,
    "alpha": 15,
    "prob_shift": [0.0, 0.4, 1.0],
    "size": 30,
    "res_path": "results2",
    "bn_base_path": "cancer.bif",
    "n_repetitions": 4,
    "n_bns": 5,
    "save_models": False,
}


@pytest.fixture(autouse=True)
def _seed():
    set_seed()


def test_build_tasks_sweeps_only_prob_shift():
    tasks = exp2.build_tasks(BASE_CONFIG)

    assert len(tasks) == len(BASE_CONFIG["prob_shift"]) * BASE_CONFIG["n_repetitions"]

    seen = set()
    for cfg, n, rep in tasks:
        # Every non-prob_shift hyperparameter is the SAME fixed scalar for
        # every task -- unlike exp1.py's grid, nothing else varies here.
        assert cfg["n_clients"] == BASE_CONFIG["n_clients"]
        assert cfg["ess"] == BASE_CONFIG["ess"]
        assert cfg["alpha"] == BASE_CONFIG["alpha"]
        assert n == BASE_CONFIG["size"]
        assert cfg["prob_shift"] in BASE_CONFIG["prob_shift"]

        key = (cfg["prob_shift"], rep)
        assert key not in seen, f"duplicate task: {key}"
        seen.add(key)

    expected = {
        (p, rep)
        for p in BASE_CONFIG["prob_shift"]
        for rep in range(BASE_CONFIG["n_repetitions"])
    }
    assert seen == expected


def test_build_tasks_output_is_accepted_by_check_unique_task_ids():
    # Shared machinery (see exp1.run_grid) must accept exp2's task shape
    # without raising.
    tasks = exp2.build_tasks(BASE_CONFIG)
    exp1._check_unique_task_ids(tasks)  # must not raise


def test_build_tasks_tasks_run_through_exp1_exp_without_error():
    # Smoke test: exp2's tasks are directly consumable by exp1.exp() (the
    # shared computation) -- only build_tasks() differs between the two
    # pipelines, everything downstream of it is identical code.
    tasks = exp2.build_tasks(BASE_CONFIG)
    cfg, n, rep = tasks[0]
    row, models, task_id = exp1.exp(cfg, n, rep)
    assert set(row.keys()) - set(exp1.GRID_KEYS) - {"size", "rep"}
    assert task_id == (
        BASE_CONFIG["n_clients"],
        BASE_CONFIG["ess"],
        cfg["prob_shift"],
        BASE_CONFIG["alpha"],
        BASE_CONFIG["size"],
        rep,
    )
