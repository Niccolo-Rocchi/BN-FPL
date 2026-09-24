"""
Pipeline 2: sweeps ONLY prob_shift (n_clients/ess/alpha/sample-size are each
fixed to a single scalar value, set in conf2.yaml) to compare the three
MOSAIC weighting schemas as distribution shift grows -- see plot2.ipynb.

Reuses exp1.py's exp()/run_grid() as-is: identical computation and
identical safe, memory-bounded parallel-execution/incremental-write engine
(see run_grid()'s docstring in exp1.py for the full rationale) as the full
grid pipeline. Only the task-building logic differs here (a 1D sweep over
prob_shift x repetitions, instead of exp1.py's 4D hyperparameter grid) --
keeping exp() itself in exactly one place avoids the two pipelines ever
silently computing something slightly different from each other.
"""
from src.config import load_config, set_seed

from exp1 import run_grid


def build_tasks(config) -> list:
    """
    One task per (prob_shift value, repetition); n_clients/ess/alpha and
    sample size are the same fixed scalars (already resolved in `config`,
    loaded from conf2.yaml) for every task.
    """
    return [
        (dict(config, prob_shift=p), config["size"], rep)
        for p in config["prob_shift"]
        for rep in range(config["n_repetitions"])
    ]


def main():

    # Set seed
    set_seed()

    # Choose configuration file
    config = load_config("conf2.yaml")
    save_models = config.get("save_models", True)
    max_tasks_per_child = config.get("max_tasks_per_child", 100)

    tasks = build_tasks(config)
    print(
        f"# {len(config['prob_shift'])} prob_shift values x "
        f"{config['n_repetitions']} repetitions = {len(tasks)} total tasks "
        f"(n_clients={config['n_clients']}, ess={config['ess']}, "
        f"alpha={config['alpha']}, size={config['size']}, "
        f"save_models={save_models})",
        flush=True,
    )

    run_grid(tasks, config["res_path"], save_models, max_tasks_per_child)


if __name__ == "__main__":
    main()
