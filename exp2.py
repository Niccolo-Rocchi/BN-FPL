"""
Sweeps ONLY prob_shift (n_clients/ess/alpha/size fixed, see conf2.yaml)
to compare the three weighting schemas as shift grows. Reuses exp1.py's
exp()/run_grid() as-is; only task-building differs.
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
