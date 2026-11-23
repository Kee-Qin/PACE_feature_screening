"""Single manuscript-ordered registry for all controlled experiment entry points."""
import hashlib

# Experiment and setting identifiers follow the manuscript tables.
CLASSIFICATION = {
    "exp1": {"source_exp": "exp1", "ps": [1, 2, 3], "scenario": "Strong additive classification"},
    "exp2": {"source_exp": "exp3", "ps": [1, 2, 3], "scenario": "Nonlinear latent classification"},
    "exp3": {"source_exp": "exp8", "ps": [1, 2, 3], "scenario": "Three-class bimodal signal"},
    "exp4": {"source_exp": "exp9", "ps": [1, 2, 3], "scenario": "Ultrahigh-dimensional classification", "is_ultra": True},
    "exp5": {"source_exp": "exp10", "ps": [2, 3], "scenario": "Mixed continuous-categorical classification"},
    "exp6": {"source_exp": "exp11", "ps": [1, 2], "scenario": "Missing feature values"},
}
REGRESSION = {
    "exp1": {"source_exp": "exp1", "ps": [2, 3], "scenario": "Nonlinear regression"},
    "exp2": {"source_exp": "exp6", "ps": [1, 2, 3], "scenario": "Ultrahigh-dimensional regression", "is_ultra": True},
    "exp3": {"source_exp": "exp7", "ps": [2, 3], "scenario": "Mixed continuous-categorical regression"},
    "exp4": {"source_exp": "exp8", "ps": [2, 3], "scenario": "Missing feature values under collinearity"},
}
PROTOCOL_ID = "paper25_order_v1"


def output_root():
    """Return the configured directory for controlled experiment outputs."""
    from src.model_path import OUTPUT_ROOT
    return OUTPUT_ROOT / 'results_simulation_paper25'


def scenarios(task):
    if task == "classification":
        return CLASSIFICATION
    if task == "regression":
        return REGRESSION
    raise ValueError(f"Unknown simulation task: {task}")


def settings(task, exp, requested=None):
    allowed = scenarios(task)[exp]["ps"]
    chosen = set(allowed if requested is None else requested)
    if not chosen or not chosen.issubset(allowed):
        raise ValueError(f"{task}/{exp}: settings must be a nonempty subset of {allowed}")
    return [s for s in allowed if s in chosen]


def experiments(task, requested="all"):
    table = scenarios(task)
    if requested in ("all", "paper"):
        return list(table)
    if requested == "plain":
        return list(table)[:4 if task == "classification" else 2]
    if requested == "mixtype_missing":
        return list(table)[4 if task == "classification" else 2:]
    chosen = {e.strip() for e in requested.split(",")}
    if not chosen or not chosen.issubset(table):
        raise ValueError(f"Unknown experiment(s) {chosen - set(table)}; valid: {list(table)}")
    return [e for e in table if e in chosen]


def repetitions(task, exp):
    return 25 if scenarios(task)[exp].get("is_ultra", False) else 50


def seed_offset(task, exp):

    # randomized regression hash reproducible across interpreter processes.
    if task == "classification":
        return 37
    source = scenarios(task)[exp]["source_exp"]
    return 1000 + int.from_bytes(hashlib.sha256(source.encode()).digest()[:4], "big") % 1000


def simulation_seed(task, exp, setting, rep):
    return 10000 * setting + rep * 37 + seed_offset(task, exp)


def execution_plan(task, exp="all", param_sets=None, n_reps=None):
    if n_reps is not None and n_reps < 1:
        raise ValueError("n_reps must be positive")
    return [(key, settings(task, key, param_sets),
             repetitions(task, key) if n_reps is None else n_reps)
            for key in experiments(task, exp)]


def ensure_output_protocol(directory, task, runner):
    import json
    from pathlib import Path
    root = Path(directory)
    marker = root / "simulation_protocol.json"
    expected = {"protocol": PROTOCOL_ID, "task": task, "runner": runner,
                "classification": CLASSIFICATION, "regression": REGRESSION,
                "seed_scheme": "setting10000_rep37_stable_source_offset_v1"}
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Incompatible simulation output directory: {root}")
        return
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Unversioned results in {root}; use a new empty output directory")
    root.mkdir(parents=True, exist_ok=True)
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(expected, stream, indent=2)
