"""Standardized run-result logging.

Every script in this project (pretraining, and every future-wave SFT/RLHF/
optimization script) writes exactly one `run_result.json` per run, with a
stable envelope around task-specific, free-form `metrics`. This is the single
mechanism `experiments/*/compare.py` scripts read from -- it's what makes
"does X beat Y" comparisons possible without each experiment needing custom
parsing logic per stage.

`variant`, `cot_enabled`, and `save_strategy` are optional/nullable because
they don't apply to every stage (e.g. pretraining has no CoT toggle and no
adapter/merge choice) -- a run_result.json missing those fields is still
valid and still comparable on whatever fields it does share with the run
it's being compared against.
"""

import datetime
import json
import os
from typing import Any, Dict, Optional


def write_run_result(
    output_dir: str,
    stage: str,
    task: str,
    modality: str,
    model_name: str,
    dataset_name: str,
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, Any],
    num_train_samples: int,
    num_eval_samples: int,
    train_runtime_seconds: float,
    architecture: Optional[str] = None,
    variant: Optional[str] = None,
    cot_enabled: Optional[bool] = None,
    save_strategy: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """Write a run's results to a standardized `run_result.json` file.

    Args:
        output_dir (str): Directory to write `run_result.json` into (created
            if missing).
        stage (str): Pipeline stage that produced this run (e.g.
            "pretraining", "supervised-finetuning", "rlhf").
        task (str): Task name (e.g. "text_classification", "ner").
        modality (str): Data modality (e.g. "text", "image", "audio").
        model_name (str): Model checkpoint id/path used for this run.
        dataset_name (str): Dataset id/path used for this run.
        hyperparameters (Dict[str, Any]): Free-form hyperparameters used for
            this run.
        metrics (Dict[str, Any]): Free-form evaluation/training metrics
            produced by this run.
        num_train_samples (int): Number of training samples used.
        num_eval_samples (int): Number of evaluation samples used.
        train_runtime_seconds (float): Wall-clock training duration in
            seconds.
        architecture (Optional[str]): Architecture family, if applicable
            (e.g. "decoder-only", "encoder-decoder").
        variant (Optional[str]): Task variant, if applicable (e.g. "cot",
            "standard").
        cot_enabled (Optional[bool]): Whether chain-of-thought was used, if
            applicable to this stage.
        save_strategy (Optional[str]): Model save strategy used, if
            applicable (see `common.model_saving`).
        timestamp (Optional[str]): ISO-8601 timestamp to record; defaults to
            the current UTC time if not given.

    Returns:
        str: The path to the written `run_result.json` file.
    """
    record = {
        "stage": stage,
        "task": task,
        "modality": modality,
        "architecture": architecture,
        "variant": variant,
        "cot_enabled": cot_enabled,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "save_strategy": save_strategy,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "num_train_samples": num_train_samples,
        "num_eval_samples": num_eval_samples,
        "train_runtime_seconds": train_runtime_seconds,
        "timestamp": timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "run_result.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[run_results] wrote {path}")
    return path
