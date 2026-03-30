"""CLI entrypoint for running one or all configured experiments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import structlog
import yaml
from dotenv import load_dotenv

load_dotenv()

from evaluation.evaluator import evaluate_experiment
from rag.pipeline import build_pipeline

logger = structlog.get_logger(__name__)

CONFIGS_DIR = Path("experiments/configs")
RESULTS_DIR = Path("experiments/results")


# ---------------------------------------------------------------------------
# Data loading (previously in shared/data_loader.py)
# ---------------------------------------------------------------------------

def load_qa_dataset(path: str | Path) -> list[dict[str, str]]:
    """Load a JSONL evaluation dataset.

    Each line must be a JSON object with at least 'query' and 'ground_truth' keys.
    An optional 'query_id' key is preserved if present.

    Args:
        path: Path to the .jsonl file.

    Returns:
        List of row dicts.
    """
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    """Load an experiment YAML config from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_result(result: dict, experiment_id: str) -> Path:
    """Save an experiment result JSON payload to the results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{experiment_id}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output_path


def run_experiment(config_path: str | Path, use_ragas: bool = False) -> Path:
    """Run a single experiment config end-to-end and save the result JSON.

    Args:
        config_path: Path to a YAML experiment config file.

    Returns:
        Path to the saved result JSON.
    """
    config = load_config(config_path)
    pipeline = build_pipeline(config)
    pipeline.prepare()

    dataset = load_qa_dataset(config["dataset"]["path"])

    # Optional: wire up the LLM judge from the evaluation config section.
    judge_llm = None
    eval_cfg = config.get("evaluation", {})
    judge_cfg = eval_cfg.get("llm_judge")
    if judge_cfg:
        try:
            from rag.config_loader import build_llm
            judge_llm = build_llm(judge_cfg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("judge_llm_unavailable", error=str(exc), fallback="heuristic")

    result = evaluate_experiment(
        pipeline=pipeline,
        dataset=dataset,
        experiment_id=config["experiment_id"],
        dataset_id=Path(config["dataset"]["path"]).stem,
        use_ragas=use_ragas,
        judge_llm=judge_llm,
    )
    output_path = save_result(result.to_dict(), config["experiment_id"])
    logger.info(
        "experiment_completed",
        experiment_id=config["experiment_id"],
        output_path=str(output_path),
        correctness=result.aggregate_metrics.mean_correctness,
    )
    return output_path


def run_all_experiments(use_ragas: bool = False) -> list[Path]:
    """Run every YAML config in the experiments/configs directory."""
    outputs: list[Path] = []
    config_paths = sorted(CONFIGS_DIR.glob("*.yaml"))
    for index, config_path in enumerate(config_paths):
        outputs.append(run_experiment(config_path, use_ragas=use_ragas))
        if index < len(config_paths) - 1:
            config = load_config(config_path)
            sleep_seconds = float(
                config.get("pipeline", {}).get("rate_limit", {}).get("sleep_between_experiments", 0)
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    return outputs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for single or batch execution."""
    parser = argparse.ArgumentParser(description="Run AI sandbox experiments.")
    parser.add_argument("--config", help="Path to a single experiment config.")
    parser.add_argument("--all", action="store_true", help="Run all experiment configs.")
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Use RAGAS LLM-as-judge scoring (requires ragas + API keys).",
    )
    return parser.parse_args()


def main() -> None:
    """Dispatch the requested experiment execution mode."""
    args = parse_args()
    if args.all:
        run_all_experiments(use_ragas=args.ragas)
        return
    if args.config:
        run_experiment(args.config, use_ragas=args.ragas)
        return
    raise SystemExit("Specify --config <path> or --all.")


if __name__ == "__main__":
    main()
