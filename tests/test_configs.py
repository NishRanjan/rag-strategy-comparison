"""Tests for YAML experiment config integrity."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_all_configs_have_required_top_level_keys() -> None:
    """Ensure every experiment config contains the required sections."""
    for config_path in Path("experiments/configs").glob("*.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "experiment_id" in config
        assert "dataset" in config
        assert "pipeline" in config
        assert "evaluation" in config
