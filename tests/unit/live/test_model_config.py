from pathlib import Path

import pytest
import yaml

from etf_backtest.live.config import load_live_config

ROOT = Path(__file__).parents[3]
MODEL = ROOT / "qmt_example/configs/live/xgboost_example_paper.yaml"


def test_model_strategy_requires_model_section(tmp_path: Path) -> None:
    payload = yaml.safe_load(MODEL.read_text(encoding="utf-8"))
    payload["strategy"].pop("model")
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="requires model"):
        load_live_config(path)
