from pathlib import Path
from unittest.mock import Mock

import pytest

import run_paper_trading
from etf_backtest.live import service

ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "qmt_example/configs/live/beginner_example_paper.yaml"


def test_main_loads_single_strategy_config_and_runs_engine(monkeypatch):
    engine = Mock()
    build = Mock(return_value=engine)
    monkeypatch.setattr(service, "build_production_runtime", build)
    assert run_paper_trading.main(["--config", str(CONFIG)]) == 0
    assert build.call_args.args[0].strategy.strategy_id == "beginner_rule"
    engine.run_forever.assert_called_once_with()


@pytest.mark.parametrize("error, code", [(KeyboardInterrupt(), 0), (RuntimeError("connection failed"), 1)])
def test_main_handles_stop_and_reports_failure(monkeypatch, capsys, error, code):
    engine = Mock()
    engine.run_forever.side_effect = error
    monkeypatch.setattr(service, "build_production_runtime", Mock(return_value=engine))
    assert run_paper_trading.main(["--config", str(CONFIG)]) == code
    if code:
        assert "RuntimeError: connection failed" in capsys.readouterr().err


def test_invalid_config_does_not_build_runtime(monkeypatch, tmp_path, capsys):
    config = tmp_path / "invalid.yaml"
    config.write_text("config_version: invalid\n", encoding="utf-8")
    build = Mock()
    monkeypatch.setattr(service, "build_production_runtime", build)
    assert run_paper_trading.main(["--config", str(config)]) == 1
    build.assert_not_called()
    assert "ValidationError" in capsys.readouterr().err
