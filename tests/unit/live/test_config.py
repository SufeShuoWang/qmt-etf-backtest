from pathlib import Path

import pytest
import yaml

from etf_backtest.experiments.config import load_system_settings
from etf_backtest.live.config import load_live_config
from etf_backtest.live.service import resolve_state_database

ROOT = Path(__file__).parents[3]
RULE = ROOT / "qmt_example/configs/live/beginner_example_paper.yaml"
MODEL = ROOT / "qmt_example/configs/live/xgboost_example_paper.yaml"


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "live.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path






def test_model_backend_bundle_suffix_and_time_order_are_strict(tmp_path: Path) -> None:
    payload = yaml.safe_load(MODEL.read_text(encoding="utf-8"))
    payload["strategy"]["model"]["bundle_path"] = "model.pt"
    with pytest.raises(ValueError, match="ubj"):
        load_live_config(_write(tmp_path, payload))
    payload = yaml.safe_load(RULE.read_text(encoding="utf-8"))
    payload["execution"]["stop_new_orders"] = "14:49:00"
    with pytest.raises(ValueError, match="submit_start"):
        load_live_config(_write(tmp_path, payload))


def test_xgboost_backend_requires_ubj_bundle(tmp_path: Path) -> None:
    payload = yaml.safe_load(MODEL.read_text(encoding="utf-8"))
    config = load_live_config(_write(tmp_path, payload))
    assert config.strategy.model.backend == "xgboost"  # type: ignore[union-attr]

    payload["strategy"]["model"]["bundle_path"] = "model.json"
    with pytest.raises(ValueError, match="ubj"):
        load_live_config(_write(tmp_path, payload))


def test_only_named_secret_environment_fields_are_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_live_config(RULE)
    monkeypatch.setenv("QMT_PAPER_ACCOUNT_ID", "paper-account")
    assert config.account.account_id() == "paper-account"
    assert str(config.miniqmt.userdata_path).startswith("C:")


def test_target_config_unifies_databases_and_needs_no_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "QMT_PAPER_ACCOUNT_ID",
        "QMT_MYSQL_PASSWORD",
        "QMT_LIVE_MYSQL_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)

    live = load_live_config(MODEL)
    system = load_system_settings(ROOT / live.account.system_path)
    state_database = resolve_state_database(live)

    assert live.account.account_id() == "66629053"
    assert system.database.host == state_database.host == "127.0.0.1"
    assert system.database.port == state_database.port == 3306
    assert system.database.database == state_database.database == "qmt_etf_quant"
    assert system.database.user == state_database.user == "root"
    assert system.database.password_env is state_database.password_env is None
    assert system.database.resolved_password() == state_database.resolved_password()
    assert "ahaailab" not in str(system.database.model_dump(mode="json"))
    assert live.strategy.initial_capital == 10_000




def test_single_rule_and_model_config():
    for path, case in ((RULE, "rule"), (MODEL, "model")):
        config = load_live_config(path)
        assert config.config_version == "4.0"
        assert config.strategy.case == case
        assert not hasattr(config, "strategies")
        assert not hasattr(config.account, "capital_pool")


@pytest.mark.parametrize("mutation", ["list", "old_version", "pool", "enabled", "negative_cash"])
def test_rejects_removed_multi_strategy_settings(tmp_path, mutation):
    payload = yaml.safe_load(RULE.read_text(encoding="utf-8"))
    if mutation == "list":
        payload["strategies"] = [payload.pop("strategy")]
    elif mutation == "old_version":
        payload["config_version"] = "3.0"
    elif mutation == "pool":
        payload["account"]["capital_pool"] = "1000000"
    elif mutation == "enabled":
        payload["strategy"]["enabled"] = True
    else:
        payload["strategy"]["initial_capital"] = "-1"
    with pytest.raises(ValueError):
        load_live_config(_write(tmp_path, payload))
