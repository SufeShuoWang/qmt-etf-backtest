import importlib.util
from contextlib import nullcontext
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import etf_backtest
import etf_backtest.application
import etf_backtest.core
import etf_backtest.live
import etf_backtest.live.service
import etf_backtest.strategy
from etf_backtest.live.broker.base import BrokerGateway
from etf_backtest.live.broker.miniqmt import MiniQmtBrokerGateway
from etf_backtest.live.config import load_live_config
from etf_backtest.live.market.base import QuoteProvider
from etf_backtest.live.market.xtdata import XtDataQuoteProvider
from etf_backtest.live.service import build_production_runtime

ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "qmt_example/configs/live/beginner_example_paper.yaml"


def test_missing_xtquant_is_delayed_until_production_adapter_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del monkeypatch
    assert all(
        module is not None
        for module in (
            etf_backtest,
            etf_backtest.application,
            etf_backtest.core,
            etf_backtest.live,
            etf_backtest.live.service,
            etf_backtest.strategy,
        )
    )
    if importlib.util.find_spec("xtquant") is not None:
        pytest.skip("target environment provides xtquant")
    with pytest.raises(RuntimeError, match="未安装 xtquant"):
        MiniQmtBrokerGateway(
            userdata_path=Path("C:/missing"),
            session_id=1,
            account_id="paper-1",
            event_queue=Queue(),
        )
    with pytest.raises(RuntimeError, match="未安装 xtquant"):
        XtDataQuoteProvider()


@pytest.mark.parametrize("filename", ["beginner_example_paper.yaml", "xgboost_example_paper.yaml"])
def test_production_builder_wires_one_rule_or_model(monkeypatch, filename):
    monkeypatch.setenv("QMT_PAPER_ACCOUNT_ID", "paper-1")
    config = load_live_config(ROOT / "qmt_example/configs/live" / filename)
    broker = Mock(spec=BrokerGateway)
    quotes = Mock(spec=QuoteProvider)
    broker_factory = Mock(return_value=broker)
    quote_factory = Mock(return_value=quotes)
    strategy_engine = Mock()
    strategy_engine.connect.return_value = nullcontext(Mock())
    monkeypatch.setattr("etf_backtest.live.service.create_database_engine", lambda config: strategy_engine)
    monkeypatch.setattr("etf_backtest.live.service.create_state_engine", lambda config: Mock())
    monkeypatch.setattr("etf_backtest.live.signals.create_repository", lambda *args, **kwargs: Mock())
    monkeypatch.setattr("etf_backtest.live.signals.resolve_universe",
                        lambda *args: SimpleNamespace(symbols=("SH.510300", "SH.518880", "SH.588000")))
    monkeypatch.setattr("etf_backtest.live.service.create_repository", lambda *args: Mock())
    runtime = build_production_runtime(config, broker_factory=broker_factory, quote_factory=quote_factory)
    assert runtime.broker is broker
    assert runtime.quote_provider is quotes
    assert runtime.event_consumer is not None
    assert runtime.jobs.strategy_runtime.spec.strategy_id == config.strategy.strategy_id
    assert runtime.jobs.strategy_runtime.spec.case == config.strategy.case
    assert runtime.jobs.strategy_runtime.signal_evaluator.__class__.__name__ == "SignalService"
    assert not hasattr(runtime.jobs, "strategy_runtimes")
    broker_factory.assert_called_once()
    quote_factory.assert_called_once()
