from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from src.data.provenance import ProvenanceLogger
from src.data.twelvedata_providers import TwelveDataBarsProvider, TwelveDataBenchmarkProvider
from src.data.yfinance_providers import YFinanceActionsProvider, YFinanceBarsProvider, YFinanceBenchmarkProvider, WikipediaUniverseProvider
from src.engine.backtest import run_backtest, run_walk_forward
from src.engine.earnings_backtest import run_earnings_backtest
from src.engine.email_report import build_email_report, send_email_report, write_email_report
from src.engine.meta_learning import model_selector, online_update
from src.engine.reporting import write_report
from src.engine.signals import normalize_signals
from src.engine.signal_engine import (
    compute_daily_scores,
    generate_buy_signals,
    record_signal_history,
    update_meta_learning,
)
from src.engine.strategy import generate_signals
from src.utils.config import ensure_dirs


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_providers(config: dict, data_dir: Path):
    cache_dir = data_dir / "cache"
    provenance = ProvenanceLogger(Path("logs/provenance.jsonl"))
    data_cfg = config.get("data", {})
    provider = str(data_cfg.get("provider", "yfinance")).lower()

    if provider == "twelvedata":
        td_cfg = data_cfg.get("twelvedata", {})
        api_key_env = td_cfg.get("api_key_env", "TWELVEDATA_API_KEY")
        api_key_raw = os.getenv(api_key_env) or ""
        # Allow comma-separated keys; use the first for backtesting.
        api_key = api_key_raw.split(",")[0].strip() if api_key_raw else None
        base_url = td_cfg.get("base_url", "https://api.twelvedata.com")
        timeout_seconds = int(td_cfg.get("timeout_seconds", 20))
        bars_provider = TwelveDataBarsProvider(
            cache_dir=cache_dir,
            provenance=provenance,
            timezone=config["bars"]["timezone"],
            open_time=config["bars"]["market_hours"]["open"],
            close_time=config["bars"]["market_hours"]["close"],
            source_interval=config["bars"]["source_interval"],
            fallback_to_sample=data_cfg.get("fallback_to_sample", False),
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        benchmark_provider = TwelveDataBenchmarkProvider(
            cache_dir=cache_dir,
            provenance=provenance,
            source_interval=config["bars"]["source_interval"],
            fallback_to_sample=data_cfg.get("fallback_to_sample", False),
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        # Avoid yfinance dependency in TwelveData mode (yfinance is unreliable/rate-limited).
        class _EmptyActionsProvider:
            def get_actions(self, symbols, start, end):
                return {str(s): pd.DataFrame() for s in symbols}
        actions_provider = _EmptyActionsProvider()
    else:
        bars_provider = YFinanceBarsProvider(
            cache_dir=cache_dir,
            provenance=provenance,
            timezone=config["bars"]["timezone"],
            open_time=config["bars"]["market_hours"]["open"],
            close_time=config["bars"]["market_hours"]["close"],
            source_interval=config["bars"]["source_interval"],
            fallback_to_sample=data_cfg.get("fallback_to_sample", False),
        )
        benchmark_provider = YFinanceBenchmarkProvider(
            cache_dir=cache_dir,
            provenance=provenance,
            source_interval=config["bars"]["source_interval"],
            fallback_to_sample=data_cfg.get("fallback_to_sample", False),
        )
        actions_provider = YFinanceActionsProvider(cache_dir=cache_dir, provenance=provenance)

    universe_provider = WikipediaUniverseProvider(
        cache_dir=cache_dir,
        provenance=provenance,
        source_url=config["universe"]["source_url"],
        as_of=config["universe"]["membership_timestamp"],
    )
    return bars_provider, actions_provider, benchmark_provider, universe_provider
def run_backtest_pipeline(config: dict) -> dict:
    data_dir = Path("data")
    report_dir = Path("reports/latest")
    ensure_dirs(data_dir, report_dir)

    # Load env vars for API keys (FMP/TwelveData/email) if present.
    env_path = Path((config.get("scheduler") or {}).get("env_path", ".env"))
    _load_dotenv(env_path)
    # Also load "../.env" if present (lets the backtesting tool reuse repo-level env).
    _load_dotenv(Path("..") / ".env")

    strategy_cfg = config.get("strategy", {})
    if not strategy_cfg.get("enabled", False):
        empty_trades = pd.DataFrame(columns=["timestamp", "symbol", "action", "shares", "price", "pnl"])
        empty_equity = pd.DataFrame(columns=["timestamp", "equity", "cash"])
        empty_trades.to_csv(report_dir / "trades.csv", index=False)
        empty_equity.to_csv(report_dir / "daily_equity.csv", index=False)
        (report_dir / "metrics.json").write_text(json.dumps({}, indent=2), encoding="utf-8")
        notes = f"{config['report']['notes']} (Backtest skipped: strategy disabled.)"
        write_report(config["report"]["title"], {}, empty_trades, notes, report_dir / "report.html")
        return {}
    bars_provider, actions_provider, benchmark_provider, universe_provider = build_providers(config, data_dir)

    override = config["universe"].get("symbols_override")
    if override:
        symbols = override
    else:
        universe = universe_provider.get_universe()
        symbols = universe.members
    start = config["backtest"]["start_date"]
    end = config["backtest"]["end_date"]
    interval = config["bars"]["interval"]

    bars = bars_provider.get_bars(symbols, start, end, interval)
    actions = actions_provider.get_actions(symbols, start, end)
    benchmark = benchmark_provider.get_benchmark(start, end, interval)

    if not bars:
        empty_trades = pd.DataFrame(columns=["timestamp", "symbol", "action", "shares", "price", "pnl"])
        empty_equity = pd.DataFrame(columns=["timestamp", "equity", "cash"])
        empty_trades.to_csv(report_dir / "trades.csv", index=False)
        empty_equity.to_csv(report_dir / "daily_equity.csv", index=False)
        (report_dir / "metrics.json").write_text(json.dumps({}, indent=2), encoding="utf-8")
        notes = f"{config['report']['notes']} (Backtest skipped: no market data.)"
        write_report(config["report"]["title"], {}, empty_trades, notes, report_dir / "report.html")
        return {}

    strategy_type = str(strategy_cfg.get("type", "native"))
    if strategy_type == "earnings_event":
        result = run_earnings_backtest(bars, benchmark, config, report_dir)
    elif config.get("backtest", {}).get("walk_forward", {}).get("enabled", False):
        result = run_walk_forward(bars, actions, benchmark, config, report_dir)
    else:
        result = run_backtest(bars, actions, benchmark, config, report_dir)
    write_report(config["report"]["title"], result.metrics, result.trades, config["report"]["notes"], report_dir / "report.html")
    subject, body = build_email_report(config, report_dir)
    write_email_report(subject, body, report_dir / "email_report.txt")
    send_email_report(config, subject, body)
    return result.metrics


def run_eod_pipeline(config: dict) -> None:
    data_dir = Path("data")
    report_dir = Path("reports/latest")
    ensure_dirs(data_dir, report_dir, Path("logs"))
    bars_provider, actions_provider, _, universe_provider = build_providers(config, data_dir)

    strategy_cfg = config.get("strategy", {})
    if not strategy_cfg.get("enabled", False):
        (report_dir / "signals.json").write_text(json.dumps([], indent=2), encoding="utf-8")
        Path("logs/eod.log").write_text(f"EOD pipeline completed at {pd.Timestamp.utcnow().isoformat()}\n", encoding="utf-8")
        subject, body = build_email_report(config, report_dir)
        write_email_report(subject, body, report_dir / "email_report.txt")
        send_email_report(config, subject, body)
        return

    override = config["universe"].get("symbols_override")
    if override:
        symbols = override
    else:
        universe = universe_provider.get_universe()
        symbols = universe.members
    end = pd.Timestamp.utcnow().date().isoformat()
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=30)).date().isoformat()

    bars = bars_provider.get_bars(symbols, start, end, config["bars"]["interval"])
    actions_provider.get_actions(symbols, start, end)

    signals = generate_signals(bars, config["strategy"])
    signals = normalize_signals(signals, int(config["portfolio"]["max_positions"]))
    output = [signal.to_json() for signal in signals]
    (report_dir / "signals.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    if config["meta_learning"]["mode"] == "online_update":
        _ = online_update({s.symbol: s.confidence for s in signals})
    else:
        _ = model_selector({"simple_momentum": [s.confidence for s in signals]})

    Path("logs/eod.log").write_text(f"EOD pipeline completed at {pd.Timestamp.utcnow().isoformat()}\n", encoding="utf-8")
    subject, body = build_email_report(config, report_dir)
    write_email_report(subject, body, report_dir / "email_report.txt")
    send_email_report(config, subject, body)


def run_signals_pipeline(config: dict) -> bool:
    if not config.get("signals", {}).get("enabled", False):
        return False

    data_dir = Path("data")
    report_dir = Path("reports/latest")
    ensure_dirs(data_dir, report_dir, Path("models"))
    bars_provider, _, _, universe_provider = build_providers(config, data_dir)

    override = config["universe"].get("symbols_override")
    if override:
        symbols = override
    else:
        universe = universe_provider.get_universe()
        symbols = universe.members

    end = pd.Timestamp.utcnow().date().isoformat()
    lookback_days = int(config.get("signals", {}).get("lookback_days", 60))
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)).date().isoformat()
    interval = config["bars"]["interval"]

    bars = bars_provider.get_bars(symbols, start, end, interval)
    status_path = report_dir / "data_status.json"
    if not bars:
        (report_dir / "signals.json").write_text(json.dumps([], indent=2), encoding="utf-8")
        _write_data_status(report_dir, "no_data", "Data fetch failed or unavailable")
        subject, body = build_email_report(config, report_dir)
        write_email_report(subject, body, report_dir / "email_report.txt")
        try:
            send_email_report(config, subject, body)
        except Exception:
            pass
        return False
    if status_path.exists():
        status_path.unlink()

    scores = compute_daily_scores(
        bars,
        lookback=int(config.get("signals", {}).get("momentum_lookback", 20)),
        mean_reversion_window=int(config.get("signals", {}).get("mean_reversion_window", 20)),
    )
    latest_prices = {symbol: frame["close"].iloc[-1] for symbol, frame in bars.items() if not frame.empty}

    state_path = Path("models/meta_state.json")
    history_path = Path("models/signal_history.jsonl")
    meta_state = update_meta_learning(scores, latest_prices, config, state_path, history_path)

    as_of = pd.Timestamp.utcnow()
    signals = generate_buy_signals(scores, meta_state.weights, config, as_of, latest_prices)
    record_signal_history(signals, scores, latest_prices, history_path)

    (report_dir / "signals.json").write_text(json.dumps([s.to_json() for s in signals], indent=2), encoding="utf-8")
    subject, body = build_email_report(config, report_dir)
    write_email_report(subject, body, report_dir / "email_report.txt")
    try:
        send_email_report(config, subject, body)
        return bool(config.get("email", {}).get("enabled", False))
    except Exception:
        return False


def _write_data_status(report_dir: Path, status: str, reason: str) -> None:
    payload = {"status": status, "reason": reason}
    (report_dir / "data_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
