from __future__ import annotations

import argparse
import yaml

from data.feature_store import FeatureStore, FeatureStoreConfig
from data.corpus_builder import CorpusBuilder, CorpusConfig
from data.universe import get_us_equity_universe


def _feature_config_from(cfg: dict) -> FeatureStoreConfig:
    paths = cfg.get("paths", {})
    return FeatureStoreConfig(
        raw_path=paths.get("raw", "data/raw"),
        processed_path=paths.get("processed", "data/processed"),
        calendar=cfg.get("calendar", "NYSE"),
        indicator_mode=cfg.get("indicator_mode", "expanded"),
        ohlcv_batch=cfg.get("ohlcv_batch", True),
        ohlcv_batch_size=cfg.get("ohlcv_batch_size", 50),
        hf_yahoo_enabled=cfg.get("hf_yahoo_enabled", True),
        hf_yahoo_cache_dir=cfg.get("hf_yahoo_cache_dir"),
        allow_source_fallbacks=cfg.get("allow_source_fallbacks", True),
        sec_fsds=cfg.get("sec_fsds", {}),
    )


def _resolve_universe(cfg: dict) -> list[str]:
    if cfg.get("universe"):
        return cfg["universe"]
    source = cfg.get("universe_source", "nasdaq_all")
    if source in {"nasdaq_all", "sec_tickers"}:
        return get_us_equity_universe(max_tickers=cfg.get("max_tickers"), source=source)
    return []


def _corpus_paths(cfg: dict) -> tuple[str, str]:
    if cfg.get("corpus_tabular_path") and cfg.get("corpus_text_path"):
        return cfg["corpus_tabular_path"], cfg["corpus_text_path"]
    corpus_cfg = cfg.get("corpus", {})
    if "tabular" in corpus_cfg and "text" in corpus_cfg:
        return corpus_cfg["tabular"], corpus_cfg["text"]
    base = cfg.get("paths", {}).get("corpus", "data/corpus")
    return f"{base}/tabular.parquet", f"{base}/text_corpus.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    build_data = sub.add_parser("build-data")
    build_data.add_argument("--config", default="configs/data.yaml")

    build_corpus = sub.add_parser("build-corpus")
    build_corpus.add_argument("--config", default="configs/data.yaml")

    backtest = sub.add_parser("backtest")
    backtest.add_argument("--config", default="configs/backtest.yaml")

    args = parser.parse_args()

    if args.cmd == "build-data":
        cfg = yaml.safe_load(open(args.config))
        fs = FeatureStore(_feature_config_from(cfg))
        universe = _resolve_universe(cfg)
        refresh = cfg.get("feature_refresh", False)
        fs.build_feature_frame(universe, cfg["start"], cfg["end"], normalize=True, refresh=refresh)
        return

    if args.cmd == "build-corpus":
        cfg = yaml.safe_load(open(args.config))
        universe = _resolve_universe(cfg)
        tabular_path, text_path = _corpus_paths(cfg)
        builder = CorpusBuilder(_feature_config_from(cfg))
        builder.build(
            universe,
            CorpusConfig(
                start=cfg["start"],
                end=cfg["end"],
                output_tabular=tabular_path,
                output_text=text_path,
                generate_labels=cfg.get("generate_labels", True),
                generate_text=cfg.get("generate_text", True),
                label_window=cfg.get("label_window", 5),
            ),
        )
        return

    if args.cmd == "backtest":
        from backtest.engine import BacktestEngine, BacktestConfig
        cfg = yaml.safe_load(open(args.config))
        engine = BacktestEngine(BacktestConfig(**cfg))
        engine.run()
        return


if __name__ == "__main__":
    main()
