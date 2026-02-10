import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import yaml


def _resolve_path(base_dir, value):
    if os.path.isabs(value):
        return value
    return os.path.join(base_dir, value)


def _read_config(config_path: str) -> dict:
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle) or {}


def _state_path(base_dir: str) -> str:
    return os.path.join(base_dir, "results", "storage_state.json")


def _load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as handle:
            return json.load(handle) or {}
    except Exception:
        return {}


def _save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(state, handle, indent=2)
    os.replace(tmp, path)


def apply_storage_policy(config_path: str) -> dict:
    """
    Enforce retention and disk-minimization:
    - Prune old rows from prices/news.
    - Optionally vacuum SQLite (expensive; throttled by vacuum_frequency_days).
    - Optionally remove feature_store CSVs (if store_feature_files=false).
    - Optionally prune old model files (keep latest only).
    """
    base_dir = os.path.dirname(os.path.abspath(config_path))
    cfg = _read_config(config_path)
    storage = (cfg.get("storage") or {}) if isinstance(cfg, dict) else {}

    db_rel = ((cfg.get("data") or {}).get("cache_path")) or "./data/trading_bot.db"
    db_path = _resolve_path(base_dir, db_rel)

    results = {
        "db_path": db_path,
        "prices_deleted": 0,
        "news_deleted": 0,
        "vacuum_ran": False,
        "feature_files_deleted": 0,
        "db_archives_deleted": 0,
    }

    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        try:
            # Prices retention
            pr_days = int(storage.get("prices_retention_days", 0) or 0)
            if pr_days > 0:
                row = conn.execute("SELECT MAX(date) FROM prices").fetchone()
                max_date = row[0] if row and row[0] else None
                if max_date:
                    cutoff = (datetime.fromisoformat(str(max_date)).date() - timedelta(days=pr_days)).isoformat()
                    cur = conn.execute("DELETE FROM prices WHERE date < ?", (cutoff,))
                    results["prices_deleted"] = int(cur.rowcount or 0)

            # News retention
            nw_days = int(storage.get("news_retention_days", 0) or 0)
            if nw_days > 0:
                # news.datetime stores timestamps; compare via ISO string (works for YYYY-MM-DD...).
                cutoff_dt = (datetime.now(timezone.utc) - timedelta(days=nw_days)).date().isoformat()
                cur = conn.execute("DELETE FROM news WHERE datetime < ?", (cutoff_dt,))
                results["news_deleted"] = int(cur.rowcount or 0)

            conn.commit()
        finally:
            conn.close()

        # Vacuum throttling
        if bool(storage.get("vacuum_sqlite", False)):
            freq = int(storage.get("vacuum_frequency_days", 7) or 7)
            sp = _state_path(base_dir)
            st = _load_state(sp)
            last = st.get("last_vacuum_at")
            run = True
            if last:
                try:
                    dt = datetime.fromisoformat(last)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if datetime.now(timezone.utc) - dt < timedelta(days=freq):
                        run = False
                except Exception:
                    run = True

            if run:
                conn = sqlite3.connect(db_path)
                try:
                    conn.execute("VACUUM;")
                    results["vacuum_ran"] = True
                finally:
                    conn.close()
                st["last_vacuum_at"] = datetime.now(timezone.utc).isoformat()
                _save_state(sp, st)

    # If we're not storing feature files, delete any leftover feature CSVs.
    if not bool(storage.get("store_feature_files", True)):
        fs_dir = os.path.join(base_dir, "feature_store")
        if os.path.isdir(fs_dir):
            deleted = 0
            for fname in os.listdir(fs_dir):
                if fname.endswith("_features.csv"):
                    try:
                        os.remove(os.path.join(fs_dir, fname))
                        deleted += 1
                    except Exception:
                        pass
            results["feature_files_deleted"] = deleted

    # Model pruning (optional)
    if bool(storage.get("prune_models_keep_latest_only", False)):
        try:
            from train import ModelManager
            mm = ModelManager(config_path)
            mm.prune_models_keep_latest_only()
        except Exception:
            pass

    # Prune old DB archives to keep disk usage low.
    raw_keep = storage.get("keep_db_archives", 1)
    keep_archives = 1 if raw_keep is None else int(raw_keep)
    if keep_archives >= 0:
        data_dir = os.path.join(base_dir, "data")
        if os.path.isdir(data_dir):
            archives = sorted(
                [f for f in os.listdir(data_dir) if f.startswith("trading_bot_archive_") and f.endswith(".db")]
            )
            # Sort newest-first by filename timestamp suffix.
            archives.sort(reverse=True)
            to_delete = archives[keep_archives:] if keep_archives else archives
            deleted = 0
            for fname in to_delete:
                try:
                    os.remove(os.path.join(data_dir, fname))
                    deleted += 1
                except Exception:
                    pass
            results["db_archives_deleted"] = deleted

    return results
