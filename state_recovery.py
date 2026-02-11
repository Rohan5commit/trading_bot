import json
import logging
import os
import sqlite3

import yaml

from positions import PositionTracker

logger = logging.getLogger(__name__)


def _resolve_path(base_dir, path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(base_dir, path_value)


def _compute_target_price(entry_price, side, tp_pct):
    side = str(side or "LONG").strip().upper()
    if side == "SHORT":
        return float(entry_price) * (1.0 - float(tp_pct))
    return float(entry_price) * (1.0 + float(tp_pct))


def recover_positions_from_seed(config_path):
    """
    One-time state recovery for cloud runs:
    - If positions tables are truly empty (total rows == 0), seed from state/positions_seed.json.
    - If a table already has any rows, never touch it.
    """
    base_dir = os.path.dirname(os.path.abspath(config_path))
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle) or {}

    seed_path = _resolve_path(base_dir, "state/positions_seed.json")
    if not os.path.exists(seed_path):
        return {"seed_file": seed_path, "recovered_total": 0, "reason": "seed_missing"}

    with open(seed_path, "r") as handle:
        seed = json.load(handle) or {}

    # Ensure tables exist before checking counts.
    PositionTracker(config_path, table_name="positions")
    PositionTracker(config_path, table_name="positions_ai")

    db_path = _resolve_path(base_dir, config["data"]["cache_path"])
    tp_pct = float((config.get("trading", {}) or {}).get("take_profit_pct", 0.03) or 0.03)

    recovered = {"positions": 0, "positions_ai": 0}
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        for table_name in ("positions", "positions_ai"):
            row = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            total_rows = int(row[0] if row else 0)
            if total_rows > 0:
                continue

            rows = seed.get(table_name) or []
            if not isinstance(rows, list) or not rows:
                continue

            for item in rows:
                symbol = str(item.get("symbol", "")).strip().upper()
                side = str(item.get("side", "LONG")).strip().upper() or "LONG"
                entry_date = str(item.get("entry_date", "")).strip()
                entry_price = float(item.get("entry_price", 0.0) or 0.0)
                quantity = float(item.get("quantity", 0.0) or 0.0)
                target_price = float(item.get("target_price", 0.0) or 0.0)

                if not symbol or not entry_date or entry_price <= 0 or quantity <= 0:
                    continue
                if target_price <= 0:
                    target_price = _compute_target_price(entry_price, side, tp_pct)

                cursor.execute(
                    f"""
                    INSERT INTO {table_name}
                    (symbol, side, entry_date, entry_price, quantity, target_price, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'OPEN')
                    """,
                    (symbol, side, entry_date, entry_price, quantity, target_price),
                )
                recovered[table_name] += 1

        conn.commit()
    finally:
        conn.close()

    recovered_total = int(recovered["positions"] + recovered["positions_ai"])
    if recovered_total:
        logger.warning(
            "Recovered position state from seed file: core=%d ai=%d",
            recovered["positions"],
            recovered["positions_ai"],
        )
        return {
            "seed_file": seed_path,
            "recovered_total": recovered_total,
            "recovered_core": recovered["positions"],
            "recovered_ai": recovered["positions_ai"],
            "reason": "recovered",
        }

    return {"seed_file": seed_path, "recovered_total": 0, "reason": "no_recovery_needed"}
