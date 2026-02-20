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


def _table_exists(cursor, table_name):
    row = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return bool(row and row[0])


def _initial_capital_for_table(config, table_name):
    if table_name == "positions_ai":
        return float(((config.get("ai_trading", {}) or {}).get("initial_capital", 100000)) or 100000)
    return 100000.0


def _get_total_realized_dollars(conn, table_name):
    rows = conn.execute(
        f"SELECT side, entry_price, exit_price, quantity FROM {table_name} WHERE status='CLOSED'"
    ).fetchall()
    total = 0.0
    for side, entry_price, exit_price, qty in rows:
        try:
            entry = float(entry_price or 0.0)
            exit_ = float(exit_price or 0.0)
            q = float(qty or 0.0)
        except Exception:
            continue
        s = str(side or "LONG").strip().upper()
        if s == "SHORT":
            total += (entry - exit_) * q
        else:
            total += (exit_ - entry) * q
    return float(total)


def _load_open_positions(conn, table_name):
    rows = conn.execute(
        f"SELECT id, symbol, side, entry_price, quantity FROM {table_name} WHERE status='OPEN'"
    ).fetchall()
    out = []
    for row_id, symbol, side, entry_price, qty in rows:
        try:
            entry = float(entry_price or 0.0)
            q = float(qty or 0.0)
        except Exception:
            continue
        if entry <= 0.0 or q <= 0.0:
            continue
        out.append(
            {
                "id": int(row_id),
                "symbol": str(symbol or "").strip().upper(),
                "side": str(side or "LONG").strip().upper(),
                "entry_price": entry,
                "quantity": q,
                "notional": entry * q,
            }
        )
    return out


def _priority_rank_for_reduction(table_name, pos):
    # User-requested one-time intervention: if core is over-capitalized, shrink JNJ short first.
    if table_name == "positions" and pos.get("symbol") == "JNJ" and pos.get("side") == "SHORT":
        return 0
    # Otherwise reduce largest positions first.
    return 1


def enforce_position_cap(config_path):
    """
    Ensure each account stays within capital using current OPEN positions.
    - If invested notional > current capital, reduce open quantities until invested <= capital.
    - Core account prioritizes shrinking JNJ SHORT first when present.
    """
    base_dir = os.path.dirname(os.path.abspath(config_path))
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle) or {}

    db_path = _resolve_path(base_dir, config["data"]["cache_path"])
    if not os.path.exists(db_path):
        return {"adjusted_total": 0, "reason": "db_missing", "tables": {}}

    result = {"adjusted_total": 0, "tables": {}}
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        for table_name in ("positions", "positions_ai"):
            if not _table_exists(cursor, table_name):
                result["tables"][table_name] = {"adjusted": 0, "reason": "table_missing"}
                continue

            initial_capital = _initial_capital_for_table(config, table_name)
            realized = _get_total_realized_dollars(conn, table_name)
            current_capital = float(initial_capital + realized)
            open_positions = _load_open_positions(conn, table_name)
            invested = float(sum(p["notional"] for p in open_positions))

            overshoot = invested - current_capital
            if overshoot <= 1e-6:
                result["tables"][table_name] = {
                    "adjusted": 0,
                    "reason": "within_cap",
                    "invested_notional": invested,
                    "current_capital": current_capital,
                }
                continue

            ordered = sorted(
                open_positions,
                key=lambda p: (_priority_rank_for_reduction(table_name, p), -float(p["notional"])),
            )
            changes = []
            for pos in ordered:
                if overshoot <= 1e-6:
                    break
                reducible = min(float(pos["notional"]), float(overshoot))
                new_notional = float(pos["notional"] - reducible)
                if new_notional <= 1e-9:
                    # Remove fully if quantity goes to 0.
                    cursor.execute(f"DELETE FROM {table_name} WHERE id=?", (int(pos["id"]),))
                    new_qty = 0.0
                else:
                    new_qty = new_notional / float(pos["entry_price"])
                    cursor.execute(
                        f"UPDATE {table_name} SET quantity=? WHERE id=?",
                        (float(new_qty), int(pos["id"])),
                    )
                changes.append(
                    {
                        "id": int(pos["id"]),
                        "symbol": pos["symbol"],
                        "side": pos["side"],
                        "old_qty": float(pos["quantity"]),
                        "new_qty": float(new_qty),
                        "old_notional": float(pos["notional"]),
                        "new_notional": float(new_notional),
                    }
                )
                overshoot -= reducible

            adjusted = len(changes)
            result["adjusted_total"] += adjusted
            result["tables"][table_name] = {
                "adjusted": adjusted,
                "reason": "reduced_overshoot",
                "changes": changes,
                "invested_before": invested,
                "current_capital": current_capital,
                "remaining_overshoot": max(0.0, float(overshoot)),
            }

        conn.commit()
    finally:
        conn.close()

    return result


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
