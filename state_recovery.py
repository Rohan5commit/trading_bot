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


def _env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


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


def enforce_position_cap(config_path):
    """
    Ensure each account stays within capital using current OPEN positions.
    - If invested notional > current capital, reduce open quantities proportionally.
    - Core account special rule: keep JNJ SHORT at 5% of current capital, then scale other
      open positions to fit within the remaining capital.
    - Never zero out positions in this correction pass.
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

            # Start with current notionals as targets.
            target_notional = {int(p["id"]): float(p["notional"]) for p in open_positions}

            # Core rule: keep JNJ SHORT at 5% of account capital (if present),
            # then scale all other positions to fill the remaining capital.
            protected_ids = set()
            jnj_target_notional = None
            if table_name == "positions":
                for p in open_positions:
                    if p.get("symbol") == "JNJ" and p.get("side") == "SHORT":
                        jnj_target_notional = float(current_capital) * 0.05
                        # Only reduce via this pass; don't increase JNJ size.
                        jnj_target_notional = min(float(p["notional"]), jnj_target_notional)
                        target_notional[int(p["id"])] = jnj_target_notional
                        protected_ids.add(int(p["id"]))
                        break

            protected_total = sum(target_notional[i] for i in protected_ids)
            other_positions = [p for p in open_positions if int(p["id"]) not in protected_ids]
            other_total = float(sum(float(p["notional"]) for p in other_positions) or 0.0)
            remaining_budget = max(0.0, float(current_capital) - float(protected_total))

            # Scale non-protected positions proportionally.
            if other_total > 0.0:
                scale = min(1.0, remaining_budget / other_total)
                for p in other_positions:
                    target_notional[int(p["id"])] = float(p["notional"]) * scale

            changes = []
            for pos in open_positions:
                pos_id = int(pos["id"])
                new_notional = max(0.0, float(target_notional.get(pos_id, float(pos["notional"]))))
                # Never set position to zero in this correction pass.
                if new_notional <= 0.0:
                    continue
                new_qty = new_notional / float(pos["entry_price"])
                if abs(float(new_qty) - float(pos["quantity"])) <= 1e-12:
                    continue
                cursor.execute(
                    f"UPDATE {table_name} SET quantity=? WHERE id=?",
                    (float(new_qty), pos_id),
                )
                changes.append(
                    {
                        "id": pos_id,
                        "symbol": pos["symbol"],
                        "side": pos["side"],
                        "old_qty": float(pos["quantity"]),
                        "new_qty": float(new_qty),
                        "old_notional": float(pos["notional"]),
                        "new_notional": float(new_notional),
                    }
                )

            adjusted = len(changes)
            result["adjusted_total"] += adjusted
            invested_after = float(sum(float(target_notional.get(int(p["id"]), p["notional"])) for p in open_positions) or 0.0)
            result["tables"][table_name] = {
                "adjusted": adjusted,
                "reason": "scaled_to_cap",
                "changes": changes,
                "invested_before": invested,
                "invested_after": invested_after,
                "current_capital": current_capital,
                "remaining_overshoot": max(0.0, invested_after - float(current_capital)),
                "jnj_target_notional": jnj_target_notional,
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

    recovery_cfg = (config.get("state_recovery", {}) or {})
    enabled = _env_flag(
        "ENABLE_POSITION_SEED_RECOVERY",
        default=bool(recovery_cfg.get("enabled", False)),
    )
    if not enabled:
        return {"seed_file": None, "recovered_total": 0, "reason": "disabled"}

    seed_rel = str(recovery_cfg.get("seed_file", "state/positions_seed.json") or "state/positions_seed.json")
    seed_path = _resolve_path(base_dir, seed_rel)
    if not os.path.exists(seed_path):
        return {"seed_file": seed_path, "recovered_total": 0, "reason": "seed_missing"}

    with open(seed_path, "r") as handle:
        seed = json.load(handle) or {}

    # Ensure tables exist before checking counts.
    PositionTracker(config_path, table_name="positions")
    PositionTracker(config_path, table_name="positions_ai")

    db_path = _resolve_path(base_dir, config["data"]["cache_path"])
    tp_pct = float((config.get("trading", {}) or {}).get("take_profit_pct", 0.03) or 0.03)

    allow_core = _env_flag(
        "ENABLE_CORE_POSITION_SEED_RECOVERY",
        default=bool(recovery_cfg.get("allow_core_seed", False)),
    )
    allow_ai = _env_flag(
        "ENABLE_AI_POSITION_SEED_RECOVERY",
        default=bool(recovery_cfg.get("allow_ai_seed", False)),
    )

    allowed_tables = []
    if allow_core:
        allowed_tables.append("positions")
    if allow_ai:
        allowed_tables.append("positions_ai")
    if not allowed_tables:
        return {"seed_file": seed_path, "recovered_total": 0, "reason": "no_tables_enabled"}

    recovered = {"positions": 0, "positions_ai": 0}
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        for table_name in allowed_tables:
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


def purge_seeded_open_positions(config_path):
    """
    Remove stale OPEN rows that exactly match seed entries by (symbol, side, entry_date).
    This is a safety cleanup for previously seeded cloud states.
    """
    base_dir = os.path.dirname(os.path.abspath(config_path))
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle) or {}

    recovery_cfg = (config.get("state_recovery", {}) or {})
    enabled = _env_flag(
        "PURGE_SEEDED_OPEN_POSITIONS",
        default=bool(recovery_cfg.get("purge_seeded_open_positions", True)),
    )
    if not enabled:
        return {"seed_file": None, "purged_total": 0, "reason": "disabled"}

    seed_rel = str(recovery_cfg.get("seed_file", "state/positions_seed.json") or "state/positions_seed.json")
    seed_path = _resolve_path(base_dir, seed_rel)
    if not os.path.exists(seed_path):
        return {"seed_file": seed_path, "purged_total": 0, "reason": "seed_missing"}

    with open(seed_path, "r") as handle:
        seed = json.load(handle) or {}

    db_path = _resolve_path(base_dir, config["data"]["cache_path"])
    if not os.path.exists(db_path):
        return {"seed_file": seed_path, "purged_total": 0, "reason": "db_missing"}

    seed_keys = {}
    for table_name in ("positions", "positions_ai"):
        rows = list(seed.get(table_name) or [])
        rows.extend(seed.get(f"legacy_seed_{table_name}") or [])
        keys = set()
        for item in rows:
            symbol = str(item.get("symbol", "")).strip().upper()
            side = str(item.get("side", "LONG")).strip().upper() or "LONG"
            entry_date = str(item.get("entry_date", "")).strip()
            if symbol and entry_date:
                keys.add((symbol, side, entry_date))
        seed_keys[table_name] = keys

    if not any(seed_keys.values()):
        return {"seed_file": seed_path, "purged_total": 0, "reason": "seed_empty"}

    purged = {"positions": 0, "positions_ai": 0}
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        for table_name in ("positions", "positions_ai"):
            if not _table_exists(cursor, table_name):
                continue
            keys = seed_keys.get(table_name) or set()
            if not keys:
                continue

            rows = cursor.execute(
                f"SELECT id, symbol, side, entry_date FROM {table_name} WHERE status='OPEN'"
            ).fetchall()
            if not rows:
                continue

            to_delete = []
            for row_id, symbol, side, entry_date in rows:
                key = (
                    str(symbol or "").strip().upper(),
                    str(side or "LONG").strip().upper(),
                    str(entry_date or "").strip(),
                )
                if key in keys:
                    to_delete.append(int(row_id))

            if not to_delete:
                continue

            placeholders = ",".join(["?"] * len(to_delete))
            cursor.execute(f"DELETE FROM {table_name} WHERE id IN ({placeholders})", to_delete)
            purged[table_name] = len(to_delete)

        conn.commit()
    finally:
        conn.close()

    purged_total = int(purged["positions"] + purged["positions_ai"])
    if purged_total:
        logger.warning(
            "Purged seeded OPEN positions: core=%d ai=%d",
            purged["positions"],
            purged["positions_ai"],
        )
        return {
            "seed_file": seed_path,
            "purged_total": purged_total,
            "purged_core": purged["positions"],
            "purged_ai": purged["positions_ai"],
            "reason": "purged",
        }

    return {"seed_file": seed_path, "purged_total": 0, "reason": "none_matched"}
