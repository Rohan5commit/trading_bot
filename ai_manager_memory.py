from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


class AIManagerMemory:
    """Shared AI-manager memory persisted in the trading SQLite DB.

    Both the full trained-model path and the distilled fallback write to the
    same journal so either backend can resume with the same recent manager
    context.
    """

    RUNS_TABLE = "ai_manager_runs"
    DECISIONS_TABLE = "ai_manager_decisions"

    def __init__(self, db_path: str, lookback_days: int = 120):
        self.db_path = str(db_path or "").strip()
        self.lookback_days = max(1, int(lookback_days or 120))
        self.available = bool(self.db_path)
        if self.available:
            self._init_tables()

    @classmethod
    def from_config(cls, config: dict | None, *, lookback_days: int | None = None) -> "AIManagerMemory":
        cfg = dict(config or {})
        data_cfg = dict(cfg.get("data") or {})
        db_path = str(data_cfg.get("cache_path") or "").strip()
        if db_path and not os.path.isabs(db_path):
            base_dir = str(
                cfg.get("_config_base_dir")
                or cfg.get("__config_base_dir__")
                or cfg.get("config_base_dir")
                or os.getcwd()
            ).strip()
            db_path = os.path.join(base_dir, db_path)
        router_cfg = dict((cfg.get("ai_trading") or {}).get("runtime_router") or {})
        resolved_lookback = lookback_days if lookback_days is not None else int(router_cfg.get("memory_lookback_days", 120) or 120)
        return cls(db_path=db_path, lookback_days=resolved_lookback)

    def _connect(self) -> sqlite3.Connection:
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _init_tables(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.RUNS_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date TEXT NOT NULL,
                    backend_selected TEXT,
                    requested_mode TEXT,
                    model_used TEXT,
                    ok INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    candidates_seen INTEGER,
                    candidates_scored INTEGER,
                    target_positions INTEGER,
                    notes_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.DECISIONS_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date TEXT NOT NULL,
                    backend_selected TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT,
                    weight REAL,
                    confidence REAL,
                    score REAL,
                    label TEXT,
                    reason TEXT,
                    extra_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _json(self, value: Any) -> str:
        return json.dumps(value or {}, sort_keys=True)

    def record_run(
        self,
        *,
        run_date: str,
        backend_selected: str,
        requested_mode: str,
        model_used: str,
        ok: bool,
        error: str | None = None,
        candidates_seen: int | None = None,
        candidates_scored: int | None = None,
        target_positions: int | None = None,
        notes: dict[str, Any] | None = None,
    ) -> None:
        if not self.available:
            return
        conn = self._connect()
        try:
            conn.execute(
                f"""
                INSERT INTO {self.RUNS_TABLE} (
                    run_date, backend_selected, requested_mode, model_used, ok, error,
                    candidates_seen, candidates_scored, target_positions, notes_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run_date or ""),
                    str(backend_selected or ""),
                    str(requested_mode or ""),
                    str(model_used or ""),
                    1 if ok else 0,
                    None if error in (None, "") else str(error),
                    None if candidates_seen is None else int(candidates_seen),
                    None if candidates_scored is None else int(candidates_scored),
                    None if target_positions is None else int(target_positions),
                    self._json(notes),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def record_trade_plan(
        self,
        *,
        run_date: str,
        backend_selected: str,
        trades: list[dict[str, Any]],
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.available:
            return
        rows = []
        for trade in list(trades or []):
            if not isinstance(trade, dict):
                continue
            symbol = str(trade.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            rows.append(
                (
                    str(run_date or ""),
                    str(backend_selected or ""),
                    symbol,
                    str(trade.get("side") or "").strip().upper(),
                    float(trade.get("weight", 0.0) or 0.0),
                    float(trade.get("confidence", 0.0) or 0.0),
                    float(trade.get("score", 0.0) or 0.0),
                    str(trade.get("label") or "").strip().upper(),
                    str(trade.get("reason") or "").strip(),
                    self._json({**dict(extra or {}), "source": str(backend_selected or "")}),
                )
            )
        if not rows:
            return
        conn = self._connect()
        try:
            conn.executemany(
                f"""
                INSERT INTO {self.DECISIONS_TABLE} (
                    run_date, backend_selected, symbol, side, weight, confidence, score, label, reason, extra_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def recent_backend_summary(self) -> dict[str, Any]:
        if not self.available:
            return {"last_backend": None, "success_rate_by_backend": {}}
        conn = self._connect()
        try:
            last_row = conn.execute(
                f"SELECT backend_selected FROM {self.RUNS_TABLE} ORDER BY id DESC LIMIT 1"
            ).fetchone()
            rows = conn.execute(
                f"""
                SELECT backend_selected, COUNT(*) AS runs, AVG(ok) AS success_rate
                FROM {self.RUNS_TABLE}
                WHERE run_date >= ?
                GROUP BY backend_selected
                """,
                (self._cutoff_date_str(),),
            ).fetchall()
        finally:
            conn.close()

        summary = {}
        for backend_selected, runs, success_rate in rows:
            backend_key = str(backend_selected or "unknown").strip() or "unknown"
            summary[backend_key] = {
                "runs": int(runs or 0),
                "success_rate": float(success_rate or 0.0),
            }
        return {
            "last_backend": str(last_row[0]).strip() if last_row and last_row[0] else None,
            "success_rate_by_backend": summary,
        }

    def symbol_side_bias(self) -> dict[tuple[str, str], dict[str, float]]:
        if not self.available:
            return {}
        cutoff = self._cutoff_date_str()
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT symbol, COALESCE(side, 'LONG') AS side, realized_pnl, realized_pnl_dollars
                FROM positions_ai
                WHERE status='CLOSED' AND COALESCE(exit_date, entry_date, '') >= ?
                """,
                (cutoff,),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        finally:
            conn.close()

        grouped: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for symbol, side, realized_pnl, realized_pnl_dollars in rows:
            key = (str(symbol or "").strip().upper(), str(side or "LONG").strip().upper())
            if not key[0]:
                continue
            grouped.setdefault(key, []).append((float(realized_pnl or 0.0), float(realized_pnl_dollars or 0.0)))

        result: dict[tuple[str, str], dict[str, float]] = {}
        for key, samples in grouped.items():
            count = len(samples)
            avg_return = sum(item[0] for item in samples) / float(count)
            wins = sum(1 for item in samples if item[0] > 0.0)
            win_rate = wins / float(count)
            pnl_scale = _clamp(avg_return * 4.0, -0.6, 0.6)
            win_scale = _clamp((win_rate - 0.5) * 0.8, -0.4, 0.4)
            confidence = _clamp(count / 6.0, 0.0, 1.0)
            result[key] = {
                "bias": _clamp((pnl_scale + win_scale) * confidence, -0.75, 0.75),
                "confidence": confidence,
                "avg_return": avg_return,
                "win_rate": win_rate,
                "samples": float(count),
            }
        return result

    def build_context(self) -> dict[str, Any]:
        backend_summary = self.recent_backend_summary()
        bias = self.symbol_side_bias()
        strongest = []
        for (symbol, side), payload in sorted(
            bias.items(),
            key=lambda item: abs(float(item[1].get("bias", 0.0) or 0.0)),
            reverse=True,
        )[:10]:
            strongest.append({
                "symbol": symbol,
                "side": side,
                "bias": float(payload.get("bias", 0.0) or 0.0),
                "confidence": float(payload.get("confidence", 0.0) or 0.0),
            })
        return {
            "lookback_days": self.lookback_days,
            "last_backend": backend_summary.get("last_backend"),
            "success_rate_by_backend": backend_summary.get("success_rate_by_backend", {}),
            "top_symbol_biases": strongest,
        }

    def _cutoff_date_str(self) -> str:
        return (datetime.now(timezone.utc) - timedelta(days=self.lookback_days)).date().isoformat()
