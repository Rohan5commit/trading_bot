import logging
import os
import re

from trained_model_client import TrainedModelTradeClient

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LABEL_TO_SCORE = {
    "STRONG_BUY": 2.0,
    "BUY": 1.0,
    "NEUTRAL": 0.0,
    "SELL": -1.0,
    "STRONG_SELL": -2.0,
}
_DIRECTIONAL_LABELS = ("STRONG_BUY", "BUY", "SELL", "STRONG_SELL")


def _enforce_english_reason(reason: str, side: str) -> str:
    text = str(reason or "").strip()
    side_u = str(side or "LONG").upper()
    fallback = (
        "Model confidence and downside profile support a short setup."
        if side_u == "SHORT"
        else "Model confidence and upside profile support a long setup."
    )
    if not text:
        return fallback
    if _CJK_RE.search(text):
        return fallback
    return text


def _normalize_candidates(candidates, limit=80):
    rows = []
    seen = set()
    lim = max(1, int(limit or 1))
    for item in list(candidates or []):
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol") or "").strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        row = dict(item)
        row["symbol"] = sym
        rows.append(row)
        if len(rows) >= lim:
            break
    return rows


def _neutral_breakout_score(prediction: dict, ai_cfg: dict) -> dict | None:
    if not bool((ai_cfg or {}).get("neutral_breakout_enabled", True)):
        return None
    probs = prediction.get("class_probabilities")
    if not isinstance(probs, dict):
        return None

    normalized = {}
    for label, value in probs.items():
        key = str(label or "").strip().upper()
        if key not in _LABEL_TO_SCORE:
            continue
        try:
            normalized[key] = float(value)
        except (TypeError, ValueError):
            continue
    if not normalized:
        return None

    neutral_prob = max(0.0, float(normalized.get("NEUTRAL", 0.0) or 0.0))
    directional = [(label, max(0.0, float(normalized.get(label, 0.0) or 0.0))) for label in _DIRECTIONAL_LABELS]
    directional = [item for item in directional if item[1] > 0.0]
    if not directional:
        return None
    best_label, best_prob = max(directional, key=lambda item: item[1])

    min_prob = float((ai_cfg or {}).get("neutral_breakout_min_prob", 0.22) or 0.22)
    max_gap = float((ai_cfg or {}).get("neutral_breakout_max_gap", 0.10) or 0.10)
    if best_prob < min_prob:
        return None
    if (neutral_prob - best_prob) > max_gap:
        return None

    score = float(_LABEL_TO_SCORE.get(best_label, 0.0))
    if score == 0.0:
        return None
    confidence = min(0.99, max(float(prediction.get("confidence", 0.0) or 0.0), best_prob))
    return {
        "label": best_label,
        "score": score,
        "confidence": confidence,
        "neutral_prob": neutral_prob,
        "directional_prob": best_prob,
    }


def _pick_predictions(predictions, max_positions=10, allow_shorts=True, max_shorts=5):
    max_positions = max(0, int(max_positions or 0))
    if max_positions <= 0:
        return []

    long_rows = [p for p in predictions if p["side"] == "LONG"]
    short_rows = [p for p in predictions if p["side"] == "SHORT"]

    long_rows = sorted(long_rows, key=lambda x: (x["strength"], x["confidence"]), reverse=True)
    short_rows = sorted(short_rows, key=lambda x: (x["strength"], x["confidence"]), reverse=True)

    short_slots = 0
    if bool(allow_shorts) and int(max_shorts or 0) > 0 and short_rows:
        short_slots = min(int(max_shorts or 0), max_positions // 2 if max_positions > 1 else 1, len(short_rows))
    long_slots = min(max_positions - short_slots, len(long_rows))
    if long_slots <= 0 and long_rows:
        long_slots = min(max_positions, len(long_rows))
        short_slots = 0

    picked = long_rows[:long_slots] + short_rows[:short_slots]
    picked_symbols = {p["symbol"] for p in picked}

    if len(picked) < max_positions:
        for row in sorted(predictions, key=lambda x: (x["strength"], x["confidence"]), reverse=True):
            if row["symbol"] in picked_symbols:
                continue
            if row["side"] == "SHORT" and (not allow_shorts or short_slots >= int(max_shorts or 0)):
                continue
            picked.append(row)
            picked_symbols.add(row["symbol"])
            if row["side"] == "SHORT":
                short_slots += 1
            if len(picked) >= max_positions:
                break

    return picked


def _weights_from_predictions(predictions, min_total_weight=0.90):
    if not predictions:
        return []
    rows = []
    total = 0.0
    for row in predictions:
        raw_weight = max(0.01, min(1.0, float(row["confidence"]) * (abs(float(row["score"])) / 2.0)))
        total += raw_weight
        rows.append({**row, "raw_weight": raw_weight})

    if total <= 0:
        even = 1.0 / float(len(rows))
        return [{**row, "weight": even} for row in rows]

    scale = 1.0
    if 0.0 < total < float(min_total_weight or 0.90):
        scale = float(min_total_weight or 0.90) / total
    total_weight = min(1.0, total * scale)
    if total_weight <= 0:
        total_weight = 1.0

    scaled_total = sum(row["raw_weight"] * scale for row in rows)
    if scaled_total <= 0:
        even = total_weight / float(len(rows))
        return [{**row, "weight": even} for row in rows]

    return [
        {
            **row,
            "weight": min(1.0, (row["raw_weight"] * scale / scaled_total) * total_weight),
        }
        for row in rows
    ]


def propose_trades_with_llm(config, candidates, max_positions=10, allow_shorts=True, max_shorts=5):
    """
    Trained-model-only AI trading path.
    Returns: (trades:list[dict], status:dict)
    """
    ai_cfg = config.get("ai_trading", {}) if isinstance(config, dict) else {}
    model_cfg = dict(ai_cfg.get("trained_model") or {})
    client = TrainedModelTradeClient(ai_cfg)
    status = {
        "enabled": bool(ai_cfg.get("enabled", False)),
        "ok": False,
        "error": None,
        "decision_engine": ai_cfg.get("decision_engine", "trained_model"),
        "backend": getattr(client, "backend", None),
        "model": client.model_identifier,
        "model_used": None,
    }

    max_positions = int(max_positions or 0)
    if max_positions <= 0:
        status["ok"] = True
        status["skipped_reason"] = "no_slots"
        return [], status

    prompt_limit = int(ai_cfg.get("prompt_candidates_limit", 80) or 80)
    prompt_candidates = _normalize_candidates(candidates, prompt_limit)
    status["candidates_seen"] = len(prompt_candidates)
    if not prompt_candidates:
        status["ok"] = True
        status["skipped_reason"] = "no_candidates"
        return [], status

    if not client.is_ready():
        status["error"] = getattr(client, "last_error", None) or "Trained model is not configured."
        return [], status

    ready_timeout_seconds = int(
        os.getenv("TRAINED_MODEL_READY_TIMEOUT_SECONDS")
        or model_cfg.get("ready_timeout_seconds", 1200)
        or 1200
    )
    ready_poll_seconds = float(
        os.getenv("TRAINED_MODEL_READY_POLL_SECONDS")
        or model_cfg.get("ready_poll_seconds", 15)
        or 15
    )
    try:
        client.wait_until_ready(
            timeout_seconds=ready_timeout_seconds,
            poll_seconds=ready_poll_seconds,
        )
    except Exception as exc:
        status["error"] = str(exc)
        status["model_used"] = getattr(client, "last_model_used", None) or client.model_identifier
        return [], status

    predictions = []
    predictions_seen = 0
    neutral_predictions = 0
    neutral_breakouts = 0
    failures = []
    batch_predictions = client.predict_candidates(prompt_candidates)
    for candidate, prediction in zip(prompt_candidates, batch_predictions):
        if prediction is None:
            failures.append(
                {
                    "symbol": candidate.get("symbol"),
                    "error": getattr(client, "last_error", None) or "prediction_failed",
                }
            )
            continue

        predictions_seen += 1
        score = float(prediction.get("score", 0.0) or 0.0)
        confidence = max(0.0, min(1.0, float(prediction.get("confidence", 0.0) or 0.0)))
        reason = prediction.get("reason")
        breakout_applied = False
        if score == 0.0:
            breakout = _neutral_breakout_score(prediction, ai_cfg)
            if breakout is None:
                neutral_predictions += 1
                continue
            score = float(breakout["score"])
            confidence = float(breakout["confidence"])
            breakout_applied = True
            neutral_breakouts += 1

        side = "LONG" if score > 0 else "SHORT"
        if side == "SHORT" and (not bool(allow_shorts) or int(max_shorts or 0) <= 0):
            continue

        strength = max(0.01, abs(score) * max(confidence, 0.05))
        if breakout_applied:
            reason = (
                f"{_enforce_english_reason(reason, side)} "
                "(neutral tie-break from trained-model class probabilities)"
            )
        predictions.append(
            {
                "symbol": candidate["symbol"],
                "side": side,
                "score": score,
                "confidence": confidence,
                "strength": strength,
                "reason": _enforce_english_reason(reason, side),
                "label": prediction.get("label"),
            }
        )

    status["model_used"] = getattr(client, "last_model_used", None) or client.model_identifier
    status["prediction_failures"] = failures[:10]
    status["candidates_scored"] = len(predictions)
    status["predictions_seen"] = predictions_seen
    status["neutral_predictions"] = neutral_predictions
    status["neutral_breakouts"] = neutral_breakouts

    if not predictions:
        if predictions_seen > 0 and neutral_predictions == predictions_seen:
            status["ok"] = True
            status["skipped_reason"] = "all_neutral"
            status["error"] = None
            return [], status
        status["error"] = getattr(client, "last_error", None) or "No usable trained-model predictions."
        return [], status

    picked = _pick_predictions(
        predictions,
        max_positions=max_positions,
        allow_shorts=allow_shorts,
        max_shorts=max_shorts,
    )
    if not picked:
        status["ok"] = True
        status["skipped_reason"] = "no_tradeable_signals"
        status["error"] = None
        return [], status

    min_total_weight = float(ai_cfg.get("min_total_weight", 0.90) or 0.90)
    weighted = _weights_from_predictions(picked, min_total_weight=min_total_weight)

    trades = [
        {
            "symbol": row["symbol"],
            "side": row["side"],
            "weight": float(row["weight"]),
            "reason": row["reason"],
        }
        for row in weighted
    ]
    status["ok"] = True
    status["total_weight"] = float(sum(float(t.get("weight", 0.0) or 0.0) for t in trades))
    status["min_total_weight"] = min_total_weight
    return trades, status
