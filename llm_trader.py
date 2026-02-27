import json
import logging
import re

from llm_sentiment import NvidiaChatClient, _extract_json

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _enforce_english_reason(reason: str, side: str) -> str:
    text = str(reason or "").strip()
    side_u = str(side or "LONG").upper()
    fallback = (
        "Momentum and risk profile support a short setup."
        if side_u == "SHORT"
        else "Momentum and risk profile support a long setup."
    )
    if not text:
        return fallback
    if _CJK_RE.search(text):
        return fallback
    return text


def _score_candidate(candidate: dict) -> float:
    """
    Deterministic score from raw price inputs when LLM is unavailable.
    Positive => long bias, negative => short bias.
    """
    closes = candidate.get("closes_tail") or []
    try:
        closes = [float(x) for x in closes if x is not None]
    except Exception:
        closes = []
    if len(closes) >= 2 and closes[0] > 0:
        return (closes[-1] / closes[0]) - 1.0
    try:
        last_close = float(candidate.get("last_close"))
    except Exception:
        last_close = 0.0
    return 0.0 if last_close <= 0 else 1e-9


def _rule_based_fallback_trades(candidates, max_positions=10, allow_shorts=True, max_shorts=5):
    """
    Last-resort deterministic fallback so AI entries are never blocked by LLM outages.
    """
    max_positions = max(0, int(max_positions or 0))
    if max_positions <= 0:
        return []

    # Keep one row per symbol.
    rows = []
    seen = set()
    for c in list(candidates or []):
        sym = str((c or {}).get("symbol") or "").strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        score = _score_candidate(c or {})
        rows.append({"symbol": sym, "score": float(score)})

    if not rows:
        return []

    rows_sorted = sorted(rows, key=lambda x: x["score"], reverse=True)
    negatives = [r for r in rows_sorted if r["score"] < 0]
    positives = [r for r in rows_sorted if r["score"] >= 0]

    shorts_slots = 0
    if bool(allow_shorts) and int(max_shorts or 0) > 0 and negatives:
        shorts_slots = min(int(max_shorts or 0), max_positions // 2, len(negatives))
    long_slots = max_positions - shorts_slots

    picked = []
    picked_symbols = set()

    # Longs from strongest scores first.
    for row in rows_sorted:
        if len(picked) >= long_slots:
            break
        sym = row["symbol"]
        if sym in picked_symbols:
            continue
        picked.append(
            {
                "symbol": sym,
                "side": "LONG",
                "score": row["score"],
                "reason": f"Rule-based fallback: positive trend score {row['score']:.2%}.",
            }
        )
        picked_symbols.add(sym)

    # Shorts from most negative scores.
    if shorts_slots > 0:
        for row in sorted(negatives, key=lambda x: x["score"]):
            if len([p for p in picked if p["side"] == "SHORT"]) >= shorts_slots:
                break
            sym = row["symbol"]
            if sym in picked_symbols:
                continue
            picked.append(
                {
                    "symbol": sym,
                    "side": "SHORT",
                    "score": row["score"],
                    "reason": f"Rule-based fallback: negative trend score {row['score']:.2%}.",
                }
            )
            picked_symbols.add(sym)

    if not picked:
        return []

    w = 1.0 / float(len(picked))
    return [
        {
            "symbol": p["symbol"],
            "side": p["side"],
            "weight": w,
            "reason": _enforce_english_reason(p["reason"], p["side"]),
        }
        for p in picked
    ]


def propose_trades_with_llm(config, candidates, max_positions=10, allow_shorts=True, max_shorts=5):
    """
    candidates: list[dict] with at minimum: symbol
      You may include any extra fields (e.g., price/returns/volatility/news/sentiment).
    returns: (trades:list[dict], status:dict)
      trade dict: symbol, side(LONG/SHORT), weight(0..1), reason
    """
    llm_cfg = config.get("llm", {})
    # AI strategy can override model + key env (use a separate reasoning model/key if configured).
    ai_cfg = config.get("ai_trading", {}) if isinstance(config, dict) else {}
    base_llm_cfg = dict(llm_cfg)
    llm_cfg = dict(base_llm_cfg)
    llm_cfg["model"] = ai_cfg.get("llm_model", llm_cfg.get("model"))
    llm_cfg["api_key_env"] = ai_cfg.get("api_key_env", llm_cfg.get("api_key_env"))
    llm_cfg["fallback_models"] = ai_cfg.get("llm_fallback_models", llm_cfg.get("fallback_models", []))
    client = NvidiaChatClient(llm_cfg)
    status = {
        "enabled": bool(llm_cfg.get("enabled", False)),
        "ok": False,
        "error": None,
        "model": llm_cfg.get("model"),
        "fallback_models": llm_cfg.get("fallback_models", []),
        "api_key_env": llm_cfg.get("api_key_env"),
    }

    max_positions = int(max_positions or 0)
    if max_positions <= 0:
        return [], {"enabled": True, "ok": True, "error": None}

    allow_shorts = bool(allow_shorts)
    max_shorts = int(max_shorts or 0)
    min_total_weight = float(ai_cfg.get("min_total_weight", 0.90) or 0.90)
    min_total_weight = max(0.0, min(1.0, min_total_weight))

    # Keep prompt compact but "strategy-free": include the provided candidates as-is (bounded).
    prompt_limit = int(ai_cfg.get("prompt_candidates_limit", 80) or 80)
    prompt_candidates = list(candidates or [])[:max(1, prompt_limit)]
    status["candidates_seen"] = len(prompt_candidates)
    if not prompt_candidates:
        status["ok"] = True
        status["skipped_reason"] = "no_candidates"
        return [], status

    system_msg = (
        "You are a trading decision engine. "
        "Return ONLY valid JSON (no markdown). "
        "All 'reason' text must be in English only. "
        "Pick a diversified set of trades from the provided candidate list. "
        "There is no fixed strategy; use your own judgement based on the provided fields. "
        "If you want momentum/mean-reversion/volatility signals, derive them yourself from raw inputs "
        "(e.g., the provided close series) rather than assuming any precomputed indicator is present."
    )
    rules = [
        f"- Choose up to {max_positions} total trades.",
        f"- Shorts are {'allowed' if allow_shorts else 'not allowed'}.",
    ]
    if allow_shorts:
        rules.append(f"- Choose up to {max_shorts} SHORT trades.")
    rules.extend([
        "- Each trade must be one of the provided symbols.",
        "- Output weights that sum to <= 1.0.",
        "- Use side LONG or SHORT.",
        "- Include a short reason per trade (English only).",
        "",
        "JSON schema:",
        "{\"trades\": [{\"symbol\": \"AAPL\", \"side\": \"LONG\", \"weight\": 0.10, \"reason\": \"...\"}], \"notes\": \"...\"}",
    ])

    user_msg = "Candidates:\n" + json.dumps(prompt_candidates, indent=2) + "\n\nRules:\n" + "\n".join(rules)
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]

    # Primary: AI-specific model/key. Fallback: default LLM config/key if overrides fail.
    attempted = []
    response_text = None
    fallback_to_default = False

    candidate_cfgs = [("ai", llm_cfg)]
    has_override = (
        llm_cfg.get("model") != base_llm_cfg.get("model")
        or llm_cfg.get("api_key_env") != base_llm_cfg.get("api_key_env")
        or llm_cfg.get("fallback_models", []) != base_llm_cfg.get("fallback_models", [])
    )
    if has_override:
        candidate_cfgs.append(("default", base_llm_cfg))

    for label, cfg in candidate_cfgs:
        candidate_client = NvidiaChatClient(cfg)
        if not candidate_client.is_ready():
            key_env = getattr(candidate_client, "api_key_env", "NVIDIA_API_KEY")
            attempted.append(f"{label}: missing {key_env}")
            continue

        response_text = candidate_client.chat(messages)
        if response_text is not None:
            client = candidate_client
            fallback_to_default = (label == "default")
            break

        attempted.append(f"{label}: {getattr(candidate_client, 'last_error', None) or 'LLM call failed.'}")

    if response_text is None:
        fallback_trades = _rule_based_fallback_trades(
            prompt_candidates,
            max_positions=max_positions,
            allow_shorts=allow_shorts,
            max_shorts=max_shorts,
        )
        if fallback_trades:
            status["ok"] = True
            status["fallback_mode"] = "rule_based"
            status["error"] = " | ".join(attempted) if attempted else (getattr(client, "last_error", None) or "LLM call failed.")
            status["model_used"] = getattr(client, "last_model_used", None)
            status["total_weight"] = float(sum(float(t.get("weight", 0.0) or 0.0) for t in fallback_trades))
            return fallback_trades, status
        status["error"] = " | ".join(attempted) if attempted else (getattr(client, "last_error", None) or "LLM call failed.")
        status["model_used"] = getattr(client, "last_model_used", None)
        return [], status

    if fallback_to_default:
        status["fallback_to_default_llm"] = True
        status["model"] = base_llm_cfg.get("model")
        status["fallback_models"] = base_llm_cfg.get("fallback_models", [])
        status["api_key_env"] = base_llm_cfg.get("api_key_env")

    status["model_used"] = getattr(client, "last_model_used", None) or status.get("model")
    parsed = _extract_json(response_text)
    if not isinstance(parsed, dict):
        fallback_trades = _rule_based_fallback_trades(
            prompt_candidates,
            max_positions=max_positions,
            allow_shorts=allow_shorts,
            max_shorts=max_shorts,
        )
        if fallback_trades:
            status["ok"] = True
            status["fallback_mode"] = "rule_based_after_parse_error"
            status["error"] = "LLM response missing JSON."
            status["total_weight"] = float(sum(float(t.get("weight", 0.0) or 0.0) for t in fallback_trades))
            return fallback_trades, status
        status["error"] = "LLM response missing JSON."
        return [], status

    trades = parsed.get("trades", [])
    if not isinstance(trades, list):
        fallback_trades = _rule_based_fallback_trades(
            prompt_candidates,
            max_positions=max_positions,
            allow_shorts=allow_shorts,
            max_shorts=max_shorts,
        )
        if fallback_trades:
            status["ok"] = True
            status["fallback_mode"] = "rule_based_after_schema_error"
            status["error"] = "LLM response JSON missing 'trades' list."
            status["total_weight"] = float(sum(float(t.get("weight", 0.0) or 0.0) for t in fallback_trades))
            return fallback_trades, status
        status["error"] = "LLM response JSON missing 'trades' list."
        return [], status

    allowed = {c.get("symbol") for c in prompt_candidates if c.get("symbol")}
    cleaned = []
    total_weight = 0.0
    shorts_count = 0
    for t in trades:
        if not isinstance(t, dict):
            continue
        sym = (t.get("symbol") or "").strip().upper()
        side = (t.get("side") or "LONG").strip().upper()
        if sym not in allowed:
            continue
        if side not in {"LONG", "SHORT"}:
            continue
        if side == "SHORT" and (not allow_shorts or max_shorts <= 0):
            continue
        try:
            w = float(t.get("weight", 0.0))
        except (TypeError, ValueError):
            w = 0.0
        if w <= 0:
            continue
        if side == "SHORT":
            if shorts_count >= max_shorts:
                continue
            shorts_count += 1
        cleaned.append({
            "symbol": sym,
            "side": side,
            "weight": w,
            "reason": _enforce_english_reason(t.get("reason"), side),
        })
        total_weight += w
        if len(cleaned) >= max_positions:
            break

    if not cleaned:
        fallback_trades = _rule_based_fallback_trades(
            prompt_candidates,
            max_positions=max_positions,
            allow_shorts=allow_shorts,
            max_shorts=max_shorts,
        )
        if fallback_trades:
            status["ok"] = True
            status["fallback_mode"] = "rule_based_after_empty_trades"
            status["error"] = "LLM returned no usable trades."
            status["total_weight"] = float(sum(float(t.get("weight", 0.0) or 0.0) for t in fallback_trades))
            return fallback_trades, status
        status["error"] = "LLM returned no usable trades."
        return [], status

    if total_weight > 1.0:
        scale = 1.0 / total_weight
        for t in cleaned:
            t["weight"] = t["weight"] * scale
        total_weight = 1.0

    # Encourage fuller capital deployment from AI when it proposes low aggregate weights.
    if 0.0 < total_weight < min_total_weight:
        scale = min_total_weight / total_weight
        for t in cleaned:
            t["weight"] = t["weight"] * scale
        total_weight = min_total_weight

    status["ok"] = True
    status["total_weight"] = total_weight
    status["min_total_weight"] = min_total_weight
    return cleaned, status
