import json
import logging

from llm_sentiment import NvidiaChatClient, _extract_json

logger = logging.getLogger(__name__)


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
    llm_cfg = dict(llm_cfg)
    llm_cfg["model"] = ai_cfg.get("llm_model", llm_cfg.get("model"))
    llm_cfg["api_key_env"] = ai_cfg.get("api_key_env", llm_cfg.get("api_key_env"))
    client = NvidiaChatClient(llm_cfg)
    status = {
        "enabled": bool(llm_cfg.get("enabled", False)),
        "ok": False,
        "error": None,
        "model": llm_cfg.get("model"),
        "api_key_env": llm_cfg.get("api_key_env"),
    }
    if not client.is_ready():
        key_env = getattr(client, "api_key_env", "NVIDIA_API_KEY")
        status["error"] = f"LLM unavailable or {key_env} missing."
        return [], status

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

    system_msg = (
        "You are a trading decision engine. "
        "Return ONLY valid JSON (no markdown). "
        "Pick a diversified set of trades from the provided candidate list. "
        "There is no fixed strategy; use your own judgement based on the provided fields."
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
        "- Include a short reason per trade.",
        "",
        "JSON schema:",
        "{\"trades\": [{\"symbol\": \"AAPL\", \"side\": \"LONG\", \"weight\": 0.10, \"reason\": \"...\"}], \"notes\": \"...\"}",
    ])

    user_msg = "Candidates:\n" + json.dumps(prompt_candidates, indent=2) + "\n\nRules:\n" + "\n".join(rules)
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    response_text = client.chat(messages)
    if response_text is None:
        status["error"] = getattr(client, "last_error", None) or "LLM call failed."
        return [], status
    parsed = _extract_json(response_text)
    if not isinstance(parsed, dict):
        status["error"] = "LLM response missing JSON."
        return [], status

    trades = parsed.get("trades", [])
    if not isinstance(trades, list):
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
        cleaned.append({"symbol": sym, "side": side, "weight": w, "reason": (t.get("reason") or "").strip()})
        total_weight += w
        if len(cleaned) >= max_positions:
            break

    if not cleaned:
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
