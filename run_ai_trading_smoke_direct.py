import json
import os
import re
from datetime import datetime

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_trader import _enforce_english_reason, _pick_predictions, _weights_from_predictions
from run_ai_trading_smoke import build_candidates, load_config

LABEL_TO_SCORE = {
    "STRONG_BUY": 2.0,
    "BUY": 1.0,
    "NEUTRAL": 0.0,
    "SELL": -1.0,
    "STRONG_SELL": -2.0,
}
LABEL_RE = re.compile(r"\b(STRONG_BUY|BUY|NEUTRAL|SELL|STRONG_SELL)\b", re.IGNORECASE)


def _extract_json(text: str):
    if not text:
        return None
    text = str(text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return None


def _parse_label(text: str):
    match = LABEL_RE.search(str(text or ""))
    if not match:
        return None
    return match.group(1).upper()


def _candidate_prompt(candidate):
    return "\n".join(
        [
            f"TICKER: {candidate.get('symbol')}",
            f"DATE: {candidate.get('as_of_date') or candidate.get('last_date')}",
            f"LAST_CLOSE: {candidate.get('last_close')}",
            f"CLOSES_TAIL: {candidate.get('closes_tail')}",
            f"RETURN_1D: {candidate.get('return_1d')}",
            f"RETURN_5D: {candidate.get('return_5d')}",
            f"RETURN_10D: {candidate.get('return_10d')}",
            f"VOLATILITY_20D: {candidate.get('volatility_20d')}",
            f"DIST_MA_20: {candidate.get('dist_ma_20')}",
            f"DIST_MA_50: {candidate.get('dist_ma_50')}",
            f"RSI_14: {candidate.get('rsi_14')}",
            f"VOLUME_RATIO: {candidate.get('volume_ratio')}",
            f"NEWS_COUNT_7D: {candidate.get('news_count_7d')}",
            f"NEWS_SENTIMENT_7D: {candidate.get('news_sentiment_7d')}",
            "QUESTION: Classify the expected 5-day return as STRONG_BUY | BUY | NEUTRAL | SELL | STRONG_SELL.",
            'Return only compact JSON: {"label":"BUY","confidence":0.63,"reason":"short english phrase"}',
        ]
    )


def _load_runtime():
    base_model = os.getenv("TRAINED_MODEL_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    adapter_path = os.getenv("TRAINED_MODEL_ADAPTER_PATH", "_smoke_artifacts/lora_solid_adapter")
    cpu_threads = max(1, int(os.getenv("TRAINED_MODEL_CPU_THREADS", "4") or 4))
    torch.set_num_threads(cpu_threads)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return model, tokenizer


def _predict_one(model, tokenizer, candidate):
    system = "You are the trained AI trading decision engine. Return only valid compact JSON with label, confidence, and a very short reason."
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": _candidate_prompt(candidate)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    prompt_len = encoded["input_ids"].shape[-1]
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
    parsed = _extract_json(text) or {}
    label = str(parsed.get("label") or "").strip().upper() or (_parse_label(parsed.get("reason") or text) or "NEUTRAL")
    confidence = parsed.get("confidence")
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.5 if label == "NEUTRAL" else 0.65
    return {
        "symbol": candidate["symbol"],
        "label": label,
        "score": LABEL_TO_SCORE.get(label, 0.0),
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": _enforce_english_reason(parsed.get("reason") or text, "LONG" if LABEL_TO_SCORE.get(label, 0.0) > 0 else "SHORT"),
        "raw_text": text,
    }


def main():
    config = load_config()
    tickers = [s.strip().upper() for s in os.getenv("AI_SMOKE_TICKERS", "AAPL").split(",") if s.strip()]
    candidates, failures = build_candidates(config, tickers)
    model, tokenizer = _load_runtime()

    predictions = []
    for candidate in candidates:
        pred = _predict_one(model, tokenizer, candidate)
        if float(pred.get("score", 0.0)) == 0.0:
            continue
        side = "LONG" if pred["score"] > 0 else "SHORT"
        predictions.append(
            {
                "symbol": candidate["symbol"],
                "side": side,
                "score": float(pred["score"]),
                "confidence": float(pred["confidence"]),
                "strength": max(0.01, abs(float(pred["score"])) * max(float(pred["confidence"]), 0.05)),
                "reason": pred["reason"],
                "label": pred["label"],
                "raw_text": pred["raw_text"],
            }
        )

    ai_cfg = config.get("ai_trading", {}) if isinstance(config, dict) else {}
    picked = _pick_predictions(
        predictions,
        max_positions=min(int(ai_cfg.get("max_positions", 10) or 10), max(1, len(predictions) or 1)),
        allow_shorts=bool(ai_cfg.get("allow_shorts", True)),
        max_shorts=int(ai_cfg.get("max_shorts", 5) or 5),
    )
    weighted = _weights_from_predictions(picked, min_total_weight=float(ai_cfg.get("min_total_weight", 0.90) or 0.90)) if picked else []
    trades = [
        {
            "symbol": row["symbol"],
            "side": row["side"],
            "weight": float(row["weight"]),
            "reason": row["reason"],
            "label": row["label"],
        }
        for row in weighted
    ]

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tickers": tickers,
        "candidates_built": len(candidates),
        "candidate_failures": failures,
        "status": {
            "enabled": True,
            "ok": True,
            "error": None,
            "decision_engine": "trained_model",
            "backend": "github_actions_cpu_direct",
            "model": os.getenv("TRAINED_MODEL_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            "model_used": os.getenv("TRAINED_MODEL_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            "candidates_seen": len(candidates),
            "candidates_scored": len(predictions),
        },
        "predictions": predictions,
        "trades": trades,
    }

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"ai_smoke_direct_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json")
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))
    if not candidates:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
