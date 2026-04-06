from __future__ import annotations

import json
import os
from typing import Dict

from openai import OpenAI


class NIMClient:
    def __init__(self, model: str = "meta/llama-3.1-8b-instruct") -> None:
        base_url = os.getenv("NIM_API_BASE", "https://integrate.api.nvidia.com/v1")
        api_key = os.getenv("NVIDIA_NIM_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_NIM_API_KEY not set")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def classify(self, prompt: str) -> Dict:
        system = (
            "You are a quant model. Return ONLY JSON with keys: "
            "label (one of STRONG_SELL, SELL, NEUTRAL, BUY, STRONG_BUY) and confidence (0-1)."
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        content = resp.choices[0].message.content
        return self._parse_json(content)

    @staticmethod
    def _parse_json(text: str) -> Dict:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in NIM response")
        return json.loads(text[start : end + 1])
