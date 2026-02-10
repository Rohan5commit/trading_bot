from __future__ import annotations

import re
from pathlib import Path


_INDICATOR_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_]\w*)\s*=\s*ta\.(?P<fn>sma|ema|rsi)\(\s*(?P<src>[A-Za-z_]\w*)\s*,\s*(?P<length>\d+)\s*\)"
)
_CROSS_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_]\w*)\s*=\s*ta\.(?P<fn>crossover|crossunder)\(\s*(?P<left>[A-Za-z_]\w*)\s*,\s*(?P<right>[A-Za-z_]\w*)\s*\)"
)
_COMPARE_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<left>[A-Za-z_]\w*)\s*(?P<op>>=|<=|>|<)\s*(?P<right>[\d\.]+)\s*$"
)
_ENTRY_RE = re.compile(r"strategy\.entry\((?P<args>.+)\)")
_EXIT_RE = re.compile(r"strategy\.(close|exit)\((?P<args>.+)\)")
_WHEN_RE = re.compile(r"when\s*=\s*(?P<cond>[A-Za-z_]\w*|ta\.\w+\([^)]+\))")
_IF_RE = re.compile(r"^\s*if\s*\(\s*(?P<cond>.+)\s*\)\s*$")
_CROSS_INLINE_RE = re.compile(r"ta\.(?P<fn>crossover|crossunder)\(\s*(?P<left>[A-Za-z_]\w*)\s*,\s*(?P<right>[A-Za-z_]\w*)\s*\)")
_COMPARE_INLINE_RE = re.compile(r"(?P<left>[A-Za-z_]\w*)\s*(?P<op>>=|<=|>|<)\s*(?P<right>[\d\.]+)")


class UnsupportedPineScriptError(RuntimeError):
    pass


def load_pine_script(path: str | Path) -> str:
    script_path = Path(path)
    if not script_path.exists():
        raise FileNotFoundError(f"Pine script not found: {script_path}")
    return script_path.read_text(encoding="utf-8")


def pine_to_strategy_config(pine_script: str) -> dict:
    indicators: dict[str, dict] = {}
    conditions: dict[str, dict] = {}
    entry_cond = None
    exit_cond = None
    pending_if = None

    for raw_line in pine_script.splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue

        if "strategy.short" in line or "strategy.entry" in line and "strategy.short" in line:
            raise UnsupportedPineScriptError("Short strategies are not supported.")

        indicator_match = _INDICATOR_RE.match(line)
        if indicator_match:
            groups = indicator_match.groupdict()
            indicators[groups["name"]] = {
                "type": groups["fn"],
                "source": groups["src"],
                "length": int(groups["length"]),
            }
            continue

        cross_match = _CROSS_RE.match(line)
        if cross_match:
            groups = cross_match.groupdict()
            conditions[groups["name"]] = {
                "op": groups["fn"],
                "left": groups["left"],
                "right": groups["right"],
            }
            continue

        compare_match = _COMPARE_RE.match(line)
        if compare_match:
            groups = compare_match.groupdict()
            conditions[groups["name"]] = {
                "op": groups["op"],
                "left": groups["left"],
                "right": float(groups["right"]),
            }
            continue

        if_match = _IF_RE.match(line)
        if if_match:
            pending_if = if_match.group("cond")
            continue

        entry_match = _ENTRY_RE.search(line)
        if entry_match:
            entry_cond = _extract_condition(entry_match.group("args"), pending_if, conditions)
            pending_if = None
            continue

        exit_match = _EXIT_RE.search(line)
        if exit_match:
            exit_cond = _extract_condition(exit_match.group("args"), pending_if, conditions)
            pending_if = None
            continue

        pending_if = None

    if not entry_cond:
        raise UnsupportedPineScriptError("No entry condition detected in Pine script.")

    return {
        "type": "pine_parsed",
        "indicators": indicators,
        "entry": entry_cond,
        "exit": exit_cond,
    }


def _extract_condition(arg_text: str, pending_if: str | None, conditions: dict) -> dict:
    when_match = _WHEN_RE.search(arg_text)
    if when_match:
        token = when_match.group("cond").strip()
    elif pending_if:
        token = pending_if.strip()
    else:
        token = ""

    if token in conditions:
        return conditions[token]

    cross_inline = _CROSS_INLINE_RE.search(token)
    if cross_inline:
        groups = cross_inline.groupdict()
        return {"op": groups["fn"], "left": groups["left"], "right": groups["right"]}

    compare_inline = _COMPARE_INLINE_RE.search(token)
    if compare_inline:
        groups = compare_inline.groupdict()
        return {"op": groups["op"], "left": groups["left"], "right": float(groups["right"])}

    raise UnsupportedPineScriptError(f"Unsupported entry/exit condition: {token or arg_text}")
