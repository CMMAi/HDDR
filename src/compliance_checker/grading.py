"""Deterministic rule-based grading of an advised standard against the ground truth.

For check items with a known GT format, the rules below decide correctness by checking
that the GT values appear in the answer:

  樁徑及樁長      L and D (converted to m) equal the GT values (L may be "a/b")
  主筋直徑、伸展長度  GT diameter (mm) and GT length (cm) both appear in the answer
  搭接長度        GT lap length appears (cm, or m converted)
  箍筋直徑、間距    GT diameter (mm) and GT spacing (cm) appear (mm / other units do not count)
  鋼筋籠間隔器     GT count per section and GT spacing (m or cm) appear; a bare "@300"
                  without unit fails

An answer may list several values for one quantity (per section, per zone, ...); it is
correct as long as the GT value is among them. If the GT itself lists several values
(e.g. "L=45/58m"), all of them must appear.

rule_grade() returns None when no rule applies to the check item or the answer cannot
be parsed; callers then fall back to the LLM judge.
"""

from __future__ import annotations

import re
from collections.abc import Callable

NUM = r"(\d+(?:\.\d+)?)"
BAR = "[Øøφ⌀∅]"


def _to_m(value: str, unit: str | None) -> float:
    return float(value) / 100 if unit == "cm" else float(value)


def _lengths(text: str, key: str) -> list[float] | None:
    """Values of `key=...` in metres, supporting 'a/b' lists and m/cm units."""
    m = re.search(rf"{key}\s*[=＝]\s*([\d./]+)\s*(cm|m)?", text)
    if not m:
        return None
    return sorted(_to_m(v, m[2]) for v in m[1].split("/") if v)


def _pile_size(gt: str, a: str) -> bool | None:
    gl, gd = _lengths(gt, "L"), _lengths(gt, "D")
    if gl is None or gd is None:
        return None
    al, ad = _lengths(a, "L"), _lengths(a, "D")
    if al is None or ad is None:
        return None
    return al == gl and ad == gd


def _main_bar(gt: str, a: str) -> bool | None:
    gd, gl = re.search(rf"直徑={NUM}mm", gt), re.search(rf"長度={NUM}cm", gt)
    if not gd or not gl:
        return None
    dias = re.findall(rf"{NUM}mm", a)
    lens = re.findall(rf"{NUM}cm", a)
    if not dias or not lens:
        return None
    return gd[1] in dias and gl[1] in lens


def _lap_length(gt: str, a: str) -> bool | None:
    g = re.search(rf"{NUM}cm", gt)
    if not g:
        return None
    lens = [float(x) for x in re.findall(rf"{NUM}cm", a)] + [
        float(x) * 100 for x in re.findall(rf"{NUM}m(?!m)", a)
    ]
    return float(g[1]) in lens


def _hoop(gt: str, a: str) -> bool | None:
    gd, gs = re.search(rf"直徑={NUM}mm", gt), re.search(rf"間距={NUM}cm", gt)
    if not gd or not gs:
        return None
    dia = re.escape(gd[1])
    spacing = re.escape(gs[1])
    # the bar symbol may follow the diameter ("19Ø") or precede it ("Ø19", "φ19@100cm")
    has_dia = re.search(rf"(?<![\d.]){dia}(mm|{BAR})", a) or re.search(rf"{BAR}{dia}(?![\d.])", a)
    return bool(has_dia) and bool(re.search(rf"(?<![\d.]){spacing}cm", a))


def _spacer(gt: str, a: str) -> bool | None:
    gc, gs = re.search(r"(\d+)個", gt), re.search(rf"間距{NUM}m", gt)
    if not gc or not gs:
        return None
    count = gc[1]
    spacing_m = float(gs[1])
    has_count = bool(re.search(rf"(?<!\d){count}(個|支|處)", a))
    m_hits = [float(x) for x in re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)m(?!m)", a)]
    cm_hits = [float(x) / 100 for x in re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)cm", a)]
    has_spacing = spacing_m in m_hits + cm_hits
    return has_count and has_spacing


# Exact check-item names → rule. Exact matching keeps rules from firing on similarly
# named items in other datasets (e.g. 基礎鋼筋及模板組立-主筋直徑、間距).
RULES: dict[str, Callable[[str, str], bool | None]] = {
    "★基樁樁徑及樁長": _pile_size,
    "★鋼筋籠加工組合-主筋直徑、伸展長度": _main_bar,
    "★鋼筋籠加工組合-搭接長度": _lap_length,
    "★鋼筋籠加工組合-箍筋直徑、間距": _hoop,
    "★鋼筋籠加工組合-鋼筋籠間隔器": _spacer,
}


def rule_grade(check_item: str, gt: str, answer) -> bool | None:
    """True/False by rule; None when no rule applies or the answer cannot be parsed."""
    rule = RULES.get(check_item.strip())
    if rule is None:
        return None
    if answer is None:
        return False
    return rule(re.sub(r"\s", "", gt), re.sub(r"\s", "", str(answer)))
