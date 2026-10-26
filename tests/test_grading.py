"""Tests for compliance_checker.grading — rule-based GT matching."""

from __future__ import annotations

import pytest

from compliance_checker.grading import rule_grade

PILE = "★基樁樁徑及樁長"
MAIN = "★鋼筋籠加工組合-主筋直徑、伸展長度"
LAP = "★鋼筋籠加工組合-搭接長度"
HOOP = "★鋼筋籠加工組合-箍筋直徑、間距"
SPACER = "★鋼筋籠加工組合-鋼筋籠間隔器"


@pytest.mark.parametrize(
    ("check_item", "gt", "answer", "expected"),
    [
        # 樁徑及樁長: unit conversion and a/b lengths
        (PILE, "L=18m;D=1.2m", "L=18m;D=120cm", True),
        (PILE, "L=45/58m;D=1.2m", "L=58/45m; D=1.2m", True),
        (PILE, "L=18m;D=1.2m", "L=22m;D=1.2m", False),
        (PILE, "L=64;D=1.8m", "L=64m;D=1.8m", True),
        (PILE, "L=18m;D=1.2m", "樁長18公尺", None),
        # 主筋直徑、伸展長度: extra detail is fine
        (MAIN, "依設計圖主筋直徑=28mm、長度=95cm", "主筋D28(28mm)，伸展長度95cm，共30支", True),
        (MAIN, "依設計圖主筋直徑=28mm、長度=95cm", "主筋直徑=28mm、長度=85cm", False),
        (MAIN, "依設計圖主筋直徑=28mm、長度=95cm", "主筋D28", None),
        # 搭接長度: m converted to cm
        (LAP, "依設計圖主筋搭接長度≧165cm", "搭接長度≧1.65m", True),
        (LAP, "依設計圖主筋搭接長度≧165cm", "搭接長度≧155cm", False),
        # 箍筋直徑、間距: 80 mm / 8 cm do not count as 80 cm
        (HOOP, "依設計圖箍筋直徑=16mm、間距=80cm", "箍筋直徑=16mm、間距=80cm", True),
        (HOOP, "依設計圖箍筋直徑=16mm、間距=80cm", "箍筋直徑=16mm、間距=8cm", False),
        (HOOP, "依設計圖箍筋直徑=16mm、間距=80cm", "箍筋直徑=16mm、間距=80mm", False),
        (HOOP, "依設計圖箍筋直徑=16mm、間距=80cm", "內環箍筋Ø16、間距80 cm；外環箍筋Ø16、間距15 cm", True),
        (HOOP, "依設計圖箍筋直徑=16mm、間距=80cm", "內環箍筋φ16@80cm；外環箍筋2(φ16)@15cm", True),
        (HOOP, "依設計圖箍筋直徑=16mm、間距=80cm", "內環箍筋Ø160、間距80cm", False),
        # 鋼筋籠間隔器
        (SPACER, "每斷面至少6個、間距2.5m", "每斷面6個、間距2.5m", True),
        (SPACER, "每斷面至少6個、間距2.5m", "每斷面6個，間距250cm，D16間隔器", True),
        (SPACER, "每斷面至少6個、間距2.5m", "每斷面6個、@250", False),
        (SPACER, "每斷面至少6個、間距2.5m", "每斷面4個、間距2.5m", False),
    ],
)
def test_rule_grade(check_item: str, gt: str, answer: str, expected: bool | None) -> None:
    assert rule_grade(check_item, gt, answer) is expected


def test_unknown_check_item_returns_none() -> None:
    # Similar name from the retaining-wall dataset must not hit the pile rule.
    wall_item = "★基礎鋼筋及模板組立-主筋直徑、間距"
    assert rule_grade(wall_item, "編號S1：依設計圖主筋直徑=16mm、間距=20cm", "16mm 20cm") is None


def test_missing_answer_is_wrong() -> None:
    assert rule_grade(PILE, "L=18m;D=1.2m", None) is False


def test_evaluate_correctness_uses_rule_without_llm_call() -> None:
    from compliance_checker.ablation.experiments import evaluate_correctness

    # client=None: any LLM call would raise, proving the rule path short-circuits.
    ok, stage, method = evaluate_correctness(
        None,  # type: ignore[arg-type]
        "o4-mini",
        "L=18m;D=1.2m",
        "L=18m;D=1.2m",
        check_item=PILE,
    )
    assert (ok, stage, method) == (True, None, "rule")
