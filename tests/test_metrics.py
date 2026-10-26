"""Tests for compliance_checker.metrics — cost/latency tracking.

TDD: these tests are written BEFORE the implementation.
"""

from __future__ import annotations

import pytest

# ── StageMetrics ─────────────────────────────────────────────────────────────


class TestStageMetrics:
    def test_fields_are_accessible(self) -> None:
        from compliance_checker.metrics import StageMetrics

        s = StageMetrics(
            stage="compliance_check",
            elapsed_s=2.5,
            prompt_tokens=1000,
            completion_tokens=300,
            model="gpt-4o-2024-08-06",
        )
        assert s.stage == "compliance_check"
        assert s.elapsed_s == pytest.approx(2.5)
        assert s.prompt_tokens == 1000
        assert s.completion_tokens == 300
        assert s.model == "gpt-4o-2024-08-06"

    def test_total_tokens(self) -> None:
        from compliance_checker.metrics import StageMetrics

        s = StageMetrics(
            stage="rerank",
            elapsed_s=1.0,
            prompt_tokens=500,
            completion_tokens=100,
            model="gpt-4o-2024-08-06",
        )
        assert s.total_tokens == 600

    def test_zero_tokens(self) -> None:
        from compliance_checker.metrics import StageMetrics

        s = StageMetrics(
            stage="embedding",
            elapsed_s=0.1,
            prompt_tokens=0,
            completion_tokens=0,
            model="text-embedding-3-small",
        )
        assert s.total_tokens == 0


# ── calculate_cost ───────────────────────────────────────────────────────────


class TestCalculateCost:
    def test_gpt4o_cost(self) -> None:
        from compliance_checker.metrics import StageMetrics, calculate_cost

        stages = [
            StageMetrics(
                stage="check",
                elapsed_s=1.0,
                prompt_tokens=1_000_000,
                completion_tokens=1_000_000,
                model="gpt-4o-2024-08-06",
            )
        ]
        # input: $2.50/M, output: $10.00/M → total $12.50
        cost = calculate_cost(stages)
        assert cost == pytest.approx(12.50)

    def test_o4mini_cost(self) -> None:
        from compliance_checker.metrics import StageMetrics, calculate_cost

        stages = [
            StageMetrics(
                stage="check",
                elapsed_s=1.0,
                prompt_tokens=1_000_000,
                completion_tokens=1_000_000,
                model="o4-mini-2025-04-16",
            )
        ]
        # input: $1.10/M, output: $4.40/M → total $5.50
        cost = calculate_cost(stages)
        assert cost == pytest.approx(5.50)

    def test_embedding_cost(self) -> None:
        from compliance_checker.metrics import StageMetrics, calculate_cost

        stages = [
            StageMetrics(
                stage="embed",
                elapsed_s=0.2,
                prompt_tokens=1_000_000,
                completion_tokens=0,
                model="text-embedding-3-small",
            )
        ]
        # input: $0.02/M, output: $0.00/M → total $0.02
        cost = calculate_cost(stages)
        assert cost == pytest.approx(0.02)

    def test_multiple_stages_accumulate(self) -> None:
        from compliance_checker.metrics import StageMetrics, calculate_cost

        stages = [
            StageMetrics("a", 1.0, 100_000, 50_000, "gpt-4o-2024-08-06"),
            StageMetrics("b", 1.0, 100_000, 50_000, "gpt-4o-2024-08-06"),
        ]
        # Each stage: 0.10M*$2.50 + 0.05M*$10 = $0.25 + $0.50 = $0.75 → ×2 = $1.50
        cost = calculate_cost(stages)
        assert cost == pytest.approx(1.50)

    def test_empty_stages_returns_zero(self) -> None:
        from compliance_checker.metrics import calculate_cost

        assert calculate_cost([]) == pytest.approx(0.0)

    def test_unknown_model_returns_zero_cost(self) -> None:
        from compliance_checker.metrics import StageMetrics, calculate_cost

        stages = [
            StageMetrics("x", 1.0, 999_999, 999_999, "unknown-model-xyz")
        ]
        assert calculate_cost(stages) == pytest.approx(0.0)


# ── ItemMetrics ──────────────────────────────────────────────────────────────


class TestItemMetrics:
    def _make_stages(self) -> list:
        from compliance_checker.metrics import StageMetrics

        return [
            StageMetrics("ddr", 1.5, 800, 200, "gpt-4o-2024-08-06"),
            StageMetrics("compliance", 3.0, 1200, 400, "gpt-4o-2024-08-06"),
            StageMetrics("evaluator", 2.0, 900, 300, "gpt-4o-2024-08-06"),
        ]

    def test_total_elapsed(self) -> None:
        from compliance_checker.metrics import ItemMetrics

        item = ItemMetrics("form_01", 0, self._make_stages())
        assert item.total_elapsed_s == pytest.approx(6.5)

    def test_total_prompt_tokens(self) -> None:
        from compliance_checker.metrics import ItemMetrics

        item = ItemMetrics("form_01", 0, self._make_stages())
        assert item.total_prompt_tokens == 2900

    def test_total_completion_tokens(self) -> None:
        from compliance_checker.metrics import ItemMetrics

        item = ItemMetrics("form_01", 0, self._make_stages())
        assert item.total_completion_tokens == 900

    def test_estimated_cost_delegates_to_calculate_cost(self) -> None:
        from compliance_checker.metrics import ItemMetrics, calculate_cost

        stages = self._make_stages()
        item = ItemMetrics("form_01", 0, stages)
        assert item.estimated_cost_usd == pytest.approx(calculate_cost(stages))

    def test_empty_stages(self) -> None:
        from compliance_checker.metrics import ItemMetrics

        item = ItemMetrics("form_01", 0, [])
        assert item.total_elapsed_s == pytest.approx(0.0)
        assert item.total_prompt_tokens == 0
        assert item.total_completion_tokens == 0
        assert item.estimated_cost_usd == pytest.approx(0.0)


# ── FormMetrics ──────────────────────────────────────────────────────────────


class TestFormMetrics:
    def _make_item_metrics(self) -> list:
        from compliance_checker.metrics import ItemMetrics, StageMetrics

        return [
            ItemMetrics(
                "form_01",
                i,
                [StageMetrics("compliance", 2.0, 1000, 300, "gpt-4o-2024-08-06")],
            )
            for i in range(3)
        ]

    def test_total_elapsed_sums_page_retrieval_and_items(self) -> None:
        from compliance_checker.metrics import FormMetrics, StageMetrics

        page_stage = StageMetrics("page_retrieval", 5.0, 500, 0, "gpt-4o-2024-08-06")
        items = self._make_item_metrics()
        form = FormMetrics("form_01", [page_stage], items)

        # page: 5.0s + items: 3 × 2.0s = 11.0s
        assert form.total_elapsed_s == pytest.approx(11.0)

    def test_total_tokens_sums_page_and_items(self) -> None:
        from compliance_checker.metrics import FormMetrics, StageMetrics

        page_stage = StageMetrics("page_retrieval", 1.0, 200, 50, "gpt-4o-2024-08-06")
        items = self._make_item_metrics()
        form = FormMetrics("form_01", [page_stage], items)

        # page: 200+50=250, items: 3×(1000+300)=3900
        assert form.total_prompt_tokens == 200 + 3 * 1000
        assert form.total_completion_tokens == 50 + 3 * 300

    def test_estimated_cost_is_sum_of_all_stages(self) -> None:
        from compliance_checker.metrics import FormMetrics, StageMetrics, calculate_cost

        page_stages = [StageMetrics("page_retrieval", 1.0, 200, 50, "gpt-4o-2024-08-06")]
        items = self._make_item_metrics()
        form = FormMetrics("form_01", page_stages, items)

        all_stages = page_stages + [s for item in items for s in item.stages]
        assert form.estimated_cost_usd == pytest.approx(calculate_cost(all_stages))

    def test_no_items(self) -> None:
        from compliance_checker.metrics import FormMetrics, StageMetrics

        page_stage = StageMetrics("page_retrieval", 3.0, 400, 0, "gpt-4o-2024-08-06")
        form = FormMetrics("form_01", [page_stage], [])
        assert form.total_elapsed_s == pytest.approx(3.0)
        assert form.total_prompt_tokens == 400


# ── format_summary ────────────────────────────────────────────────────────────


class TestFormatSummary:
    """format_summary(config_name, model, forms) → multi-line string."""

    def _make_form_list(self) -> list:
        from compliance_checker.metrics import FormMetrics, ItemMetrics, StageMetrics

        items = [
            ItemMetrics(
                "form_01",
                i,
                [StageMetrics("compliance", 2.0, 1000, 300, "gpt-4o-2024-08-06")],
            )
            for i in range(5)
        ]
        page_stage = StageMetrics("page_retrieval", 4.0, 500, 0, "gpt-4o-2024-08-06")
        return [FormMetrics("form_01", [page_stage], items)]

    def test_returns_string(self) -> None:
        from compliance_checker.metrics import format_summary

        forms = self._make_form_list()
        result = format_summary("TSPR_DDC_DDR_Rev", "gpt-4o-2024-08-06", forms)
        assert isinstance(result, str)

    def test_contains_config_name(self) -> None:
        from compliance_checker.metrics import format_summary

        forms = self._make_form_list()
        result = format_summary("TSPR_DDC_DDR_Rev", "gpt-4o-2024-08-06", forms)
        assert "TSPR_DDC_DDR_Rev" in result

    def test_contains_model_name(self) -> None:
        from compliance_checker.metrics import format_summary

        forms = self._make_form_list()
        result = format_summary("TSPR_DDC_DDR_Rev", "gpt-4o-2024-08-06", forms)
        assert "gpt-4o-2024-08-06" in result

    def test_contains_per_form_and_per_item_sections(self) -> None:
        from compliance_checker.metrics import format_summary

        forms = self._make_form_list()
        result = format_summary("TSPR_DDC_DDR_Rev", "gpt-4o-2024-08-06", forms)
        lower = result.lower()
        assert "per-form" in lower or "per form" in lower
        assert "per-item" in lower or "per item" in lower

    def test_empty_forms_list(self) -> None:
        from compliance_checker.metrics import format_summary

        result = format_summary("TSPR_DDC_DDR_Rev", "gpt-4o-2024-08-06", [])
        assert isinstance(result, str)
