"""Latency and token-cost tracking for the compliance-checking pipeline.

Usage
-----
Wrap each pipeline stage with ``time.perf_counter()`` and collect
``StageMetrics`` objects.  Pass them to ``ItemMetrics`` / ``FormMetrics``
for aggregation, then call ``format_summary()`` to produce a text summary.

Non-API stages (YOLO, ResNet+KMeans, Neo4j search) use default values of
zero for token fields — only ``elapsed_s`` is meaningful for those stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Pricing table — USD per 1 million tokens (input / output).
# OpenAI list prices used in our experiments; update them to the current rates.
# Models missing from this table are costed at $0.
# ---------------------------------------------------------------------------
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
    "o4-mini-2025-04-16": {"input": 1.10, "output": 4.40},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
}

# ---------------------------------------------------------------------------
# Stage name constants — covers both LLM and non-API pipeline stages.
# ---------------------------------------------------------------------------
STAGE_NEO4J_EMBEDDING = "neo4j_embedding"
STAGE_NEO4J_SEARCH = "neo4j_search"
STAGE_YOLO_CROP = "yolo_crop"
STAGE_RESNET_KMEANS = "resnet_kmeans"
STAGE_DDREVIEW_RERANK = "ddreview_rerank"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage (LLM call or non-API computation).

    For non-API stages set prompt_tokens=0, completion_tokens=0, model="".
    """

    stage: str
    elapsed_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def calculate_cost(stages: list[StageMetrics]) -> float:
    """Return total USD cost for a list of stage metrics."""
    total = 0.0
    for s in stages:
        pricing = PRICING.get(s.model)
        if pricing is None:
            continue
        total += (s.prompt_tokens / 1_000_000) * pricing["input"]
        total += (s.completion_tokens / 1_000_000) * pricing["output"]
    return total


@dataclass
class ItemMetrics:
    """Aggregated metrics for one check-item across all its pipeline stages."""

    form_filename: str
    item_idx: int
    stages: list[StageMetrics] = field(default_factory=list)

    def elapsed_excluding(self, *stage_names: str) -> float:
        """Sum elapsed_s for all stages not in stage_names."""
        return sum(s.elapsed_s for s in self.stages if s.stage not in stage_names)

    @property
    def total_elapsed_s(self) -> float:
        return sum(s.elapsed_s for s in self.stages)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.stages)

    @property
    def total_completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.stages)

    @property
    def estimated_cost_usd(self) -> float:
        return calculate_cost(self.stages)


@dataclass
class FormMetrics:
    """Aggregated metrics for one inspection form (page retrieval + all items)."""

    form_filename: str
    page_stages: list[StageMetrics] = field(default_factory=list)
    items: list[ItemMetrics] = field(default_factory=list)

    @property
    def _all_stages(self) -> list[StageMetrics]:
        item_stages = [s for item in self.items for s in item.stages]
        return self.page_stages + item_stages

    @property
    def page_elapsed_s(self) -> float:
        return sum(s.elapsed_s for s in self.page_stages)

    @property
    def total_elapsed_s(self) -> float:
        return sum(s.elapsed_s for s in self._all_stages)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self._all_stages)

    @property
    def total_completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self._all_stages)

    @property
    def estimated_cost_usd(self) -> float:
        return calculate_cost(self._all_stages)

    @property
    def amortized_item_elapsed_s(self) -> float:
        """Per-item elapsed including page retrieval cost amortized across items."""
        n = len(self.items)
        if n == 0:
            return 0.0
        avg_item = sum(i.total_elapsed_s for i in self.items) / n
        return avg_item + self.page_elapsed_s / n

    def amortized_item_excl(
        self, *stage_names: str
    ) -> tuple[float, float, float, float]:
        """Amortized per-item (elapsed_s, prompt_tok, compl_tok, cost_usd) excluding stage_names."""
        n = len(self.items)
        if n == 0:
            return 0.0, 0.0, 0.0, 0.0

        def item_elapsed(i: ItemMetrics) -> float:
            return sum(s.elapsed_s for s in i.stages if s.stage not in stage_names)

        def item_prompt(i: ItemMetrics) -> int:
            return sum(s.prompt_tokens for s in i.stages if s.stage not in stage_names)

        def item_compl(i: ItemMetrics) -> int:
            return sum(s.completion_tokens for s in i.stages if s.stage not in stage_names)

        def item_cost(i: ItemMetrics) -> float:
            return calculate_cost([s for s in i.stages if s.stage not in stage_names])

        page_prompt = sum(s.prompt_tokens for s in self.page_stages)
        page_compl = sum(s.completion_tokens for s in self.page_stages)
        page_cost = calculate_cost(self.page_stages)

        return (
            sum(item_elapsed(i) for i in self.items) / n + self.page_elapsed_s / n,
            sum(item_prompt(i) for i in self.items) / n + page_prompt / n,
            sum(item_compl(i) for i in self.items) / n + page_compl / n,
            sum(item_cost(i) for i in self.items) / n + page_cost / n,
        )

    @property
    def amortized_item_prompt_tokens(self) -> float:
        n = len(self.items)
        if n == 0:
            return 0.0
        page_prompt = sum(s.prompt_tokens for s in self.page_stages)
        avg_item = sum(i.total_prompt_tokens for i in self.items) / n
        return avg_item + page_prompt / n

    @property
    def amortized_item_completion_tokens(self) -> float:
        n = len(self.items)
        if n == 0:
            return 0.0
        page_compl = sum(s.completion_tokens for s in self.page_stages)
        avg_item = sum(i.total_completion_tokens for i in self.items) / n
        return avg_item + page_compl / n

    @property
    def amortized_item_cost_usd(self) -> float:
        n = len(self.items)
        if n == 0:
            return 0.0
        page_cost = calculate_cost(self.page_stages)
        avg_item = sum(i.estimated_cost_usd for i in self.items) / n
        return avg_item + page_cost / n


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def format_summary(
    config_name: str,
    model: str,
    forms: list[FormMetrics],
) -> str:
    """Return a multi-line latency and token-cost summary."""
    if not forms:
        return (
            f"Configuration: {config_name}\n"
            f"Model: {model}\n"
            "No forms processed.\n"
        )

    n_forms = len(forms)
    n_items_per_form = len(forms[0].items) if forms else 0

    # Per-form averages
    form_page_elapsed = _avg([f.page_elapsed_s for f in forms])
    form_page_prompt = _avg([sum(s.prompt_tokens for s in f.page_stages) for f in forms])
    form_page_completion = _avg([sum(s.completion_tokens for s in f.page_stages) for f in forms])
    form_page_cost = _avg([calculate_cost(f.page_stages) for f in forms])

    form_cc_elapsed = _avg([sum(i.total_elapsed_s for i in f.items) for f in forms])
    form_cc_prompt = _avg([sum(i.total_prompt_tokens for i in f.items) for f in forms])
    form_cc_completion = _avg([sum(i.total_completion_tokens for i in f.items) for f in forms])
    form_cc_cost = _avg([sum(i.estimated_cost_usd for i in f.items) for f in forms])

    form_total_elapsed = _avg([f.total_elapsed_s for f in forms])
    form_total_prompt = _avg([f.total_prompt_tokens for f in forms])
    form_total_completion = _avg([f.total_completion_tokens for f in forms])
    form_total_cost = _avg([f.estimated_cost_usd for f in forms])

    # Per-item averages — amortized: page retrieval cost divided across items
    item_elapsed = _avg([f.amortized_item_elapsed_s for f in forms])
    item_prompt = _avg([f.amortized_item_prompt_tokens for f in forms])
    item_completion = _avg([f.amortized_item_completion_tokens for f in forms])
    item_cost = _avg([f.amortized_item_cost_usd for f in forms])

    # Pipeline-only: exclude correctness_eval (evaluation overhead, not production cost)
    excl = [f.amortized_item_excl("correctness_eval") for f in forms]
    item_elapsed_p = _avg([e[0] for e in excl])
    item_prompt_p = _avg([e[1] for e in excl])
    item_completion_p = _avg([e[2] for e in excl])
    item_cost_p = _avg([e[3] for e in excl])

    lines = [
        f"Configuration: {config_name}",
        f"Model: {model}",
        f"Forms: {n_forms} | Items per form: {n_items_per_form}",
        "",
        "Per-Form Averages:",
        f"  Page Retrieval:          {form_page_elapsed:>6.1f} s  "
        f"{int(form_page_prompt):>8,} prompt tok  "
        f"{int(form_page_completion):>8,} compl tok  "
        f"${form_page_cost:.4f}",
        f"  Compliance (all items):  {form_cc_elapsed:>6.1f} s  "
        f"{int(form_cc_prompt):>8,} prompt tok  "
        f"{int(form_cc_completion):>8,} compl tok  "
        f"${form_cc_cost:.4f}",
        f"  Total per form:          {form_total_elapsed:>6.1f} s  "
        f"{int(form_total_prompt):>8,} prompt tok  "
        f"{int(form_total_completion):>8,} compl tok  "
        f"${form_total_cost:.4f}",
        "",
        "Per-Item Averages (amortized page retrieval):",
        f"  w/  correctness_eval:    {item_elapsed:>6.1f} s  "
        f"{int(item_prompt):>8,} prompt tok  "
        f"{int(item_completion):>8,} compl tok  "
        f"${item_cost:.4f}",
        f"  w/o correctness_eval:    {item_elapsed_p:>6.1f} s  "
        f"{int(item_prompt_p):>8,} prompt tok  "
        f"{int(item_completion_p):>8,} compl tok  "
        f"${item_cost_p:.4f}",
    ]

    return "\n".join(lines) + "\n"
