r"""
Run the ablation configurations on the Retaining Wall, Full Casing Pile and
Open Ditch datasets.

Page retrieval modes:
    greedy   Neo4j similarity search → top-1 result, no LLM verification.
    vanilla  Neo4j similarity search → single-stage LLM (content check only).
    tspr     Neo4j similarity search → Two-Stage Page Retrieval
             (Stage 1: position check, Stage 2: content check).

Module ablation (--set full; cumulative, ≤3 evaluator rounds):
    Name              page_mode  DDC    DDR    Review
    ──────────────────────────────────────────────────
    Greedy            greedy      -      -      -
    Vanilla           vanilla     -      -      -
    DDC               vanilla     ✓      -      -
    TSPR_DDC          tspr        ✓      -      -
    TSPR_DDC_DDR      tspr        ✓      ✓      -
    TSPR_DDC_DDR_Rev  tspr        ✓      ✓      ✓

Reflection-count ablation (--set reflection; all modules, max_reflections 0–3):
    Reflection_0      tspr        ✓      ✓      ✓   (no evaluator loop)
    Reflection_1      tspr        ✓      ✓      ✓   (≤1 evaluator round)
    Reflection_2      tspr        ✓      ✓      ✓   (≤2 evaluator rounds)
    Reflection_3      tspr        ✓      ✓      ✓   (≤3 evaluator rounds; = TSPR_DDC_DDR_Rev)

Prerequisites:
    • OpenAI API key, and a Neo4j KG built first with build-kg.

Usage:
    run-ablation --dataset pile --set full --model_name o4-mini-2025-04-16 \\
        --openai_api_key $OPENAI_API_KEY \\
        --neo4j_password $NEO4J_PASSWORD

    Outputs are written to output/<dataset>_<model>_<timestamp>/ (override with --run_root).
"""

import base64
import json
import os
import sys
import threading
import time
from typing import Any
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

from compliance_checker.clustering import (
    detailed_drawing_adjuestment,
    group_images,
)
from compliance_checker.detection import crop_page, getModel
from compliance_checker.grading import rule_grade
from compliance_checker.metrics import (
    STAGE_DDREVIEW_RERANK,
    STAGE_NEO4J_EMBEDDING,
    STAGE_NEO4J_SEARCH,
    STAGE_RESNET_KMEANS,
    STAGE_YOLO_CROP,
    FormMetrics,
    ItemMetrics,
    StageMetrics,
    format_summary,
)
from compliance_checker.prompts import (
    create_prompt_for_compliance_check,
    create_prompt_for_compliance_evaluator,
    create_prompt_for_correctness_eval,
    create_prompt_for_page_retrieval_detail,
    create_prompt_for_page_retrieval_position,
    create_prompt_for_page_selection_with_detail_drawings,
    create_prompt_for_page_selection_with_whole_page,
    create_prompt_for_rerank,
)

load_dotenv()

_NEO4J_PASSWORD_REQUIRED_MSG = (
    "Page retrieval requires --neo4j_password (or NEO4J_PASSWORD in .env)."
)

# Module-level bound loggers — sinks are wired in main()
status_log = logger.bind(log_type="status")
form_log = logger.bind(log_type="form")

# ── Datasets ────────────────────────────────────────────────────────────

# Each drawing book is a directory under DRAWINGS_ROOT holding
# <DrawingNumber>/<DrawingNumber>.png and <DrawingNumber>/Detailed Drawings/.
DRAWINGS_ROOT = Path("data/drawings")
BOOK_DIRS = sorted(p for p in DRAWINGS_ROOT.glob("*/") if p.is_dir())


def find_book_dir(page: str) -> Path | None:
    """Return the book directory that contains page PNG, or None if not found."""
    for book_dir in BOOK_DIRS:
        if (book_dir / page / f"{page}.png").exists():
            return book_dir
    return None


DATASET_DIRS = {
    "wall": "data/test_forms_retaining_wall",
    "pile": "data/test_forms_full_casing_pile",
    "ditch": "data/test_forms_open_ditch",
}

# page_mode: how the page is selected for each form
# do_seg:    whether to segment the page into detailed drawings (DDC)
# do_sel:    whether to rerank and select the best drawing (DDR)
# do_review: whether to cluster and adjust minority drawings (DDReview)
CONFIGURATIONS = [
    # ─── Full pipeline (requires Neo4j KG) ──────────────────────────────────
    {
        "name": "Greedy",
        "page_mode": "greedy",
        "do_seg": False,
        "do_sel": False,
        "do_review": False,
        "do_reflection": True,
    },
    {
        "name": "Vanilla",
        "page_mode": "vanilla",
        "do_seg": False,
        "do_sel": False,
        "do_review": False,
        "do_reflection": True,
    },
    {
        "name": "DDC",
        "page_mode": "vanilla",
        "do_seg": True,
        "do_sel": False,
        "do_review": False,
        "do_reflection": True,
    },
    {
        "name": "TSPR_DDC",
        "page_mode": "tspr",
        "do_seg": True,
        "do_sel": False,
        "do_review": False,
        "do_reflection": True,
    },
    {
        "name": "TSPR_DDC_DDR",
        "page_mode": "tspr",
        "do_seg": True,
        "do_sel": True,
        "do_review": False,
        "do_reflection": True,
    },
    {
        "name": "TSPR_DDC_DDR_Rev",
        "page_mode": "tspr",
        "do_seg": True,
        "do_sel": True,
        "do_review": True,
        "do_reflection": True,
    },
    # ─── Reflection-count ablation (full pipeline, vary max_reflections) ─────
    {
        "name": "Reflection_0",
        "page_mode": "tspr",
        "do_seg": True,
        "do_sel": True,
        "do_review": True,
        "max_reflections": 0,
    },
    {
        "name": "Reflection_1",
        "page_mode": "tspr",
        "do_seg": True,
        "do_sel": True,
        "do_review": True,
        "max_reflections": 1,
    },
    {
        "name": "Reflection_2",
        "page_mode": "tspr",
        "do_seg": True,
        "do_sel": True,
        "do_review": True,
        "max_reflections": 2,
    },
    {
        "name": "Reflection_3",
        "page_mode": "tspr",
        "do_seg": True,
        "do_sel": True,
        "do_review": True,
        "max_reflections": 3,
    },
]

NEEDS_NEO4J = {"greedy", "vanilla", "tspr"}

# ── Parallel execution ────────────────────────────────────────────────────────
MAX_WORKERS = 20
_yolo_lock = threading.Lock()

# ── API helpers ────────────────────────────────────────────────────────────────


def getResponse(client: OpenAI, prompt: dict, model_name: str) -> tuple[str, int, int]:  # noqa: N802
    """Call OpenAI with retry on any exception. Returns (text, prompt_tokens, completion_tokens)."""
    is_reasoning = model_name.startswith("o")
    while True:
        try:
            kwargs: dict = {
                "model": model_name,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
            }
            if not is_reasoning:
                kwargs["temperature"] = 0.0
                kwargs["max_completion_tokens"] = 4000
            resp = client.chat.completions.create(**kwargs)
            return (
                resp.choices[0].message.content,
                resp.usage.prompt_tokens,
                resp.usage.completion_tokens,
            )
        except Exception as err:
            status_log.warning(f"API error: {err}, retrying in 1s")
            time.sleep(1)


# ── Page retrieval ─────────────────────────────────────────────────────────────


def _similarity_search(
    client: OpenAI, kg, query: str, top_k: int
) -> tuple[list[dict], list[StageMetrics]]:
    """Run Neo4j vector similarity search.

    Returns (results, stage_metrics) where results is a list of {title, number}
    dicts and stage_metrics contains one entry each for the embedding API call
    and the Neo4j query.
    """
    stage_metrics: list[StageMetrics] = []

    t0 = time.perf_counter()
    emb_response = client.embeddings.create(input=[query], model="text-embedding-3-small")
    embedding = emb_response.data[0].embedding
    stage_metrics.append(
        StageMetrics(
            stage=STAGE_NEO4J_EMBEDDING,
            elapsed_s=time.perf_counter() - t0,
            prompt_tokens=emb_response.usage.prompt_tokens,
            model="text-embedding-3-small",
        )
    )

    t0 = time.perf_counter()
    results = kg.query(
        """
        CALL db.index.vector.queryNodes($index_name, $top_k, $question_embedding)
        YIELD node, score
        RETURN node.title AS title, node.number AS number
        """,
        params={
            "question_embedding": embedding,
            "index_name": "references",
            "top_k": top_k,
        },
    )
    stage_metrics.append(
        StageMetrics(
            stage=STAGE_NEO4J_SEARCH,
            elapsed_s=time.perf_counter() - t0,
        )
    )

    return results, stage_metrics


def _ensure_detailed_drawings(
    yolo_model,
    page_png: Path,
    detailed_dir: Path,
) -> StageMetrics | None:
    """Crop page into detailed drawings if not already done.

    Returns a StageMetrics for the YOLO inference when it actually runs,
    or None when the directory already contains PNGs (idempotent).
    """
    detailed_dir.mkdir(parents=True, exist_ok=True)
    with _yolo_lock:
        if not any(p.suffix == ".png" for p in detailed_dir.iterdir()):
            t0 = time.perf_counter()
            crop_page(yolo_model, str(page_png), str(detailed_dir))
            return StageMetrics(stage=STAGE_YOLO_CROP, elapsed_s=time.perf_counter() - t0)
    return None


def retrieve_page(  # noqa: PLR0911 PLR0912 PLR0913 PLR0915
    page_mode: str,
    do_seg: bool,
    form_data: dict,
    client: OpenAI,
    model_name: str,
    yolo_model,
    kg,
    top_k: int = 10,
) -> tuple[str | None, bool, list[StageMetrics]]:
    """
    Select a page number for the given form.

    Returns:
        (page_number, page_retrieval_correct, stage_metrics)
        page_number: None if retrieval failed.
        page_retrieval_correct: True if selected page == ground_truth_page.
        stage_metrics: list of StageMetrics for each LLM call made.

    """
    stage_metrics: list[StageMetrics] = []
    gt_page = form_data["ground_truth_page"]

    # ── Neo4j similarity search ──────────────────────────────────────────────
    position = form_data["position"]
    form_name = form_data["form"]
    query = f"{form_name} {position}"
    results, sim_stages = _similarity_search(client, kg, query, top_k)
    stage_metrics.extend(sim_stages)
    if not results:
        return None, False, stage_metrics

    status_log.info(
        f"Similarity search top-{top_k} for '{position}': {[r['number'] for r in results[:top_k]]}",
    )

    if page_mode == "greedy":
        for result in results[:top_k]:
            # result[number] is the page number, e.g. "G-101"
            number = result["number"]
            if find_book_dir(number) is not None:
                return number, (number == gt_page), stage_metrics
            status_log.warning(f"Page PNG not found in any book dir: {number}, skipping")
        return None, False, stage_metrics

    # Build check-items summary for content verification prompts
    items_summary = " | ".join(
        f"{item['check_item']}: {item['check_standard']}"
        for item in form_data["check_items"]
    )

    form_info = f"{form_name} {position}"

    for result in results[:top_k]:
        title = result["title"]
        number = result["number"]
        book_dir = find_book_dir(number)
        if book_dir is None:
            status_log.warning(f"Page PNG not found in any book dir: {number}, skipping")
            continue
        page_png = book_dir / number / f"{number}.png"
        detailed_dir = book_dir / number / "Detailed Drawings"

        if do_seg:
            yolo_stage = _ensure_detailed_drawings(yolo_model, page_png, detailed_dir)
            if yolo_stage is not None:
                stage_metrics.append(yolo_stage)

        if page_mode == "vanilla":
            if do_seg:
                # DDC: LLM sees all segmented sub-drawings, single-stage decision
                content_prompt = create_prompt_for_page_selection_with_detail_drawings(
                    title,
                    form_info,
                    items_summary,
                    str(detailed_dir),
                )
            else:
                # Vanilla: LLM sees the whole page, single-stage decision
                content_prompt = create_prompt_for_page_selection_with_whole_page(
                    title,
                    form_info,
                    items_summary,
                    str(page_png),
                )
            t0 = time.perf_counter()
            response, prompt_tok, compl_tok = getResponse(client, content_prompt, model_name)
            stage_metrics.append(
                StageMetrics(
                    stage="vanilla_content_check",
                    elapsed_s=time.perf_counter() - t0,
                    prompt_tokens=prompt_tok,
                    completion_tokens=compl_tok,
                    model=model_name,
                )
            )
            jr = json.loads(response)
            decision = jr["Decision"]
            # A true decision returns this page; otherwise the next candidate is checked.
            if isinstance(decision, str):
                decision = decision.lower() == "true"
            form_log.info(f"Vanilla(do_seg={do_seg}) content check {number}:\n{response}")
            status_log.info(f"Vanilla(do_seg={do_seg}) content check {number}: {decision}")
            if decision:
                return number, (number == gt_page), stage_metrics

        # "tspr" = Two-Stage Page Retrieval
        elif page_mode == "tspr":
            # Stage 1: position check (always uses segmented drawings)
            pos_prompt = create_prompt_for_page_retrieval_position(
                title,
                form_name,
                position,
                str(detailed_dir),
            )
            t0 = time.perf_counter()
            response, prompt_tok, compl_tok = getResponse(client, pos_prompt, model_name)
            stage_metrics.append(
                StageMetrics(
                    stage="tspr_position_check",
                    elapsed_s=time.perf_counter() - t0,
                    prompt_tokens=prompt_tok,
                    completion_tokens=compl_tok,
                    model=model_name,
                )
            )
            jr = json.loads(response)
            decision = jr["Decision"]
            if isinstance(decision, str):
                decision = decision.lower() == "true"
            form_log.info(f"TSPR position check {number}:\n{response}")
            status_log.info(f"TSPR position check {number}: {decision}")
            if not decision:
                continue

            # Stage 2: content check (uses segmented drawings)
            content_prompt = create_prompt_for_page_retrieval_detail(
                title,
                form_name,
                position,
                items_summary,
                str(detailed_dir),
            )
            t0 = time.perf_counter()
            response, prompt_tok, compl_tok = getResponse(client, content_prompt, model_name)
            stage_metrics.append(
                StageMetrics(
                    stage="tspr_content_check",
                    elapsed_s=time.perf_counter() - t0,
                    prompt_tokens=prompt_tok,
                    completion_tokens=compl_tok,
                    model=model_name,
                )
            )
            jr = json.loads(response)
            decision = jr["Decision"]
            if isinstance(decision, str):
                decision = decision.lower() == "true"
            form_log.info(f"TSPR content check {number}:\n{response}")
            status_log.info(f"TSPR content check {number}: {decision}")
            if decision:
                return number, (number == gt_page), stage_metrics

    return None, False, stage_metrics


# ── Drawing retrieval ──────────────────────────────────────────────────────────


def rerank_by_llm(
    client: OpenAI,
    model_name: str,
    check_item_standard: str,
    candidates: list[str],
) -> tuple[int, str, str | None, list[StageMetrics]]:
    """Score each candidate image. Returns (best_score, best_path, reasoning, stage_metrics)."""
    stage_metrics: list[StageMetrics] = []
    scores, reasonings = [], []
    for image_path in tqdm(candidates, leave=False, desc="rerank"):
        with Path(image_path).open("rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        prompt = create_prompt_for_rerank(check_item_standard, b64)
        t0 = time.perf_counter()
        response, prompt_tok, compl_tok = getResponse(client, prompt, model_name)
        stage_metrics.append(
            StageMetrics(
                stage="ddr_rerank",
                elapsed_s=time.perf_counter() - t0,
                prompt_tokens=prompt_tok,
                completion_tokens=compl_tok,
                model=model_name,
            )
        )
        jr = json.loads(response)
        form_log.info(f"Scoring {image_path}:\n{response}")
        status_log.info(f"Rerank {image_path}: score={jr['FinalScore']}")
        if jr["FinalScore"] == 10:  # noqa: PLR2004
            return 10, image_path, jr["ScoringReasoning"], stage_metrics
        scores.append(jr["FinalScore"])
        reasonings.append(jr["ScoringReasoning"])

    combined = sorted(
        zip(scores, candidates, reasonings, strict=False),
        key=lambda x: x[0],
        reverse=True,
    )
    return combined[0][0], combined[0][1], combined[0][2], stage_metrics


# ── Compliance checking ────────────────────────────────────────────────────────


def run_compliance_with_reflection(
    client: OpenAI,
    model_name: str,
    check_item_standard: str,
    image_candidates: list[str],
    max_reflections: int = 3,
    log_prefix: str = "",
) -> tuple[str | None, str | None, str | None, list[StageMetrics], list[dict[str, Any]]]:
    """
    Run compliance checker + optional evaluator loop (agentic reflection).

    max_reflections caps the number of evaluator→checker rounds AFTER the
    initial compliance check:
      - 0: single checker call, no evaluator (ablation: reflection off)
      - N>0: up to N evaluator rounds; stop early when evaluator approves

    Returns (compliance, advised_standard, reasoning, stage_metrics, reflection_iterations).
    """
    stage_metrics: list[StageMetrics] = []
    reflection_iterations: list[dict[str, Any]] = []
    reasoning = None
    compliance = None
    advised_standard = None
    feedback = None
    decision = False
    count = 0
    prefix = f"{log_prefix} " if log_prefix else ""

    while True:
        cc_prompt = create_prompt_for_compliance_check(
            check_item_standard,
            image_candidates,
            reasoning,
            compliance,
            advised_standard,
            feedback,
        )
        success = False
        while not success:
            t0 = time.perf_counter()
            response, prompt_tok, compl_tok = getResponse(client, cc_prompt, model_name)
            stage_metrics.append(
                StageMetrics(
                    stage="compliance_check",
                    elapsed_s=time.perf_counter() - t0,
                    prompt_tokens=prompt_tok,
                    completion_tokens=compl_tok,
                    model=model_name,
                )
            )
            try:
                jr = json.loads(response)
                reasoning = jr["Reasoning"]
                compliance = jr["ComplianceWithRegulations"]
                advised_standard = jr["AdvisedStandard"]
                reflection_iterations.append(
                    {
                        "iteration": count,
                        "has_prior_feedback": feedback is not None,
                        "compliance_check": jr,
                        "compliance_check_raw": response,
                    }
                )
                form_log.info(f"{prefix}[iter {count}] Compliance Check Result:\n{response}")
                success = True
            except Exception as e:
                status_log.warning(f"Compliance checker parse error: {e}\nRaw response: {response}")

        if max_reflections <= 0:
            break

        eval_prompt = create_prompt_for_compliance_evaluator(
            check_item_standard,
            image_candidates,
            reasoning,
            compliance,
            advised_standard,
        )
        success = False
        while not success:
            t0 = time.perf_counter()
            eval_response, prompt_tok, compl_tok = getResponse(client, eval_prompt, model_name)
            stage_metrics.append(
                StageMetrics(
                    stage="evaluator",
                    elapsed_s=time.perf_counter() - t0,
                    prompt_tokens=prompt_tok,
                    completion_tokens=compl_tok,
                    model=model_name,
                )
            )
            try:
                jr = json.loads(eval_response)
                decision = jr["Decision"]
                feedback = jr["Feedback"]
                reflection_iterations[-1]["evaluator"] = jr
                reflection_iterations[-1]["evaluator_raw"] = eval_response
                form_log.info(f"{prefix}[iter {count}] Evaluation Result:\n{eval_response}")
                success = True
            except Exception as e:
                status_log.warning(f"Evaluator parse error: {e}")

        count += 1
        if isinstance(decision, str):
            decision = decision.lower() == "true"
        if decision or count >= max_reflections:
            break

    return compliance, advised_standard, reasoning, stage_metrics, reflection_iterations


def _write_reflection_iterations_file(
    reflection_dir: Path,
    trace_doc: dict[str, Any],
) -> None:
    """Persist per-form reflection iterations as a single JSON file."""
    out_path = reflection_dir / f"{trace_doc['form_filename']}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(trace_doc, f, indent=2, ensure_ascii=False)


def evaluate_correctness(
    client: OpenAI,
    model_name: str,
    ground_truth: str,
    advised_standard: str | None,
    check_item: str | None = None,
) -> tuple[bool, StageMetrics | None, str]:
    """Compare advised_standard vs ground_truth.

    Uses the deterministic rule in compliance_checker.grading when one exists for
    check_item and the answer parses; otherwise falls back to the LLM judge.

    Returns (is_correct, stage_metrics, method) where method is "rule" or "llm".
    """
    if not advised_standard:
        return False, None, "rule"
    if check_item is not None:
        ruled = rule_grade(check_item, ground_truth, advised_standard)
        if ruled is not None:
            return ruled, None, "rule"
    prompt = create_prompt_for_correctness_eval(ground_truth, advised_standard)
    result: bool = False
    success = False
    stage: StageMetrics | None = None
    while not success:
        t0 = time.perf_counter()
        response, prompt_tok, compl_tok = getResponse(client, prompt, model_name)
        stage = StageMetrics(
            stage="correctness_eval",
            elapsed_s=time.perf_counter() - t0,
            prompt_tokens=prompt_tok,
            completion_tokens=compl_tok,
            model=model_name,
        )
        try:
            jr = json.loads(response)
            raw = jr["decision"]
            result = raw.lower() == "true" if isinstance(raw, str) else bool(raw)
            success = True
        except Exception as e:
            status_log.warning(f"Correctness eval parse error: {e}")
    return result, stage, "llm"


# ── Main experiment loop ───────────────────────────────────────────────────────


def run_configuration(  # noqa: PLR0912 PLR0913 PLR0915
    config: dict,
    client: OpenAI,
    model_name: str,
    yolo_model,
    input_json_dir: str,
    output_dir: str,
    kg=None,
    top_k: int = 10,
) -> tuple[int, int, int, int]:
    """
    Run one configuration. Returns (page_correct, page_total, cc_correct, cc_total).

    page_correct/total: page retrieval accuracy.
    cc_correct/total: compliance checking accuracy.
    Also writes latency_cost_report.json and latency_cost_summary.txt to output_dir.
    """
    page_mode = config["page_mode"]
    do_seg = config["do_seg"]
    do_sel = config["do_sel"]
    do_review = config["do_review"]
    # Reflection_* configs set max_reflections; the others use do_reflection
    # (True → 3 rounds, False → 0).
    if "max_reflections" in config:
        max_reflections = config["max_reflections"]
    else:
        max_reflections = 3 if config.get("do_reflection", True) else 0
    config_name = config["name"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all forms
    input_json_path = Path(input_json_dir)
    form_files = sorted(p for p in input_json_path.iterdir() if p.suffix == ".json")
    forms = []
    for json_path in form_files:
        with json_path.open() as f:
            data = json.load(f)
        data["_filename"] = json_path.stem
        forms.append(data)
    forms_by_name = {form_data["_filename"]: form_data for form_data in forms}

    # Seed output files
    for form_data in forms:
        fname = form_data["_filename"]
        out = {k: v for k, v in form_data.items() if k != "_filename"}
        out_seed = output_path / f"{fname}.json"
        with out_seed.open("w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    num_items = len(forms[0]["check_items"])
    reflection_dir: Path | None = None
    reflection_docs: dict[str, dict[str, Any]] = {}
    if max_reflections >= 0:
        reflection_dir = output_path / "reflection_iterations"
        reflection_dir.mkdir(parents=True, exist_ok=True)
        reflection_docs = {
            form_data["_filename"]: {
                "form_filename": form_data["_filename"],
                "form": form_data["form"],
                "construction": form_data["construction"],
                "position": form_data["position"],
                "items": [None] * num_items,
            }
            for form_data in forms
        }

    # ── Phase A: retrieve page for each form ─────────────────────────────────
    page_per_form: dict[str, str | None] = {}
    page_stages_per_form: dict[str, list[StageMetrics]] = {}
    page_correct_count = 0
    page_total_count = 0

    def _retrieve_one(form_data: dict) -> tuple[str, str | None, bool, list[StageMetrics]]:
        fname = form_data["_filename"]
        page, correct, page_stages = retrieve_page(
            page_mode,
            do_seg,
            form_data,
            client,
            model_name,
            yolo_model,
            kg,
            top_k,
        )
        return fname, page, correct, page_stages

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_retrieve_one, fd): fd for fd in forms}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"{config_name} page retrieval",
        ):
            fd = futures[future]
            fname = fd["_filename"]
            try:
                fname, page, correct, page_stages = future.result()
            except Exception as e:
                status_log.error(f"[{config_name}] {fname}: page retrieval failed: {e}")
                page_per_form[fname] = None
                page_stages_per_form[fname] = []
                page_total_count += 1
                continue
            page_per_form[fname] = page
            page_stages_per_form[fname] = page_stages
            page_total_count += 1
            if correct:
                page_correct_count += 1
            status_log.info(f"[{config_name}] {fname}: page={page}, correct={correct}")

    # ── Phase B: per-check-item loop (needed for cross-form review) ───────────
    cc_correct_count = 0
    cc_total_count = 0

    # item_metrics_per_form[fname][item_idx] = ItemMetrics
    item_metrics_per_form: dict[str, dict[int, ItemMetrics]] = {
        form_data["_filename"]: {} for form_data in forms
    }
    # ddr_stages_per_form_item collected during DDR phase
    ddr_stages_per_form_item: dict[str, dict[int, list[StageMetrics]]] = {}

    for item_idx in tqdm(range(num_items), desc=f"{config_name} compliance"):
        # Collect image candidates (and optionally select top-1)
        selected_per_form: dict[str, list[str]] = {}
        # review_stages_per_form: DDReview stage metrics distributed per form
        review_stages_per_form: dict[str, list[StageMetrics]] = {}

        def _ddr_one(
            form_data: dict,
            _item_idx: int = item_idx,
        ) -> tuple[str, list[str] | None, list[StageMetrics], list[StageMetrics], str | None]:
            """Returns (fname, selected_images, yolo_stages, ddr_stages, ddr_reasoning)."""
            fname = form_data["_filename"]
            page = page_per_form[fname]
            if page is None:
                return fname, None, [], [], None

            book_dir = find_book_dir(page)
            if book_dir is None:
                status_log.warning(
                    f"Phase B: page PNG not found in any book dir: {page}, skipping {fname}",
                )
                return fname, None, [], [], None
            page_png = book_dir / page / f"{page}.png"
            detailed_dir = book_dir / page / "Detailed Drawings"

            yolo_stages: list[StageMetrics] = []
            if do_seg:
                yolo_stage = _ensure_detailed_drawings(yolo_model, page_png, detailed_dir)
                if yolo_stage is not None:
                    yolo_stages.append(yolo_stage)
                candidates = sorted(
                    (str(p) for p in detailed_dir.iterdir() if p.suffix == ".png"),
                    key=lambda p: (0 if Path(p).name.startswith("table") else 1, p),
                )
            else:
                candidates = [str(page_png)]

            if do_sel:
                item_info = form_data["check_items"][_item_idx]
                check_item_std = (
                    f"check item: {item_info['check_item']} and check standard: "
                    f"{item_info['check_standard']} extracted from form "
                    f"{form_data['form']} in construction {form_data['construction']} "
                    f"at position {form_data['position']}"
                )
                _, image_path, ddr_reasoning, ddr_stages = rerank_by_llm(
                    client,
                    model_name,
                    check_item_std,
                    candidates,
                )
                return fname, [image_path], yolo_stages, ddr_stages, ddr_reasoning
            return fname, candidates, yolo_stages, [], None

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(_ddr_one, fd) for fd in forms]
            for future in as_completed(futures):
                try:
                    fname, selected, yolo_stages, ddr_stages, ddr_reasoning = future.result()
                except Exception as e:
                    status_log.error(f"[{config_name}] item {item_idx} DDR failed: {e}")
                    continue
                if selected is None:
                    continue
                for ys in yolo_stages:
                    # YOLO ran for the first time for this form (page retrieval did not crop it).
                    # Append to page_stages so it's counted as retrieval overhead.
                    page_stages_per_form.setdefault(fname, []).append(ys)
                if do_sel:
                    status_log.info(
                        f"[{config_name}] {fname} item {item_idx} DDR reasoning: {ddr_reasoning}",
                    )
                    ddr_stages_per_form_item.setdefault(fname, {})[item_idx] = ddr_stages
                selected_per_form[fname] = selected

        # Review: cluster across forms and adjust minority
        if do_review and do_sel and selected_per_form:
            n_participating = len(selected_per_form)

            t0_resnet = time.perf_counter()
            labels, majority_label = group_images(selected_per_form)
            majority = [f for f, lbl in labels.items() if lbl == majority_label]
            minority = [f for f, lbl in labels.items() if lbl != majority_label]
            reference_paths = [selected_per_form[f][0] for f in majority]
            resnet_elapsed = time.perf_counter() - t0_resnet
            status_log.info(
                f"[{config_name}] item {item_idx}: majority={majority}, minority={minority}",
            )

            t0_rerank = time.perf_counter()
            for fname in minority:
                form_data = next(d for d in forms if d["_filename"] == fname)
                page = page_per_form[fname]
                if page is None:
                    continue
                book = find_book_dir(page)
                if book is None:
                    continue
                detailed_dir = book / page / "Detailed Drawings"
                all_cands = sorted(str(p) for p in detailed_dir.iterdir() if p.suffix == ".png")
                selected_per_form[fname] = detailed_drawing_adjuestment(
                    reference_paths,
                    all_cands,
                )
            rerank_elapsed = time.perf_counter() - t0_rerank

            # Amortize review cost across all participating forms
            per_form_resnet = resnet_elapsed / n_participating
            per_form_rerank = rerank_elapsed / n_participating
            for fname in selected_per_form:
                review_stages_per_form[fname] = [
                    StageMetrics(stage=STAGE_RESNET_KMEANS, elapsed_s=per_form_resnet),
                    StageMetrics(stage=STAGE_DDREVIEW_RERANK, elapsed_s=per_form_rerank),
                ]

        # Compliance check and evaluate (parallel per form)
        def _compliance_one(
            form_data: dict,
            _item_idx: int = item_idx,
            _selected: dict = selected_per_form,
        ) -> tuple[
            str,
            str | None,
            str | None,
            str | None,
            list[StageMetrics],
            list[dict[str, Any]],
            StageMetrics | None,
            bool,
            str | None,
            str,
            list[str],
        ]:
            """Returns (fname, compliance, advised_std, reasoning,
                        cc_stages, reflection_iters, correctness_stage,
                        is_correct, grading_method, gt, images)."""
            fname = form_data["_filename"]
            item_info = form_data["check_items"][_item_idx]
            check_item_std = (
                f"check item: {item_info['check_item']} and check standard: "
                f"{item_info['check_standard']} extracted from form "
                f"{form_data['form']} in construction {form_data['construction']} "
                f"at position {form_data['position']}"
            )
            gt = item_info.get("check_standard_gt", "")
            images = _selected[fname]

            compliance, advised_std, reasoning, cc_stages, reflection_iters = run_compliance_with_reflection(
                client,
                model_name,
                check_item_std,
                images,
                max_reflections,
                log_prefix=f"[{config_name}][{fname}][item {_item_idx}]",
            )
            correctness_stage: StageMetrics | None = None
            is_correct = False
            grading_method: str | None = None
            if gt:
                is_correct, correctness_stage, grading_method = evaluate_correctness(
                    client, model_name, gt, advised_std, check_item=item_info["check_item"]
                )
            return (
                fname,
                compliance,
                advised_std,
                reasoning,
                cc_stages,
                reflection_iters,
                correctness_stage,
                is_correct,
                grading_method,
                gt,
                images,
            )

        forms_to_run = [fd for fd in forms if fd["_filename"] in selected_per_form]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(_compliance_one, fd) for fd in forms_to_run]
            for future in as_completed(futures):
                try:
                    (
                        fname,
                        compliance,
                        advised_std,
                        reasoning,
                        cc_stages,
                        reflection_iters,
                        correctness_stage,
                        is_correct,
                        grading_method,
                        gt,
                        images,
                    ) = future.result()
                except Exception as e:
                    status_log.error(f"[{config_name}] item {item_idx} compliance failed: {e}")
                    continue

                status_log.info(
                    f"[{config_name}] {fname} item {item_idx}: "
                    f"compliance={compliance}, advised={advised_std}",
                )

                # Assemble per-item metrics
                item_stage_list: list[StageMetrics] = []
                item_stage_list += ddr_stages_per_form_item.get(fname, {}).get(item_idx, [])
                item_stage_list += review_stages_per_form.get(fname, [])
                item_stage_list += cc_stages
                if correctness_stage is not None:
                    item_stage_list.append(correctness_stage)

                cc_total_count += 1
                if gt:
                    if is_correct:
                        cc_correct_count += 1
                    status_log.info(
                        f"[{config_name}] {fname} item {item_idx}: "
                        f"correct={is_correct} ({grading_method}), gt={gt}",
                    )

                item_metrics_per_form[fname][item_idx] = ItemMetrics(
                    form_filename=fname,
                    item_idx=item_idx,
                    stages=item_stage_list,
                )

                # Persist result (serial, main thread — each form has its own file)
                out_path = output_path / f"{fname}.json"
                with out_path.open() as f:
                    out_form = json.load(f)
                page = page_per_form[fname]
                origin = images[0] if do_sel else page
                out_form["check_items"][item_idx]["origin"] = origin
                out_form["check_items"][item_idx]["compliance_with_regulations"] = compliance
                out_form["check_items"][item_idx]["advised_standard"] = advised_std
                out_form["check_items"][item_idx]["reasoning"] = reasoning
                with out_path.open("w") as f:
                    json.dump(out_form, f, indent=2, ensure_ascii=False)

                if reflection_dir is not None:
                    source_form = forms_by_name[fname]
                    item_info = source_form["check_items"][item_idx]
                    reflection_docs[fname]["items"][item_idx] = {
                        "item_idx": item_idx,
                        "check_item": item_info["check_item"],
                        "check_standard": item_info["check_standard"],
                        "check_standard_gt": item_info.get("check_standard_gt", ""),
                        "selected_images": images,
                        "final_result": {
                            "compliance_with_regulations": compliance,
                            "advised_standard": advised_std,
                            "reasoning": reasoning,
                            "ground_truth_match": is_correct if gt else None,
                            "grading_method": grading_method,
                        },
                        "iterations": reflection_iters,
                    }
                    _write_reflection_iterations_file(reflection_dir, reflection_docs[fname])

        # Items whose page/detail drawing could not be located count as wrong
        # in the compliance accuracy.
        for form_data in forms:
            fname = form_data["_filename"]
            if fname in selected_per_form:
                continue
            cc_total_count += 1
            page = page_per_form.get(fname)
            reason = "page retrieval failed" if page is None else f"page PNG missing ({page})"
            status_log.info(
                f"[{config_name}] {fname} item {item_idx}: "
                f"no drawing found ({reason}) — counted as wrong in denominator",
            )

    # ── Write latency / cost report ───────────────────────────────────────────
    form_metrics_list: list[FormMetrics] = []
    for form_data in forms:
        fname = form_data["_filename"]
        fm = FormMetrics(
            form_filename=fname,
            page_stages=page_stages_per_form.get(fname, []),
            items=list(item_metrics_per_form[fname].values()),
        )
        form_metrics_list.append(fm)

    _write_latency_cost_report(output_path, config_name, model_name, form_metrics_list)

    return page_correct_count, page_total_count, cc_correct_count, cc_total_count


def _write_latency_cost_report(
    output_path: Path,
    config_name: str,
    model_name: str,
    form_metrics_list: list[FormMetrics],
) -> None:
    """Write latency_cost_report.json and latency_cost_summary.txt to output_path."""
    # Machine-readable JSON
    report = {
        "config": config_name,
        "model": model_name,
        "forms": [
            {
                "form_filename": fm.form_filename,
                "page_retrieval": [
                    {
                        "stage": s.stage,
                        "elapsed_s": s.elapsed_s,
                        "prompt_tokens": s.prompt_tokens,
                        "completion_tokens": s.completion_tokens,
                        "model": s.model,
                    }
                    for s in fm.page_stages
                ],
                "total_elapsed_s": fm.total_elapsed_s,
                "total_prompt_tokens": fm.total_prompt_tokens,
                "total_completion_tokens": fm.total_completion_tokens,
                "estimated_cost_usd": fm.estimated_cost_usd,
                "items": [
                    {
                        "item_idx": item.item_idx,
                        "total_elapsed_s": item.total_elapsed_s,
                        "total_prompt_tokens": item.total_prompt_tokens,
                        "total_completion_tokens": item.total_completion_tokens,
                        "estimated_cost_usd": item.estimated_cost_usd,
                        "stages": [
                            {
                                "stage": s.stage,
                                "elapsed_s": s.elapsed_s,
                                "prompt_tokens": s.prompt_tokens,
                                "completion_tokens": s.completion_tokens,
                                "model": s.model,
                            }
                            for s in item.stages
                        ],
                    }
                    for item in fm.items
                ],
            }
            for fm in form_metrics_list
        ],
    }
    report_path = output_path / "latency_cost_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    status_log.info(f"Latency/cost report written to {report_path}")

    # Human-readable summary
    summary = format_summary(config_name, model_name, form_metrics_list)
    summary_path = output_path / "latency_cost_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    status_log.info(f"Latency/cost summary written to {summary_path}")
    print(summary)


# ── Entry point ────────────────────────────────────────────────────────────────


def _short_model_name(model_name: str) -> str:
    """Strip version dates from model name: 'gpt-4o-2024-08-06' → 'gpt4o'."""
    parts = []
    for part in model_name.split("-"):
        if part.isdigit() and len(part) == 4:  # hit a year token, stop
            break
        parts.append(part)
    return "".join(parts)


def main() -> None:
    parser = ArgumentParser(description="Run ablation experiment configurations.")
    parser.add_argument(
        "--dataset",
        default="wall",
        choices=["wall", "pile", "ditch"],
        help="'wall' = Retaining Wall, 'pile' = Full Casing Pile, 'ditch' = Open Ditch",
    )
    parser.add_argument(
        "--set",
        default="full",
        help="Which configs to run: 'full' | 'reflection' | index",
    )
    parser.add_argument(
        "--model_name",
        default="gpt-4o-2024-08-06",
        help="OpenAI model: 'gpt-4o-2024-08-06' or 'o4-mini-2025-04-16'",
    )
    parser.add_argument("--openai_api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument(
        "--neo4j_uri",
        default=os.getenv("NEO4J_URI", ""),
    )
    parser.add_argument(
        "--neo4j_username",
        default=os.getenv("NEO4J_USERNAME", "neo4j"),
    )
    parser.add_argument("--neo4j_password", default=os.getenv("NEO4J_PASSWORD", ""))
    parser.add_argument(
        "--neo4j_database",
        default=os.getenv("NEO4J_DATABASE", "neo4j"),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of pages retrieved by similarity search.",
    )
    parser.add_argument(
        "--input_json_dir",
        default=None,
        help="Read the forms from this folder instead of the dataset's default folder.",
    )
    parser.add_argument(
        "--run_root",
        default=None,
        help="Output directory for this run (default: output/<dataset>_<model>_<timestamp>).",
    )
    args = parser.parse_args()

    if args.set == "full":
        configs_to_run = [c for c in CONFIGURATIONS if not c["name"].startswith("Reflection_")]
    elif args.set == "reflection":
        configs_to_run = [c for c in CONFIGURATIONS if c["name"].startswith("Reflection_")]
    else:
        configs_to_run = [CONFIGURATIONS[int(args.set)]]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_model = _short_model_name(args.model_name)
    run_root = (
        Path(args.run_root)
        if args.run_root
        else Path("output") / f"{args.dataset}_{short_model}_{timestamp}"
    )
    log_dir = run_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Wire up loguru sinks — remove default stderr sink first
    logger.remove()
    logger.add(
        log_dir / "status.log",
        filter=lambda record: record["extra"].get("log_type") == "status",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG",
        encoding="utf-8",
    )
    logger.add(
        log_dir / "form.log",
        filter=lambda record: record["extra"].get("log_type") == "form",
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        level="DEBUG",
        encoding="utf-8",
    )
    logger.add(
        sys.stderr,
        filter=lambda record: record["extra"].get("log_type") == "status",
        level="INFO",
    )

    client = OpenAI(api_key=args.openai_api_key)

    # Initialize YOLO model only if any config needs segmentation
    yolo_model = None
    if any(c["do_seg"] for c in configs_to_run):
        yolo_model = getModel()

    # Initialize Neo4j only if any config needs it
    kg = None
    if any(c["page_mode"] in NEEDS_NEO4J for c in configs_to_run):
        if not args.neo4j_password:
            raise ValueError(_NEO4J_PASSWORD_REQUIRED_MSG)
        kg = Neo4jGraph(
            url=args.neo4j_uri,
            username=args.neo4j_username,
            password=args.neo4j_password,
            database=args.neo4j_database,
        )

    results = []
    for config in configs_to_run:
        output_dir = str(run_root / config["name"])
        page_correct, page_total, cc_correct, cc_total = run_configuration(
            config=config,
            client=client,
            model_name=args.model_name,
            yolo_model=yolo_model,
            input_json_dir=args.input_json_dir or DATASET_DIRS[args.dataset],
            output_dir=output_dir,
            kg=kg,
            top_k=args.top_k,
        )
        page_acc = page_correct / page_total * 100 if page_total > 0 else None
        cc_acc = cc_correct / cc_total * 100 if cc_total > 0 else 0.0
        results.append((config["name"], page_acc, cc_correct, cc_total, cc_acc))
        status_log.info(
            f"RESULT {config['name']}: "
            f"page={f'{page_acc:.1f}%' if page_acc is not None else 'N/A'}, "
            f"compliance={cc_correct}/{cc_total}={cc_acc:.1f}%",
        )
        print(
            f"  {config['name']}: "
            f"page={'N/A' if page_acc is None else f'{page_acc:.1f}%'}, "
            f"compliance={cc_correct}/{cc_total}={cc_acc:.1f}%",
        )

    # Summary table
    summary_lines = [
        "\n" + "=" * 72,
        f"Ablation Results — dataset={args.dataset}, model={args.model_name}",
        "=" * 72,
        f"{'Configuration':<26} {'Page Retr.':>11} {'Correct':>8} {'Total':>7} {'Compliance':>11}",
        "-" * 72,
    ]
    for name, page_acc, cc_correct, cc_total, cc_acc in results:
        page_str = "N/A" if page_acc is None else f"{page_acc:.1f}%"
        summary_lines.append(
            f"{name:<26} {page_str:>11} {cc_correct:>8} {cc_total:>7} {cc_acc:>10.1f}%",
        )
    summary_lines.append("=" * 72)
    summary = "\n".join(summary_lines)
    print(summary)
    (run_root / "summary.txt").write_text(summary + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
