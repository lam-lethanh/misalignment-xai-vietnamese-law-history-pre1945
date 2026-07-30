import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.research_analysis.ahp_misalignment.lhms_metric import LHMSMetricEngine


def _token_length(text: str) -> int:
    return len((text or "").split())


def _normalize_for_nli(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _compute_full_reference_contradiction(engine: LHMSMetricEngine, reference: str, response: str) -> float:
    reference_sentences = engine._split_sentences(reference)
    if not reference_sentences:
        return 0.0
    if not response:
        return 0.0

    response_windows = engine._build_sliding_windows(response, window_size=2)
    if not response_windows:
        return 0.0

    contradiction_scores: List[float] = []
    for window in response_windows:
        window_sentences = engine._split_sentences(window)
        if not window_sentences:
            continue
        hypothesis = " ".join(window_sentences)
        premise = _normalize_for_nli(" ".join(reference_sentences))
        contradiction_scores.append(engine._predict_contradiction(premise, hypothesis))

    return float(np.max(contradiction_scores)) if contradiction_scores else 0.0


def _compute_retrieval_contradiction(
    engine: LHMSMetricEngine,
    reference: str,
    response: str,
    k: int = 3,
    similarity_threshold: float = 0.4,
    use_hybrid_search: bool = True,
    use_reranker: bool = True,
) -> float:
    reference_sentences = engine._split_sentences(reference)
    if not reference_sentences:
        return 0.0
    if not response:
        return 0.0

    response_windows = engine._build_sliding_windows(response, window_size=2)
    if not response_windows:
        return 0.0

    contradiction_scores: List[float] = []
    for window in response_windows:
        window_sentences = engine._split_sentences(window)
        if not window_sentences:
            continue
        hypothesis = " ".join(window_sentences)
        top_reference = engine._select_relevant_reference_sentences(
            hypothesis,
            reference_sentences,
            k=k,
            similarity_threshold=similarity_threshold,
            use_hybrid_search=use_hybrid_search,
            use_reranker=use_reranker,
        )
        if not top_reference:
            contradiction_scores.append(0.0)
            continue

        premise = _normalize_for_nli(" ".join(top_reference)) if top_reference else _normalize_for_nli(" ".join(reference_sentences[:k]))
        contradiction_scores.append(engine._predict_contradiction(premise, hypothesis))

    return float(np.max(contradiction_scores)) if contradiction_scores else 0.0


def analyze_best_k_lhms(
    reference: str,
    response: str,
    ks: Optional[Sequence[int]] = None,
    engine: Optional[LHMSMetricEngine] = None,
    similarity_threshold: float = 0.4,
    use_hybrid_search: bool = True,
    use_reranker: bool = True,
) -> Dict[str, object]:
    engine = engine or LHMSMetricEngine(local_files_only=True)
    ks = list(ks or [2, 3, 4, 5])
    rows: List[Dict[str, float]] = []

    reference_token_count = _token_length(reference)
    full_reference_score = None
    eligible_for_full_reference = reference_token_count <= 512
    if eligible_for_full_reference:
        full_reference_score = _compute_full_reference_contradiction(engine, reference, response)

    for k in ks:
        retrieval_score = _compute_retrieval_contradiction(
            engine,
            reference,
            response,
            k=k,
            similarity_threshold=similarity_threshold,
            use_hybrid_search=use_hybrid_search,
            use_reranker=use_reranker,
        )
        rows.append({
            "k": int(k),
            "retrieval_contradiction": float(retrieval_score),
        })

    best_row = None
    if full_reference_score is not None and rows:
        best_row = min(
            rows,
            key=lambda item: abs(float(full_reference_score) - float(item["retrieval_contradiction"])),
        )
    best_k = int(best_row["k"]) if best_row is not None else None

    similarity = None
    if full_reference_score is not None and best_row is not None:
        delta = abs(float(full_reference_score) - float(best_row["retrieval_contradiction"]))
        similarity = max(0.0, 1.0 - delta)

    return {
        "reference_token_count": reference_token_count,
        "eligible_for_full_reference": eligible_for_full_reference,
        "full_reference_contradiction": full_reference_score,
        "rows": rows,
        "best_k": best_k,
        "similarity_to_full_reference": similarity,
    }


def evaluate_results_file(results_file: Path, ks: Optional[Sequence[int]] = None, engine: Optional[LHMSMetricEngine] = None) -> List[Dict[str, object]]:
    engine = engine or LHMSMetricEngine(local_files_only=True)
    with results_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    rows: List[Dict[str, object]] = []
    total_samples = len(data.get("results", []))
    print(f"    Processing {results_file.name}: {total_samples} samples")
    for idx, item in enumerate(data.get("results", [])):
        reference = item.get("expected_answer") or ""
        response = item.get("model_response") or ""
        rows.append(analyze_best_k_lhms(reference, response, ks=ks, engine=engine))
        if (idx + 1) % 50 == 0 or idx + 1 == total_samples:
            print(f"      {idx + 1}/{total_samples} samples processed")
    return rows


def _build_report_lines(per_file_rows: List[Dict[str, object]]) -> List[str]:
    report_lines = [
        "# Best-K LHMS Analysis",
        "",
        "## Goal",
        "Compare contradiction scores from the full reference against retrieval-based premise selection with different k values on references shorter than or equal to 512 tokens.",
        "",
        "## Summary by file",
        "",
        "| file | samples | eligible_short_refs | avg_reference_tokens | avg_full_reference_contradiction | avg_similarity_to_full_reference | preferred_k |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for file_row in per_file_rows:
        samples = file_row["samples"]
        reference_tokens = [item["reference_token_count"] for item in samples if "reference_token_count" in item]
        eligible_samples = [item for item in samples if item.get("eligible_for_full_reference")]
        full_scores = [item["full_reference_contradiction"] for item in eligible_samples if item.get("full_reference_contradiction") is not None]
        similarities = [item["similarity_to_full_reference"] for item in eligible_samples if item.get("similarity_to_full_reference") is not None]
        preferred_k = None
        if eligible_samples:
            preferred_candidates = [item for item in eligible_samples if item.get("best_k") is not None]
            if preferred_candidates:
                preferred_k = Counter([item["best_k"] for item in preferred_candidates]).most_common(1)[0][0]
        report_lines.append(
            f"| {file_row['file']} | {len(samples)} | {len(eligible_samples)} | {float(np.mean(reference_tokens)):.1f} | {float(np.mean(full_scores)):.4f} | {float(np.mean(similarities)):.4f} | {preferred_k if preferred_k is not None else '-'} |"
        )

    report_lines.extend([
        "",
        "## Mean retrieval contradiction and similarity by k",
        "",
        "| file | k | eligible_samples | mean_retrieval_contradadiction | mean_similarity_to_full_reference | mean_abs_delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])

    for file_row in per_file_rows:
        samples = [item for item in file_row["samples"] if item.get("eligible_for_full_reference")]
        for k in sorted({child["k"] for sample in samples for child in sample.get("rows", [])}):
            values = [child["retrieval_contradiction"] for sample in samples for child in sample.get("rows", []) if child["k"] == k]
            similarities = [max(0.0, 1.0 - abs(float(sample["full_reference_contradiction"]) - float(child["retrieval_contradiction"]))) for sample in samples for child in sample.get("rows", []) if child["k"] == k]
            abs_deltas = [abs(float(sample["full_reference_contradiction"]) - float(child["retrieval_contradiction"])) for sample in samples for child in sample.get("rows", []) if child["k"] == k]
            report_lines.append(
                f"| {file_row['file']} | {k} | {len(values)} | {float(np.mean(values)):.4f} | {float(np.mean(similarities)):.4f} | {float(np.mean(abs_deltas)):.4f} |"
            )

    return report_lines


def build_best_k_lhms_report(
    results_dir: Path,
    output_path: Path,
    ks: Optional[Sequence[int]] = None,
    batch_size: Optional[int] = None,
    batch_index: Optional[int] = None,
    file_names: Optional[Sequence[str]] = None,
    partial_dir: Optional[Path] = None,
) -> Path:
    candidate_paths = sorted(results_dir.glob("*.json"))
    if file_names:
        selected_names = set(file_names)
        candidate_paths = [path for path in candidate_paths if path.name in selected_names]

    if batch_size is not None and batch_index is not None:
        start = batch_index * batch_size
        end = start + batch_size
        candidate_paths = candidate_paths[start:end]

    print(f"Processing {len(candidate_paths)} file(s) from {results_dir}")
    per_file_rows = []
    for path in candidate_paths:
        print(f"  -> {path.name}")
        file_samples = evaluate_results_file(path, ks=ks)
        per_file_rows.append({
            "file": path.name,
            "samples": file_samples,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(_build_report_lines(per_file_rows)), encoding="utf-8")

    if partial_dir is not None:
        partial_dir.mkdir(parents=True, exist_ok=True)
        batch_tag = f"batch_{batch_index if batch_index is not None else 0}"
        partial_path = partial_dir / f"{batch_tag}.json"
        partial_path.write_text(json.dumps(per_file_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved partial summary to {partial_path}")

    return output_path


def merge_partial_reports(partial_dir: Path, output_path: Path) -> Path:
    partial_paths = sorted(partial_dir.glob("batch_*.json"))
    if not partial_paths:
        raise FileNotFoundError(f"No partial reports found in {partial_dir}")

    merged_rows: List[Dict[str, object]] = []
    for partial_path in partial_paths:
        with partial_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        merged_rows.extend(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(_build_report_lines(merged_rows)), encoding="utf-8")
    return output_path


def main(
    results_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    ks: Optional[Sequence[int]] = None,
    batch_size: Optional[int] = None,
    batch_index: Optional[int] = None,
    file_names: Optional[Sequence[str]] = None,
    partial_dir: Optional[Path] = None,
    merge_partials: bool = False,
) -> Path:
    base_dir = Path(results_dir or Path(__file__).resolve().parents[3] / "experiments" / "results")
    output_file = output_path or Path(__file__).resolve().parent / "output" / "best_k_lhms_report.md"
    if merge_partials:
        return merge_partial_reports(partial_dir or output_file.parent / "partials", output_file)
    return build_best_k_lhms_report(
        base_dir,
        output_file,
        ks=ks,
        batch_size=batch_size,
        batch_index=batch_index,
        file_names=file_names,
        partial_dir=partial_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a best-k LHMS report from result files")
    parser.add_argument("--results-dir", type=Path, default=None, help="Directory containing result JSON files")
    parser.add_argument("--output-path", type=Path, default=None, help="Where to write the markdown report")
    parser.add_argument("--ks", type=int, nargs="+", default=None, help="List of k values to evaluate")
    parser.add_argument("--batch-size", type=int, default=None, help="Process result files in batches of this size")
    parser.add_argument("--batch-index", type=int, default=None, help="Which batch number to process (0-based)")
    parser.add_argument("--files", nargs="+", default=None, help="Specific result filenames to process")
    parser.add_argument("--partial-dir", type=Path, default=None, help="Directory to save partial batch summaries")
    parser.add_argument("--merge-partials", action="store_true", help="Merge previously saved partial batch reports")
    args = parser.parse_args()
    main(
        results_dir=args.results_dir,
        output_path=args.output_path,
        ks=args.ks,
        batch_size=args.batch_size,
        batch_index=args.batch_index,
        file_names=args.files,
        partial_dir=args.partial_dir,
        merge_partials=args.merge_partials,
    )
