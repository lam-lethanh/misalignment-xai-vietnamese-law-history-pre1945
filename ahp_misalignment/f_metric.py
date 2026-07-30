import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


class FMetricEngine:
    """Compute a factuality score using embedding similarity and entity overlap."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name
        self._model = None
        if SentenceTransformer is not None:
            try:
                self._model = SentenceTransformer(model_name or "bkai-foundation-models/vietnamese-bi-encoder")
            except Exception:
                self._model = None

    def _normalize(self, text: str) -> str:
        if text is None:
            return ""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _extract_entities(self, text: str) -> set:
        if not text:
            return set()
        text = text.strip()
        candidates = set(re.findall(r"[A-ZĐÂĂÊÔƠƯÀẢÃÁẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ][a-zđâăêôơưàảãáạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]+(?:\s+[A-ZĐÂĂÊÔƠƯÀẢÃÁẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ][a-zđâăêôơưàảãáạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]+)*", text))
        tokens = re.findall(r"\b\d{4}\b", text)
        tokens.extend(re.findall(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b", text))
        candidates.update(tokens)
        return {self._normalize(item) for item in candidates if item}

    def _token_overlap(self, expected: str, actual: str) -> float:
        expected_tokens = set(self._normalize(expected).split())
        actual_tokens = set(self._normalize(actual).split())
        if not expected_tokens or not actual_tokens:
            return 0.0
        return len(expected_tokens & actual_tokens) / len(expected_tokens | actual_tokens)

    def compute_entity_f1(self, expected: str, actual: str) -> float:
        expected_entities = self._extract_entities(expected)
        actual_entities = self._extract_entities(actual)
        if not expected_entities and not actual_entities:
            return 0.0
        if not expected_entities or not actual_entities:
            return 0.0
        intersection = expected_entities & actual_entities
        precision = len(intersection) / len(actual_entities) if actual_entities else 0.0
        recall = len(intersection) / len(expected_entities) if expected_entities else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def compute_embedding_similarity(self, expected: str, actual: str) -> float:
        if not expected or not actual:
            return 0.0
        if self._model is None:
            return self._token_overlap(expected, actual)
        emb_expected = self._model.encode(expected, convert_to_numpy=True)
        emb_actual = self._model.encode(actual, convert_to_numpy=True)
        cosine = float(np.dot(emb_expected, emb_actual) / (np.linalg.norm(emb_expected) * np.linalg.norm(emb_actual)))
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    def compute_f_score(self, expected: str, actual: str, embedding_weight: float = 0.6) -> float:
        embedding_similarity = self.compute_embedding_similarity(expected, actual)
        entity_f1 = self.compute_entity_f1(expected, actual)
        combined = (embedding_weight * embedding_similarity) + ((1.0 - embedding_weight) * entity_f1)
        return max(0.0, min(1.0, 1.0 - combined))

    def evaluate_file(self, results_file: Path) -> Dict[str, float]:
        with results_file.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        scores = []
        for item in data.get("results", []):
            expected = item.get("expected_answer") or ""
            actual = item.get("model_response") or ""
            scores.append(self.compute_f_score(expected, actual))

        return {
            "avg_f_score": float(np.mean(scores)) if scores else 0.0,
            "max_f_score": float(np.max(scores)) if scores else 0.0,
            "min_f_score": float(np.min(scores)) if scores else 0.0,
        }


def compute_entity_f1(expected: str, actual: str) -> float:
    return FMetricEngine().compute_entity_f1(expected, actual)


def compute_embedding_similarity(expected: str, actual: str) -> float:
    return FMetricEngine().compute_embedding_similarity(expected, actual)


def compute_f_score(expected: str, actual: str, embedding_weight: float = 0.6) -> float:
    return FMetricEngine().compute_f_score(expected, actual, embedding_weight=embedding_weight)


def run_f_weight_sensitivity(base_embedding_weight: float = 0.5, deltas: Optional[List[float]] = None) -> List[Dict[str, float]]:
    if deltas is None:
        deltas = [i * 0.04 for i in range(-15, 16)]

    scenarios = []
    for delta in deltas:
        weight = base_embedding_weight * (1.0 + delta)
        if weight <= 0.0 or weight >= 1.0:
            continue
        scenarios.append({"delta": delta, "embedding_weight": weight})
    return scenarios


def _get_model_name(path: Path) -> str:
    name = path.name.lower()
    if "chatgpt" in name:
        return "chatgpt-4o-mini-8b"
    if "gemma" in name:
        return "gemma2-9b"
    if "llama" in name:
        return "llama-2-7b"
    if "qwen" in name:
        return "qwen-2.5-7b"
    if "vietcuna" in name:
        return "vietcuna-7b"
    return path.stem


def compute_ci_cr(base_ranking: List[str], scenario_ranking: List[str]) -> Tuple[float, float]:
    if not base_ranking or not scenario_ranking:
        return 0.0, 0.0

    base_rank_map = {model: idx + 1 for idx, model in enumerate(base_ranking)}
    scenario_rank_map = {model: idx + 1 for idx, model in enumerate(scenario_ranking)}
    models = list(base_rank_map.keys())

    deviations = [abs(scenario_rank_map[model] - base_rank_map[model]) for model in models]
    consistency_index = float(np.mean(deviations)) if deviations else 0.0
    max_possible_deviation = max(1, len(models) - 1)
    consistency_ratio = consistency_index / max_possible_deviation if max_possible_deviation > 0 else 0.0
    return consistency_index, consistency_ratio


def summarize_f_by_weight(results_dir: Path, base_embedding_weight: float = 0.5, deltas: Optional[List[float]] = None) -> List[Dict[str, float]]:
    engine = FMetricEngine()
    sensitivity = run_f_weight_sensitivity(base_embedding_weight=base_embedding_weight, deltas=deltas)
    rows = []

    base_model_scores: Dict[str, List[float]] = {}
    for path in sorted(results_dir.glob("*.json")):
        model_name = _get_model_name(path)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        file_scores = []
        for item in data.get("results", []):
            expected = item.get("expected_answer") or ""
            actual = item.get("model_response") or ""
            score = engine.compute_f_score(expected, actual, embedding_weight=base_embedding_weight)
            file_scores.append(score)
        if file_scores:
            base_model_scores.setdefault(model_name, []).extend(file_scores)

    base_avg_by_model = {model: float(np.mean(scores)) for model, scores in base_model_scores.items()}
    base_ranking = [model for model, _ in sorted(base_avg_by_model.items(), key=lambda item: item[1])]

    for scenario in sensitivity:
        weight = float(scenario["embedding_weight"])
        per_model_scores: Dict[str, List[float]] = {}
        all_scores = []

        for path in sorted(results_dir.glob("*.json")):
            model_name = _get_model_name(path)
            file_scores = []
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            for item in data.get("results", []):
                expected = item.get("expected_answer") or ""
                actual = item.get("model_response") or ""
                score = engine.compute_f_score(expected, actual, embedding_weight=weight)
                file_scores.append(score)
                all_scores.append(score)

            if file_scores:
                per_model_scores.setdefault(model_name, []).extend(file_scores)

        if all_scores:
            avg_by_model = {model: float(np.mean(scores)) for model, scores in per_model_scores.items()}
            ranking = [model for model, _ in sorted(avg_by_model.items(), key=lambda item: item[1])]
            ranking_text = " > ".join(f"{model} ({score:.4f})" for model, score in sorted(avg_by_model.items(), key=lambda item: item[1]))
            ci, cr = compute_ci_cr(base_ranking, ranking)
            rows.append({
                "delta": scenario["delta"],
                "embedding_weight": weight,
                "avg_f_score": float(np.mean(all_scores)),
                "max_f_score": float(np.max(all_scores)),
                "min_f_score": float(np.min(all_scores)),
                "model_ranking": ranking_text,
                "ci": ci,
                "cr": cr,
            })

    return rows


def build_f_report(results_dir: Path, output_path: Path, base_embedding_weight: float = 0.5) -> Path:
    engine = FMetricEngine()
    summary_rows = []
    for path in sorted(results_dir.glob("*.json")):
        metrics = engine.evaluate_file(path)
        summary_rows.append({"file": path.name, **metrics})

    sensitivity = run_f_weight_sensitivity(base_embedding_weight=base_embedding_weight)
    weight_summaries = summarize_f_by_weight(results_dir, base_embedding_weight=base_embedding_weight)
    report_lines = [
        "# F Metric Report",
        "",
        "## Objective",
        "This report computes factual misalignment F using an embedding-based similarity component and an entity-overlap component.",
        "",
        "## Formula",
        "F = 1 - [embedding_weight * embedding_similarity + (1 - embedding_weight) * entity_f1]",
        "",
        "## Base configuration",
        f"- Base embedding weight: {base_embedding_weight:.2f}",
        "",
        "## Weight sensitivity",
        "",
        "| delta | embedding_weight |",
        "| ---: | ---: |",
    ]
    for row in sensitivity:
        report_lines.append(f"| {row['delta']:+.2f} | {row['embedding_weight']:.2f} |")

    report_lines.extend([
        "",
        "## F by embedding weight",
        "",
        "| delta | embedding_weight | avg_f_score | max_f_score | min_f_score |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in weight_summaries:
        report_lines.append(f"| {row['delta']:+.2f} | {row['embedding_weight']:.2f} | {row['avg_f_score']:.4f} | {row['max_f_score']:.4f} | {row['min_f_score']:.4f} |")

    report_lines.extend([
        "",
        "## Model ranking by embedding weight",
        "",
        "| delta | embedding_weight | model_ranking |",
        "| ---: | ---: | --- |",
    ])
    for row in weight_summaries:
        report_lines.append(f"| {row['delta']:+.2f} | {row['embedding_weight']:.2f} | {row['model_ranking']} |")

    report_lines.extend([
        "",
        "## Ranking stability analysis for the F weighting scheme",
        "",
        "Because F uses only two components, the classic AHP CI/CR calculation for a 2x2 matrix is not informative. To make the choice of embedding weight more interpretable, the table below reports a stability score based on how much the model ranking changes when the embedding weight is varied.",
        "",
        "| delta | embedding_weight | Ranking Stability Index | Stability Ratio |",
        "| ---: | ---: | ---: | ---: |",
    ])
    for row in weight_summaries:
        report_lines.append(f"| {row['delta']:+.2f} | {row['embedding_weight']:.2f} | {row['ci']:.4f} | {row['cr']:.4f} |")

    report_lines.extend([
        "",
        "## Summary by file",
        "",
        "| file | avg_f_score | max_f_score | min_f_score |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in summary_rows:
        report_lines.append(f"| {row['file']} | {row['avg_f_score']:.4f} | {row['max_f_score']:.4f} | {row['min_f_score']:.4f} |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    return output_path


def main(results_dir: Optional[Path] = None, output_path: Optional[Path] = None, base_embedding_weight: float = 0.5) -> Path:
    base_dir = Path(results_dir or Path(__file__).resolve().parents[3] / "experiments" / "results")
    output_file = output_path or Path(__file__).resolve().parent / "output" / "f_metric_report.md"
    return build_f_report(base_dir, output_file, base_embedding_weight=base_embedding_weight)


if __name__ == "__main__":
    main()
