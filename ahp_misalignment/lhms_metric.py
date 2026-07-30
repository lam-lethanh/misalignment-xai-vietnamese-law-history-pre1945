import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    CrossEncoder = None
    SentenceTransformer = None

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
except Exception:  # pragma: no cover - optional dependency
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    pipeline = None


class LHMSMetricEngine:
    """Compute a Legal-Historical Misconcept Score (LHMS) from concept overlap and misconception detection."""

    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        nli_model_name: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        local_files_only: bool = False,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.nli_model_name = nli_model_name or "lizNguyen235/phobert-nli-vi"
        self._embedding_model = None
        self._nli_pipeline = None
        self._nli_model = None
        self._nli_tokenizer = None
        self._nli_error = None
        self._cross_encoder = None
        self._cross_encoder_error = None
        self._cache_dir = Path(
            cache_dir
            or os.environ.get("HF_HOME")
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
            or str(Path(__file__).resolve().parents[3] / ".cache" / "huggingface")
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._local_files_only = local_files_only
        if SentenceTransformer is not None:
            try:
                self._embedding_model = SentenceTransformer(
                    embedding_model_name or "bkai-foundation-models/vietnamese-bi-encoder",
                    cache_folder=str(self._cache_dir),
                    local_files_only=self._local_files_only,
                )
            except Exception:
                self._embedding_model = None

    def _normalize(self, text: str) -> str:
        if text is None:
            return ""
        text = text.lower().strip()
        text = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _tokenize(self, text: str) -> List[str]:
        return [token for token in self._normalize(text).split() if token]

    def _split_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        parts = re.split(r"(?<=[.!?;])\s+", text.strip())
        return [part.strip() for part in parts if part and part.strip()]

    def _build_sliding_windows(self, text: str, window_size: int = 2) -> List[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        windows: List[str] = []
        for start in range(0, len(sentences)):
            window = sentences[start:start + window_size]
            if window:
                windows.append(" ".join(window))
        return windows

    def _normalize_for_retrieval(self, text: str) -> str:
        return self._normalize(text)

    def _tokenize_for_retrieval(self, text: str) -> List[str]:
        return [token for token in self._normalize_for_retrieval(text).split() if token]

    def _bm25_scores(self, query: str, candidates: List[str]) -> List[float]:
        if not query or not candidates:
            return [0.0] * len(candidates)

        query_tokens = self._tokenize_for_retrieval(query)
        if not query_tokens:
            return [0.0] * len(candidates)

        docs = [self._tokenize_for_retrieval(candidate) for candidate in candidates]
        if not docs:
            return [0.0] * len(candidates)

        doc_lengths = [len(doc) for doc in docs]
        avg_doc_length = float(np.mean(doc_lengths)) if doc_lengths else 0.0
        total_docs = len(docs)
        doc_frequency = Counter()
        for doc_tokens in docs:
            doc_frequency.update(set(doc_tokens))

        k1 = 1.5
        b = 0.75
        scores: List[float] = []
        for doc_tokens, doc_length in zip(docs, doc_lengths):
            if not doc_tokens:
                scores.append(0.0)
                continue
            score = 0.0
            for token in set(query_tokens):
                if token not in doc_frequency:
                    continue
                df = doc_frequency[token]
                idf = math.log(((total_docs - df + 0.5) / (df + 0.5)) + 1.0)
                tf = doc_tokens.count(token)
                denominator = tf + k1 * (1.0 - b + b * (doc_length / avg_doc_length if avg_doc_length > 0 else 0.0))
                score += idf * ((tf * (k1 + 1.0)) / denominator)
            scores.append(float(score))
        return scores

    def _load_cross_encoder(self):
        if self._cross_encoder is not None:
            return self._cross_encoder
        if CrossEncoder is None:
            self._cross_encoder_error = "sentence_transformers CrossEncoder is not available"
            return None
        try:
            self._cross_encoder = CrossEncoder(
                "BAAI/bge-reranker-v2-m3",
                cache_folder=str(self._cache_dir),
                local_files_only=self._local_files_only,
            )
            self._cross_encoder_error = None
            return self._cross_encoder
        except Exception as exc:  # pragma: no cover - optional dependency
            self._cross_encoder_error = str(exc)
            self._cross_encoder = None
            return None

    def _rerank_candidates(self, hypothesis: str, candidates: List[str], top_n: int = 10) -> List[Tuple[float, str]]:
        if not candidates:
            return []
        cross_encoder = self._load_cross_encoder()
        if cross_encoder is None:
            return [
                (self._cosine_similarity(hypothesis, candidate), candidate)
                for candidate in candidates[:top_n]
            ]

        try:
            pairs = [(hypothesis, candidate) for candidate in candidates[:top_n]]
            scores = cross_encoder.predict(pairs, batch_size=32, show_progress_bar=False)
            reranked: List[Tuple[float, str]] = []
            for candidate, score in zip(candidates[:top_n], scores):
                normalized_score = float(score)
                normalized_score = max(0.0, min(1.0, normalized_score))
                reranked.append((normalized_score, candidate))
            reranked.sort(key=lambda item: item[0], reverse=True)
            return reranked
        except Exception as exc:  # pragma: no cover - optional dependency
            self._cross_encoder_error = str(exc)
            return [
                (self._cosine_similarity(hypothesis, candidate), candidate)
                for candidate in candidates[:top_n]
            ]

    def _select_relevant_reference_sentences(
        self,
        hypothesis: str,
        reference_sentences: List[str],
        k: int = 3,
        similarity_threshold: float = 0.4,
        use_hybrid_search: bool = True,
        use_reranker: bool = True,
        top_n: int = 10,
        k_smooth: int = 60,
    ) -> List[str]:
        if not reference_sentences:
            return []
        if not hypothesis:
            return []

        dense_scores = [(self._cosine_similarity(hypothesis, sentence), sentence) for sentence in reference_sentences]
        dense_scores.sort(key=lambda item: item[0], reverse=True)
        dense_ranked = [sentence for _, sentence in dense_scores]

        if use_hybrid_search:
            bm25_scores = self._bm25_scores(hypothesis, reference_sentences)
            bm25_ranked = [
                sentence for _, sentence in sorted(
                    zip(bm25_scores, reference_sentences),
                    key=lambda item: item[0],
                    reverse=True,
                )
            ]
            rrf_scores: Dict[str, float] = {sentence: 0.0 for sentence in reference_sentences}
            for rank, sentence in enumerate(dense_ranked, start=1):
                rrf_scores[sentence] += 1.0 / (k_smooth + rank)
            for rank, sentence in enumerate(bm25_ranked, start=1):
                rrf_scores[sentence] += 1.0 / (k_smooth + rank)
            hybrid_ranked = [
                sentence for sentence, _ in sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
            ]
            candidate_pool = hybrid_ranked[:top_n]
        else:
            candidate_pool = dense_ranked[:top_n]

        if use_reranker and candidate_pool:
            reranked_candidates = self._rerank_candidates(hypothesis, candidate_pool, top_n=top_n)
            filtered_candidates = [
                sentence for score, sentence in reranked_candidates if score >= similarity_threshold
            ]
            if filtered_candidates:
                return filtered_candidates[:k]
            return []

        filtered_candidates = [
            sentence for sentence in candidate_pool if self._cosine_similarity(hypothesis, sentence) >= similarity_threshold
        ]
        if filtered_candidates:
            return filtered_candidates[:k]
        return []

    def _encode_text(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(64, dtype=float)
        if self._embedding_model is not None:
            try:
                vec = self._embedding_model.encode(text, convert_to_numpy=True)
                return np.asarray(vec, dtype=float)
            except Exception:
                pass
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(64, dtype=float)
        vector = np.zeros(64, dtype=float)
        for token in tokens:
            vector[abs(hash(token)) % 64] += 1.0
        return vector

    def _cosine_similarity(self, left: str, right: str) -> float:
        left_vec = self._encode_text(left)
        right_vec = self._encode_text(right)
        left_norm = np.linalg.norm(left_vec)
        right_norm = np.linalg.norm(right_vec)
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        score = float(np.dot(left_vec, right_vec) / (left_norm * right_norm))
        return max(0.0, min(1.0, (score + 1.0) / 2.0))

    def extract_key_phrases(self, text: str) -> List[str]:
        if not text:
            return []
        candidates = []
        for sentence in self._split_sentences(text):
            matches = re.findall(r"[A-Za-zÀ-ỹ]+(?:\s+[A-Za-zÀ-ỹ]+){0,3}", sentence.lower())
            for match in matches:
                phrase = match.strip()
                if len(phrase.split()) >= 2 and len(phrase.split()) <= 4:
                    candidates.append(phrase)
        if not candidates:
            tokens = self._tokenize(text)
            if len(tokens) >= 2:
                candidates = [" ".join(tokens[i:i + 2]) for i in range(max(1, len(tokens) - 1))]
        unique_phrases = []
        seen = set()
        for phrase in candidates:
            if phrase not in seen:
                unique_phrases.append(phrase)
                seen.add(phrase)
        return unique_phrases[:10]

    def compute_key_concept_overlap(self, expected: str, actual: str) -> float:
        expected_phrases = self.extract_key_phrases(expected)
        actual_phrases = self.extract_key_phrases(actual)
        if not expected_phrases and not actual_phrases:
            return 0.0
        if not expected_phrases or not actual_phrases:
            return 0.0

        precision_scores = [
            max(self._cosine_similarity(actual_phrase, expected_phrase) for expected_phrase in expected_phrases)
            for actual_phrase in actual_phrases
        ]
        recall_scores = [
            max(self._cosine_similarity(expected_phrase, actual_phrase) for actual_phrase in actual_phrases)
            for expected_phrase in expected_phrases
        ]

        precision = float(np.mean(precision_scores)) if precision_scores else 0.0
        recall = float(np.mean(recall_scores)) if recall_scores else 0.0
        if precision + recall == 0.0:
            return 0.0
        return float(2.0 * precision * recall / (precision + recall))

    def compute_misconception_score(self, reference: str, response: str) -> float:
        reference_sentences = self._split_sentences(reference)
        if not reference_sentences:
            return 0.0
        if not response:
            return 0.0

        contradiction_scores: List[float] = []
        for window in self._build_sliding_windows(response, window_size=2):
            window_sentences = self._split_sentences(window)
            if not window_sentences:
                continue
            hypothesis = " ".join(window_sentences)
            scored_reference = []
            for sentence in reference_sentences:
                scored_reference.append((self._cosine_similarity(hypothesis, sentence), sentence))
            scored_reference.sort(key=lambda item: item[0], reverse=True)
            top_reference = [sentence for _, sentence in scored_reference[:3]]
            premise = " ".join(top_reference) if top_reference else " ".join(reference_sentences[:3])
            contradiction_scores.append(self._predict_contradiction(premise, hypothesis))

        return float(np.max(contradiction_scores)) if contradiction_scores else 0.0

    def _load_nli_pipeline(self):
        if self._nli_pipeline is not None:
            return self._nli_pipeline
        if pipeline is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            self._nli_error = "transformers is not available"
            return None

        try:
            self._nli_pipeline = pipeline(
                "text-classification",
                model=self.nli_model_name,
                tokenizer=self.nli_model_name,
                cache_dir=str(self._cache_dir),
                local_files_only=self._local_files_only,
                top_k=None,
                device=-1,
            )
            self._nli_error = None
            return self._nli_pipeline
        except Exception as exc:  # pragma: no cover - network/cache dependent
            self._nli_error = str(exc)
            self._nli_pipeline = None
            return None

    def _load_nli_model_and_tokenizer(self):
        if self._nli_model is not None and self._nli_tokenizer is not None:
            return self._nli_model, self._nli_tokenizer
        if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
            self._nli_error = "transformers/torch is not available"
            return None, None

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.nli_model_name,
                cache_dir=str(self._cache_dir),
                local_files_only=self._local_files_only,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.nli_model_name,
                cache_dir=str(self._cache_dir),
                local_files_only=self._local_files_only,
            )
            model.eval()
            self._nli_tokenizer = tokenizer
            self._nli_model = model
            self._nli_error = None
            return model, tokenizer
        except Exception as exc:  # pragma: no cover - network/cache dependent
            self._nli_error = str(exc)
            self._nli_model = None
            self._nli_tokenizer = None
            return None, None

    def _extract_label_scores(self, labels: Optional[List[str]], scores: Optional[List[float]]) -> Dict[str, float]:
        if not labels or not scores:
            return {}
        label_scores: Dict[str, float] = {}
        for label, score in zip(labels, scores):
            if label is None:
                continue
            label_scores[str(label).strip().lower()] = float(score)
        return label_scores

    def _score_from_label_scores(self, label_scores: Dict[str, float]) -> float:
        contradiction_score = 0.0
        for label, score in label_scores.items():
            if "contradict" in label:
                contradiction_score = max(contradiction_score, score)

        entailment_score = label_scores.get("entailment") or label_scores.get("entail") or label_scores.get("entailment_score")
        if contradiction_score <= 0.0 and entailment_score is not None:
            contradiction_score = max(0.0, 1.0 - entailment_score)

        return max(0.0, min(1.0, contradiction_score))

    def inspect_nli_pair(self, premise: str, hypothesis: str) -> Dict[str, object]:
        model, tokenizer = self._load_nli_model_and_tokenizer()
        if model is None or tokenizer is None:
            return {"raw_scores": {}, "contradiction_score": 0.0, "error": self._nli_error}

        try:
            encoded = tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[0]
            if hasattr(model, "config") and hasattr(model.config, "id2label"):
                labels = [str(model.config.id2label[idx]).strip().lower() for idx in range(len(probabilities))]
                scores = [float(probability.item()) for probability in probabilities]
            else:
                labels = []
                scores = []
            label_scores = self._extract_label_scores(labels, scores)
            return {
                "raw_scores": label_scores,
                "contradiction_score": self._score_from_label_scores(label_scores),
                "error": None,
            }
        except Exception as exc:  # pragma: no cover - network/cache dependent
            self._nli_error = str(exc)
            return {"raw_scores": {}, "contradiction_score": 0.0, "error": str(exc)}

    def _predict_contradiction(self, premise: str, hypothesis: str) -> float:
        return self.inspect_nli_pair(premise, hypothesis).get("contradiction_score", 0.0)

    def compute_lhms_score(self, expected: str, actual: str, concept_weight: float = 0.5) -> float:
        concept_overlap = self.compute_key_concept_overlap(expected, actual)
        misconception = self.compute_misconception_score(expected, actual)
        mismatch = max(0.0, min(1.0, 1.0 - concept_overlap))
        combined = (concept_weight * mismatch) + ((1.0 - concept_weight) * misconception)
        return max(0.0, min(1.0, combined))

    def evaluate_file(self, results_file: Path) -> Dict[str, float]:
        with results_file.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        scores = []
        for item in data.get("results", []):
            expected = item.get("expected_answer") or ""
            actual = item.get("model_response") or ""
            scores.append(self.compute_lhms_score(expected, actual))

        return {
            "avg_lhms_score": float(np.mean(scores)) if scores else 0.0,
            "max_lhms_score": float(np.max(scores)) if scores else 0.0,
            "min_lhms_score": float(np.min(scores)) if scores else 0.0,
        }


def compute_key_concept_overlap(expected: str, actual: str) -> float:
    return LHMSMetricEngine().compute_key_concept_overlap(expected, actual)


def compute_misconception_score(reference: str, response: str) -> float:
    return LHMSMetricEngine().compute_misconception_score(reference, response)


def compute_lhms_score(expected: str, actual: str, concept_weight: float = 0.5) -> float:
    return LHMSMetricEngine().compute_lhms_score(expected, actual, concept_weight=concept_weight)


def run_lhms_weight_sensitivity(base_concept_weight: float = 0.5, deltas: Optional[List[float]] = None) -> List[Dict[str, float]]:
    if deltas is None:
        deltas = [-0.20, -0.10, 0.0, 0.10, 0.20]

    scenarios = []
    for delta in deltas:
        weight = base_concept_weight + delta
        if weight <= 0.0 or weight >= 1.0:
            continue
        scenarios.append({"delta": delta, "concept_weight": weight})
    return scenarios


def summarize_lhms_by_weight(results_dir: Path, base_concept_weight: float = 0.5, deltas: Optional[List[float]] = None) -> List[Dict[str, float]]:
    engine = LHMSMetricEngine()
    sensitivity = run_lhms_weight_sensitivity(base_concept_weight=base_concept_weight, deltas=deltas)
    rows: List[Dict[str, float]] = []

    for scenario in sensitivity:
        weight = float(scenario["concept_weight"])
        all_scores = []
        for path in sorted(results_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            for item in data.get("results", []):
                expected = item.get("expected_answer") or ""
                actual = item.get("model_response") or ""
                all_scores.append(engine.compute_lhms_score(expected, actual, concept_weight=weight))

        rows.append({
            "delta": float(scenario["delta"]),
            "concept_weight": weight,
            "avg_lhms_score": float(np.mean(all_scores)) if all_scores else 0.0,
            "max_lhms_score": float(np.max(all_scores)) if all_scores else 0.0,
            "min_lhms_score": float(np.min(all_scores)) if all_scores else 0.0,
        })

    return rows


def build_lhms_report(results_dir: Path, output_path: Path, base_concept_weight: float = 0.5) -> Path:
    engine = LHMSMetricEngine()
    summary_rows = []
    for path in sorted(results_dir.glob("*.json")):
        metrics = engine.evaluate_file(path)
        summary_rows.append({"file": path.name, **metrics})

    weight_summaries = summarize_lhms_by_weight(results_dir, base_concept_weight=base_concept_weight)
    report_lines = [
        "# LHMS Metric Report",
        "",
        "## Objective",
        "This report computes a Legal-Historical Misconcept Score (LHMS) from semantic concept overlap and contradiction-based misconception detection.",
        "",
        "## Formula",
        "LHMS = concept_weight * (1 - KeyConceptOverlap) + (1 - concept_weight) * MisconceptionScore",
        "",
        "## Base configuration",
        f"- Base concept weight: {base_concept_weight:.2f}",
        "",
        "## LHMS by concept weight",
        "",
        "| delta | concept_weight | avg_lhms_score | max_lhms_score | min_lhms_score |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in weight_summaries:
        report_lines.append(
            f"| {row['delta']:+.2f} | {row['concept_weight']:.2f} | {row['avg_lhms_score']:.4f} | {row['max_lhms_score']:.4f} | {row['min_lhms_score']:.4f} |"
        )

    report_lines.extend([
        "",
        "## Summary by file",
        "",
        "| file | avg_lhms_score | max_lhms_score | min_lhms_score |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in summary_rows:
        report_lines.append(f"| {row['file']} | {row['avg_lhms_score']:.4f} | {row['max_lhms_score']:.4f} | {row['min_lhms_score']:.4f} |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_lines), encoding="utf-8")
    return output_path


def main(results_dir: Optional[Path] = None, output_path: Optional[Path] = None, base_concept_weight: float = 0.5) -> Path:
    base_dir = Path(results_dir or Path(__file__).resolve().parents[3] / "experiments" / "results")
    output_file = output_path or Path(__file__).resolve().parent / "output" / "lhms_metric_report.md"
    return build_lhms_report(base_dir, output_file, base_concept_weight=base_concept_weight)


if __name__ == "__main__":
    main()
