import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def _get_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =============================================================================
# 1. Atomic Claim Extractor (dùng LLM)
# =============================================================================
class ClaimExtractor:
    """Tách atomic claims từ văn bản bằng Qwen2.5-3B-Instruct."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        cache_dir: Optional[Path] = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir or Path(
            os.environ.get("HF_HOME")
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
            or str(Path(__file__).resolve().parents[3] / ".cache" / "huggingface")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = _get_torch_device()
        self.local_files_only = local_files_only

        print(f"[ClaimExtractor] Loading {self.model_name} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
            local_files_only=self.local_files_only,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
            local_files_only=self.local_files_only,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        print("[ClaimExtractor] Loaded.")

    def _build_messages(self, text: str) -> List[Dict[str, str]]:
        system_content = """
Bạn là chuyên gia tách các atomic claim phục vụ kiểm chứng thông tin (fact-checking).

Nhiệm vụ:
Tách đoạn văn đầu vào thành các atomic claim.

Định nghĩa:
Atomic claim là một khẳng định chỉ chứa một thông tin thực tế duy nhất và có thể được kiểm chứng độc lập.

Quy tắc:

- Mỗi atomic claim chỉ được chứa MỘT khẳng định thực tế.
- Mỗi atomic claim phải có thể hiểu đầy đủ mà không cần tham chiếu đến các claim khác.
- Nếu một câu chứa nhiều thông tin độc lập thì phải tách thành nhiều atomic claim.
- Nếu một câu liệt kê nhiều nguyên nhân, mục đích, hành động, sự kiện, đặc điểm hoặc hệ quả thì phải tách mỗi thông tin thành một atomic claim riêng.
- Giữ nguyên ý nghĩa của văn bản gốc.
- Giữ nguyên tên riêng, ngày tháng, số liệu, địa điểm, tổ chức và các thực thể khác.
- Không suy diễn, không giải thích, không tóm tắt và không bổ sung thông tin mới.
- Không được bỏ sót bất kỳ thông tin thực tế nào.
- Được phép chỉnh sửa rất nhỏ chỉ để thay thế các đại từ hoặc cụm tham chiếu (ví dụ: "đây", "đó", "điều này", "ông", "bà", "họ") bằng đối tượng mà chúng đề cập đến, nhằm giúp mỗi atomic claim có thể đứng độc lập.
- Không gộp nhiều thông tin thực tế vào cùng một atomic claim.
- Chỉ xuất ra các atomic claim.
- Mỗi dòng chỉ chứa một atomic claim.
- Không đánh số.
- Không sử dụng ký hiệu đầu dòng.
- Nếu một sự kiện có nhiều thuộc tính (ví dụ: thời gian, địa điểm, chủ thể, đối tượng) thì giữ chúng trong cùng một atomic claim, miễn là chúng cùng mô tả một sự kiện duy nhất.
- Không tạo các atomic claim chứa các cụm tham chiếu không rõ nghĩa như "điều này", "điểm yếu này", "đó", "việc này". Nếu cần, hãy thay bằng đối tượng được nhắc đến trong văn bản.

Example 1
Input:
Nguyễn Du sinh năm 1765 và là tác giả Truyện Kiều.
Output:
Nguyễn Du sinh năm 1765.
Nguyễn Du là tác giả Truyện Kiều.

Example 2
Input:
Nguyễn Hữu Cầu tiến sang Kinh Bắc. Đây là biểu hiện của tinh thần dũng cảm và ý chí kiên cường.
Output:
Nguyễn Hữu Cầu tiến sang Kinh Bắc.
Quyết định tiến sang Kinh Bắc là biểu hiện của tinh thần dũng cảm.
Quyết định tiến sang Kinh Bắc là biểu hiện của ý chí kiên cường.
""".strip()
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": text},
        ]

    def extract_atomic_claims(self, text: str, max_new_tokens: int = 512) -> List[str]:
        if not text or not text.strip():
            return []

        messages = self._build_messages(text)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device != "cpu":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        return self._parse_claims(generated)

    @staticmethod
    def _parse_claims(raw_text: str) -> List[str]:
        lines = re.split(r"[\n\r]+", raw_text.strip())
        claims = []
        for line in lines:
            line = re.sub(r"^\s*[-*\d.\)]+\s*", "", line).strip()
            if line:
                claims.append(line)
        return claims


# =============================================================================
# 2. Claim Verifier dùng NLI mạnh (mDeBERTa-v3)
# =============================================================================
class ClaimVerifier:
    """
    Dùng NLI model mạnh để kiểm tra Fact Coverage.
    
    Premise   = toàn bộ câu trả lời dài của LLM (actual)
    Hypothesis = từng atomic claim (lấy từ đáp án mẫu)
    """

    def __init__(
        self,
        model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        cache_dir: Optional[Path] = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir or Path(
            os.environ.get("HF_HOME")
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
            or str(Path(__file__).resolve().parents[3] / ".cache" / "huggingface")
        )
        self.device = _get_torch_device()
        self.local_files_only = local_files_only

        print(f"[ClaimVerifier] Loading {self.model_name} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
            local_files_only=self.local_files_only,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir),
            local_files_only=self.local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()
        print("[ClaimVerifier] Loaded.")

        # Map label
        self.id2label = {
            int(k): str(v).lower()
            for k, v in self.model.config.id2label.items()
        }

    @torch.no_grad()
    def verify_claim(self, claim: str, llm_answer: str) -> Dict[str, float]:
        """
        Premise    = llm_answer (toàn bộ câu trả lời của LLM)
        Hypothesis = claim      (atomic claim từ đáp án mẫu)
        """
        if not claim or not llm_answer:
            return {
                "entailment": 0.0,
                "neutral": 0.0,
                "contradiction": 1.0,
                "score": 0.0,
            }

        inputs = self.tokenizer(
            llm_answer,          # premise
            claim,               # hypothesis
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits[0]
        probs = F.softmax(logits, dim=-1).cpu()

        result = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
        for idx, prob in enumerate(probs):
            label = self.id2label.get(idx, "")
            if "entail" in label:
                result["entailment"] = float(prob)
            elif "contra" in label:
                result["contradiction"] = float(prob)
            elif "neutral" in label:
                result["neutral"] = float(prob)

        # Fact coverage score: ưu tiên entailment, phạt contradiction
        score = result["entailment"]
        if result["contradiction"] > 0.5:
            score = 0.0

        result["score"] = score
        return result

    def truth_score(self, claim: str, llm_answer: str) -> float:
        return self.verify_claim(claim, llm_answer)["score"]


# =============================================================================
# 3. Fact Coverage Metric
# =============================================================================
class FactualityMetric:
    """
    Fact Coverage:
    1. Tách atomic claims từ đáp án mẫu (expected / ground-truth)
    2. Với mỗi claim, dùng NLI kiểm tra xem toàn bộ câu trả lời của LLM có support claim đó không
    """

    def __init__(
        self,
        extractor: Optional[ClaimExtractor] = None,
        verifier: Optional[ClaimVerifier] = None,
    ) -> None:
        self.extractor = extractor or ClaimExtractor()
        self.verifier = verifier or ClaimVerifier()

    def compute_factuality(self, expected: str, actual: str) -> float:
        claims = self.extractor.extract_atomic_claims(expected)
        if not claims:
            return 0.0
        scores = [self.verifier.truth_score(claim, actual) for claim in claims]
        return float(sum(scores) / len(scores))

    def full_item_report(self, item: Dict) -> Dict:
        expected = self._get_expected_text(item)
        actual = self._get_actual_text(item)

        claims = self.extractor.extract_atomic_claims(expected)
        results = []
        for claim in claims:
            nli = self.verifier.verify_claim(claim, actual)
            results.append({
                "claim": claim,
                "entailment": nli["entailment"],
                "neutral": nli["neutral"],
                "contradiction": nli["contradiction"],
                "score": nli["score"],
            })

        avg_score = (
            float(sum(r["score"] for r in results) / len(results))
            if results else 0.0
        )

        return {
            "expected": expected,
            "actual": actual,
            "claims": results,
            "fact_coverage": avg_score,
            "num_claims": len(results),
        }

    @staticmethod
    def _get_expected_text(item: Dict) -> str:
        return (
            item.get("expected_answer")
            or item.get("reference_answer")
            or item.get("ground_truth")
            or item.get("target_answer")
            or item.get("answer_key")
            or ""
        )

    @staticmethod
    def _get_actual_text(item: Dict) -> str:
        return item.get("final_answer") or item.get("answer") or ""


# =============================================================================
# 4. Runner
# =============================================================================
def run_json_file(
    json_path: Path,
    sample_size: int = 5,
    local_files_only: bool = False,
    output_json: Optional[Path] = None,
) -> List[Dict]:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    items = data.get("results", [])[:sample_size]

    metric = FactualityMetric(
        extractor=ClaimExtractor(local_files_only=local_files_only),
        verifier=ClaimVerifier(local_files_only=local_files_only),
    )

    reports = []
    print(f"\nRunning Fact Coverage on {len(items)} items\n")
    print("=" * 80)

    for idx, item in enumerate(items, start=1):
        report = metric.full_item_report(item)
        reports.append(report)

        print(f"\n--- ITEM {idx} ---")
        print(f"Expected : {report['expected'][:180]}{'...' if len(report['expected']) > 180 else ''}")
        print(f"Actual   : {report['actual'][:180]}{'...' if len(report['actual']) > 180 else ''}")
        print(f"Fact Coverage: {report['fact_coverage']:.4f}   ({report['num_claims']} claims)")
        print("-" * 40)
        for j, c in enumerate(report["claims"], start=1):
            print(f"{j}. [{c['score']:.3f}] {c['claim']}")
            print(f"   entail={c['entailment']:.3f} | neutral={c['neutral']:.3f} | contra={c['contradiction']:.3f}")
        print()

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as fh:
            json.dump(reports, fh, ensure_ascii=False, indent=2)
        print(f"Saved report → {output_json}")

    return reports


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python f_metric_nli.py <input.json> [output.json]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    run_json_file(
        json_path=input_path,
        sample_size=200,
        local_files_only=True,      # đổi thành True sau khi đã tải model
        output_json=output_path,
    )