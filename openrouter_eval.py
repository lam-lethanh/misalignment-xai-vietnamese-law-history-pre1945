import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


DEFAULT_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = "openai/gpt-4o-mini-2024-07-18"


def normalize_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    aliases = {
        "zero": "zero-shot",
        "zero-shot": "zero-shot",
        "few": "few-shot",
        "few-shot": "few-shot",
        "cot": "cot",
        "explicit-cot": "cot",
    }
    return aliases.get(normalized, normalized)


def build_prompt(question: str, mode: str, examples: Optional[List[Dict[str, str]]] = None, language: str = "vi") -> str:
    normalized_mode = normalize_mode(mode)
    if normalized_mode == "zero-shot":
        return f"""Bạn là trợ lý trả lời câu hỏi lịch sử và pháp luật bằng tiếng Việt.

Câu hỏi: {question}

Trả lời ngắn gọn, rõ ràng và chính xác."""

    if normalized_mode == "few-shot":
        example_block = ""
        if examples:
            for idx, ex in enumerate(examples, 1):
                example_block += f"\nVí dụ {idx}:\nCâu hỏi: {ex['question']}\nTrả lời: {ex['answer']}"
        else:
            example_block = "\nVí dụ: Câu hỏi: Hà Nội là thủ đô của Việt Nam? Trả lời: Có, Hà Nội là thủ đô của Việt Nam."
        return f"""Bạn là trợ lý trả lời câu hỏi lịch sử và pháp luật bằng tiếng Việt.

Dưới đây là một vài ví dụ trước khi trả lời:
{example_block}

Câu hỏi hiện tại: {question}

Trả lời bằng tiếng Việt, ngắn gọn và chính xác."""

    if normalized_mode == "cot":
        return f"""Bạn là trợ lý trả lời câu hỏi lịch sử và pháp luật bằng tiếng Việt.

Hãy suy luận từng bước rồi đưa ra câu trả lời cuối cùng.
Sử dụng định dạng sau:
SUY LUẬN: <suy luận>
ĐÁP ÁN: <câu trả lời cuối cùng>

Câu hỏi: {question}"""

    raise ValueError(f"Unsupported mode: {mode}")


def extract_reasoning_and_answer(response_text: str) -> Dict[str, str]:
    if "SUY LUẬN:" in response_text and "ĐÁP ÁN:" in response_text:
        parts_reasoning = response_text.split("SUY LUẬN:", 1)
        if len(parts_reasoning) > 1:
            rest = parts_reasoning[1]
            parts_answer = rest.split("ĐÁP ÁN:", 1)
            cot = parts_answer[0].strip() if len(parts_answer) > 1 else ""
            answer = parts_answer[1].strip() if len(parts_answer) > 1 else ""
            if cot or answer:
                return {"chain_of_thought": cot, "final_answer": answer}

    for marker in ["Do vậy,", "Vì vậy,", "Tóm lại", "Kết luận:", "Kết luận là", "Câu trả lời là:", "ANSWER:"]:
        if marker in response_text:
            parts = response_text.split(marker, 1)
            return {"chain_of_thought": parts[0].strip(), "final_answer": parts[1].strip() if len(parts) > 1 else response_text}

    paragraphs = response_text.split("\n\n")
    if len(paragraphs) >= 2:
        return {"chain_of_thought": "\n\n".join(paragraphs[:-1]).strip(), "final_answer": paragraphs[-1].strip()}

    return {"chain_of_thought": response_text, "final_answer": response_text}


def call_openrouter(prompt: str, api_key: str, model: str, timeout: int = 120) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com",
        "X-Title": "openrouter-eval",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Bạn là một trợ lý thông minh, trả lời bằng tiếng Việt và chú ý đến độ chính xác."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    start = time.perf_counter()
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=timeout)
    latency = time.perf_counter() - start

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text}")

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return {"response": content, "latency": latency, "usage": usage, "error": None}


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_results(results: Dict[str, Any], output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def run_experiment(dataset: List[Dict[str, Any]], api_key: str, model: str, output_path: str, modes: List[str], sample_size: Optional[int], limit: Optional[int]) -> Dict[str, Any]:
    if sample_size is not None:
        dataset = dataset[:sample_size]
    if limit is not None:
        dataset = dataset[:limit]

    overall_results: Dict[str, Any] = {
        "metadata": {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "dataset_size": len(dataset),
            "modes": modes,
        },
        "results": {},
    }

    normalized_modes = [normalize_mode(mode) for mode in modes]

    for mode in normalized_modes:
        mode_key = "zero" if mode == "zero-shot" else "few" if mode == "few-shot" else "cot"
        mode_results = []
        for idx, item in enumerate(dataset):
            question = item.get("question") or item.get("question_text") or ""
            expected_answer = item.get("answer") or item.get("expected_answer") or ""
            question_id = item.get("id") or item.get("question_id") or f"Q{idx:06d}"

            examples = None
            if mode == "few-shot" and idx < len(dataset):
                examples = [
                    {
                        "question": dataset[max(0, idx - 1)].get("question") or dataset[max(0, idx - 1)].get("question_text") or "",
                        "answer": dataset[max(0, idx - 1)].get("answer") or dataset[max(0, idx - 1)].get("expected_answer") or "",
                    }
                    for _ in range(1)
                ]
                # Keep it lightweight and avoid empty examples.
                if not any(ex["question"] for ex in examples):
                    examples = None

            prompt = build_prompt(question=question, mode=mode, examples=examples)

            try:
                response_data = call_openrouter(prompt=prompt, api_key=api_key, model=model)
                response_text = response_data.get("response") or ""
                parsed = extract_reasoning_and_answer(response_text)
                result_entry = {
                    "question_id": question_id,
                    "question": question,
                    "expected_answer": expected_answer,
                    "model_response": response_text,
                    "metadata": {
                        "latency": response_data.get("latency"),
                        "usage": response_data.get("usage"),
                        "error": response_data.get("error"),
                        "test_mode": mode_key,
                    },
                    "chain_of_thought": parsed.get("chain_of_thought", response_text),
                    "final_answer": parsed.get("final_answer", response_text),
                }
            except Exception as exc:  # pragma: no cover - network/runtime issue
                result_entry = {
                    "question_id": question_id,
                    "question": question,
                    "expected_answer": expected_answer,
                    "model_response": "",
                    "metadata": {
                        "latency": None,
                        "usage": {},
                        "error": str(exc),
                        "test_mode": mode_key,
                    },
                    "chain_of_thought": "",
                    "final_answer": "",
                }

            mode_results.append(result_entry)

        overall_results["results"][mode_key] = mode_results

    save_results(overall_results, output_path)
    return overall_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenRouter evaluation for zero-shot, few-shot, and CoT strategies")
    parser.add_argument("--input", required=True, help="Path to dataset JSON file")
    parser.add_argument("--output", default="experiments/results/openrouter_results.json", help="Path to save output JSON")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", DEFAULT_API_KEY), help="OpenRouter API key")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model name")
    parser.add_argument("--modes", default="zero,few,cot", help="Comma-separated modes: zero/few/cot")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of items to process")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of questions to process")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = load_dataset(args.input)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    run_experiment(dataset=dataset, api_key=args.api_key, model=args.model, output_path=args.output, modes=modes, sample_size=args.sample_size, limit=args.limit)
    print(f"Saved results to {args.output}")
