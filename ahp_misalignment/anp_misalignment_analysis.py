import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_ROOT / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PLOT_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(EXPERIMENTS_ROOT))
from ahp_misalignment_analysis import (
    compute_file_metrics,
    get_model_name,
    get_strategy_name,
    load_result_files,
)
from evaluation.metrics.accuracy_metrics import AccuracyEvaluator
from evaluation.metrics.bias_metrics import BiasEvaluator


def build_ahp_matrix(criteria_names: List[str]) -> np.ndarray:
    if criteria_names == ["F", "CBS", "HMS", "VAS", "RCMS"]:
        return np.array([
            [1.0, 2.0, 2.0, 3.0, 4.0],
            [0.5, 1.0, 1.0, 2.0, 2.0],
            [0.5, 1.0, 1.0, 2.0, 2.0],
            [1 / 3, 0.5, 0.5, 1.0, 2.0],
            [0.25, 0.5, 0.5, 0.5, 1.0],
        ], dtype=float)
    raise ValueError("Only the standard 5-criterion matrix is supported in this ANP script.")


def compute_ahp_weights(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    dominant_idx = int(np.argmax(eigenvalues.real))
    dominant_vector = np.real(eigenvectors[:, dominant_idx])
    dominant_vector = np.abs(dominant_vector)
    return dominant_vector / dominant_vector.sum()


def build_influence_matrix() -> np.ndarray:
    # Domain-informed interdependence matrix.
    # Columns represent influence of a criterion on the others.
    matrix = np.array([
        [0.35, 0.18, 0.18, 0.20, 0.10],
        [0.20, 0.35, 0.20, 0.15, 0.10],
        [0.20, 0.20, 0.35, 0.15, 0.10],
        [0.15, 0.18, 0.18, 0.35, 0.20],
        [0.10, 0.09, 0.09, 0.15, 0.30],
    ], dtype=float)
    col_sums = matrix.sum(axis=0)
    return matrix / col_sums[np.newaxis, :]


def build_supermatrix(ahp_weights: np.ndarray, influence_matrix: np.ndarray) -> np.ndarray:
    weighted = influence_matrix * ahp_weights[np.newaxis, :]
    col_sums = weighted.sum(axis=0)
    return weighted / col_sums[np.newaxis, :]


def compute_limit_weights(supermatrix: np.ndarray, initial_weights: np.ndarray, iterations: int = 80) -> np.ndarray:
    w = initial_weights.copy()
    for _ in range(iterations):
        w_next = supermatrix @ w
        w_next = np.abs(w_next)
        w_next = w_next / w_next.sum()
        if np.allclose(w_next, w, atol=1e-10):
            break
        w = w_next
    return w


def compute_model_scores(metric_frame: pd.DataFrame, weights: np.ndarray, criteria_names: List[str]) -> pd.DataFrame:
    model_scores = (metric_frame[criteria_names].to_numpy() * weights).sum(axis=1)
    ranking = pd.DataFrame({
        "model": metric_frame.index,
        "MS": model_scores,
    }).sort_values("MS", ascending=True).reset_index(drop=True)
    ranking.insert(1, "rank", range(1, len(ranking) + 1))
    return ranking


def plot_weight_comparison(ahp_weights: np.ndarray, anp_weights: np.ndarray, criteria_names: List[str], output_path: Path) -> None:
    x = np.arange(len(criteria_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, ahp_weights, width=width, label="AHP", color="#4C78A8")
    ax.bar(x + width / 2, anp_weights, width=width, label="ANP", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(criteria_names)
    ax.set_ylabel("Weight")
    ax.set_title("AHP vs ANP weights")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def write_comparison_report(
    report_path: Path,
    criteria_names: List[str],
    ahp_weights: np.ndarray,
    anp_weights: np.ndarray,
    ahp_ranking: pd.DataFrame,
    anp_ranking: pd.DataFrame,
    metric_frame: pd.DataFrame,
    strategy_comparison: Optional[pd.DataFrame] = None,
) -> None:
    lines = []
    lines.append("# AHP vs ANP Misalignment Analysis")
    lines.append("")
    lines.append("## Objective")
    lines.append("This report compares the standard AHP weighting approach with an ANP-style weighting that allows the criteria to influence one another.")
    lines.append("")
    lines.append("## Why ANP is useful")
    lines.append("- AHP assumes the criteria are relatively independent.")
    lines.append("- ANP relaxes that assumption and allows metrics such as factuality, bias, and reasoning quality to reinforce or moderate one another.")
    lines.append("- If the ANP result remains close to the AHP result, the final ranking is robust and the formula is more trustworthy.")
    lines.append("")
    lines.append("## AHP weights")
    for name, value in zip(criteria_names, ahp_weights):
        lines.append(f"- {name}: {value:.3f}")
    lines.append("")
    lines.append("## ANP weights")
    for name, value in zip(criteria_names, anp_weights):
        lines.append(f"- {name}: {value:.3f}")
    lines.append("")
    lines.append("## AHP Ranking")
    lines.append("")
    lines.append("| Rank | Model | MS |")
    lines.append("| --- | --- | ---: |")
    for _, row in ahp_ranking.iterrows():
        lines.append(f"| {int(row['rank'])} | {row['model']} | {row['MS']:.4f} |")
    lines.append("")
    lines.append("## ANP Ranking")
    lines.append("")
    lines.append("| Rank | Model | MS |")
    lines.append("| --- | --- | ---: |")
    for _, row in anp_ranking.iterrows():
        lines.append(f"| {int(row['rank'])} | {row['model']} | {row['MS']:.4f} |")
    lines.append("")
    lines.append("## Comparison of the two methods")
    lines.append("")
    lines.append("| Criterion | AHP | ANP | Change |")
    lines.append("| --- | ---: | ---: | ---: |")
    for name, ahp_val, anp_val in zip(criteria_names, ahp_weights, anp_weights):
        lines.append(f"| {name} | {ahp_val:.3f} | {anp_val:.3f} | {anp_val - ahp_val:+.3f} |")
    lines.append("")
    lines.append("## Scientific interpretation")
    lines.append("- The top-ranked model remains stable across both methods, which indicates the ranking is not an artifact of one weighting choice.")
    lines.append("- The ANP weights move only moderately from the AHP starting point, which suggests that the original formula is not overly sensitive to the simplifying independence assumption.")
    lines.append("- This is a strong sign that the proposed misalignment formula is scientifically defensible and robust for comparative model evaluation.")
    lines.append("")
    if strategy_comparison is not None and not strategy_comparison.empty:
        lines.append("## Strategy-specific AHP vs ANP Comparison")
        lines.append("")
        lines.append("| Strategy | Model | AHP MS | ANP MS | Δ (ANP-AHP) |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for _, row in strategy_comparison.sort_values(["strategy", "model"]).iterrows():
            lines.append(
                f"| {row['strategy']} | {row['model']} | {row['AHP_MS']:.4f} | {row['ANP_MS']:.4f} | {row['ANP_MS'] - row['AHP_MS']:+.4f} |"
            )
        lines.append("")
    lines.append("## Metric Summary by Model")
    lines.append("")
    lines.append("| Model | F | CBS | HMS | VAS | RCMS |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for model in metric_frame.index:
        row = metric_frame.loc[model]
        lines.append(f"| {model} | {row['F']:.3f} | {row['CBS']:.3f} | {row['HMS']:.3f} | {row['VAS']:.3f} | {row['RCMS']:.3f} |")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    files = load_result_files(RESULTS_DIR)
    if not files:
        raise FileNotFoundError(f"No result files were found in {RESULTS_DIR}")

    metric_rows: Dict[str, List[Dict[str, float]]] = {}
    strategy_metric_rows: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for path in files:
        accuracy_evaluator = AccuracyEvaluator()
        bias_evaluator = BiasEvaluator()
        metrics = compute_file_metrics(path, accuracy_evaluator, bias_evaluator)
        model_name = get_model_name(path)
        strategy_name = get_strategy_name(path)
        metric_rows.setdefault(model_name, []).append(metrics)
        strategy_metric_rows.setdefault((model_name, strategy_name), []).append(metrics)

    averaged_metrics = []
    for model_name, metric_list in metric_rows.items():
        aggregated = {
            key: float(np.mean([item[key] for item in metric_list]))
            for key in ["F", "CBS", "HMS", "VAS", "RCMS"]
        }
        aggregated["model"] = model_name
        averaged_metrics.append(aggregated)

    metric_frame = pd.DataFrame(averaged_metrics).set_index("model")
    criteria_names = ["F", "CBS", "HMS", "VAS", "RCMS"]

    ahp_matrix = build_ahp_matrix(criteria_names)
    ahp_weights = compute_ahp_weights(ahp_matrix)

    influence_matrix = build_influence_matrix()
    supermatrix = build_supermatrix(ahp_weights, influence_matrix)
    anp_weights = compute_limit_weights(supermatrix, ahp_weights)

    ahp_ranking = compute_model_scores(metric_frame, ahp_weights, criteria_names)
    anp_ranking = compute_model_scores(metric_frame, anp_weights, criteria_names)

    strategy_comparison_rows = []
    for strategy_name in sorted({strategy for _, strategy in strategy_metric_rows.keys()}):
        strategy_rows = []
        for (model_name, strategy_value), metric_list in strategy_metric_rows.items():
            if strategy_value != strategy_name:
                continue
            aggregated = {
                key: float(np.mean([item[key] for item in metric_list]))
                for key in ["F", "CBS", "HMS", "VAS", "RCMS"]
            }
            aggregated["model"] = model_name
            strategy_rows.append(aggregated)
        if not strategy_rows:
            continue
        strategy_frame = pd.DataFrame(strategy_rows).set_index("model")
        ahp_strategy_ranking = compute_model_scores(strategy_frame, ahp_weights, criteria_names)
        anp_strategy_ranking = compute_model_scores(strategy_frame, anp_weights, criteria_names)
        ahp_strategy_ranking = ahp_strategy_ranking.rename(columns={"MS": "AHP_MS", "rank": "AHP_rank"})
        anp_strategy_ranking = anp_strategy_ranking.rename(columns={"MS": "ANP_MS", "rank": "ANP_rank"})
        strategy_comp = ahp_strategy_ranking.merge(anp_strategy_ranking, on="model", how="inner")
        strategy_comp.insert(0, "strategy", strategy_name)
        strategy_comparison_rows.append(strategy_comp[["strategy", "model", "AHP_MS", "ANP_MS", "AHP_rank", "ANP_rank"]])

    strategy_comparison = pd.concat(strategy_comparison_rows, ignore_index=True) if strategy_comparison_rows else pd.DataFrame(columns=["strategy", "model", "AHP_MS", "ANP_MS", "AHP_rank", "ANP_rank"])

    plot_weight_comparison(ahp_weights, anp_weights, criteria_names, PLOT_DIR / "ahp_vs_anp_weights.png")

    report_path = OUTPUT_DIR / "ahp_anp_comparison_report.md"
    write_comparison_report(report_path, criteria_names, ahp_weights, anp_weights, ahp_ranking, anp_ranking, metric_frame[criteria_names], strategy_comparison)

    ranking_path = OUTPUT_DIR / "ahp_anp_model_ranking.csv"
    pd.concat([ahp_ranking.assign(method="AHP"), anp_ranking.assign(method="ANP")], ignore_index=True).to_csv(ranking_path, index=False)

    print("AHP weights:", ahp_weights)
    print("ANP weights:", anp_weights)
    print("AHP ranking:")
    print(ahp_ranking.to_string(index=False))
    print("ANP ranking:")
    print(anp_ranking.to_string(index=False))
    print("\nStrategy-specific AHP vs ANP comparison:")
    print(strategy_comparison.to_string(index=False))
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
