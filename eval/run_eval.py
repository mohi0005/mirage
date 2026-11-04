"""
run_eval.py — Evaluation script for MirageML detection rules.

Runs the analyzer (rules only, no LLM) on labelled snippets in eval/cases/
and reports per-rule and overall precision, recall, and F1 score.

Usage:
    python eval/run_eval.py

Output:
    eval/results.json  — machine-readable results
    docs/metrics.md    — human-readable report
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzer.parser import parse_source
from analyzer.flow import FlowTracker
from analyzer.rules import run_all_rules


def load_cases(cases_dir: Path, labels_path: Path) -> tuple[dict, dict]:
    """Load test cases and their labels."""
    with open(labels_path, "r") as f:
        labels = json.load(f)

    cases = {}
    for filename, expected_rules in labels.items():
        filepath = cases_dir / filename
        if filepath.exists():
            cases[filename] = filepath.read_text(encoding="utf-8")
        else:
            print(f"  WARNING: {filename} not found in {cases_dir}")

    return cases, labels


def run_evaluation(cases: dict, labels: dict) -> dict:
    """Run all rules on all cases and compute metrics."""
    # Per-rule tracking
    per_rule: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    all_rule_ids = {"LG001", "LG002", "LG003", "LG004", "LG005", "LG006", "LG007", "LG008"}

    total_cases = len(cases)
    buggy_cases = sum(1 for v in labels.values() if v)
    clean_cases = sum(1 for v in labels.values() if not v)
    false_positives_on_clean = 0
    results_detail = []

    for filename, code in sorted(cases.items()):
        expected = set(labels.get(filename, []))
        is_clean = len(expected) == 0

        # Run analysis
        source, cell_map, tree = parse_source(code, filename)
        if tree is None:
            print(f"  SKIP: {filename} — parse error")
            continue

        tracker = FlowTracker()
        tracker.analyze(tree)
        findings = run_all_rules(tree, source, tracker)
        detected = set(f.rule_id for f in findings)

        # Compute TP/FP/FN per rule
        for rule_id in all_rule_ids:
            if rule_id in expected and rule_id in detected:
                per_rule[rule_id]["tp"] += 1
            elif rule_id not in expected and rule_id in detected:
                per_rule[rule_id]["fp"] += 1
            elif rule_id in expected and rule_id not in detected:
                per_rule[rule_id]["fn"] += 1

        # Track false positives on clean cases
        if is_clean and detected:
            false_positives_on_clean += 1
            print(f"  FALSE POSITIVE on {filename}: {detected}")

        status = "OK" if detected == expected else "MISS"
        results_detail.append({
            "file": filename,
            "expected": sorted(expected),
            "detected": sorted(detected),
            "match": detected == expected,
            "is_clean": is_clean,
        })

        print(f"  {status} {filename}: expected={sorted(expected)}, detected={sorted(detected)}")

    # Compute per-rule metrics
    per_rule_results = {}
    total_tp = total_fp = total_fn = 0
    for rule_id in sorted(all_rule_ids):
        stats = per_rule[rule_id]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_rule_results[rule_id] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    # Overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    overall_f1 = (2 * overall_precision * overall_recall /
                  (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0)

    return {
        "total_cases": total_cases,
        "buggy_cases": buggy_cases,
        "clean_cases": clean_cases,
        "overall_precision": round(overall_precision, 4),
        "overall_recall": round(overall_recall, 4),
        "overall_f1": round(overall_f1, 4),
        "false_positives_on_clean": false_positives_on_clean,
        "per_rule": per_rule_results,
        "details": results_detail,
    }


def write_results(results: dict, results_path: Path, metrics_path: Path) -> None:
    """Write results to JSON and Markdown."""
    # JSON
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results written to {results_path}")

    # Markdown
    md_lines = [
        "# MirageML Evaluation Metrics",
        "",
        "> **Note:** This is a small, self-built evaluation set designed to verify",
        "> rule correctness, not an industry benchmark.",
        "",
        "## Overall Results",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total cases | {results['total_cases']} |",
        f"| Buggy cases | {results['buggy_cases']} |",
        f"| Clean cases | {results['clean_cases']} |",
        f"| **Overall Precision** | **{results['overall_precision']*100:.1f}%** |",
        f"| **Overall Recall** | **{results['overall_recall']*100:.1f}%** |",
        f"| **Overall F1** | **{results['overall_f1']*100:.1f}%** |",
        f"| False positives on clean code | {results['false_positives_on_clean']} |",
        "",
        "## Per-Rule Results",
        "",
        "| Rule | TP | FP | FN | Precision | Recall | F1 |",
        "|------|----|----|-----|-----------|--------|-----|",
    ]

    for rule_id, stats in results["per_rule"].items():
        md_lines.append(
            f"| {rule_id} | {stats['tp']} | {stats['fp']} | {stats['fn']} "
            f"| {stats['precision']*100:.0f}% | {stats['recall']*100:.0f}% "
            f"| {stats['f1']*100:.0f}% |"
        )

    md_lines.extend(["", "---", f"*Generated by run_eval.py*"])

    with open(metrics_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  Metrics written to {metrics_path}")


def main():
    project_root = Path(__file__).resolve().parent.parent
    cases_dir = project_root / "eval" / "cases"
    labels_path = project_root / "eval" / "labels.json"
    results_path = project_root / "eval" / "results.json"
    metrics_path = project_root / "docs" / "metrics.md"

    if not labels_path.exists():
        print("ERROR: eval/labels.json not found")
        sys.exit(1)

    print("MirageML Evaluation")
    print("=" * 50)

    cases, labels = load_cases(cases_dir, labels_path)
    print(f"Loaded {len(cases)} cases ({sum(1 for v in labels.values() if v)} buggy, "
          f"{sum(1 for v in labels.values() if not v)} clean)")
    print()

    results = run_evaluation(cases, labels)
    print()
    print(f"Overall: P={results['overall_precision']*100:.1f}% "
          f"R={results['overall_recall']*100:.1f}% "
          f"F1={results['overall_f1']*100:.1f}%")
    print(f"False positives on clean code: {results['false_positives_on_clean']}")

    write_results(results, results_path, metrics_path)


if __name__ == "__main__":
    main()
