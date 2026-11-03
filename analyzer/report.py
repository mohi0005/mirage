"""
report.py — Finding dataclass, severity ranking, quality score, and grade.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# Severity weights for quality score
SEVERITY_WEIGHTS = {"HIGH": 25, "MEDIUM": 10, "LOW": 3}
SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

# Grade thresholds
GRADE_THRESHOLDS = [
    (90, "A", "Excellent — no significant leakage detected"),
    (75, "B", "Good — minor issues found"),
    (60, "C", "Fair — some leakage risks detected"),
    (40, "D", "Poor — significant leakage risks"),
    (0,  "F", "Critical — major data leakage detected"),
]


@dataclass
class Finding:
    """A single detected issue in the analyzed code."""
    rule_id: str
    title: str
    severity: str  # HIGH, MEDIUM, LOW
    line: int
    end_line: Optional[int] = None
    cell: Optional[int] = None  # for notebooks
    snippet: str = ""
    why: str = ""
    fix_hint: str = ""
    # LLM-generated fields (filled later)
    impact: str = ""
    explanation: str = ""
    fixed_code: str = ""
    explanation_source: str = "rule-based"  # "rule-based" or "gemini"

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        d = asdict(self)
        # Remove None values for clean JSON
        return {k: v for k, v in d.items() if v is not None}


def build_report(findings: list[Finding], cell_map: Optional[dict[int, int]] = None) -> dict:
    """Build a complete review report from findings.

    Args:
        findings: List of Finding objects from rule checks.
        cell_map: Optional line→cell mapping for notebooks.

    Returns:
        Report dict with score, grade, summary, and findings.
    """
    # Attach cell numbers for notebooks
    if cell_map:
        for f in findings:
            if f.cell is None and f.line in cell_map:
                f.cell = cell_map[f.line]

    # Sort by severity (HIGH first), then by line number
    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 99), f.line))

    # Calculate quality score
    penalty = sum(SEVERITY_WEIGHTS.get(f.severity, 0) for f in findings)
    score = max(0, 100 - penalty)

    # Determine grade
    grade = "F"
    grade_label = "Critical"
    for threshold, g, label in GRADE_THRESHOLDS:
        if score >= threshold:
            grade = g
            grade_label = label
            break

    # Build summary
    high = sum(1 for f in findings if f.severity == "HIGH")
    medium = sum(1 for f in findings if f.severity == "MEDIUM")
    low = sum(1 for f in findings if f.severity == "LOW")

    if not findings:
        summary = "No data leakage or ML pipeline issues detected. Great job!"
    else:
        parts = []
        if high:
            parts.append(f"{high} high-severity")
        if medium:
            parts.append(f"{medium} medium-severity")
        if low:
            parts.append(f"{low} low-severity")
        summary = f"Found {', '.join(parts)} issue{'s' if len(findings) != 1 else ''}."

    return {
        "score": score,
        "grade": grade,
        "grade_label": grade_label,
        "summary": summary,
        "findings": [f.to_dict() for f in findings],
        "counts": {"high": high, "medium": medium, "low": low, "total": len(findings)},
    }
