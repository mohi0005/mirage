"""
llm.py — Gemini explanation + fix generation with fallback.

Sends detected findings to Google Gemini for natural-language explanations
and corrected code snippets. Falls back to rule-based fix_hint if the API
key is missing, the call fails, or the response is malformed.

The LLM NEVER decides whether a bug exists. It only explains and fixes
bugs that were already detected by deterministic rules.
"""

from __future__ import annotations

import json
import os
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


def enhance_findings(findings: list[dict]) -> tuple[list[dict], bool]:
    """Enhance findings with Gemini explanations.

    Args:
        findings: List of finding dicts from the analyzer.

    Returns:
        (enhanced_findings, llm_used) — The findings with LLM fields populated,
        and a boolean indicating whether the LLM was actually used.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or not findings:
        return findings, False

    try:
        return _call_gemini(findings, api_key)
    except Exception as e:
        logger.warning(f"Gemini API call failed: {e}. Falling back to rule-based explanations.")
        return findings, False


def _call_gemini(findings: list[dict], api_key: str) -> tuple[list[dict], bool]:
    """Call Gemini API to get explanations for findings."""
    try:
        from google import genai
    except ImportError:
        logger.warning("google-genai not installed. Falling back to rule-based explanations.")
        return findings, False

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    # Build the prompt with all findings
    finding_summaries = []
    for i, f in enumerate(findings):
        finding_summaries.append({
            "index": i,
            "rule_id": f.get("rule_id", ""),
            "title": f.get("title", ""),
            "snippet": f.get("snippet", ""),
            "line": f.get("line", 0),
            "fix_hint": f.get("fix_hint", ""),
        })

    system_prompt = (
        "You are a machine learning code review assistant. "
        "You may only explain and fix the detected issues provided. "
        "Do not invent new issues, libraries, or files. "
        "For each finding, return a JSON object with keys: impact, explanation, fixed_code. "
        "- impact: one sentence on real-world consequences. "
        "- explanation: 2-3 sentences explaining the bug clearly. "
        "- fixed_code: a corrected version of the snippet (just the relevant lines). "
        "Return a JSON array of objects in the same order as the input findings."
    )

    user_prompt = (
        f"Here are {len(findings)} detected issues in an ML pipeline. "
        f"Explain each and provide a fix.\n\n"
        f"{json.dumps(finding_summaries, indent=2)}"
    )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
        ),
    )

    text = response.text.strip()
    explanations = _parse_response(text, len(findings))

    if explanations is None:
        # Retry once
        logger.info("First Gemini parse failed, retrying...")
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt + "\n\nIMPORTANT: Return ONLY a valid JSON array, no markdown fences.",
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
            ),
        )
        text = response.text.strip()
        explanations = _parse_response(text, len(findings))

    if explanations is None:
        logger.warning("Could not parse Gemini response after retry.")
        return findings, False

    # Merge explanations into findings
    for i, f in enumerate(findings):
        if i < len(explanations):
            exp = explanations[i]
            f["impact"] = exp.get("impact", "")
            f["explanation"] = exp.get("explanation", "")
            f["fixed_code"] = exp.get("fixed_code", "")
            f["explanation_source"] = "gemini"

    return findings, True


def _parse_response(text: str, expected_count: int) -> list[dict[str, Any]] | None:
    """Parse Gemini response into a list of explanation dicts.

    Strips code fences and validates JSON structure.
    """
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array in the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(parsed, list):
        return None

    # Validate each entry has the expected keys
    valid = []
    for item in parsed:
        if isinstance(item, dict):
            valid.append({
                "impact": str(item.get("impact", "")),
                "explanation": str(item.get("explanation", "")),
                "fixed_code": str(item.get("fixed_code", "")),
            })

    return valid if valid else None
