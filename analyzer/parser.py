"""
parser.py — Parse .py source code and extract code cells from .ipynb notebooks.

Handles:
  - Plain Python source text  →  (source, cell_map=None)
  - Jupyter .ipynb JSON       →  (merged source, cell_map mapping line → cell#)
"""

from __future__ import annotations

import ast
import json
from typing import Optional


def parse_source(text: str, filename: str = "<string>") -> tuple[str, Optional[dict[int, int]], Optional[ast.Module]]:
    """Parse Python source or a Jupyter notebook.

    Args:
        text: Raw file content (Python source or notebook JSON).
        filename: Original filename, used to detect .ipynb.

    Returns:
        (source, cell_map, ast_tree)
        - source: Python code string ready for AST analysis.
        - cell_map: {line_number: cell_index} for notebooks, None for .py.
        - ast_tree: Parsed AST module, or None if parsing failed.
    """
    if filename.endswith(".ipynb"):
        source, cell_map = _extract_notebook(text)
    else:
        source = text
        cell_map = None

    tree = _safe_parse(source, filename)
    return source, cell_map, tree


def _extract_notebook(text: str) -> tuple[str, dict[int, int]]:
    """Extract code cells from a Jupyter notebook JSON string.

    Returns merged source and a mapping of line numbers to cell indices (1-based).
    """
    try:
        nb = json.loads(text)
    except json.JSONDecodeError:
        return text, {}

    cells = nb.get("cells", [])
    lines: list[str] = []
    cell_map: dict[int, int] = {}
    current_line = 1

    for cell_idx, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "code":
            continue
        source_lines = cell.get("source", [])
        # source can be a single string or list of strings
        if isinstance(source_lines, str):
            source_lines = source_lines.splitlines(True)
        for src_line in source_lines:
            cell_map[current_line] = cell_idx
            lines.append(src_line if src_line.endswith("\n") else src_line + "\n")
            current_line += 1
        # Add a blank line between cells to keep AST line numbers valid
        lines.append("\n")
        current_line += 1

    return "".join(lines), cell_map


def _safe_parse(source: str, filename: str) -> Optional[ast.Module]:
    """Parse Python source into an AST, returning None on failure."""
    try:
        return ast.parse(source, filename=filename)
    except SyntaxError:
        return None
