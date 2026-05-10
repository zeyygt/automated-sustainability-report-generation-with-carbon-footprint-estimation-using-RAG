"""LLM-based emission formula extraction with safe arithmetic evaluation.

Flow:
  1. Collect plain text from all non-spreadsheet documents in a session.
  2. Ask the LLM to identify the emission calculation formula and express it
     as a Python arithmetic expression with named variables.
  3. Store the result as an ExtractedFormula.
  4. DataEngine uses safe_eval() to evaluate the formula against per-district
     consumption values instead of the hardcoded multiplication.
  5. If no formula is found (no API key, LLM returns null, parse error),
     DataEngine falls back to the default  consumption × emission_factor  logic.
"""

from __future__ import annotations

import ast
import json
import operator
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ParsedDocument


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ExtractedFormula:
    """An emission formula parsed from a document by the LLM."""
    expression: str
    constants: dict[str, float] = field(default_factory=dict)
    variable_hints: dict[str, str] = field(default_factory=dict)
    confidence: str = "medium"
    source_text: str = ""


# ── Safe arithmetic evaluator ─────────────────────────────────────────────────

_ALLOWED_OPS: dict = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.USub: operator.neg,
}


def safe_eval(expression: str, variables: dict[str, float]) -> float:
    """Evaluate an arithmetic expression allowing only named variables and +,-,*,/,**.

    Raises ValueError for any unsupported syntax (no builtins, no calls, no imports).
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid formula syntax: {expression!r}") from exc
    return _eval_node(tree.body, variables)


def _eval_node(node: ast.expr, variables: dict[str, float]) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"Unknown variable '{node.id}' in formula. Available: {list(variables)}")
        return float(variables[node.id])

    if isinstance(node, ast.BinOp):
        op_fn = _ALLOWED_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.left, variables), _eval_node(node.right, variables))

    if isinstance(node, ast.UnaryOp):
        op_fn = _ALLOWED_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.operand, variables))

    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


# ── LLM extractor ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a formula extraction assistant for sustainability reporting.

Your task: read the document text and identify any formula used to calculate
carbon footprint or CO2 emissions. Express it as a Python arithmetic expression
using only +, -, *, /, ** operators and snake_case variable names.

Return a JSON object with:
  - "formula": Python arithmetic expression string (no function calls, no imports)
  - "constants": dict of variable names that are fixed numeric values in the formula
  - "variable_hints": dict describing what each non-constant variable represents
  - "confidence": "high", "medium", or "low"
  - "source_text": the exact text snippet the formula was derived from

If the document contains no clear emission/carbon formula, return {"formula": null}.

Example input:
  "Total carbon footprint = electricity use × 0.75 kgCO2/kWh + gas use × 2.85 kgCO2/m3 - renewable offset × 0.10"

Example output:
{
  "formula": "electricity * electricity_factor + gas * gas_factor - renewable * renewable_offset",
  "constants": {"electricity_factor": 0.75, "gas_factor": 2.85, "renewable_offset": 0.10},
  "variable_hints": {
    "electricity": "electricity consumption in kWh",
    "gas": "natural gas consumption in m3",
    "renewable": "renewable energy offset in kWh"
  },
  "confidence": "high",
  "source_text": "Total carbon footprint = electricity use × 0.75 kgCO2/kWh + gas use × 2.85 kgCO2/m3 - renewable offset × 0.10"
}"""


class LLMFormulaExtractor:
    """Send document text to an LLM and extract the emission formula from it."""

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o"

    def extract_from_documents(self, documents: list[ParsedDocument]) -> ExtractedFormula | None:
        """Return an ExtractedFormula if the LLM finds one, else None (triggers default logic)."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        text = _collect_text(documents)
        if not text.strip():
            return None

        return self._call_llm(text, api_key)

    def extract_from_text(self, text: str) -> ExtractedFormula | None:
        """Run LLM extraction on an arbitrary text string (used for RAG-assisted fallback)."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or not text.strip():
            return None
        return self._call_llm(text, api_key)

    def _call_llm(self, text: str, api_key: str) -> ExtractedFormula | None:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, timeout=30.0)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"Document text:\n\n{text[:3000]}"},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=512,
            )
            raw = response.choices[0].message.content or ""
            data = json.loads(raw)
        except Exception:
            return None

        if not data or not data.get("formula"):
            return None

        try:
            constants = {k: float(v) for k, v in (data.get("constants") or {}).items()}
        except (TypeError, ValueError):
            constants = {}

        formula = ExtractedFormula(
            expression=data["formula"],
            constants=constants,
            variable_hints=data.get("variable_hints") or {},
            confidence=data.get("confidence") or "medium",
            source_text=data.get("source_text") or "",
        )

        # Validate the expression is safe before accepting it
        dummy = {**constants, "electricity": 0.0, "gas": 0.0, "natural_gas": 0.0,
                 "direct_emissions": 0.0, "renewable": 0.0, "transport": 0.0, "waste": 0.0}
        try:
            safe_eval(formula.expression, dummy)
        except ValueError:
            return None

        return formula


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_text(documents: list[ParsedDocument]) -> str:
    from .models import ElementType
    TEXT_TYPES = {ElementType.PARAGRAPH, ElementType.LIST_ITEM}
    parts: list[str] = []
    for doc in documents:
        for element in doc.elements:
            if element.element_type in TEXT_TYPES and element.text:
                parts.append(element.text)
    return "\n".join(parts)
