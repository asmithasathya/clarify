"""Alternative retrieval trajectory generation for weak-claim verification."""

from __future__ import annotations

import re
from typing import Any

from src.llm.generator import BaseGenerator
from src.llm.prompts import query_rewrite_prompt, schema_instruction, search_plan_prompt
from src.llm.schemas import QueryRewriteSchema, SearchPlanSchema


STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def _keyword_phrase(text: str, limit: int = 6) -> str:
    tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    kept = [token for token in tokens if token not in STOPWORDS]
    return " ".join(kept[:limit])


class AlternativeTrajectoryBuilder:
    def __init__(self, config: dict[str, Any], generator: BaseGenerator | None = None) -> None:
        self.config = config
        self.generator = generator

    def rewrite_query(self, question: str, weak_claim: str, state: str) -> QueryRewriteSchema:
        year = self.config.get("project", {}).get("year", 2021)
        if self.generator is None or not self.config.get("ablations", {}).get("query_rewrite", True):
            heuristic = f"{state} housing law {year} {_keyword_phrase(weak_claim)} {_keyword_phrase(question, limit=4)}"
            return QueryRewriteSchema(
                rewritten_query=heuristic.strip(),
                justification="Heuristic fallback rewrite using state, year, weak claim terms, and question context.",
            )
        return self.generator.generate_structured(
            query_rewrite_prompt(
                question=question,
                weak_claim=weak_claim,
                state=state,
                year=year,
                schema_text=schema_instruction(QueryRewriteSchema),
            ),
            QueryRewriteSchema,
        )

    def build_search_plan(self, question: str, weak_claim: str, state: str) -> SearchPlanSchema:
        year = self.config.get("project", {}).get("year", 2021)
        if self.generator is None or not self.config.get("verify", {}).get("use_search_plan", True):
            keyword_phrase = _keyword_phrase(weak_claim)
            return SearchPlanSchema(
                steps=[
                    {"query": f"{state} housing law {year} exact rule {keyword_phrase}", "purpose": "rule"},
                    {"query": f"{state} housing law {year} exception {keyword_phrase}", "purpose": "exception"},
                    {"query": f"{state} housing law {year} definition {keyword_phrase}", "purpose": "definition"},
                ]
            )
        return self.generator.generate_structured(
            search_plan_prompt(
                question=question,
                weak_claim=weak_claim,
                state=state,
                year=year,
                schema_text=schema_instruction(SearchPlanSchema),
            ),
            SearchPlanSchema,
        )

    def build_queries(self, question: str, weak_claim: str, state: str) -> dict[str, Any]:
        rewritten = self.rewrite_query(question=question, weak_claim=weak_claim, state=state)
        use_multi = self.config.get("verify", {}).get("use_multi_query_fusion", True)
        if use_multi:
            plan = self.build_search_plan(question=question, weak_claim=weak_claim, state=state)
            queries = [step.query for step in plan.steps]
        else:
            plan = None
            queries = [rewritten.rewritten_query]
        return {
            "rewritten_query": rewritten.rewritten_query,
            "rewrite_justification": rewritten.justification,
            "search_plan": plan.model_dump() if plan else None,
            "queries": queries,
        }
