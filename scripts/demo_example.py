"""Small smoke demo for the revise_verify pipeline."""

from __future__ import annotations

import json

import typer

from src.data.schema import LegalQAExample, RetrievedPassage
from src.llm.generator import MockGenerator
from src.methods.revise_verify import run_revise_verify
from src.verify.abstain import AbstentionPolicy
from src.verify.claim_extraction import ClaimExtractor
from src.verify.claim_scoring import build_support_scorer
from src.verify.revise import RevisionEngine
from src.verify.support_checker import LLMSupportJudge
from src.retrieval.query_rewrite import AlternativeTrajectoryBuilder


app = typer.Typer(add_completion=False)


class TinyRetriever:
    def __init__(self) -> None:
        self.initial = [
            RetrievedPassage(
                doc_id="CA-001",
                state="California",
                citation="CA Civ. Code 1946.1",
                text="A periodic tenancy may be terminated by written notice served in advance.",
                dense_score=0.56,
                rerank_score=0.25,
                fused_score=0.60,
                rank=1,
            ),
            RetrievedPassage(
                doc_id="CA-002",
                state="California",
                citation="CA Civ. Code 1161",
                text="A tenant may be subject to unlawful detainer procedures after specified notice requirements are met.",
                dense_score=0.51,
                rerank_score=0.12,
                fused_score=0.54,
                rank=2,
            ),
        ]
        self.verification = [
            RetrievedPassage(
                doc_id="CA-001",
                state="California",
                citation="CA Civ. Code 1946.1",
                text="A landlord must give written notice before terminating a month-to-month tenancy, subject to specified notice periods.",
                dense_score=0.70,
                rerank_score=1.10,
                fused_score=0.88,
                rank=1,
            ),
            RetrievedPassage(
                doc_id="CA-003",
                state="California",
                citation="CA Civ. Proc. Code 1162",
                text="Notice may be served personally, by substituted service, or by posting and mailing as provided by statute.",
                dense_score=0.63,
                rerank_score=0.90,
                fused_score=0.82,
                rank=2,
            ),
        ]

    def retrieve(self, query: str, *, state: str | None = None, top_k: int | None = None):
        if "written notice" in query.lower():
            return self.verification[: top_k or len(self.verification)]
        return self.initial[: top_k or len(self.initial)]

    def retrieve_multi(self, queries, *, state: str | None = None, top_k: int | None = None, rerank_query: str | None = None):
        return self.verification[: top_k or len(self.verification)]


def _demo_responder(messages):
    prompt = messages[-1]["content"]
    if "Extract 2-5 atomic support claims" in prompt:
        return json.dumps(
            {
                "claims": [
                    {
                        "claim_text": "California requires written notice before terminating a periodic tenancy.",
                        "claim_type": "procedural",
                        "importance_score": 0.95,
                        "span_text_from_original_explanation": "California requires written notice before termination.",
                    },
                    {
                        "claim_text": "The answer depends on the tenancy type and statutory notice period.",
                        "claim_type": "rule",
                        "importance_score": 0.65,
                        "span_text_from_original_explanation": "The rule depends on the tenancy type and notice period.",
                    },
                ]
            }
        )
    if "Rewrite the search query" in prompt:
        return json.dumps(
            {
                "rewritten_query": "California housing law 2021 written notice terminate month-to-month tenancy",
                "justification": "Targets the specific statutory notice rule for termination.",
            }
        )
    if "Create a compact legal search plan" in prompt:
        return json.dumps(
            {
                "steps": [
                    {"query": "California housing law 2021 written notice termination rule", "purpose": "rule"},
                    {"query": "California housing law 2021 exceptions to written notice termination", "purpose": "exception"},
                    {"query": "California housing law 2021 definition of service of notice", "purpose": "definition"},
                ]
            }
        )
    if "Revise the answer minimally" in prompt:
        return json.dumps(
            {
                "answer": "No",
                "explanation": "No. For California housing law as of 2021, a landlord generally must provide written notice before terminating a periodic tenancy, and the exact notice period depends on the tenancy details.",
                "cited_statute_ids": ["CA-001", "CA-003"],
                "citations": ["CA Civ. Code 1946.1", "CA Civ. Proc. Code 1162"],
                "confidence": 0.82,
                "confidence_bucket": "high",
                "revision_notes": "Strengthened the notice requirement and narrowed the explanation to periodic tenancies.",
            }
        )
    return json.dumps(
        {
            "answer": "No",
            "explanation": "No. California generally requires written notice before termination, although the initial explanation is brief.",
            "cited_statute_ids": ["CA-001"],
            "citations": ["CA Civ. Code 1946.1"],
            "confidence": 0.58,
            "confidence_bucket": "medium",
        }
    )


@app.command()
def main() -> None:
    typer.echo("Research demo only. This repository targets housing law as of 2021 and is not legal advice.")
    config = {
        "project": {"year": 2021, "seed": 42},
        "retrieval": {"top_k": 2},
        "verify": {
            "support_threshold": 0.62,
            "narrow_threshold": 0.48,
            "retrieval_weight": 0.7,
            "lexical_weight": 0.2,
            "gold_weight": 0.1,
            "min_claims": 2,
            "max_claims": 5,
            "use_search_plan": True,
            "use_multi_query_fusion": True,
        },
        "ablations": {
            "claim_decomposition": True,
            "query_rewrite": True,
            "second_pass_verification": True,
            "abstain": True,
            "support_scorer": "retrieval_only",
        },
        "generation": {"json_max_retries": 0},
    }
    example = LegalQAExample(
        example_id="demo-001",
        question="Can my landlord terminate my month-to-month tenancy in California without written notice?",
        answer="No",
        state="California",
        statutes=["CA-001", "CA Civ. Code 1946.1"],
        citation=["CA Civ. Code 1946.1"],
        excerpt=[],
    )
    generator = MockGenerator(responder=_demo_responder, config=config)
    retriever = TinyRetriever()
    judge = LLMSupportJudge(config, generator=None)
    support_scorer = build_support_scorer(config, judge=judge)
    result = run_revise_verify(
        example,
        generator=generator,
        retriever=retriever,
        config=config,
        claim_extractor=ClaimExtractor(config, generator=generator),
        support_scorer=support_scorer,
        trajectory_builder=AlternativeTrajectoryBuilder(config, generator=generator),
        revision_engine=RevisionEngine(config, generator=generator),
        abstention_policy=AbstentionPolicy(config, generator=None),
    )
    typer.echo(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
