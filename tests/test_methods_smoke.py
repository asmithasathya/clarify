import json

from src.data.schema import LegalQAExample, RetrievedPassage
from src.llm.generator import MockGenerator
from src.methods.closed_book import run_closed_book
from src.methods.hedge import run_hedge
from src.methods.rag_direct import run_rag_direct
from src.methods.revise_verify import run_revise_verify
from src.retrieval.query_rewrite import AlternativeTrajectoryBuilder
from src.verify.abstain import AbstentionPolicy
from src.verify.claim_extraction import ClaimExtractor
from src.verify.claim_scoring import build_support_scorer
from src.verify.revise import RevisionEngine
from src.verify.support_checker import LLMSupportJudge


class TinyRetriever:
    def __init__(self):
        self.base = [
            RetrievedPassage(
                doc_id="CA-001",
                state="California",
                citation="CA Civ. Code 1946.1",
                text="A landlord must give written notice before terminating a periodic tenancy.",
                dense_score=0.65,
                rerank_score=0.8,
                fused_score=0.82,
                rank=1,
            )
        ]
        self.second = [
            RetrievedPassage(
                doc_id="CA-003",
                state="California",
                citation="CA Civ. Proc. Code 1162",
                text="Notice may be served personally, by substituted service, or by posting and mailing.",
                dense_score=0.61,
                rerank_score=1.0,
                fused_score=0.84,
                rank=1,
            )
        ]

    def retrieve(self, query, *, state=None, top_k=None):
        return (self.second if "served" in query.lower() else self.base)[: top_k or 1]

    def retrieve_multi(self, queries, *, state=None, top_k=None, rerank_query=None):
        return (self.base + self.second)[: top_k or 2]


def make_config():
    return {
        "project": {"year": 2021},
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
            "reranker": False,
            "second_pass_verification": True,
            "abstain": True,
            "support_scorer": "retrieval_only",
        },
        "generation": {"json_max_retries": 0},
    }


def make_example(example_id="ex-1"):
    return LegalQAExample(
        example_id=example_id,
        question="Can my landlord terminate my month-to-month tenancy in California without written notice?",
        answer="No",
        state="California",
        statutes=["CA-001", "CA-003"],
        citation=["CA Civ. Code 1946.1"],
        excerpt=[],
    )


def responder(messages):
    prompt = messages[-1]["content"]
    if "Answer the following housing-law question for California as of 2021." in prompt:
        return json.dumps(
            {
                "answer": "No",
                "explanation": "No. California generally requires written notice before termination.",
                "confidence": 0.55,
                "confidence_bucket": "medium",
            }
        )
    if "using only the supplied statute passages" in prompt and "Perform" not in prompt:
        return json.dumps(
            {
                "answer": "No",
                "explanation": "No. California generally requires written notice before termination.",
                "cited_statute_ids": ["CA-001"],
                "citations": ["CA Civ. Code 1946.1"],
                "confidence": 0.60,
                "confidence_bucket": "medium",
            }
        )
    if "Extract 2-5 atomic support claims" in prompt:
        return json.dumps(
            {
                "claims": [
                    {
                        "claim_text": "California requires written notice before terminating a periodic tenancy.",
                        "claim_type": "procedural",
                        "importance_score": 0.9,
                        "span_text_from_original_explanation": "California generally requires written notice before termination.",
                    },
                    {
                        "claim_text": "The answer applies to month-to-month tenancies.",
                        "claim_type": "rule",
                        "importance_score": 0.7,
                        "span_text_from_original_explanation": "California generally requires written notice before termination.",
                    },
                ]
            }
        )
    if "Rewrite the search query" in prompt:
        return json.dumps(
            {
                "rewritten_query": "California housing law 2021 written notice month-to-month tenancy termination",
                "justification": "Targets the exact rule language for periodic tenancy termination.",
            }
        )
    if "Create a compact legal search plan" in prompt:
        return json.dumps(
            {
                "steps": [
                    {"query": "California housing law 2021 written notice termination rule", "purpose": "rule"},
                    {"query": "California housing law 2021 notice exceptions periodic tenancy", "purpose": "exception"},
                ]
            }
        )
    if "Revise the answer minimally" in prompt:
        return json.dumps(
            {
                "answer": "No",
                "explanation": "No. California generally requires written notice before terminating a periodic tenancy, and service rules are addressed separately by statute.",
                "cited_statute_ids": ["CA-001", "CA-003"],
                "citations": ["CA Civ. Code 1946.1", "CA Civ. Proc. Code 1162"],
                "confidence": 0.81,
                "confidence_bucket": "high",
                "revision_notes": "Added claim-specific support from service-of-notice language.",
            }
        )
    if "Draft either a narrow answer or an abstention message" in prompt:
        return json.dumps(
            {
                "answer": "Abstain",
                "explanation": "Abstain. The available evidence does not fully support the answer.",
                "narrow_answer": False,
                "requested_verification": "Need a direct statutory basis.",
            }
        )
    raise AssertionError(f"Unexpected prompt: {prompt[:120]}")


def test_methods_smoke():
    config = make_config()
    example = make_example()
    generator = MockGenerator(responder=responder, config=config)
    retriever = TinyRetriever()
    judge = LLMSupportJudge(config, generator=None)
    support_scorer = build_support_scorer(config, judge=judge)

    closed = run_closed_book(example, generator=generator, config=config)
    rag = run_rag_direct(example, generator=generator, retriever=retriever, config=config, support_scorer=support_scorer)
    hedge = run_hedge(example, generator=generator, retriever=retriever, config=config, support_scorer=support_scorer)
    revise = run_revise_verify(
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

    assert closed.predicted_answer == "No"
    assert rag.predicted_answer == "No"
    assert hedge.predicted_answer == "No"
    assert revise.predicted_answer == "No"
    assert revise.support_score is not None
    assert revise.trace["weakest_claim"]["claim_text"]
