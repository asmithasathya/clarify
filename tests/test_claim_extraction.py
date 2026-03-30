from src.llm.generator import MockGenerator
from src.verify.claim_extraction import ClaimExtractor


def test_claim_extraction_validates_schema():
    generator = MockGenerator(
        responses=[
            """{
              "claims": [
                {
                  "claim_text": "California requires written notice before terminating a periodic tenancy.",
                  "claim_type": "procedural",
                  "importance_score": 0.9,
                  "span_text_from_original_explanation": "California requires written notice."
                },
                {
                  "claim_text": "The notice period depends on the tenancy type.",
                  "claim_type": "rule",
                  "importance_score": 0.6,
                  "span_text_from_original_explanation": "The notice period depends on the tenancy type."
                }
              ]
            }"""
        ],
        config={"generation": {"json_max_retries": 0}, "verify": {"min_claims": 2, "max_claims": 5}},
    )
    extractor = ClaimExtractor(
        {"generation": {"json_max_retries": 0}, "verify": {"min_claims": 2, "max_claims": 5}, "ablations": {"claim_decomposition": True}, "project": {"year": 2021}},
        generator=generator,
    )
    claims = extractor.extract(
        question="Can a landlord terminate without notice?",
        explanation="California requires written notice. The notice period depends on tenancy type.",
        state="California",
    )
    assert len(claims) == 2
    assert claims[0].claim_type == "procedural"
    assert 0.0 <= claims[0].importance_score <= 1.0

