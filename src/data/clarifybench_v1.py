"""Synthetic but fixed multi-turn clarification benchmark."""

from __future__ import annotations

from pathlib import Path

from src.data.report_data import write_dataset_split, write_manifest
from src.data.schema import DialogueExample
from src.utils.io import write_json


SCENARIO_BANK = {
    "travel": [
        {
            "referent": "the refundable hotel in Kyoto",
            "goal": "a four-night October foliage trip with your mother",
            "constraints": "needs elevator access, Kyoto Station access, and a budget under $180 per night",
            "answer": "Book the refundable Kyoto Station hotel now because it fits the accessibility and budget constraints for the October trip.",
            "checklist": ["Ask what 'it' refers to.", "Confirm timing and accessibility constraints."],
        },
        {
            "referent": "the evening train pass for Switzerland",
            "goal": "a five-day Swiss rail itinerary with heavy mountain travel",
            "constraints": "wants flexible same-day changes and already has lodging booked",
            "answer": "Buy the flexible Swiss rail pass because the itinerary involves frequent route changes across several mountain towns.",
            "checklist": ["Clarify which pass is under discussion.", "Ask how flexible the itinerary must be."],
        },
        {
            "referent": "the direct flight to Lisbon",
            "goal": "a spring break visit with one checked bag and no long layovers",
            "constraints": "wants to arrive rested for a morning family event",
            "answer": "Take the direct Lisbon flight because avoiding layovers matters more than a small fare difference before the morning event.",
            "checklist": ["Disambiguate the option under discussion.", "Confirm arrival priorities."],
        },
        {
            "referent": "the hostel near the trailhead",
            "goal": "a low-budget hiking weekend in Zion",
            "constraints": "no car, early trail access, and total lodging budget under $90 per night",
            "answer": "Keep the trailhead hostel because it is the only option that preserves early hiking access without needing a car.",
            "checklist": ["Clarify the lodging option.", "Ask about transport and budget."],
        },
        {
            "referent": "the afternoon ferry to Vancouver Island",
            "goal": "a two-day island trip with a rental bike",
            "constraints": "must avoid late arrivals because the bike shop closes at 6 p.m.",
            "answer": "Take the earlier ferry instead of the afternoon one so the rental bike can still be collected before the shop closes.",
            "checklist": ["Clarify which departure is being discussed.", "Ask about arrival deadlines."],
        },
        {
            "referent": "the airport hotel in Seoul",
            "goal": "an overnight layover before a 7 a.m. departure",
            "constraints": "needs a free shuttle and a quiet room for one night",
            "answer": "Book the airport hotel because the free shuttle and quiet overnight stay matter more than being downtown for a short layover.",
            "checklist": ["Clarify the property under discussion.", "Confirm the layover constraints."],
        },
    ],
    "finance": [
        {
            "referent": "the old 401(k)",
            "goal": "simplify retirement accounts before starting graduate school",
            "constraints": "wants low fees and a single target-date fund",
            "answer": "Roll the old 401(k) into a low-fee rollover IRA so the accounts are consolidated before graduate school starts.",
            "checklist": ["Clarify which account 'it' refers to.", "Ask about fees and account simplicity."],
        },
        {
            "referent": "the emergency fund in a checking account",
            "goal": "keep six months of expenses liquid",
            "constraints": "needs same-day access but wants a better yield",
            "answer": "Move the emergency fund from checking to a high-yield savings account that still provides quick access.",
            "checklist": ["Clarify the asset under discussion.", "Confirm liquidity requirements."],
        },
        {
            "referent": "the company stock grant",
            "goal": "reduce concentration risk after a recent vest",
            "constraints": "already has enough exposure through salary and future grants",
            "answer": "Sell a meaningful portion of the company stock grant to reduce concentration risk while keeping a smaller upside position.",
            "checklist": ["Ask what holding 'it' refers to.", "Confirm concentration concerns."],
        },
        {
            "referent": "the 0% balance-transfer offer",
            "goal": "pay down credit-card debt over twelve months",
            "constraints": "can make fixed monthly payments but cannot afford a missed-payment penalty",
            "answer": "Use the 0% balance-transfer offer only if the fixed monthly payoff amount is realistic and autopay is enabled to avoid penalties.",
            "checklist": ["Clarify which offer is under discussion.", "Ask about payoff capacity."],
        },
        {
            "referent": "the inherited brokerage account",
            "goal": "repurpose the money for a home down payment in two years",
            "constraints": "cannot tolerate a large drawdown before buying",
            "answer": "Shift the inherited brokerage account toward lower-volatility assets because the house down payment is only two years away.",
            "checklist": ["Clarify the account in question.", "Confirm the time horizon."],
        },
        {
            "referent": "the student loan refinance offer",
            "goal": "lower monthly payments while keeping flexibility",
            "constraints": "may pursue public-service work in the next year",
            "answer": "Avoid refinancing the student loans if public-service forgiveness remains plausible, because refinancing would remove that option.",
            "checklist": ["Clarify which debt product is being discussed.", "Ask about forgiveness eligibility."],
        },
    ],
    "health": [
        {
            "referent": "the iron supplement",
            "goal": "keep energy up during marathon training",
            "constraints": "currently taking it at night because it causes daytime nausea",
            "answer": "Keep taking the iron supplement at night if that avoids nausea, and pair it with the clinician's existing dosing guidance.",
            "checklist": ["Clarify what 'it' refers to.", "Ask about the side effect driving the schedule."],
        },
        {
            "referent": "the physical therapy routine",
            "goal": "recover from mild knee irritation without stopping all activity",
            "constraints": "can commit twenty minutes daily but not a full gym session",
            "answer": "Stick with the physical therapy routine and build consistency with the twenty-minute daily block before adding more load.",
            "checklist": ["Clarify the routine under discussion.", "Confirm time limits and recovery goal."],
        },
        {
            "referent": "the caffeine gel",
            "goal": "avoid stomach issues during long rides",
            "constraints": "tolerates caffeine in small amounts but not on an empty stomach",
            "answer": "Use the caffeine gel only with food and test it in training, because the current issue is stomach tolerance rather than race-day performance.",
            "checklist": ["Clarify the item being discussed.", "Ask what causes the concern."],
        },
        {
            "referent": "the earlier bedtime plan",
            "goal": "shift sleep earlier before a new morning-shift job",
            "constraints": "currently falls asleep after midnight and wakes groggy",
            "answer": "Keep the earlier bedtime plan, but move the schedule gradually and anchor the wake time for the new morning shift.",
            "checklist": ["Clarify which routine change is under discussion.", "Confirm current sleep timing."],
        },
        {
            "referent": "the home strength circuit",
            "goal": "rebuild baseline strength after six sedentary months",
            "constraints": "only has resistance bands and two adjustable dumbbells",
            "answer": "Use the home strength circuit because it matches the available equipment and current deconditioned baseline.",
            "checklist": ["Clarify the routine being referenced.", "Confirm equipment limits."],
        },
        {
            "referent": "the breathing drill",
            "goal": "reduce panic during presentations",
            "constraints": "needs something discreet that can be done minutes before speaking",
            "answer": "Keep the breathing drill and rehearse it before presentations because the main need is a discreet pre-talk reset.",
            "checklist": ["Clarify the technique under discussion.", "Confirm when it needs to be used."],
        },
    ],
    "software": [
        {
            "referent": "the auth module",
            "goal": "ship token refresh support this sprint",
            "constraints": "refresh logic is duplicated in two services and caused one recent bug",
            "answer": "Refactor the auth module before shipping because the duplicated refresh logic already caused a bug and is the main release risk.",
            "checklist": ["Clarify which component 'it' refers to.", "Ask about release risk and duplication."],
        },
        {
            "referent": "the analytics job",
            "goal": "cut nightly runtime below thirty minutes",
            "constraints": "current bottleneck is a full-table scan on every run",
            "answer": "Rework the analytics job around incremental processing because the full-table scan is the clear runtime bottleneck.",
            "checklist": ["Clarify the system under discussion.", "Confirm the performance bottleneck."],
        },
        {
            "referent": "the React migration",
            "goal": "modernize the dashboard without freezing feature work",
            "constraints": "only one engineer knows the old templating system well",
            "answer": "Phase the React migration behind feature boundaries instead of a big-bang rewrite because team knowledge of the old stack is concentrated.",
            "checklist": ["Clarify the initiative being referenced.", "Ask about staffing and migration risk."],
        },
        {
            "referent": "the cache invalidation layer",
            "goal": "fix stale product data after checkout",
            "constraints": "the bug only appears under concurrent updates",
            "answer": "Audit and simplify the cache invalidation layer first, because concurrent updates are producing stale reads after checkout.",
            "checklist": ["Clarify the subsystem under discussion.", "Confirm the failure mode."],
        },
        {
            "referent": "the monorepo split",
            "goal": "speed up CI and isolate deploys by team",
            "constraints": "shared types are changing weekly across services",
            "answer": "Delay the monorepo split until the shared types stabilize, because weekly interface churn would create immediate coordination overhead.",
            "checklist": ["Clarify the architectural change under discussion.", "Ask about dependency churn."],
        },
        {
            "referent": "the search prototype",
            "goal": "demo semantic search for the sales team next month",
            "constraints": "no labeled relevance data exists yet",
            "answer": "Ship a narrow search prototype with a curated evaluation set first because there is no labeled relevance data for a broader launch.",
            "checklist": ["Clarify the project being referenced.", "Confirm data limitations and timeline."],
        },
    ],
    "shopping": [
        {
            "referent": "the 13-inch laptop",
            "goal": "replace an aging machine for travel-heavy consulting work",
            "constraints": "needs long battery life, 16GB RAM, and lightweight carry",
            "answer": "Skip the 13-inch laptop if it only has 8GB RAM; buy the lightest 16GB model that still delivers strong battery life for travel work.",
            "checklist": ["Clarify which product 'it' refers to.", "Ask about travel and memory needs."],
        },
        {
            "referent": "the mirrorless camera body",
            "goal": "start with portraits and indoor family photos",
            "constraints": "budget is tight and lens selection matters more than exotic video features",
            "answer": "Choose the mirrorless body with the strongest affordable portrait-lens ecosystem rather than paying extra for unused video features.",
            "checklist": ["Clarify the product under discussion.", "Confirm the actual shooting priority."],
        },
        {
            "referent": "the standing desk converter",
            "goal": "reduce back pain in a small apartment office",
            "constraints": "cannot fit a full motorized desk and needs room for two monitors",
            "answer": "Buy the standing desk converter because it fits the apartment and keeps the dual-monitor setup without replacing the whole desk.",
            "checklist": ["Clarify the item being referenced.", "Ask about space and monitor constraints."],
        },
        {
            "referent": "the used road bike",
            "goal": "commute fifteen miles daily with occasional weekend rides",
            "constraints": "needs reliable gearing and fender clearance more than race performance",
            "answer": "Buy the used road bike only if it has clearance for commuting accessories and a recent service history, because reliability matters more than speed.",
            "checklist": ["Clarify the product being considered.", "Confirm the commute-oriented requirements."],
        },
        {
            "referent": "the air purifier",
            "goal": "reduce bedroom allergies during spring",
            "constraints": "room is small and noise at night matters a lot",
            "answer": "Choose the quieter purifier sized for a small bedroom instead of the loud higher-throughput model, because nighttime noise is the tighter constraint.",
            "checklist": ["Clarify which product is under discussion.", "Ask about room size and noise tolerance."],
        },
        {
            "referent": "the espresso grinder",
            "goal": "upgrade home espresso consistency",
            "constraints": "kitchen space is tight and single-dosing is preferred",
            "answer": "Pick the compact single-dose grinder because counter space and espresso consistency matter more than hopper capacity.",
            "checklist": ["Clarify the tool under discussion.", "Confirm the space and workflow priorities."],
        },
    ],
}


LEXICAL_BANK = {
    "travel": [
        ("pass", "a national rail pass for intercity train travel", "Buy the rail pass because you will make multiple long-distance train trips in one week."),
        ("class", "the train seat class rather than a tour package", "Choose standard seat class because comfort matters, but first class would not justify the price gap on this route."),
        ("connection", "a flight layover connection rather than a social contact", "Take the longer connection because a missed layover would disrupt the trip more than a short wait."),
        ("base", "the home city for day trips rather than a military base", "Use Osaka as the trip base because it supports cheaper lodging and easy day trips."),
        ("transfer", "moving between airports rather than changing hotels", "Avoid the airport transfer if the savings are small because the cross-city move adds too much risk."),
        ("direct", "a nonstop flight rather than booking direct with the airline", "Prefer the nonstop flight because the traveler needs the most reliable arrival option."),
    ],
    "finance": [
        ("principal", "the loan principal rather than school principal income", "Pay down the loan principal once the emergency fund is complete because reducing interest cost is the next priority."),
        ("fund", "a low-cost index fund rather than a sinking-fund envelope", "Use the low-cost index fund for long-term investing because the money is not needed soon."),
        ("charge", "the annual credit-card fee rather than an accusation", "Keep the card only if the annual fee is offset by benefits you will actually use."),
        ("bond", "a short-term bond fund rather than a single bond", "Use a short-term bond fund because the savings are earmarked for a near-term purchase."),
        ("margin", "brokerage leverage rather than business profit margin", "Avoid margin entirely because the goal is stable investing, not leveraged risk."),
        ("yield", "the savings-account yield rather than crop yield", "Move the cash to the higher-yield savings account because liquidity still matters."),
    ],
    "health": [
        ("load", "weekly training load rather than a literal object load", "Increase training load gradually because the current goal is steady progression without flare-ups."),
        ("dose", "the caffeine dose rather than a prescription change", "Keep the caffeine dose low and test it with food before race day."),
        ("cycle", "a sleep cycle rather than a menstrual cycle", "Protect complete sleep cycles by moving bedtime earlier in smaller increments."),
        ("set", "a set of strength exercises rather than a fixed plan", "Add one more exercise set only if recovery remains solid the next day."),
        ("recovery", "athletic recovery rather than disease recovery", "Prioritize lighter recovery sessions because the recent fatigue points to overload."),
        ("baseline", "baseline fitness rather than lab baseline values", "Rebuild baseline fitness with short consistent sessions before adding intensity."),
    ],
    "software": [
        ("branch", "a git branch rather than a roadmap branch", "Open a separate git branch for the risky fix so it does not block the safer sprint work."),
        ("stack", "the application stack rather than a call stack trace", "Use the existing Python backend stack because the team can ship faster without a rewrite."),
        ("service", "a deployable backend service rather than customer service", "Split the workload into a separate service only if the scaling profile truly differs from the main app."),
        ("port", "a network port rather than moving the app to a new platform", "Keep the existing network port mapping and fix the proxy rules instead of changing ports again."),
        ("state", "frontend application state rather than server geography", "Centralize the frontend state because duplicate local state is causing stale UI behavior."),
        ("pipeline", "the CI pipeline rather than a data-processing system", "Shorten the CI pipeline before adding more checks because latency is already blocking merges."),
    ],
    "shopping": [
        ("case", "a protective carrying case rather than the reason for buying", "Buy the padded carrying case because travel protection is the real use case."),
        ("support", "warranty support rather than ergonomic support", "Pay extra for stronger warranty support because the device is for client work."),
        ("light", "lighter carry weight rather than a built-in light", "Choose the lighter model because you will carry it daily."),
        ("fit", "shoe fit rather than style fit", "Pick the pair with the better fit even if the style is less interesting."),
        ("cover", "insurance coverage rather than a physical cover", "Add the accidental-damage coverage because replacement cost would be painful."),
        ("band", "watch band rather than product tier banding", "Buy the more comfortable watch band because all-day wear matters more than appearance."),
    ],
}


REFERENTIAL_REQUESTS = {
    "travel": "Should I book it now or wait another week?",
    "finance": "Should I move it now or leave it where it is?",
    "health": "Is it okay to keep doing it this way?",
    "software": "Can I ship it this sprint or should I refactor first?",
    "shopping": "Should I keep it or switch to something else?",
}

UNDERSPECIFIED_REQUESTS = {
    "travel": "Help me plan this trip.",
    "finance": "What should I do with this money?",
    "health": "What routine should I follow?",
    "software": "What should I build here?",
    "shopping": "What should I buy?",
}

MISSING_CONTEXT_REQUESTS = {
    "travel": "Can you make this plan work?",
    "finance": "Can you help me make this decision?",
    "health": "Can you help me manage this better?",
    "software": "How should I solve this release problem?",
    "shopping": "Help me choose the right option.",
}

EXTRA_DEV_FAMILIES = {
    ("referential", "travel"),
    ("referential", "finance"),
    ("referential", "health"),
    ("lexical", "software"),
    ("lexical", "shopping"),
    ("lexical", "travel"),
    ("underspecified", "finance"),
    ("underspecified", "software"),
    ("missing_context", "health"),
    ("missing_context", "shopping"),
}


def _scenario_hidden_text(scenario: dict[str, str]) -> str:
    return (
        f"Goal: {scenario['goal']}\n"
        f"Constraints: {scenario['constraints']}\n"
        f"Reference object: {scenario['referent']}\n"
        f"Recommended action: {scenario['answer']}"
    )


def _referential_example(domain: str, index: int, scenario: dict[str, str]) -> DialogueExample:
    return DialogueExample(
        example_id=f"clarifybench-v1-{index:03d}",
        dataset_name="clarifybench",
        user_request=REFERENTIAL_REQUESTS[domain],
        hidden_context=_scenario_hidden_text(scenario),
        gold_clarification_needed=True,
        gold_answer=scenario["answer"],
        simulated_user_reply=(
            f"I mean {scenario['referent']} for {scenario['goal']}, and the main constraint is that it {scenario['constraints']}."
        ),
        ambiguity_type="referential",
        domain=domain,
        checklist=scenario["checklist"],
        metadata={"family": "referential", "domain": domain},
    )


def _underspecified_example(domain: str, index: int, scenario: dict[str, str]) -> DialogueExample:
    return DialogueExample(
        example_id=f"clarifybench-v1-{index:03d}",
        dataset_name="clarifybench",
        user_request=UNDERSPECIFIED_REQUESTS[domain],
        hidden_context=_scenario_hidden_text(scenario),
        gold_clarification_needed=True,
        gold_answer=scenario["answer"],
        simulated_user_reply=(
            f"The context is {scenario['goal']}, and the main constraints are that it {scenario['constraints']}."
        ),
        ambiguity_type="underspecified",
        domain=domain,
        checklist=[
            "Ask about the user's goal.",
            "Ask about the main constraints before answering.",
        ],
        metadata={"family": "underspecified", "domain": domain},
    )


def _missing_context_example(domain: str, index: int, scenario: dict[str, str]) -> DialogueExample:
    return DialogueExample(
        example_id=f"clarifybench-v1-{index:03d}",
        dataset_name="clarifybench",
        user_request=MISSING_CONTEXT_REQUESTS[domain],
        hidden_context=_scenario_hidden_text(scenario),
        gold_clarification_needed=True,
        gold_answer=scenario["answer"],
        simulated_user_reply=(
            f"The hidden context is that this is about {scenario['goal']}, and the constraints are that it {scenario['constraints']}."
        ),
        ambiguity_type="missing_context",
        domain=domain,
        checklist=[
            "Ask for the missing context driving the decision.",
            "Surface the concrete constraints before recommending a plan.",
        ],
        metadata={"family": "missing_context", "domain": domain},
    )


def _lexical_example(domain: str, index: int, lexical: tuple[str, str, str]) -> DialogueExample:
    term, meaning, answer = lexical
    return DialogueExample(
        example_id=f"clarifybench-v1-{index:03d}",
        dataset_name="clarifybench",
        user_request=f"What should I do about the {term} here?",
        hidden_context=f"The ambiguous term '{term}' refers to {meaning}. Recommended action: {answer}",
        gold_clarification_needed=True,
        gold_answer=answer,
        simulated_user_reply=f"By '{term}', I mean {meaning}.",
        ambiguity_type="lexical",
        domain=domain,
        checklist=[
            f"Clarify what '{term}' means in this context.",
            "Avoid answering until the intended sense is clear.",
        ],
        metadata={"family": "lexical", "domain": domain, "term": term},
    )


def build_clarifybench_v1() -> list[DialogueExample]:
    examples: list[DialogueExample] = []
    example_index = 1
    domains = list(SCENARIO_BANK)
    for variant in range(6):
        for ambiguity_type in ("referential", "lexical", "underspecified", "missing_context"):
            for domain in domains:
                if ambiguity_type == "lexical":
                    example = _lexical_example(domain, example_index, LEXICAL_BANK[domain][variant])
                else:
                    scenario = SCENARIO_BANK[domain][variant]
                    if ambiguity_type == "referential":
                        example = _referential_example(domain, example_index, scenario)
                    elif ambiguity_type == "underspecified":
                        example = _underspecified_example(domain, example_index, scenario)
                    else:
                        example = _missing_context_example(domain, example_index, scenario)
                example.split_name = "test"
                if variant == 0:
                    example.split_name = "dev"
                elif variant == 3 and (ambiguity_type, domain) in EXTRA_DEV_FAMILIES:
                    example.split_name = "dev"
                example.metadata["variant"] = variant + 1
                examples.append(example)
                example_index += 1
    return examples


def write_clarifybench_v1(output_dir: str | Path) -> dict[str, object]:
    rows = build_clarifybench_v1()
    output_dir = Path(output_dir)
    split_map = {
        "full": rows,
        "dev": [row for row in rows if row.split_name == "dev"],
        "test": [row for row in rows if row.split_name == "test"],
    }
    manifest_splits: dict[str, object] = {}
    for split_name, split_rows in split_map.items():
        info = write_dataset_split(
            split_rows,
            output_dir / f"clarifybench_v1_{split_name}.jsonl",
            expected_dataset="clarifybench",
            require_split=(split_name != "full"),
        )
        manifest_splits[split_name] = info

    dataset_stats = {
        "full": manifest_splits["full"]["stats"],
        "dev": manifest_splits["dev"]["stats"],
        "test": manifest_splits["test"]["stats"],
    }
    write_json(dataset_stats, output_dir / "clarifybench_v1_stats.json")
    write_manifest(
        output_dir / "clarifybench_v1_manifest.json",
        dataset_name="clarifybench",
        splits=manifest_splits,
        extra={"version": "clarifybench-v1"},
    )
    return {
        "n_examples": len(rows),
        "dev_examples": len(split_map["dev"]),
        "test_examples": len(split_map["test"]),
    }
