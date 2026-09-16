from tests._loaders import load_inference_module


def _engine():
    inference = load_inference_module()
    engine = object.__new__(inference.SuggestionEngine)
    engine._always_actionable_customer = True
    return engine


def test_parse_response_valid_json_with_prefix():
    engine = _engine()
    raw = 'Here is JSON: {"type":"objection","suggestion":"ROI response","reasoning_short":"price pushback","confidence":0.83}'
    parsed = engine._parse_response(raw, "orig")
    assert parsed["type"] == "objection"
    assert parsed["suggestion"] == "ROI response"
    assert parsed["reasoning_short"] == "price pushback"
    assert parsed["confidence"] == 0.83


def test_parse_response_invalid_type_with_suggestion_becomes_actionable():
    engine = _engine()
    raw = '{"type":"something_else","suggestion":"x","confidence":0.3}'
    parsed = engine._parse_response(raw, "orig")
    assert parsed["type"] == "question"
    assert parsed["confidence"] == 0.3


def test_parse_response_malformed_json_is_safe_default():
    engine = _engine()
    parsed = engine._parse_response("{not-json", "orig")
    assert parsed == {
        "type": "none",
        "suggestion": "",
        "reasoning_short": "",
        "confidence": 0.0,
        "customer_name": "",
        "address": "",
        "pain_points": [],
        "buying_temperature": "",
    }


def test_parse_response_accepts_wrapped_response_object():
    engine = _engine()
    raw = (
        '{"response":{"type":"question","suggestion":"Clarify integration steps.",'
        '"reasoning_short":"Customer asked implementation details","confidence":0.74}}'
    )
    parsed = engine._parse_response(raw, "orig")
    assert parsed["type"] == "question"
    assert parsed["suggestion"] == "Clarify integration steps."
    assert parsed["reasoning_short"] == "Customer asked implementation details"
    assert parsed["confidence"] == 0.74


def test_parse_response_ignores_prose_before_and_after_json():
    engine = _engine()
    raw = (
        'Sure, here you go:\n{"type":"buying_signal","suggestion":"Offer onboarding timeline.",'
        '"reasoning_short":"Customer asked next steps","confidence":0.9}\nThanks!'
    )
    parsed = engine._parse_response(raw, "orig")
    assert parsed["type"] == "buying_signal"
    assert parsed["suggestion"] == "Offer onboarding timeline."
    assert parsed["reasoning_short"] == "Customer asked next steps"
    assert parsed["confidence"] == 0.9


def test_parse_response_accepts_intent_and_message_aliases():
    engine = _engine()
    raw = '{"intent":"objection","message":"Reframe around ROI and timeline.","reason":"Budget concern","confidence":0.8}'
    parsed = engine._parse_response(raw, "orig")
    assert parsed["type"] == "objection"
    assert parsed["suggestion"] == "Reframe around ROI and timeline."
    assert parsed["reasoning_short"] == "Budget concern"
    assert parsed["confidence"] == 0.8


def test_parse_response_promotes_action_text_when_type_missing():
    engine = _engine()
    raw = '{"action":"Ask one qualifying question before quoting."}'
    parsed = engine._parse_response(raw, "orig")
    assert parsed["type"] == "question"
    assert parsed["suggestion"] == "Ask one qualifying question before quoting."


def test_ensure_actionable_result_promotes_none_to_question():
    engine = _engine()
    result = {"type": "none", "suggestion": "", "reasoning_short": "", "confidence": 0.0}
    promoted = engine._ensure_actionable_result(result, "I need to think about this", [])
    assert promoted["type"] == "question"
    assert "$99 initial" in promoted["suggestion"]
    assert promoted["confidence"] >= 0.35


def test_ensure_actionable_result_respects_feature_flag():
    engine = _engine()
    engine._always_actionable_customer = False
    result = {"type": "none", "suggestion": "", "reasoning_short": "", "confidence": 0.0}
    unchanged = engine._ensure_actionable_result(result, "I need to think about this", [])
    assert unchanged["type"] == "none"
    assert unchanged["suggestion"] == ""


def test_parse_response_repairs_invalid_escaped_underscores():
    engine = _engine()
    raw = '{"intent":"question","message":"total\\_cost is $1,150","confidence":0.7}'
    parsed = engine._parse_response(raw, "orig")
    assert parsed["type"] == "question"
    assert parsed["suggestion"] == "total_cost is $1,150"
    assert parsed["confidence"] == 0.7


def test_parse_response_repairs_other_invalid_escapes():
    engine = _engine()
    raw = '{"intent":"question","message":"month\\-to\\-month is not offered","confidence":0.6}'
    parsed = engine._parse_response(raw, "orig")
    assert parsed["type"] == "question"
    assert parsed["suggestion"] == "month-to-month is not offered"
    assert parsed["confidence"] == 0.6


def test_business_rule_spouse_smokescreen_gets_pullback_question():
    engine = _engine()
    result = {"type": "none", "suggestion": "", "reasoning_short": "", "confidence": 0.1}
    out = engine._ensure_actionable_result(result, "I need to go talk to my husband first.", [])
    assert out["type"] == "objection"
    assert "what would your husband/wife need to hear" in out["suggestion"].lower()


def test_business_rule_pricing_question_gets_pricing_anchors():
    engine = _engine()
    result = {"type": "none", "suggestion": "", "reasoning_short": "", "confidence": 0.1}
    out = engine._ensure_actionable_result(result, "How much total would this cost?", [])
    assert out["type"] == "question"
    assert "$175" in out["suggestion"]
    assert "$150" in out["suggestion"]
    assert "24 months" in out["suggestion"]
    assert "18" in out["suggestion"]
    assert "12" in out["suggestion"]


def test_business_rule_blocks_terms_under_12_months():
    engine = _engine()
    result = {
        "type": "question",
        "suggestion": "We can do 2 months if that helps.",
        "reasoning_short": "",
        "confidence": 0.4,
    }
    out = engine._ensure_actionable_result(result, "Can I do two months?", [])
    assert out["type"] == "objection"
    assert "don't offer terms below 12 months" in out["suggestion"].lower()


def test_offer_progress_tracks_best_and_steps():
    engine = _engine()
    turns = [
        {"speaker": "salesperson", "transcript": "We start at 24 months, $175 initial and $150 bimonthly."},
        {"speaker": "salesperson", "transcript": "I can do $99 initial and keep bimonthly at $150."},
        {"speaker": "salesperson", "transcript": "If needed, I can do $99 initial and $120 bimonthly."},
    ]
    progress = engine._derive_offer_progress(turns)
    assert progress["best"]["initial"] == 99
    assert progress["best"]["bimonthly"] == 120
    assert progress["best"]["term_months"] == 24
    assert progress["rac_steps"] >= 2


def test_business_rule_prevents_regressive_offer():
    engine = _engine()
    turns = [
        {"speaker": "salesperson", "transcript": "I offered $99 initial and $120 bimonthly."},
    ]
    result = {
        "type": "question",
        "suggestion": "Let's do $175 initial and $150 every two months on 24 months.",
        "reasoning_short": "",
        "confidence": 0.7,
    }
    out = engine._ensure_actionable_result(result, "What can you do?", turns)
    assert "$99 initial" in out["suggestion"]
    assert "$120 every two months" in out["suggestion"]


def test_rac_fallback_goes_to_quarterly_after_three_steps():
    engine = _engine()
    turns = [
        {"speaker": "salesperson", "transcript": "I can do $99 initial and $150 bimonthly."},
        {"speaker": "salesperson", "transcript": "I can also do 18 months."},
        {"speaker": "salesperson", "transcript": "I can do $59 initial and $99 bimonthly."},
    ]
    result = {"type": "none", "suggestion": "", "reasoning_short": "", "confidence": 0.2}
    out = engine._ensure_actionable_result(result, "Still too high for me.", turns)
    assert "quarterly" in out["suggestion"].lower()


# ── Deal notepad: extraction, reattachment, merge, package summary ────────────

def test_parse_response_extracts_notepad_fields_when_present():
    engine = _engine()
    raw = (
        '{"type":"question","suggestion":"Sure.","confidence":0.7,'
        '"customer_name":"Sarah","address":"123 Main St","pain_points":["worried about setup time"]}'
    )
    parsed = engine._parse_response(raw, "orig")
    assert parsed["customer_name"] == "Sarah"
    assert parsed["address"] == "123 Main St"
    assert parsed["pain_points"] == ["worried about setup time"]


def test_parse_response_defaults_notepad_fields_when_absent():
    engine = _engine()
    parsed = engine._parse_response('{"type":"question","suggestion":"Sure."}', "orig")
    assert parsed["customer_name"] == ""
    assert parsed["address"] == ""
    assert parsed["pain_points"] == []


def test_parse_response_malformed_json_still_has_notepad_defaults():
    engine = _engine()
    parsed = engine._parse_response("{not-json", "orig")
    assert parsed["customer_name"] == ""
    assert parsed["address"] == ""
    assert parsed["pain_points"] == []


def test_parse_response_ignores_non_list_pain_points():
    engine = _engine()
    parsed = engine._parse_response(
        '{"type":"question","suggestion":"Sure.","pain_points":"not a list"}', "orig"
    )
    assert parsed["pain_points"] == []


def test_finalize_result_reattaches_extraction_after_business_rule_override():
    engine = _engine()
    # Simulate a guardrail override (a fresh dict, as _apply_business_rules returns)
    # that knows nothing about extraction fields.
    guardrail_result = {
        "type": "objection",
        "suggestion": "Let's stay at your best offered terms so far.",
        "reasoning_short": "Prevented regressive offer.",
        "confidence": 0.75,
    }
    parsed = {
        "type": "question",
        "suggestion": "original",
        "customer_name": "Sarah",
        "address": "123 Main St",
        "pain_points": ["worried about setup time"],
    }
    out = engine._finalize_result(guardrail_result, parsed)
    assert out["type"] == "objection"  # guardrail's type/suggestion win
    assert out["suggestion"] == "Let's stay at your best offered terms so far."
    assert out["customer_name"] == "Sarah"  # extraction fields still survive
    assert out["address"] == "123 Main St"
    assert out["pain_points"] == ["worried about setup time"]


def test_merge_notepad_overwrites_only_on_nonempty():
    engine = _engine()
    notepad = {"customer_name": "Sarah", "address": "", "pain_points": []}
    out = engine.merge_notepad(notepad, {"customer_name": "", "address": "123 Main St"})
    assert out["customer_name"] == "Sarah"  # not regressed by empty update
    assert out["address"] == "123 Main St"


def test_merge_notepad_dedupes_and_caps_pain_points():
    engine = _engine()
    notepad = {"customer_name": "", "address": "", "pain_points": []}
    for i in range(8):
        notepad = engine.merge_notepad(notepad, {"pain_points": [f"concern {i}"]})
    notepad = engine.merge_notepad(notepad, {"pain_points": ["concern 3"]})  # duplicate
    assert len(notepad["pain_points"]) == 6
    assert notepad["pain_points"] == [f"concern {i}" for i in range(2, 8)]


def test_describe_best_offer_formats_current_ladder():
    engine = _engine()
    turns = [
        {"speaker": "salesperson", "transcript": "We start at 24 months, $175 initial and $150 bimonthly."},
        {"speaker": "salesperson", "transcript": "I can do $99 initial and keep bimonthly at $150."},
    ]
    summary = engine.describe_best_offer(turns)
    assert "$99 initial" in summary
    assert "$150/2mo" in summary
    assert "24-month term" in summary


def test_describe_best_offer_defaults_with_no_turns():
    engine = _engine()
    summary = engine.describe_best_offer([])
    assert "$175 initial" in summary
    assert "24-month term" in summary


# ── Buying temperature ─────────────────────────────────────────────────────────

def test_parse_response_accepts_valid_buying_temperature():
    engine = _engine()
    parsed = engine._parse_response(
        '{"type":"buying_signal","suggestion":"Let\'s get started.","buying_temperature":"hot"}',
        "orig",
    )
    assert parsed["buying_temperature"] == "hot"


def test_parse_response_rejects_invalid_buying_temperature():
    engine = _engine()
    parsed = engine._parse_response(
        '{"type":"question","suggestion":"Sure.","buying_temperature":"lukewarm"}', "orig"
    )
    assert parsed["buying_temperature"] == ""


def test_merge_notepad_buying_temperature_overwrites_only_on_valid_value():
    engine = _engine()
    notepad = {"customer_name": "", "address": "", "pain_points": [], "buying_temperature": ""}
    out = engine.merge_notepad(notepad, {"buying_temperature": "hot"})
    assert out["buying_temperature"] == "hot"

    # Empty/invalid updates never regress a known temperature.
    out = engine.merge_notepad(out, {"buying_temperature": ""})
    assert out["buying_temperature"] == "hot"
    out = engine.merge_notepad(out, {"buying_temperature": "bogus"})
    assert out["buying_temperature"] == "hot"


# ── Deterministic notepad-field triggers ───────────────────────────────────────

def test_detect_triggered_fields_address_ask_then_given_captures_full_answer():
    engine = _engine()
    prior = [{"speaker": "salesperson", "transcript": "Great, what's your service address?"}]
    out = engine.detect_triggered_fields(prior, "customer", "123 Main Street, Springfield, IL 62704")
    assert out["address"] == "123 Main Street, Springfield, IL 62704"


def test_detect_triggered_fields_standalone_address_pattern():
    engine = _engine()
    out = engine.detect_triggered_fields([], "customer", "you can reach me at 456 Oak Avenue anytime")
    assert out["address"] == "456 Oak Avenue"


def test_detect_triggered_fields_name_prefers_clean_standalone_match():
    engine = _engine()
    prior = [{"speaker": "salesperson", "transcript": "Can I get your name?"}]
    out = engine.detect_triggered_fields(prior, "customer", "this is Sarah Jennings")
    assert out["customer_name"] == "Sarah Jennings"


def test_detect_triggered_fields_name_falls_back_to_full_answer_when_asked():
    engine = _engine()
    prior = [{"speaker": "salesperson", "transcript": "Can I get your name?"}]
    out = engine.detect_triggered_fields(prior, "customer", "Sarah Jennings, nice to meet you")
    assert out["customer_name"] == "Sarah Jennings, nice to meet you"


def test_detect_triggered_fields_pain_point_ask_then_given():
    engine = _engine()
    prior = [{"speaker": "salesperson", "transcript": "What's holding you back from moving forward?"}]
    out = engine.detect_triggered_fields(prior, "customer", "I just don't want a long contract")
    assert out["pain_points"] == ["I just don't want a long contract"]


def test_detect_triggered_fields_ignores_salesperson_turns():
    engine = _engine()
    out = engine.detect_triggered_fields([], "salesperson", "my name is Bob and my address is 1 Elm St")
    assert out == {"customer_name": "", "address": "", "pain_points": []}


def test_detect_triggered_fields_no_match_without_pattern_or_ask():
    engine = _engine()
    out = engine.detect_triggered_fields([], "customer", "I think that works for us")
    assert out == {"customer_name": "", "address": "", "pain_points": []}


# ── Pain-point keyword detection (pest control domain) ─────────────────────────

def test_detect_triggered_fields_captures_pest_mentions():
    engine = _engine()
    out = engine.detect_triggered_fields([], "customer", "We've been seeing a lot of ants and spiders lately")
    assert out["pain_points"] == ["pest: ants", "pest: spiders"]


def test_detect_triggered_fields_canonicalizes_pest_variants():
    engine = _engine()
    out = engine.detect_triggered_fields([], "customer", "we have mice and also a couple rats in the garage")
    assert out["pain_points"] == ["pest: rodents"]


def test_detect_triggered_fields_captures_price_concern():
    engine = _engine()
    out = engine.detect_triggered_fields([], "customer", "Honestly that's too expensive for us right now")
    assert "price concern" in out["pain_points"]


def test_detect_triggered_fields_captures_safety_concern():
    engine = _engine()
    out = engine.detect_triggered_fields([], "customer", "Is this safe to use with my dog and kids around?")
    assert "pet/people safety concern" in out["pain_points"]


def test_detect_triggered_fields_combines_pest_and_price_capped_at_three():
    engine = _engine()
    out = engine.detect_triggered_fields(
        [], "customer", "We have ants, spiders, and roaches, and honestly it's too expensive"
    )
    assert len(out["pain_points"]) == 3
    assert out["pain_points"][:3] == ["pest: ants", "pest: spiders", "pest: roaches"]


def test_detect_triggered_fields_salesperson_pest_mention_ignored():
    engine = _engine()
    out = engine.detect_triggered_fields([], "salesperson", "A lot of customers deal with ants and spiders")
    assert out["pain_points"] == []


# ── Frequency objection guardrail fix ───────────────────────────────────────────

def test_frequency_objection_is_not_hijacked_by_pricing_override():
    engine = _engine()
    progress = engine._derive_offer_progress([])
    result = {"type": "objection", "suggestion": "", "reasoning_short": "", "confidence": 0.5}
    out = engine._apply_business_rules(result, "I don't want monthly treatments, maybe a couple times a year", progress)
    assert out is None  # defers to the LLM instead of the canned pricing paragraph


def test_pure_pricing_question_still_gets_deterministic_answer():
    engine = _engine()
    progress = engine._derive_offer_progress([])
    result = {"type": "question", "suggestion": "", "reasoning_short": "", "confidence": 0.5}
    out = engine._apply_business_rules(result, "How much does this cost per month?", progress)
    assert out is not None
    assert "$175" in out["suggestion"]


# ── Quarterly service-frequency tier ────────────────────────────────────────────

def test_detect_offer_from_text_captures_quarterly_frequency():
    engine = _engine()
    detected = engine._detect_offer_from_text("We can switch you to quarterly visits instead.")
    assert detected["service_frequency"] == "quarterly"


def test_derive_offer_progress_tracks_quarterly_after_mention():
    engine = _engine()
    turns = [{"speaker": "salesperson", "transcript": "I can move you to quarterly treatments instead."}]
    progress = engine._derive_offer_progress(turns)
    assert progress["best"]["service_frequency"] == "quarterly"


def test_describe_best_offer_reflects_quarterly_tier():
    engine = _engine()
    turns = [{"speaker": "salesperson", "transcript": "Let's do quarterly visits at $150."}]
    summary = engine.describe_best_offer(turns)
    assert "quarterly" in summary.lower()


def test_pricing_override_uses_quarterly_cadence_when_active():
    engine = _engine()
    turns = [{"speaker": "salesperson", "transcript": "I can move you to quarterly treatments instead."}]
    progress = engine._derive_offer_progress(turns)
    result = {"type": "question", "suggestion": "", "reasoning_short": "", "confidence": 0.5}
    out = engine._apply_business_rules(result, "How much does that cost?", progress)
    assert "quarterly" in out["suggestion"].lower()


# ── Deterministic buying temperature ────────────────────────────────────────────

def test_buying_temperature_hot_cue_overrides_history():
    engine = _engine()
    out = engine.detect_buying_temperature("Ok let's do it, how do we get started?", ["objection", "objection"])
    assert out == "hot"


def test_buying_temperature_cold_cue_overrides_history():
    engine = _engine()
    out = engine.detect_buying_temperature("I'll think about it and get back to you", ["buying_signal"])
    assert out == "cold"


def test_buying_temperature_hot_from_repeated_buying_signal_intents():
    engine = _engine()
    out = engine.detect_buying_temperature("What's included in that?", ["question", "buying_signal", "buying_signal"])
    assert out == "hot"


def test_buying_temperature_hot_when_latest_turn_is_buying_signal():
    engine = _engine()
    out = engine.detect_buying_temperature("Sounds good", ["objection", "question", "buying_signal"])
    assert out == "hot"


def test_buying_temperature_cold_from_repeated_objections_without_buying_signal():
    engine = _engine()
    out = engine.detect_buying_temperature("I'm still not sure about this", ["objection", "objection", "question"])
    assert out == "cold"


def test_buying_temperature_warm_default_with_mixed_signal():
    engine = _engine()
    out = engine.detect_buying_temperature("How does onboarding work?", ["objection", "buying_signal", "question"])
    assert out == "warm"


def test_buying_temperature_empty_with_no_history():
    engine = _engine()
    out = engine.detect_buying_temperature("hello", [])
    assert out == ""
