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
    assert "quarterly billing" in out["suggestion"].lower()


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
