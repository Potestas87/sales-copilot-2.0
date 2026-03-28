from tests._loaders import load_inference_module


def _engine():
    inference = load_inference_module()
    return object.__new__(inference.SuggestionEngine)


def test_parse_response_valid_json_with_prefix():
    engine = _engine()
    raw = 'Here is JSON: {"type":"objection","suggestion":"ROI response","reasoning_short":"price pushback","confidence":0.83}'
    parsed = engine._parse_response(raw, "orig")
    assert parsed["type"] == "objection"
    assert parsed["suggestion"] == "ROI response"
    assert parsed["reasoning_short"] == "price pushback"
    assert parsed["confidence"] == 0.83


def test_parse_response_invalid_type_falls_back_to_none():
    engine = _engine()
    raw = '{"type":"something_else","suggestion":"x","confidence":0.3}'
    parsed = engine._parse_response(raw, "orig")
    assert parsed["type"] == "none"
    assert parsed["confidence"] == 0.3


def test_parse_response_malformed_json_is_safe_default():
    engine = _engine()
    parsed = engine._parse_response("{not-json", "orig")
    assert parsed == {
        "type": "none",
        "suggestion": "",
        "reasoning_short": "",
        "confidence": 0.0,
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
