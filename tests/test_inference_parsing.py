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
