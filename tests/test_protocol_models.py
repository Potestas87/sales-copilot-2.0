import pytest

from tests._loaders import load_client_protocol, load_server_protocol


def test_server_protocol_rejects_invalid_speaker():
    protocol = load_server_protocol()
    with pytest.raises(Exception):
        protocol.UtteranceMessage.model_validate(
            {
                "type": "utterance",
                "speaker": "prospect",
                "sample_rate": 16000,
                "ts_ms": 123,
                "audio_b64": "YWJj",
            }
        )


def test_server_inference_model_has_reasoning_and_latency():
    protocol = load_server_protocol()
    msg = protocol.InferenceMessage(
        speaker="customer",
        transcript="hello",
        intent="question",
        suggestion="test",
        reasoning_short="short reason",
        confidence=0.7,
        latency_ms=1280.5,
    )
    dumped = msg.model_dump()
    assert dumped["reasoning_short"] == "short reason"
    assert dumped["latency_ms"] == pytest.approx(1280.5)


def test_client_protocol_parses_optional_fields():
    protocol = load_client_protocol()
    parsed = protocol.InferenceMessage.from_json(
        '{"type":"inference","speaker":"salesperson","transcript":"t","intent":"none","suggestion":"","reasoning_short":"r","confidence":0.2,"latency_ms":900}'
    )
    assert parsed.speaker == "salesperson"
    assert parsed.reasoning_short == "r"
    assert parsed.latency_ms == pytest.approx(900.0)
