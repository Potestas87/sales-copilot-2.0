import json

import numpy as np

from tests._loaders import load_client_websocket_module


class FakeLoop:
    def __init__(self, *, closed: bool = False, raise_on_call: bool = False):
        self._closed = closed
        self._raise_on_call = raise_on_call
        self.calls = []

    def is_closed(self):
        return self._closed

    def call_soon_threadsafe(self, fn, *args):
        if self._raise_on_call:
            raise RuntimeError("loop unavailable")
        self.calls.append((fn, args))
        fn(*args)


def test_speaker_routing_serializes_customer_and_salesperson():
    mod = load_client_websocket_module()
    client = mod.WebSocketClient(on_response=lambda _d: None)
    client._running = True
    client._loop = FakeLoop()

    captured = []
    client._enqueue_payload = lambda payload: captured.append(json.loads(payload))

    client.send(np.zeros(320, dtype=np.float32), speaker="customer")
    client.send(np.zeros(320, dtype=np.float32), speaker="salesperson")

    assert [p["speaker"] for p in captured] == ["customer", "salesperson"]
    assert all(p["type"] == "utterance" for p in captured)


def test_send_drops_safely_when_loop_closed():
    mod = load_client_websocket_module()
    client = mod.WebSocketClient(on_response=lambda _d: None)
    client._running = True
    client._loop = FakeLoop(closed=True)

    client.send(np.zeros(320, dtype=np.float32), speaker="customer")
    metrics = client._snapshot_metrics()
    assert metrics["dropped_loop_unavailable"] >= 1


def test_send_handles_call_soon_threadsafe_race():
    mod = load_client_websocket_module()
    client = mod.WebSocketClient(on_response=lambda _d: None)
    client._running = True
    client._loop = FakeLoop(raise_on_call=True)

    # Should not raise even if loop becomes unavailable between checks.
    client.send(np.zeros(320, dtype=np.float32), speaker="customer")
    metrics = client._snapshot_metrics()
    assert metrics["dropped_loop_unavailable"] >= 1
