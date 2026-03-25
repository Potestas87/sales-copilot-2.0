import importlib.util
import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_server_protocol():
    return _load_module("server_protocol_test", ROOT / "server" / "protocol.py")


def load_client_protocol():
    return _load_module("client_protocol_test", ROOT / "client" / "protocol.py")


def load_inference_module():
    # inference.py imports llama_cpp at module import time; stub it for tests.
    fake_llama_cpp = types.ModuleType("llama_cpp")

    class DummyLlama:
        pass

    fake_llama_cpp.Llama = DummyLlama
    sys.modules.setdefault("llama_cpp", fake_llama_cpp)

    # prompts.py imports yaml at module import time; keep tests lightweight.
    fake_yaml = types.ModuleType("yaml")
    fake_yaml.safe_load = lambda *_args, **_kwargs: {}
    sys.modules.setdefault("yaml", fake_yaml)

    server_dir = str(ROOT / "server")
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    return _load_module("server_inference_test", ROOT / "server" / "inference.py")


def load_client_websocket_module():
    # websocket_client.py expects to import "protocol" from client/.
    client_dir = str(ROOT / "client")
    if client_dir not in sys.path:
        sys.path.insert(0, client_dir)
    sys.modules.pop("protocol", None)
    return _load_module("client_websocket_test", ROOT / "client" / "websocket_client.py")
