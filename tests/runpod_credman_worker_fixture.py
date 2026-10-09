"""Isolated synthetic worker harness. No live CLI mode, store, network or GPU."""

import base64
import ctypes  # noqa: F401 - stdlib initialization before auditing native effect calls
import hashlib
import importlib.util
import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

request = json.loads(sys.stdin.buffer.read())
tempfile.tempdir = str(Path(request["config"]).parent)
helpers = request.pop("test_helpers")
for path in helpers:
    sys.path.insert(0, path)
spec = importlib.util.spec_from_file_location(
    "synthetic_bootstrap", Path(request["qwen"]) / "scripts/runpod_credman_startup.py"
)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
events, store_calls, rest_methods, owner_calls = [], [], [], []
failures = []


def trace(frame, event, arg):
    if frame.f_code.co_name == "_consume" and event == "exception":
        failures.append({"type": type(arg[1]).__name__, "line": frame.f_lineno})
    return trace


sys.settrace(trace)
buffer = bytearray(b"dummy-provider-credential")
active = []


def audit(event, args):
    if event in {"ctypes.dlopen", "socket.connect", "socket.bind", "subprocess.Popen"}:
        raise AssertionError("LIVE_EFFECT_FORBIDDEN")


sys.addaudithook(audit)


@contextmanager
def synthetic_context(credential, provider, trial):
    from qmc_runpod.production_gateway import ProductionGateway

    from test_runpod_trial import fixture

    class Store:
        def __init__(self, target, *, enabled):
            assert target == "AIEngineeringFactory/RunPod/provider/v2" and enabled is True

        def read(self):
            store_calls.append(True)
            return buffer

    class Socket:
        def __init__(self):
            self.wire = bytearray()

    @contextmanager
    def serve(gateway, **kwargs):
        active.append(gateway)
        yield ("127.0.0.1", 0)

    @contextmanager
    def unwatched():
        yield

    def http(address, path, key, payload=None, headers=None):
        gateway = active[-1]
        gateway._authorize(path, {"authorization": "Bearer " + key})
        if path == "/healthz":
            return {"alive": True}
        if path == "/v1/models":
            return {"data": [{"id": "qwen-27b"}]}
        headers = {k.lower(): v for k, v in headers.items()}
        sock = Socket()
        with (
            patch.object(
                gateway,
                "_json_write",
                side_effect=lambda s, code, value, **kw: s.wire.extend(json.dumps(value).encode()),
            ),
            patch.object(gateway, "_watch_client", return_value=unwatched()),
            patch.object(gateway, "_guard", side_effect=lambda s, ticket: gateway.host.chat_fence(ticket)),
        ):
            gateway._chat(sock, payload, headers["x-request-id"], headers["x-intent-token"], [False])
        return json.loads(sock.wire)

    with (
        fixture() as (_, external, factories),
        patch.object(credential, "current_windows_sid", return_value="synthetic-owner"),
        patch.object(credential, "WindowsCredentialStore", Store),
        patch.object(ProductionGateway, "serve", serve),
        patch.object(trial, "_http", side_effect=http),
    ):
        yield
        rest_methods.extend(method for method, _, _ in external.rest.calls)


@contextmanager
def synthetic_console():
    import test_c1_flow as core
    from test_single_user_host import CONTROL, INFERENCE

    def secret(role):
        owner_calls.append(role)
        return {
            "model": "dummy-model-credential",
            "inference": INFERENCE,
            "control": CONTROL,
            "journal signing key hex": core.KEY.hex(),
        }[role]

    def answer(prompt):
        if prompt.startswith("Verified NEW peer proof JSON for "):
            owner_calls.append("peer")
            sid, pod = prompt[len("Verified NEW peer proof JSON for ") : -2].split("/")
            public = b"public-verified-host-fixture-bytes"
            hosts = Path(request["config"]).parent / "known_hosts"
            hosts.write_text("[8.8.8.8]:12345 ssh-ed25519 " + base64.b64encode(public).decode() + "\n")
            return json.dumps(
                dict(
                    session_id=sid,
                    pod_id=pod,
                    address="8.8.8.8",
                    port=12345,
                    known_hosts=str(hosts),
                    fingerprint="SHA256:"
                    + base64.b64encode(hashlib.sha256(public).digest()).decode().rstrip("="),
                    verified_out_of_band=True,
                )
            )
        if prompt.startswith("Exact terminate approval; type "):
            owner_calls.append("terminate")
            return prompt[len("Exact terminate approval; type ") : -2]
        raise AssertionError("unexpected owner input")

    yield secret, answer


status = bootstrap._consume(
    request, events.append, _test_context=synthetic_context, _console_factory=synthetic_console
)
print(
    json.dumps(
        {
            "status": status,
            "events": events,
            "reads": len(store_calls),
            "wiped": buffer == bytearray(len(buffer)),
            "methods": rest_methods,
            "owner_inputs": owner_calls,
            "failures": failures,
        }
    )
)
