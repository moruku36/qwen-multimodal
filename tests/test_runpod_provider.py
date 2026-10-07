"""Provider/OWUI preparation and official REST v1 contracts; never live effects."""
import asyncio
import base64
from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import runpod_provider as provider
from runpod_trial import main, run_injected
import test_runpod_trial as old

AVAILABLE = importlib.util.find_spec("qmc_runpod") is not None
SOURCE = b'''@app.post('/api/chat/completions')
@app.post('/api/v1/chat/completions')
async def chat_completion(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user),
):
    response = await chat_completion_handler(request, form_data, user)
    return response
app.state.CHAT_COMPLETION_HANDLER = chat_completion
'''


def config(root):
    return dict(schema="runpod-owner-v1", account="dummy-account", account_attested=True,
        image="dummy-image@sha256:" + "1" * 64, gpu_type="NVIDIA A100 80GB PCIe",
        gpu_rate="1.80", storage_rate="0.05", overhead="0", rest_ip="8.8.8.8",
        ssh_executable=str(root / "dummy-ssh"), ssh_key=str(root / "dummy-key-reference"),
        journal_path=str(root / "journal.sqlite"), head_path=str(root / "head.json"),
        journal_domain="provider-fixture", export_root=str(root / "export"),
        subject="fixture-user", owui_source_sha256=hashlib.sha256(SOURCE).hexdigest(), owui_version="0.11.4")


class ConfigTests(unittest.TestCase):
    def test_validation_is_pure_and_reports_missing_names_only(self):
        example = json.loads((Path(__file__).resolve().parents[1] / "docs/runpod-operator.example.json").read_text())
        with patch("getpass.getpass", side_effect=AssertionError("prompt")), patch("pathlib.Path.open", side_effect=AssertionError("filesystem")):
            result = provider.validate_config(example)
        self.assertFalse(result["ready"])
        self.assertIn("account", result["missing"])
        self.assertIn("account_attested", result["invalid"])

    def test_unknown_secret_fields_and_wrong_types_are_refused(self):
        c = config(Path(tempfile.gettempdir()))
        self.assertTrue(provider.validate_config(c)["ready"])
        self.assertFalse(provider.validate_config(dict(c, provider_token="NEVER_ECHO"))["ready"])
        for key, value in [("gpu_type", []), ("gpu_rate", float("nan")), ("account_attested", "true"), ("image", "tag:latest"), ("rest_ip", "127.0.0.1")]:
            self.assertIn(key, provider.validate_config(dict(c, **{key: value}))["invalid"])

    def test_cost_cap_and_reserve_are_validated(self):
        self.assertFalse(provider.validate_config(dict(config(Path(tempfile.gettempdir())), gpu_rate="20"))["ready"])

    def test_missing_existing_state_refuses_before_secret_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = config(Path(tmp))
            with self.assertRaisesRegex(RuntimeError, "EXISTING_APPROVED_PATHS_REQUIRED"), provider.open_injection(c, secret_reader=lambda role: self.fail("prompt")):
                self.fail("yielded")
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_serve_requires_owner_start_before_reading_config(self):
        with patch("pathlib.Path.open", side_effect=AssertionError("config")), patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(main(["serve", "--config", "not-read"]), 2)
            self.assertEqual(json.loads(output.getvalue())["code"], "OWNER_START_REQUIRED")

    def test_validate_cli_exposes_only_expected_missing_field_names(self):
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            code = main(["validate", "--config", str(Path(__file__).resolve().parents[1] / "docs/runpod-operator.example.json")])
        result = json.loads(output.getvalue())
        self.assertEqual(code, 2)
        self.assertEqual(result["code"], "CONFIG_INCOMPLETE")
        self.assertIn("journal_path", result["missing"])

    def test_hook_default_disabled_and_source_mismatch_refused(self):
        with self.assertRaisesRegex(RuntimeError, "OWUI_INTEGRATION_DISABLED"):
            provider.owui_dispatcher.begin(object(), SimpleNamespace(id="fixture-user"))
        with self.assertRaisesRegex(RuntimeError, "VERIFIED_OWUI_CONFIGURATION_REQUIRED"), provider.configure_owui(config(Path(tempfile.gettempdir())), observed_source_sha256="0" * 64, inference_key="unused", control_key="unused", owner_enable=True):
            self.fail("enabled")


@unittest.skipUnless(AVAILABLE, "explicit reviewed operations source required")
class ProviderContracts(unittest.TestCase):
    @contextmanager
    def opened(self, *, good_peer=False, allow_cleanup=False):
        from qmc_runpod.c1_authority import AuthorityJournal
        import test_c1_flow as core
        with tempfile.TemporaryDirectory() as tmp:
            c = config(Path(tmp))
            for key in ["ssh_executable", "ssh_key"]: Path(c[key]).write_text("public dummy reference")
            Path(c["export_root"]).mkdir()
            with AuthorityJournal(c["journal_path"], c["head_path"], key=core.KEY, domain=c["journal_domain"]).open(): pass
            def secret(role):
                if role == "journal signing key hex": return core.KEY.hex()
                return {"provider": "dummy-provider-credential", "model": "dummy-model-credential", "inference": "single-user-public-inference-fixture", "control": "single-user-public-control-fixture"}[role]
            def peer(sid, pod):
                if not good_peer: return {"session_id": "wrong"}
                public_key = b"public-verified-host-fixture-bytes"
                hosts = Path(tmp) / "known_hosts"
                hosts.write_text("[8.8.8.8]:12345 ssh-ed25519 " + base64.b64encode(public_key).decode() + "\n")
                return dict(session_id=sid, pod_id=pod, address="8.8.8.8", port=12345, known_hosts=str(hosts),
                    fingerprint="SHA256:" + base64.b64encode(hashlib.sha256(public_key).digest()).decode().rstrip("="), verified_out_of_band=True)
            def approval(scope):
                if not allow_cleanup: return None
                from qmc_runpod.ondemand import TerminationApproval
                return TerminationApproval("dummy-owner-approval", "terminate", scope.session_id, scope.pod_id, scope.receipt_sha256, scope.expires_at)
            with provider.open_injection(c, secret_reader=secret, peer_reader=peer, approval_reader=approval) as inputs:
                yield c, inputs

    def test_production_provider_to_real_factory_cli_probe_and_cleanup(self):
        with old.fixture() as (_, case, factories), self.opened(good_peer=True, allow_cleanup=True) as (_, inputs):
            result = run_injected(inputs, owner_start=True, simulated=True)
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(len(factories), 1)
            self.assertEqual([m for m, _, _ in case.rest.calls], ["POST", "DELETE", "GET"])
            state = json.dumps(inputs.journal.latest())
            self.assertNotIn(inputs.credentials.provider, state)
            self.assertNotIn(inputs.credentials.model, state)

    def test_existing_journal_memory_injection_and_bad_peer(self):
        with self.opened() as (_, inputs):
            self.assertEqual(repr(inputs), "Injection()")
            self.assertIsNone(inputs.journal.latest())
            with self.assertRaisesRegex(RuntimeError, "EXACT_VERIFIED_PEER_REQUIRED"):
                inputs.enroll_peer("s", "new-pod")

    def test_staged_hook_preserves_source_pin_and_starts_disabled(self):
        c = config(Path(tempfile.gettempdir()))
        plan = provider.stage_owui(SOURCE, c)
        self.assertEqual(plan.original, SOURCE)
        self.assertIn(b"from runpod_provider import owui_dispatcher", plan.hook)
        self.assertIn(b"qmc_ondemand_dispatcher.final_chat", plan.patched)
        with self.assertRaises(Exception): provider.stage_owui(SOURCE + b"\n", c)

    def test_official_rest_v1_methods_paths_and_bearer_contract(self):
        # Based on RunPod official POST /v1/pods, GET/DELETE /v1/pods/{podId}.
        from qmc_runpod import execution_adapters as adapters
        from qmc_runpod.c1_ports import PodSpec, ProviderPlans
        requests = []
        class Response:
            status = 200
            def __init__(self, data): self.data = data
            def read(self, size): data, self.data = self.data, b""; return data
        class Connection:
            sock = None
            def __init__(self, address, timeout): self.reply = None
            def connect(self): pass
            def request(self, method, path, body, headers):
                requests.append((method, path, body, headers))
                self.reply = Response(b'{}')
                if method == "GET": self.reply.status = 404
                if method == "DELETE": self.reply.status = 204
            def getresponse(self): return self.reply
            def close(self): pass
        rest = adapters.RunPodREST(gate=adapters.ExecutionGate(lambda effect: True), address="8.8.8.8", bearer="dummy-provider-credential")
        plans = ProviderPlans(PodSpec(account="dummy-account", image="dummy-image@sha256:" + "1" * 64))
        create = plans.create("s", "a" * 64)
        plans.acknowledge("s", "a" * 64, {"id": "new-pod"}, account="dummy-account")
        sequence = [("create", None, create), ("read", "new-pod", plans.read("s", "new-pod", "b" * 64)), ("terminate", "new-pod", plans.terminate("s", "new-pod", "c" * 64, frozenset({"new-pod"})))]
        with patch.object(adapters, "_PinnedHTTPS", Connection):
            for action, pod, req in sequence:
                effect = adapters.Effect("s", req.operation_id, action, pod, time.monotonic() + 10)
                value = rest.request(effect, req.method, req.path, req.body, threading.Event())
                if action == "read": self.assertIsNone(value)
                if action == "terminate": self.assertTrue(value)
        self.assertEqual([(m, p) for m, p, _, _ in requests], [("POST", "/v1/pods"), ("GET", "/v1/pods/new-pod"), ("DELETE", "/v1/pods/new-pod")])
        payload = json.loads(requests[0][2])
        self.assertEqual(payload["gpuCount"], 1)
        self.assertFalse(payload["interruptible"])
        self.assertEqual(payload["ports"], ["22/tcp"])
        self.assertNotIn("env", payload)
        self.assertTrue(all(h["Authorization"] == "Bearer dummy-provider-credential" for _, _, _, h in requests))

    def test_verified_staged_owui_chat_through_controller_then_owned_cleanup(self):
        from qmc_runpod.production_gateway import ProductionGateway
        with old.fixture() as (_, case, _), self.opened(good_peer=True, allow_cleanup=True) as (c, inputs):
            staged = provider.stage_owui(SOURCE, c)
            module = ModuleType("open_webui.qmc_ondemand_hook")
            module.dispatcher = provider.owui_dispatcher
            package = ModuleType("open_webui")
            observed = []; errors = []; threads = []
            original = ProductionGateway.serve
            @contextmanager
            def serve(gateway, **kwargs):
                with original(gateway, **kwargs) as address:
                    def chat():
                        try:
                            with provider.configure_owui(c, observed_source_sha256=c["owui_source_sha256"], inference_key=inputs.credentials.inference, control_key=inputs.credentials.control, owner_enable=True, _fixture_port=address[1], _response_factory=lambda kind, value: (kind, value)):
                                app = SimpleNamespace(post=lambda path: lambda fn: fn, state=SimpleNamespace())
                                ns = dict(app=app, Request=object, Depends=lambda fn: None, get_verified_user=lambda: None)
                                async def other(*args): return "existing"
                                ns["chat_completion_handler"] = other
                                with patch.dict(sys.modules, {"open_webui": package, "open_webui.qmc_ondemand_hook": module}):
                                    exec(staged.patched, ns)
                                    request = object(); user = SimpleNamespace(id="fixture-user")
                                    payload = {"model": "qwen-27b", "messages": [{"role": "user", "content": "Reply OK."}], "stream": False}
                                    async def run():
                                        try: await ns["app"].state.CHAT_COMPLETION_HANDLER(request, payload, user)
                                        except Exception: observed.append("background_refused")
                                        else: raise AssertionError("background allowed")
                                        try: provider.owui_dispatcher.begin(request, SimpleNamespace(id="wrong-user"))
                                        except Exception: observed.append("wrong_user_refused")
                                        else: raise AssertionError("wrong user allowed")
                                        kind, value = await ns["chat_completion"](request, payload, user)
                                        assert kind == "json" and value["choices"][0]["message"]["content"] == "ok"
                                        observed.append("manual_ok")
                                        assert await ns["chat_completion"](request, dict(payload, model="other"), user) == "existing"
                                    asyncio.run(run())
                        except Exception as error: errors.append(type(error).__name__)
                    thread = threading.Thread(target=chat); threads.append(thread); thread.start()
                    yield address
                    thread.join(2)
            with patch.object(ProductionGateway, "serve", serve):
                result = run_injected(inputs, owner_start=True, simulated=True, owui=True)
            self.assertEqual(errors, [])
            self.assertTrue(all(not t.is_alive() for t in threads))
            self.assertEqual(observed, ["background_refused", "wrong_user_refused", "manual_ok"])
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual([m for m, _, _ in case.rest.calls], ["POST", "DELETE", "GET"])
            self.assertIsNone(provider.owui_dispatcher._target)


if __name__ == "__main__": unittest.main()
