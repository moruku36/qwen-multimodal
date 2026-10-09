"""Real startup routing/provider seam and supervised synthetic child contracts."""

import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import runpod_credman_startup as startup
import runpod_provider as provider
from runpod_trial import main

if __package__:
    from .test_runpod_provider import config
else:
    from test_runpod_provider import config


class StartupTests(unittest.TestCase):
    def test_owner_console_separates_input_restores_identity_and_eof_refuses(self):
        incoming, outgoing = io.StringIO("public-owner-answer\n"), io.StringIO()
        before = sys.stdin, sys.__stdin__
        seen = []

        def hidden(prompt, stream):
            self.assertIs(sys.stdin, sys.__stdin__)
            self.assertIs(stream, outgoing)
            seen.append(prompt)
            return "synthetic-hidden-fixture"

        with (
            patch("builtins.open", side_effect=[incoming, outgoing]),
            patch.object(startup.getpass, "getpass", side_effect=hidden),
            startup._owner_console() as (secret, answer),
        ):
            self.assertEqual(secret("model"), "synthetic-hidden-fixture")
            self.assertEqual(answer("Public owner confirmation: "), "public-owner-answer")
            with self.assertRaises(EOFError):
                answer("Cancel on EOF: ")
        self.assertEqual((sys.stdin, sys.__stdin__), before)
        self.assertEqual(len(seen), 1)

    def test_default_gate_refuses_before_source_or_credential(self):
        with (
            patch.object(startup, "_policy", side_effect=AssertionError("source must not load")),
            patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            code = main(["start", "--credential-approval", "not-read", "--config", "not-read"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["code"], "OWNER_START_REQUIRED")

    def test_normal_main_routes_approved_start_to_status_only_supervisor(self):
        fake = SimpleNamespace(load_approval=lambda *a, **k: {"claims_dir": str(Path("claims").resolve())})
        seen = []

        def supervised(argv, payload):
            seen.append((argv, payload))
            return dict(status="COMPLETE", worker_exit_confirmed=True, cleanup_required=False)

        with (
            patch.object(startup, "_policy", return_value=fake),
            patch.object(startup, "supervise", side_effect=supervised),
            patch.object(startup, "_owner_terminal", return_value=True),
            patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            output.isatty = lambda: True
            code = main(
                [
                    "start",
                    "--owner-start",
                    "--operations-source",
                    "operations",
                    "--config",
                    "config",
                    "--credential-approval",
                    "approval",
                    "--credential-factory-source",
                    "factory",
                    "--credential-launcher-sha256",
                    "a" * 64,
                    "--credential-claims",
                    "claims",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(seen[0][1]["command"], "start")
        self.assertNotIn("credential", seen[0][1])
        self.assertEqual(json.loads(output.getvalue())["status"], "COMPLETE")

    def test_serve_routes_with_distinct_startup_operation(self):
        with (
            patch.object(startup, "start_once", return_value={"status": "COMPLETE"}) as start,
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            self.assertEqual(
                main(
                    [
                        "serve",
                        "--owner-start",
                        "--operations-source",
                        "operations",
                        "--config",
                        "config",
                        "--credential-approval",
                        "approval",
                    ]
                ),
                0,
            )
        self.assertEqual(start.call_args.args[0].command, "serve")

    def test_provider_once_replaces_only_provider_without_prompt_fallback(self):
        authority = ModuleType("qmc_runpod.c1_authority")
        ports = ModuleType("qmc_runpod.c1_ports")
        adapters = ModuleType("qmc_runpod.execution_adapters")

        @contextmanager
        def opened():
            yield object()

        authority.AuthorityJournal = lambda *a, **k: SimpleNamespace(open=opened)
        ports.PodSpec = lambda **k: k
        ports.Rate = lambda *a: a
        adapters.SSHPeer = object
        adapters.verify_known_host = lambda *a: None
        modules = {m.__name__: m for m in (authority, ports, adapters)}
        roles, reads = [], []

        def secret(role):
            roles.append(role)
            if role == "provider":
                self.fail("provider prompt fallback")
            return "12" * 32 if role == "journal signing key hex" else "synthetic-" + role

        def read():
            reads.append(True)
            return "synthetic-provider-credential"

        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            cfg = config(Path(tmp))
            for key in ("journal_path", "head_path", "ssh_executable", "ssh_key"):
                Path(cfg[key]).write_text("synthetic public state")
            Path(cfg["export_root"]).mkdir()
            with provider.open_injection(cfg, provider_reader=read, secret_reader=secret) as inputs:
                self.assertEqual(inputs.credentials.provider, "synthetic-provider-credential")
                self.assertEqual(repr(inputs), "Injection()")
            self.assertEqual(reads, [True])
            self.assertEqual(roles, ["model", "inference", "control", "journal signing key hex"])
            with (
                self.assertRaisesRegex(RuntimeError, "EXISTING_CREDENTIAL_INPUT_REFUSED"),
                provider.open_injection(
                    cfg,
                    provider_reader=lambda: (_ for _ in ()).throw(RuntimeError("SECRET_CANARY")),
                    secret_reader=secret,
                ),
            ):
                self.fail("read failure yielded")
            self.assertEqual(roles, ["model", "inference", "control", "journal signing key hex"])

    def child(self, source, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic_worker.py"
            path.write_text("import os, sys, time\nsys.stdin.buffer.read()\n" + source)
            return startup.supervise(
                [sys.executable, "-I", "-B", str(path)],
                {},
                entry_seconds=1,
                read_seconds=0.1,
                session_seconds=0.2,
                cleanup_seconds=0.5,
                **kwargs,
            )

    def test_child_status_only_output_and_confirmed_exit(self):
        result = self.child("os.write(1,b'READING\\nLOADED\\nRUNNING\\nCOMPLETE\\n')\n")
        self.assertEqual(result, dict(status="COMPLETE", worker_exit_confirmed=True, cleanup_required=False))
        result = self.child("os.write(1,b'SYNTHETIC_SECRET_CANARY\\n')\n")
        self.assertEqual(result["status"], "REFUSED")
        self.assertNotIn("CANARY", json.dumps(result))

    def test_read_hang_kills_before_gpu_and_session_hang_requires_cleanup(self):
        result = self.child("os.write(1,b'READING\\n');time.sleep(10)\n")
        self.assertEqual(result, dict(status="TIMEOUT", worker_exit_confirmed=True, cleanup_required=False))
        result = self.child("os.write(1,b'READING\\nLOADED\\nRUNNING\\n');time.sleep(10)\n")
        self.assertEqual(result, dict(status="TIMEOUT", worker_exit_confirmed=True, cleanup_required=True))

    def test_success_token_without_clean_exit_is_refused(self):
        result = self.child("os.write(1,b'READING\\nLOADED\\nRUNNING\\nCOMPLETE\\n');time.sleep(10)\n")
        self.assertNotEqual(result["status"], "COMPLETE")
        self.assertTrue(result["worker_exit_confirmed"])
        self.assertTrue(result["cleanup_required"])


@unittest.skipUnless(
    importlib.util.find_spec("orchestrator") is not None
    and importlib.util.find_spec("qmc_runpod") is not None,
    "explicit Factory and operations public sources required",
)
class CredentialStartupIntegration(unittest.TestCase):
    def test_isolated_pinned_worker_to_actual_startup_and_owner_hooks(self):
        import qmc_runpod
        import test_c1_flow as core
        from orchestrator import runpod_startup_credential as policy
        from qmc_runpod.c1_authority import AuthorityJournal

        operations = Path(qmc_runpod.__file__).resolve().parents[1]
        qwen = Path(__file__).resolve().parents[1]
        factory = Path(policy.__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            f, q, o = (root / name for name in ("factory", "qwen", "operations"))
            for source, destination, files in (
                (factory, f, policy.FACTORY_FILES),
                (qwen, q, policy.QWEN_FILES),
            ):
                for rel in files:
                    path = destination / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / rel, path)
            shutil.copytree(
                operations / "qmc_runpod", o / "qmc_runpod", ignore=shutil.ignore_patterns("__pycache__")
            )
            claims = root / "runpod-startup-claims"
            claims.mkdir()
            cfg = config(root)
            for key in ("ssh_executable", "ssh_key"):
                Path(cfg[key]).write_text("synthetic public reference")
            Path(cfg["export_root"]).mkdir()
            with AuthorityJournal(
                cfg["journal_path"], cfg["head_path"], key=core.KEY, domain=cfg["journal_domain"]
            ).open():
                pass
            config_path = root / "config.json"
            config_path.write_text(json.dumps(cfg))
            data = dict(
                schema=policy.SCHEMA,
                owner_approved=True,
                approval_id="synthetic-worker-startup-001",
                owner_sid="synthetic-owner",
                target=policy.TARGET,
                operation="runpod.start.once",
                issued_at=time.time(),
                expires_at=time.time() + 299,
                session_seconds=6000,
                factory_root=str(f),
                qwen_root=str(q),
                operations_root=str(o),
                python_exe=sys.executable,
                claims_dir=str(claims),
                config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                pins=policy.public_pins(f, q, o),
            )
            approval = root / "synthetic-approval.json"
            approval.write_text(json.dumps(data))
            request = dict(
                factory=str(f),
                qwen=str(q),
                operations=str(o),
                config=str(config_path),
                approval=str(approval),
                claims=str(claims),
                command="start",
                policy_sha256=data["pins"]["factory:orchestrator/runpod_startup_credential.py"],
                test_helpers=[str(qwen / "tests"), str(operations / "tests")],
            )
            child = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(Path(__file__).with_name("runpod_credman_worker_fixture.py")),
                ],
                input=json.dumps(request),
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(child.returncode, 0, child.stderr)
            result = json.loads(child.stdout)
            self.assertEqual(result["status"], "COMPLETE", result)
            self.assertEqual(result["events"], ["READING", "LOADED", "RUNNING"])
            self.assertEqual(result["reads"], 1)
            self.assertTrue(result["wiped"])
            self.assertEqual(result["methods"], ["POST", "DELETE", "GET"])
            self.assertEqual(
                result["owner_inputs"],
                ["model", "inference", "control", "journal signing key hex", "peer", "terminate"],
            )
            self.assertNotIn("dummy-provider-credential", child.stdout + child.stderr)

    def test_v2_reader_through_normal_provider_real_host_and_owned_cleanup(self):
        import runpod_trial
        from orchestrator.runpod_startup_credential import ProviderOnce
        from orchestrator.windows_credential_store import wipe
        from qmc_runpod.production_gateway import ProductionGateway

        if __package__:
            from .test_runpod_provider import ProviderContracts
            from .test_runpod_trial import fixture, run_injected
        else:
            from test_runpod_provider import ProviderContracts
            from test_runpod_trial import fixture, run_injected

        buffer = bytearray(b"synthetic-provider-credential")
        calls = []

        class Socket:
            def __init__(self):
                self.wire = bytearray()

            def setblocking(self, value):
                pass

            def send(self, value):
                self.wire.extend(value)
                return len(value)

            def settimeout(self, value):
                pass

            def sendall(self, value):
                self.wire.extend(value)

        active = []

        @contextmanager
        def serve(gateway, **kwargs):
            active.append(gateway)
            yield ("127.0.0.1", 0)

        def http(address, path, key, payload=None, headers=None):
            gateway = active[-1]
            gateway._authorize(path, {"authorization": "Bearer " + key})
            if path == "/healthz":
                return {"alive": True}
            if path == "/v1/models":
                return {"data": [{"id": "qwen-27b"}]}
            sock = Socket()
            headers = {name.lower(): value for name, value in headers.items()}
            rid = headers["x-request-id"]
            with (
                patch.object(
                    gateway,
                    "_json_write",
                    side_effect=lambda s, code, value, **kw: s.wire.extend(json.dumps(value).encode()),
                ),
                patch.object(gateway, "_watch_client", return_value=contextmanager(lambda: (yield))()),
                patch.object(
                    gateway, "_guard", side_effect=lambda s, ticket: gateway.host.chat_fence(ticket)
                ),
            ):
                gateway._chat(sock, payload, rid, headers["x-intent-token"], [False])
            return json.loads(bytes(sock.wire))

        with tempfile.TemporaryDirectory() as tmp:
            claims = Path(tmp) / "claims"
            claims.mkdir()
            data = dict(
                owner_sid="synthetic-owner",
                expires_at=time.time() + 300,
                approval_id="synthetic-integration-startup-001",
            )

            def read():
                calls.append("read")
                return buffer

            credential = SimpleNamespace(current_windows_sid=lambda: "synthetic-owner", wipe=wipe)
            reader = ProviderOnce(
                data,
                revalidate=lambda: data,
                claims=claims,
                credential=credential,
                enabled=True,
                store=SimpleNamespace(read=read),
            )
            original = provider.open_injection

            @contextmanager
            def injected(cfg, **options):
                with original(cfg, provider_reader=reader.read, **options) as inputs:
                    yield inputs

            case = ProviderContracts()
            with (
                fixture() as (_, external, factories),
                patch.object(provider, "open_injection", side_effect=injected),
                patch.object(ProductionGateway, "serve", serve),
                patch.object(runpod_trial, "_http", side_effect=http),
                case.opened(good_peer=True, allow_cleanup=True) as (_, inputs),
            ):
                result = run_injected(inputs, owner_start=True, simulated=True)
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(calls, ["read"])
            self.assertEqual(buffer, bytearray(len(buffer)))
            self.assertEqual([method for method, _, _ in external.rest.calls], ["POST", "DELETE", "GET"])
            self.assertEqual(len(factories), 1)
            self.assertNotIn("synthetic-provider-credential", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
