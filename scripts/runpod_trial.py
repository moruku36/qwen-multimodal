"""One explicitly injected RunPod trial. Default CLI is inert; no secret discovery."""
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
import argparse
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import sys
import time
import uuid

sys.modules.setdefault("runpod_trial", sys.modules[__name__])

@dataclass(frozen=True)
class Credentials:
    provider: str = field(repr=False)
    model: str = field(repr=False)
    inference: str = field(repr=False)
    control: str = field(repr=False)


@dataclass(frozen=True)
class Injection:
    # The provider owns opening/closing its already-approved journal.
    journal: object = field(repr=False)
    export_root: Path = field(repr=False)
    subject: str = field(repr=False)
    spec: object = field(repr=False)
    rate: object = field(repr=False)
    rest_address: str = field(repr=False)
    ssh_executable: str = field(repr=False)
    credentials: Credentials = field(repr=False)
    enroll_peer: object = field(repr=False)
    authorize_termination: object = field(repr=False)


@dataclass(frozen=True)
class TerminationScope:
    session_id: str
    pod_id: str
    receipt_sha256: str
    expires_at: float


def plan():
    return {"mode": "PREPARATION_ONLY", "total_cap_usd": "10.00",
            "work_seconds": 5400, "cleanup_seconds": 600,
            "cleanup_reserve_usd": "1.00", "idle_seconds": 300,
            "scope": "one new owned Pod; retained Pod excluded",
            "termination": "export/readback; exact individual approval; DELETE once; GET absence",
            "automatic_retries": False, "credential_discovery": False,
            "provider_required": True, "current_rate_and_account_attestation_required": True}


def _http(address, path, key, payload=None, headers=None):
    connection = http.client.HTTPConnection(*address, timeout=10)
    try:
        h = {"Authorization": "Bearer " + key, "Connection": "close", **(headers or {})}
        wire = None if payload is None else json.dumps(payload).encode()
        if wire is not None:
            h["Content-Type"] = "application/json"
        connection.request("GET" if payload is None else "POST", path, body=wire, headers=h)
        response = connection.getresponse()
        data = response.read(65537)
        if response.status != 200 or len(data) > 65536:
            raise RuntimeError("bounded_response_refused")
        return json.loads(data)
    finally:
        connection.close()


def run_injected(inputs, *, owner_start=False, simulated=False):
    """Uses only caller-injected values. Never serializes inputs or exception text.

    owner_start is an explicit call-site gate, not evidence of owner consent.
    The operator must have the next-start approval and reviewed provider first.
    """
    result = {"mode": "DUMMY" if simulated else "LIVE", "status": "REFUSED",
              "health_ok": False, "chat_ok": False, "export_readback_ok": False,
              "termination_absence_confirmed": False, "cleanup_required": False,
              "total_cap_usd": "10.00"}
    if owner_start is not True:
        result["code"] = "OWNER_START_REQUIRED"
        return result
    if type(inputs) is not Injection or type(inputs.credentials) is not Credentials:
        result["code"] = "TYPED_INJECTION_REQUIRED"
        return result
    # Import only after explicit start, from the reviewed operations source.
    from qmc_runpod.c1_ports import PodSpec, Rate
    from qmc_runpod.execution_adapters import SSHPeer
    from qmc_runpod.ondemand import REQUIRED_ACTIONS, TerminationApproval, UsageLimits, UsageScope
    from qmc_runpod.single_user_owui import build_service

    host = None
    sid = "trial-" + uuid.uuid4().hex
    try:
        if (type(inputs.spec) is not PodSpec or type(inputs.rate) is not Rate
                or not callable(inputs.enroll_peer) or not callable(inputs.authorize_termination)
                or inputs.spec.volume_gb != 80 or inputs.spec.container_gb != 20
                or inputs.spec.gpu_type not in {"NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"}
                or inputs.rate.gpu_usd_per_hour <= 0 or inputs.rate.storage_usd_per_hour < 0
                or inputs.rate.overhead_usd < 0):
            raise RuntimeError("approved_profile_required")
        now = time.time()
        limits = UsageLimits.create(max_runtime_seconds=5400, max_usd="10.00",
            expires_at=now + 6000, scope=UsageScope(sid, "new-pod", REQUIRED_ACTIONS),
            cleanup_runtime_seconds=600, cleanup_usd="1.00")
        # Rate evaluates worst-case full work/cleanup against the same total cap.
        if inputs.rate.cost(now, now + 6000) > Decimal("10.00"):
            raise RuntimeError("full_window_exceeds_cap")
        keys = inputs.credentials
        host, gateway, _ = build_service(journal=inputs.journal, export_root=inputs.export_root,
            subject=inputs.subject, spec=inputs.spec, rate=inputs.rate, enabled=True,
            rest_address=inputs.rest_address, provider_bearer=keys.provider,
            ssh_executable=inputs.ssh_executable, model_bearer=keys.model,
            inference_key=keys.inference, control_key=keys.control, idle_seconds=300)
        host.approve_session(limits)
        with gateway.serve(port=0 if simulated else 19180) as address:
            _http(address, "/healthz", keys.inference)
            catalog = _http(address, "/v1/models", keys.inference)
            result["health_ok"] = any(m.get("id") == "qwen-27b" for m in catalog.get("data", []))
            if not result["health_ok"]:
                raise RuntimeError("health_refused")
            # This is one owner-triggered CLI probe, not an installed OWUI hook.
            payload = {"model": "qwen-27b", "messages": [{"role": "user", "content": "Reply OK."}], "stream": False}
            pending = host.enqueue(inputs.subject, payload)
            while host.status()["phase"] != "ready":
                phase = host.step()
                if phase == "created":
                    peer = inputs.enroll_peer(sid, host.status()["pod_id"])
                    if type(peer) is not SSHPeer:
                        raise RuntimeError("typed_peer_required")
                    host.register_peer(sid, peer)
                elif phase not in {"bootstrapped", "ready"}:
                    raise RuntimeError("startup_unconfirmed")
            headers = host.ready_request(pending)
            reply = _http(address, "/v1/chat/completions", keys.inference, payload, headers)
            # Do not persist or print reply/prompt/auth/provider responses.
            content = reply.get("choices", [{}])[0].get("message", {}).get("content")
            result["chat_ok"] = type(content) is str and 0 < len(content) <= 8192
            if not result["chat_ok"]:
                raise RuntimeError("chat_unconfirmed")
            result["status"] = "PROBE_COMPLETE"
    except Exception:
        result["code"] = "TRIAL_REFUSED_OR_UNCONFIRMED"
    finally:
        if host is not None:
            try:
                state = host.status()
                if state["session_id"] == sid and state["pod_id"] is not None:
                    result["cleanup_required"] = True
                    host.stop(sid)
                    if host.step() == "exporting":
                        host.export(sid)
                    if host.status()["phase"] == "approval_pending" and host.readback(sid):
                        result["export_readback_ok"] = True
                        # Internal receipt only crosses to the explicit owner hook.
                        with host._lock:
                            row = host._row(sid)
                            scope = TerminationScope(sid, row["pod_id"], row["receipt"]["sha256"], row["cleanup_deadline"])
                        approval = inputs.authorize_termination(scope)
                        if type(approval) is not TerminationApproval:
                            raise RuntimeError("individual_termination_approval_required")
                        host.approve_cleanup(approval)
                        result["termination_absence_confirmed"] = host.terminate(sid) == "absent"
                        result["cleanup_required"] = not result["termination_absence_confirmed"]
                        if result["cleanup_required"]:
                            result["code"] = "TERMINATION_UNCONFIRMED_NO_RETRY"
                elif state["phase"] == "create_unresolved":
                    result["cleanup_required"] = True
                    result["code"] = "CREATE_UNKNOWN_NO_RETRY"
            except Exception:
                result["cleanup_required"] = True
                result["code"] = "CLEANUP_UNCONFIRMED_NO_RETRY"
            finally:
                try:
                    host.close()
                except Exception:
                    result["cleanup_required"] = True
                    result["code"] = "HOST_CLOSE_UNCONFIRMED"
    if result["health_ok"] and result["chat_ok"] and result["termination_absence_confirmed"]:
        result["status"] = "COMPLETE"
        result.pop("code", None)
    elif result["cleanup_required"]:
        result["status"] = "CLEANUP_REQUIRED"
    return result


def _provider(path, expected_sha):
    # Owner explicitly names reviewed code. No config, env, key or store scanning.
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha:
        raise RuntimeError("provider_pin_refused")
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("approved_trial_provider", loader=None))
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module.open_injection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "dummy", "start"])
    parser.add_argument("--operations-source", type=Path)
    parser.add_argument("--provider", type=Path)
    parser.add_argument("--provider-sha256")
    parser.add_argument("--owner-start", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = plan()
    elif args.command == "dummy":
        if args.operations_source is None:
            result = {"status": "REFUSED", "code": "REVIEWED_OPERATIONS_SOURCE_REQUIRED"}
        else:
            sys.path.insert(0, str(args.operations_source.resolve()))
            sys.path.insert(0, str(args.operations_source.resolve() / "tests"))
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
            from test_runpod_trial import dummy_trial
            result = dummy_trial()
    elif not args.owner_start:
        result = {"status": "REFUSED", "code": "OWNER_START_REQUIRED"}
    elif not all((args.operations_source, args.provider, args.provider_sha256)):
        result = {"status": "REFUSED", "code": "EXPLICIT_REVIEWED_PROVIDER_REQUIRED"}
    else:
        try:
            sys.path.insert(0, str(args.operations_source.resolve()))
            with _provider(args.provider, args.provider_sha256)() as inputs:
                result = run_injected(inputs, owner_start=True)
        except Exception:
            result = {"status": "REFUSED", "code": "PROVIDER_OR_TRIAL_REFUSED"}
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") == "COMPLETE" or args.command == "plan" else 2


if __name__ == "__main__":
    raise SystemExit(main())
