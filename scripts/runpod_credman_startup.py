"""Pinned Windows-v2 startup worker and status-only supervisor; default inert."""

import getpass
import hashlib
import importlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, nullcontext, suppress
from pathlib import Path
from types import ModuleType

TOKENS = {b"READING\n", b"LOADED\n", b"RUNNING\n", b"COMPLETE\n", b"REFUSED\n", b"CLEANUP_REQUIRED\n"}


def _policy(factory, expected):
    path = Path(factory) / "orchestrator/runpod_startup_credential.py"
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected:
        raise RuntimeError("CREDENTIAL_SOURCE_REFUSED")
    module = ModuleType("pinned_startup_policy")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def supervise(
    argv,
    payload,
    *,
    entry_seconds=300,
    read_seconds=27,
    session_seconds=6000,
    cleanup_seconds=3,
    popen=subprocess.Popen,
):
    """No credential-bearing stdin/env/output. Fixed status protocol only.

    Entry and native read deadlines precede GPU activity. The session deadline
    includes the existing cleanup reserve. Kill/reap failure never means success
    or absence of billable resources; OS stalls/descendants are not guaranteed.
    """
    started = time.monotonic()
    deadline = started + entry_seconds
    stage = "ENTRY"
    process = None
    result = {"status": "REFUSED", "worker_exit_confirmed": False, "cleanup_required": False}
    events = queue.Queue(maxsize=8)

    def receive():
        try:
            while True:
                token = process.stdout.readline(33)
                if not token:
                    events.put_nowait(None)
                    return
                events.put_nowait(token if token in TOKENS else b"INVALID")
                if token not in TOKENS:
                    return
        except Exception:
            with suppress(queue.Full):
                events.put_nowait(None)

    reader = None
    try:
        process = popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=False,
            close_fds=True,
            env={"SystemRoot": os.environ.get("SYSTEMROOT", "C:\\Windows")},
        )
        process.stdin.write(json.dumps(payload, allow_nan=False).encode())
        process.stdin.close()
        reader = threading.Thread(target=receive, daemon=True)
        reader.start()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result["status"] = "TIMEOUT"
                break
            try:
                token = events.get(timeout=remaining)
            except queue.Empty:
                result["status"] = "TIMEOUT"
                break
            if token == b"READING\n" and stage == "ENTRY":
                stage = "READING"
                deadline = min(deadline, time.monotonic() + read_seconds)
            elif token == b"LOADED\n" and stage == "READING":
                stage = "LOADED"
                deadline = started + entry_seconds
            elif token == b"RUNNING\n" and stage == "LOADED":
                stage = "RUNNING"
                result["cleanup_required"] = True
                deadline = time.monotonic() + session_seconds
            elif token in {b"COMPLETE\n", b"CLEANUP_REQUIRED\n", b"REFUSED\n"}:
                if token == b"COMPLETE\n" and stage != "RUNNING":
                    break
                result["status"] = token.decode("ascii").strip()
                result["cleanup_required"] = token == b"CLEANUP_REQUIRED\n" or (
                    stage == "RUNNING" and token != b"COMPLETE\n"
                )
                break
            else:
                break
    except Exception:
        result["status"] = "REFUSED"
    finally:
        reap_deadline = time.monotonic() + cleanup_seconds
        if process is not None:
            try:
                if result["status"] in {"COMPLETE", "CLEANUP_REQUIRED", "REFUSED"}:
                    process.wait(timeout=max(0.001, reap_deadline - time.monotonic()))
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=max(0.001, reap_deadline - time.monotonic()))
                result["worker_exit_confirmed"] = True
                if result["status"] == "COMPLETE" and process.returncode != 0:
                    result.update(status="REFUSED", cleanup_required=True)
            except Exception:
                if result["status"] == "COMPLETE":
                    result.update(status="REFUSED", cleanup_required=True)
                try:
                    process.kill()
                    process.wait(timeout=max(0.001, reap_deadline - time.monotonic()))
                    result["worker_exit_confirmed"] = True
                except Exception:
                    result["status"] = "EXIT_UNCONFIRMED"
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        if not result["worker_exit_confirmed"]:
            result["status"] = "EXIT_UNCONFIRMED"
        if stage == "RUNNING" and result["status"] != "COMPLETE":
            result["cleanup_required"] = True
        if reader is not None:
            reader.join(0.1)
    return result


def _owner_terminal():
    return os.name == "nt" and sys.flags.isolated and sys.stdin.isatty() and sys.stdout.isatty()


def start_once(args):
    # Code-integration approval does not grant credential access or a new Pod.
    if (
        not _owner_terminal()
        or args.owner_start is not True
        or args.command not in {"start", "serve"}
        or args.provider is not None
        or args.config is None
        or args.credential_factory_source is None
        or args.credential_claims is None
        or not args.credential_launcher_sha256
    ):
        return {"status": "REFUSED", "code": "OWNER_CREDENTIAL_APPROVAL_REQUIRED"}
    policy = _policy(args.credential_factory_source, args.credential_launcher_sha256)
    qwen = Path(__file__).resolve().parents[1]
    data = policy.load_approval(
        args.credential_approval,
        factory=args.credential_factory_source,
        qwen=qwen,
        operations=args.operations_source,
        config_path=args.config,
        command=args.command,
    )
    if Path(args.credential_claims).resolve() != Path(data["claims_dir"]).resolve():
        return {"status": "REFUSED", "code": "CLAIM_LOCATION_REFUSED"}
    request = {
        "factory": str(args.credential_factory_source),
        "qwen": str(qwen),
        "operations": str(args.operations_source),
        "config": str(args.config),
        "approval": str(args.credential_approval),
        "claims": str(args.credential_claims),
        "command": args.command,
        "policy_sha256": args.credential_launcher_sha256,
    }
    return supervise([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker"], request)


@contextmanager
def _owner_console():
    """Child-only Windows console: no secrets or owner answers cross protocol IPC."""
    with (
        open("CONIN$", encoding="utf-8") as incoming,
        open("CONOUT$", "w", encoding="utf-8", buffering=1) as outgoing,
    ):
        before = sys.stdin, sys.__stdin__
        # Windows getpass uses no-echo msvcrt only when these are the same.
        sys.stdin = sys.__stdin__ = incoming

        def answer(prompt):
            outgoing.write(prompt)
            line = incoming.readline()
            if not line:
                raise EOFError()
            return line.rstrip("\r\n")

        try:
            yield (
                lambda role: getpass.getpass(
                    "Existing " + role + " credential (hidden; not saved): ", stream=outgoing
                ),
                answer,
            )
        finally:
            sys.stdin, sys.__stdin__ = before


def _consume(request, emit, *, _test_context=None, _console_factory=_owner_console):
    """Actual pinned startup path. Private test seams never come from CLI/JSON."""
    status = "REFUSED"
    try:
        policy = _policy(request["factory"], request["policy_sha256"])

        def approved():
            return policy.load_approval(
                request["approval"],
                factory=request["factory"],
                qwen=request["qwen"],
                operations=request["operations"],
                config_path=request["config"],
                command=request["command"],
            )

        data = approved()
        if Path(request["claims"]).resolve() != Path(data["claims_dir"]).resolve():
            raise RuntimeError("CLAIM_LOCATION_REFUSED")
        sources = policy.load_sources(
            data, factory=request["factory"], qwen=request["qwen"], operations=request["operations"]
        )
        finder = policy.PinnedSources(sources)
        if any(name in sys.modules for name in sources):
            raise RuntimeError("SOURCE_ALREADY_LOADED")
        sys.meta_path.insert(0, finder)
        credential = importlib.import_module("orchestrator.runpod_local_credential")
        provider = importlib.import_module("runpod_provider")
        trial = importlib.import_module("runpod_trial")
        config_raw = Path(request["config"]).read_bytes()
        if len(config_raw) > 16384 or hashlib.sha256(config_raw).hexdigest() != data["config_sha256"]:
            raise RuntimeError("CONFIG_REFUSED")
        config = json.loads(config_raw, object_pairs_hook=policy._unique)
        with (
            _test_context(credential, provider, trial) if _test_context else nullcontext(),
            _console_factory() as (secret, answer),
        ):
            reader = policy.ProviderOnce(
                data, revalidate=approved, claims=request["claims"], credential=credential, enabled=True
            )

            def provider_read():
                emit("READING")
                key = reader.read()
                emit("LOADED")
                return key

            with provider.open_injection(
                config,
                provider_reader=provider_read,
                secret_reader=secret,
                peer_reader=lambda sid, pod: provider._peer_input(sid, pod, read_input=answer),
                approval_reader=lambda scope: provider._approve(scope, read_input=answer),
            ) as inputs:
                if not reader._allowed():
                    raise RuntimeError("STARTUP_EXPIRED")
                emit("RUNNING")
                result = trial.run_injected(inputs, owner_start=True, owui=request["command"] == "serve")
                status = "COMPLETE" if result.get("status") == "COMPLETE" else "CLEANUP_REQUIRED"
    except BaseException:
        status = "REFUSED"
    return status


def _worker():
    protocol = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.dup2(null, 2)
    os.close(null)
    status = "REFUSED"

    def emit(word):
        token = (word + "\n").encode("ascii")
        if token not in TOKENS:
            raise RuntimeError("STATUS_REFUSED")
        os.write(protocol, token)

    try:
        raw = sys.stdin.buffer.read(16385)
        request = json.loads(raw)
        fields = {"factory", "qwen", "operations", "config", "approval", "claims", "command", "policy_sha256"}
        if len(raw) > 16384 or type(request) is not dict or set(request) != fields or os.name != "nt":
            raise RuntimeError("REQUEST_REFUSED")
        status = _consume(request, emit)
    except BaseException:
        status = "REFUSED"
    finally:
        emit(status)
        os.close(protocol)
        os._exit(0 if status == "COMPLETE" else 1)


if __name__ == "__main__" and sys.flags.isolated and sys.argv[1:] == ["--worker"]:
    _worker()
