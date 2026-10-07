"""Explicit nonsecret configuration, existing journal and in-memory credential injection."""
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import getpass
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import threading

from runpod_trial import Credentials, Injection

FIELDS = frozenset({"schema", "account", "account_attested", "image", "gpu_type",
    "gpu_rate", "storage_rate", "overhead", "rest_ip", "ssh_executable", "ssh_key",
    "journal_path", "head_path", "journal_domain", "export_root", "subject",
    "owui_source_sha256", "owui_version"})
PATHS = {"ssh_executable", "ssh_key", "journal_path", "head_path", "export_root"}


def validate_config(config):
    """Pure validation: no filesystem, network, environment or secret access."""
    if type(config) is not dict or set(config) - FIELDS:
        return {"ready": False, "code": "CONFIG_SCHEMA_REFUSED", "missing": [], "invalid": []}
    missing = sorted(k for k in FIELDS if config.get(k) is None or config.get(k) == "")
    invalid = []
    for name, value in config.items():
        if name in missing:
            continue
        if name != "account_attested" and type(value) is not str:
            invalid.append(name)
            continue
        valid = True
        if name == "schema": valid = value == "runpod-owner-v1"
        elif name == "account_attested": valid = value is True
        elif name in {"account", "subject"}: valid = type(value) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value) is not None and not value.startswith("public-fixture")
        elif name == "image": valid = type(value) is str and re.fullmatch(r"[A-Za-z0-9./_-]+@sha256:[0-9a-f]{64}", value) is not None and not value.startswith("public-fixture")
        elif name == "gpu_type": valid = value in {"NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"}
        elif name in {"gpu_rate", "storage_rate", "overhead"}:
            try:
                valid = type(value) is str and len(value) <= 32
                amount = Decimal(value) if valid else Decimal("NaN")
                valid = valid and amount.is_finite() and (amount > 0 if name == "gpu_rate" else amount >= 0)
            except (InvalidOperation, ValueError): valid = False
        elif name == "rest_ip":
            try: valid = type(value) is str and ipaddress.ip_address(value).is_global
            except ValueError: valid = False
        elif name in PATHS: valid = type(value) is str and Path(value).is_absolute() and not any(c in value for c in "\r\n\0")
        elif name == "journal_domain": valid = type(value) is str and 1 <= len(value) <= 64
        elif name == "owui_source_sha256": valid = type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None
        elif name == "owui_version": valid = value == "0.11.4"
        if not valid: invalid.append(name)
    if not missing and not invalid:
        rate = Decimal(config["gpu_rate"]) + Decimal(config["storage_rate"])
        if rate * Decimal(6000) / 3600 + Decimal(config["overhead"]) > 10 or rate / 6 > 1:
            invalid.append("total_cost_and_cleanup_reserve")
        if Path(config["journal_path"]) == Path(config["head_path"]): invalid.append("independent_head")
    return {"ready": not missing and not invalid, "code": "CONFIG_READY" if not missing and not invalid else "CONFIG_INCOMPLETE",
            "missing": missing, "invalid": sorted(invalid)}


def read_config(path):
    # Only an explicitly supplied nonsecret file. Never search adjacent files.
    with Path(path).open("rb") as stream: raw = stream.read(16385)
    if len(raw) > 16384: raise RuntimeError("CONFIG_BOUND_REFUSED")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise RuntimeError("CONFIG_DUPLICATE_REFUSED")
            result[key] = value
        return result
    try: return json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    except Exception: raise RuntimeError("CONFIG_PARSE_REFUSED") from None


def _secret(role):
    # Called only after explicit owner start, never by validate/import/stage.
    return getpass.getpass("Existing " + role + " credential (hidden; not saved): ")


def _peer_input(sid, pod):
    return json.loads(input("Verified NEW peer proof JSON for " + sid + "/" + pod + ": "))


def _approve(scope):
    from qmc_runpod.ondemand import TerminationApproval
    exact = scope.session_id + ":" + scope.pod_id + ":" + scope.receipt_sha256
    if input("Exact terminate approval; type " + exact + ": ") != exact: return None
    return TerminationApproval("owner-" + scope.session_id, "terminate", scope.session_id,
                               scope.pod_id, scope.receipt_sha256, scope.expires_at)


@contextmanager
def open_injection(config, *, secret_reader=_secret, peer_reader=_peer_input, approval_reader=_approve):
    """Production provider: existing paths only, explicit memory-only secret reader."""
    if not validate_config(config)["ready"]: raise RuntimeError("CONFIG_INCOMPLETE")
    # Refuse before prompting if approved persistent infrastructure is absent.
    if not all(Path(config[k]).is_file() for k in {"journal_path", "head_path", "ssh_executable", "ssh_key"}) or not Path(config["export_root"]).is_dir():
        raise RuntimeError("EXISTING_APPROVED_PATHS_REQUIRED")
    from qmc_runpod.c1_authority import AuthorityJournal
    from qmc_runpod.c1_ports import PodSpec, Rate
    from qmc_runpod.execution_adapters import SSHPeer, verify_known_host
    try:
        keys = Credentials(*(secret_reader(role) for role in ("provider", "model", "inference", "control")))
        signing_key = bytes.fromhex(secret_reader("journal signing key hex"))
        if len(signing_key) < 32 or len(signing_key) > 128: raise ValueError()
    except Exception: raise RuntimeError("EXISTING_CREDENTIAL_INPUT_REFUSED") from None
    def enroll(sid, pod):
        proof = peer_reader(sid, pod)
        expected = {"session_id", "pod_id", "address", "port", "known_hosts", "fingerprint", "verified_out_of_band"}
        if type(proof) is not dict or set(proof) != expected or proof["session_id"] != sid or proof["pod_id"] != pod or proof["verified_out_of_band"] is not True:
            raise RuntimeError("EXACT_VERIFIED_PEER_REQUIRED")
        peer = SSHPeer(pod, proof["address"], proof["port"], proof["known_hosts"], config["ssh_key"], proof["fingerprint"])
        with Path(peer.known_hosts).open("rb") as stream: data = stream.read(4097)
        verify_known_host(peer, data)
        return peer
    try:
        with AuthorityJournal(config["journal_path"], config["head_path"], key=signing_key, domain=config["journal_domain"]).open() as journal:
            yield Injection(journal, Path(config["export_root"]), config["subject"],
                PodSpec(account=config["account"], image=config["image"], gpu_type=config["gpu_type"]),
                Rate(Decimal(config["gpu_rate"]), Decimal(config["storage_rate"]), Decimal(config["overhead"])),
                config["rest_ip"], config["ssh_executable"], keys, enroll, approval_reader)
    except Exception: raise RuntimeError("PROVIDER_OR_JOURNAL_REFUSED") from None


class OWUIHook:
    """Stable imported proxy: default denied, explicitly bound by the owner backend."""
    def __init__(self): self._target = None; self._lock = threading.Lock()
    @contextmanager
    def bind(self, target):
        with self._lock:
            if self._target is not None: raise RuntimeError("OWUI_ALREADY_BOUND")
            self._target = target
        try: yield self
        finally:
            with self._lock: self._target = None
            target.close()
    def _current(self):
        with self._lock: target = self._target
        if target is None: raise RuntimeError("OWUI_INTEGRATION_DISABLED")
        return target
    def begin(self, request, user): return self._current().begin(request, user)
    async def final_chat(self, request, form_data, user, context=None):
        return await self._current().final_chat(request, form_data, user, context=context)


owui_dispatcher = OWUIHook()


@contextmanager
def configure_owui(config, *, observed_source_sha256, inference_key, control_key,
                   owner_enable=False, _fixture_port=None, _response_factory=None):
    if owner_enable is not True or not validate_config(config)["ready"] or observed_source_sha256 != config["owui_source_sha256"]:
        raise RuntimeError("VERIFIED_OWUI_CONFIGURATION_REQUIRED")
    from qmc_runpod.single_user_owui import build_remote_dispatcher
    dispatcher = build_remote_dispatcher(subject=config["subject"], enabled=True,
        inference_key=inference_key, control_key=control_key, port=_fixture_port or 19180,
        fixture=_fixture_port is not None, response_factory=_response_factory)
    with owui_dispatcher.bind(dispatcher): yield owui_dispatcher


def stage_owui(source, config):
    # Caller explicitly provides public/reviewed source bytes; no live lookup.
    if not validate_config(config)["ready"]: raise RuntimeError("CONFIG_INCOMPLETE")
    from dataclasses import replace
    from qmc_runpod.owui_installer import build_plan
    plan = build_plan(source, version=config["owui_version"], expected_sha256=config["owui_source_sha256"])
    return replace(plan, hook=b"# Default denied; explicit reviewed backend configure_owui required.\nfrom runpod_provider import owui_dispatcher as dispatcher\n")
