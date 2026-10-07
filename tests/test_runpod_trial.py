"""Launcher regressions: real factory/host/gateway with dummy external adapters."""
from contextlib import contextmanager
from decimal import Decimal
import io
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from runpod_trial import Credentials, Injection, main, plan, run_injected

OPERATIONS_AVAILABLE = importlib.util.find_spec("qmc_runpod") is not None


@contextmanager
def fixture(*, denial=False, unknown_delete=False, bad_chat=False, over_cap=False):
    from qmc_runpod.c1_ports import PodSpec, Rate
    from qmc_runpod.execution_adapters import SSHPeer
    from qmc_runpod.ondemand import TerminationApproval
    from qmc_runpod import single_user_owui as wiring
    import test_single_user_host as existing
    case = existing.SingleUserTests()
    case.setUp()
    case.rest.unknown_delete = unknown_delete
    if bad_chat:
        def refuse(*args, **kwargs):
            raise RuntimeError("DUMMY_SECRET_MUST_NEVER_APPEAR")
        case.upstream.json = refuse
    real_factory = wiring.build_service
    factories = []
    def factory(**kwargs):
        value = real_factory(**kwargs)
        case.rest.host = case.ssh.host = value[0]
        factories.append(value[0])
        return value
    def enroll(sid, pod):
        return SSHPeer(pod, "8.8.8.8", 12345, str(case.path / "dummy-known-hosts"),
                       str(case.path / "dummy-key-reference"), "SHA256:" + "a" * 43)
    def approve(scope):
        if denial:
            return None
        return TerminationApproval("dummy-approval", "terminate", scope.session_id,
                                   scope.pod_id, scope.receipt_sha256, scope.expires_at)
    keys = Credentials("dummy-provider-credential", "dummy-model-credential",
                       existing.INFERENCE, existing.CONTROL)
    try:
        with case.journal().open() as journal, \
                patch.object(wiring, "RunPodREST", side_effect=lambda **kw: case.rest), \
                patch.object(wiring, "SSHExecution", side_effect=lambda *a, **kw: case.ssh), \
                patch.object(wiring, "LoopbackUpstream", side_effect=lambda *a, **kw: case.upstream), \
                patch.object(wiring, "build_service", side_effect=factory), \
                patch("subprocess.Popen", side_effect=AssertionError("real process forbidden")):
            inputs = Injection(journal, case.path / "export", "fixture-user",
                PodSpec(account="dummy-account", image="dummy-image@sha256:" + "1" * 64),
                Rate(gpu_usd_per_hour=Decimal("20" if over_cap else "1.80")),
                "8.8.8.8", "dummy-ssh-reference", keys, enroll, approve)
            yield inputs, case, factories
    finally:
        case.doCleanups()


def dummy_trial():
    with fixture() as (inputs, case, factories):
        result = run_injected(inputs, owner_start=True, simulated=True)
        methods = [m for m, _, _ in case.rest.calls]
        assert result["status"] == "COMPLETE", result
        assert len(factories) == 1 and methods == ["POST", "DELETE", "GET"]
        assert case.ssh.tunnels == case.ssh.closed == 1
        assert all("gfvzduwn8wej0e" not in p for _, p, _ in case.rest.calls)
        # Read the dummy journal through its owner API, not its held lock file.
        state = json.dumps(inputs.journal.latest())
        assert all(key not in state for key in (
            inputs.credentials.provider, inputs.credentials.model,
            inputs.credentials.inference, inputs.credentials.control))
        result.update(real_provider_calls=0, real_ssh_processes=0, real_credential_reads=0,
                      factory_calls=1, dummy_create_calls=1, dummy_delete_calls=1,
                      dummy_absence_reads=1, retained_pod_touched=False)
        return result


class LauncherTests(unittest.TestCase):
    @unittest.skipUnless(OPERATIONS_AVAILABLE, "explicit reviewed operations source required")
    def test_real_factory_full_trial_and_exact_single_delete(self):
        self.assertEqual(dummy_trial()["status"], "COMPLETE")

    def test_missing_owner_start_is_inert(self):
        self.assertEqual(run_injected(None)["code"], "OWNER_START_REQUIRED")

    def test_cli_start_does_not_load_provider_without_owner_start(self):
        with patch("runpod_trial._provider", side_effect=AssertionError("provider read")), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(main(["start", "--provider", "never-read"]), 2)
            self.assertEqual(json.loads(output.getvalue())["code"], "OWNER_START_REQUIRED")

    @unittest.skipUnless(OPERATIONS_AVAILABLE, "explicit reviewed operations source required")
    def test_no_individual_approval_never_deletes(self):
        with fixture(denial=True) as (inputs, case, _):
            result = run_injected(inputs, owner_start=True, simulated=True)
            self.assertTrue(result["cleanup_required"])
            self.assertTrue(result["export_readback_ok"])
            self.assertFalse(any(m == "DELETE" for m, _, _ in case.rest.calls))

    @unittest.skipUnless(OPERATIONS_AVAILABLE, "explicit reviewed operations source required")
    def test_unknown_delete_is_not_retried_or_claimed_absent(self):
        with fixture(unknown_delete=True) as (inputs, case, _):
            result = run_injected(inputs, owner_start=True, simulated=True)
            self.assertFalse(result["termination_absence_confirmed"])
            self.assertTrue(result["cleanup_required"])
            self.assertEqual(sum(m == "DELETE" for m, _, _ in case.rest.calls), 1)

    @unittest.skipUnless(OPERATIONS_AVAILABLE, "explicit reviewed operations source required")
    def test_chat_failure_still_exports_and_cleans_up_without_secret_text(self):
        with fixture(bad_chat=True) as (inputs, case, _):
            result = run_injected(inputs, owner_start=True, simulated=True)
            self.assertFalse(result["chat_ok"])
            self.assertTrue(result["termination_absence_confirmed"])
            self.assertNotIn("DUMMY_SECRET", json.dumps(result))

    @unittest.skipUnless(OPERATIONS_AVAILABLE, "explicit reviewed operations source required")
    def test_rate_above_ten_dollar_full_window_refuses_before_factory(self):
        with fixture(over_cap=True) as (inputs, case, factories):
            result = run_injected(inputs, owner_start=True, simulated=True)
            self.assertEqual(result["status"], "REFUSED")
            self.assertEqual(factories, [])
            self.assertEqual(case.rest.calls, [])

    @unittest.skipUnless(OPERATIONS_AVAILABLE, "explicit reviewed operations source required")
    def test_typed_input_and_secret_repr(self):
        self.assertEqual(run_injected({}, owner_start=True)["code"], "TYPED_INJECTION_REQUIRED")
        with fixture() as (inputs, _, _):
            self.assertEqual(repr(inputs), "Injection()")
            self.assertEqual(repr(inputs.credentials), "Credentials()")
        self.assertEqual(plan()["total_cap_usd"], "10.00")

    def test_unpinned_provider_refuses_without_exposing_exception_text(self):
        with patch("pathlib.Path.read_bytes", return_value=b"raise Exception('SECRET_SENTINEL')"), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            code = main(["start", "--owner-start", "--operations-source", ".",
                         "--provider", "explicit-reviewed-provider.py", "--provider-sha256", "0" * 64])
            self.assertEqual(code, 2)
            self.assertNotIn("SECRET_SENTINEL", output.getvalue())
            self.assertEqual(json.loads(output.getvalue())["code"], "PROVIDER_OR_TRIAL_REFUSED")


if __name__ == "__main__":
    unittest.main()
