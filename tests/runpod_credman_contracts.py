"""Run credential/startup contracts with explicit public dependency checkouts."""

import argparse
import sys
import tempfile
import unittest
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory-source", type=Path, required=True)
    parser.add_argument("--operations-source", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not (args.factory_source / "orchestrator/runpod_startup_credential.py").is_file():
        raise SystemExit("Factory startup source required")
    if not (args.operations_source / "qmc_runpod/__init__.py").is_file():
        raise SystemExit("Operations source required")
    for path in (
        args.factory_source,
        args.operations_source,
        args.operations_source / "tests",
        root / "scripts",
        root / "tests",
    ):
        sys.path.insert(0, str(path.resolve()))
    # Keep private synthetic fixtures inside the caller's disposable workspace.
    with tempfile.TemporaryDirectory(prefix="startup-contracts-", dir=root) as directory:
        previous = tempfile.tempdir
        tempfile.tempdir = directory
        try:
            suite = unittest.defaultTestLoader.loadTestsFromNames(
                ["test_runpod_credman_startup", "test_runpod_provider.ConfigTests"]
            )
            result = unittest.TextTestRunner(verbosity=2).run(suite)
        finally:
            tempfile.tempdir = previous
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
