import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "clpeak-result.json"


def load_client_module():
    spec = importlib.util.spec_from_file_location("clpeak_client", REPO_ROOT / "clpeak-client.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_fake_binary(path, exit_code=0):
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import shutil, sys\n"
        "args = sys.argv[1:]\n"
        "output = args[args.index('--output') + 1]\n"
        f"shutil.copyfile({str(FIXTURE)!r}, output)\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


class ClientWrapper(unittest.TestCase):
    def setUp(self):
        self.client = load_client_module()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.result_file = self.root / "clpeak-result.json"
        self.output_file = self.root / "clpeak-client-stderrout.txt"
        self.manifest_file = self.root / "clpeak-run-manifest.json"
        self.error_file = self.root / "clpeak-error.txt"
        self.provenance_file = self.root / "build-provenance.json"
        self.binary = self.root / "fake-clpeak"
        self.provenance_file.write_text(
            json.dumps({"commit": self.client.UPSTREAM_COMMIT}),
            encoding="utf-8",
        )

        patches = [
            mock.patch.object(self.client, "RESULT_FILE", self.result_file),
            mock.patch.object(self.client, "OUTPUT_LOG_FILE", self.output_file),
            mock.patch.object(self.client, "MANIFEST_FILE", self.manifest_file),
            mock.patch.object(self.client, "ERROR_FILE", self.error_file),
            mock.patch.object(self.client, "PROVENANCE_FILE", self.provenance_file),
            mock.patch.dict(os.environ, {"CLPEAK_BINARY": str(self.binary)}),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def run_client(self, arguments):
        return self.client.main(arguments)

    def read_manifest(self):
        return json.loads(self.manifest_file.read_text(encoding="utf-8"))

    def test_gpu_environment_is_recorded_in_manifest(self):
        write_fake_binary(self.binary)
        gpu_environment = {
            "cuda_toolkit_release": None,
            "nvidia_smi_driver_versions": None,
        }
        collect_patch = mock.patch.object(
            self.client, "collect_gpu_environment", return_value=gpu_environment
        )
        with collect_patch:
            return_code = self.run_client(["--tests", "compute", "--max-time-gpu", "25"])
        manifest = self.read_manifest()

        self.assertEqual(return_code, 0)
        self.assertEqual(manifest["gpu_environment"], gpu_environment)

    def test_cuda_driver_mismatch_rejects_run(self):
        write_fake_binary(self.binary)
        gpu_environment = {
            "cuda_toolkit_release": "12.8",
            "nvidia_smi_driver_versions": ["570.26"],
        }
        with mock.patch.object(
            self.client, "collect_gpu_environment", return_value=gpu_environment
        ):
            return_code = self.run_client(["--tests", "compute"])

        self.assertEqual(return_code, 1)
        self.assertFalse(self.result_file.exists())
        self.assertIn(
            "requires NVIDIA Linux driver >= 570.124.06",
            self.error_file.read_text(encoding="utf-8"),
        )

    def test_successful_run_creates_result_and_manifest(self):
        write_fake_binary(self.binary)
        return_code = self.run_client(["--tests", "compute", "--max-time-gpu", "25"])
        manifest = self.read_manifest()

        self.assertEqual(return_code, 0)
        self.assertTrue(manifest["success"])
        self.assertFalse(manifest["timed_out"])
        self.assertEqual(manifest["return_code"], 0)
        self.assertEqual(manifest["parsed_metric_count"], 3)
        self.assertTrue(self.result_file.is_file())
        self.assertTrue(self.output_file.is_file())
        self.assertFalse(self.error_file.exists())

    def test_stale_result_is_removed_before_run(self):
        write_fake_binary(self.binary)
        self.result_file.write_text("{}\n", encoding="utf-8")
        self.output_file.write_text("stale output\n", encoding="utf-8")
        self.manifest_file.write_text("{}\n", encoding="utf-8")
        self.error_file.write_text("stale error\n", encoding="utf-8")

        original_popen = self.client.subprocess.Popen

        def popen_and_assert_clean(command, **kwargs):
            assert not self.result_file.exists()
            assert not self.output_file.exists()
            assert not self.manifest_file.exists()
            assert not self.error_file.exists()
            return original_popen(command, **kwargs)

        with mock.patch.object(self.client.subprocess, "Popen", side_effect=popen_and_assert_clean):
            return_code = self.run_client(["--tests", "compute"])
        self.assertEqual(return_code, 0)

    def test_nonzero_exit_rejects_and_removes_result(self):
        write_fake_binary(self.binary, exit_code=1)
        return_code = self.run_client(["--tests", "compute"])
        manifest = self.read_manifest()

        self.assertEqual(return_code, 1)
        self.assertFalse(manifest["success"])
        self.assertEqual(manifest["return_code"], 1)
        self.assertFalse(self.result_file.exists())
        self.assertEqual(
            self.error_file.read_text(encoding="utf-8").strip(),
            "clpeak exited with status 1",
        )

    def test_timeout_rejects_and_removes_result(self):
        write_fake_binary(self.binary)
        process = FakeProcess("native timeout output\n")
        with mock.patch.object(self.client.subprocess, "Popen", return_value=process):
            return_code = self.run_client(["--deadline", "30"])
        manifest = self.read_manifest()

        self.assertEqual(return_code, 1)
        self.assertFalse(manifest["success"])
        self.assertTrue(manifest["timed_out"])
        self.assertEqual(manifest["return_code"], 124)
        self.assertFalse(self.result_file.exists())
        self.assertIn("timeout after 30 seconds", self.output_file.read_text(encoding="utf-8"))

    def test_signal_is_forwarded_and_rejects_result(self):
        write_fake_binary(self.binary)

        class SignallingProcess(FakeProcess):
            def communicate(self, timeout=None):
                signal.raise_signal(signal.SIGTERM)
                return self.output, None

        process = SignallingProcess("native signal output\n", return_code=143)
        with mock.patch.object(self.client.subprocess, "Popen", return_value=process):
            return_code = self.run_client(["--deadline", "30"])
        manifest = self.read_manifest()

        self.assertEqual(return_code, 1)
        self.assertFalse(manifest["success"])
        self.assertEqual(manifest["return_code"], 143)
        self.assertFalse(self.result_file.exists())
        self.assertEqual(process.sent_signals, [signal.SIGTERM])
        self.assertEqual(
            self.error_file.read_text(encoding="utf-8").strip(),
            "clpeak interrupted by signal 15",
        )


class FakeProcess:
    def __init__(self, output, return_code=0):
        self.output = output
        self.returncode = return_code
        self.killed = False
        self.sent_signals = []

    @property
    def returncode(self):
        return self._return_code

    @returncode.setter
    def returncode(self, value):
        self._return_code = value

    def communicate(self, timeout=None):
        if timeout is not None and not self.killed:
            raise subprocess.TimeoutExpired(cmd=["fake-clpeak"], timeout=timeout)
        return self.output, None

    def kill(self):
        self.killed = True
        self.returncode = 124

    def send_signal(self, signum):
        self.sent_signals.append(signum)


if __name__ == "__main__":
    unittest.main()
