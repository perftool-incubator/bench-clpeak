import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "clpeak-result.json"


def load_post_process_module():
    spec = importlib.util.spec_from_file_location(
        "clpeak_post_process", REPO_ROOT / "clpeak-post-process.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PostProcessing(unittest.TestCase):
    def setUp(self):
        self.post_process = load_post_process_module()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        original_cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, original_cwd)
        self.result_file = self.root / "clpeak-result.json"
        self.manifest_file = self.root / "clpeak-run-manifest.json"
        self.result_file.write_bytes(FIXTURE.read_bytes())
        self.samples = []
        self.metric_files = ["clpeak-metrics.json"]

        patches = [
            mock.patch.object(self.post_process, "RESULT_FILE", self.result_file),
            mock.patch.object(self.post_process, "MANIFEST_FILE", self.manifest_file),
            mock.patch.object(self.post_process, "load_toolbox_metrics", self.fake_toolbox),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def fake_toolbox(self):
        def log_sample(sequence, descriptor, names, sample):
            self.samples.append((sequence, descriptor, names, sample))

        def finish_samples():
            return self.metric_files.pop()

        return log_sample, finish_samples

    def write_manifest(self, success=True, result_hash=None):
        manifest = {
            "benchmark": "clpeak",
            "success": success,
            "return_code": 0,
            "timed_out": False,
            "upstream_commit": self.post_process.UPSTREAM_COMMIT,
            "parsed_metric_count": 3,
            "parameters": {
                "tests": "all",
                "devices": "",
                "max_time_gpu": 250,
                "max_time_cpu": 2000,
                "deadline": 900,
            },
            "launch_begin_epoch_ns": 1_000_000_000,
            "process_completion_epoch_ns": 2_000_000_000,
            "result_sha256": result_hash or sha256_file(self.result_file),
        }
        self.manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    def write_stale_output(self):
        output_file = self.root / "postprocess" / "post-process-data.json"
        output_file.parent.mkdir(exist_ok=True)
        output_file.write_text("{}\n", encoding="utf-8")

    def test_successful_result_is_indexed_into_cdm(self):
        self.write_manifest()
        self.post_process.main([])
        output_file = self.root / "postprocess" / "post-process-data.json"
        output = json.loads(output_file.read_text(encoding="utf-8"))

        self.assertEqual(len(self.samples), 3)
        first_sequence, first_descriptor, first_names, first_sample = self.samples[0]
        self.assertEqual(first_sequence, "0")
        self.assertEqual(first_descriptor["source"], "clpeak")
        self.assertEqual(first_descriptor["class"], "throughput")
        self.assertEqual(first_descriptor["type"], "flops")
        self.assertEqual(first_names["backend"], "CUDA")
        self.assertEqual(first_names["test"], "single_precision_compute")
        self.assertEqual(first_sample["begin"], 1_000)
        self.assertEqual(first_sample["end"], 2_000)
        self.assertEqual(output["benchmark"], "clpeak")
        self.assertEqual(output["periods"][0]["metric-files"], ["clpeak-metrics.json"])
        self.assertEqual(output["clpeak-summary"]["metric_count"], 3)

    def test_hash_mismatch_is_rejected(self):
        self.write_manifest(result_hash="not-the-result-hash")
        self.write_stale_output()
        with self.assertRaisesRegex(RuntimeError, "result hash does not match"):
            self.post_process.main()
        self.assertFalse((self.root / "postprocess" / "post-process-data.json").exists())

    def test_failed_manifest_is_rejected(self):
        self.write_manifest(success=False)
        with self.assertRaisesRegex(RuntimeError, "does not record a successful run"):
            self.post_process.main()
        self.assertFalse((self.root / "postprocess" / "post-process-data.json").exists())

    def test_parameter_mismatch_is_rejected(self):
        self.write_manifest()
        with self.assertRaisesRegex(RuntimeError, "parameters do not match"):
            self.post_process.main(["--deadline", "123"])
        self.assertFalse((self.root / "postprocess" / "post-process-data.json").exists())


if __name__ == "__main__":
    unittest.main()
