import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clpeak_common import (
    build_clpeak_command,
    cdm_names,
    metric_aggregation,
    metric_class,
    metric_type,
    normalize_devices,
    normalize_positive_int,
    normalize_tests,
    parse_cuda_release,
    parse_driver_versions,
    parse_clpeak_result,
    validate_nvidia_environment,
)


FIXTURE = Path(__file__).with_name("fixtures") / "clpeak-result.json"


class NormalizeTests(unittest.TestCase):
    def test_all_is_empty_flag_list(self):
        self.assertEqual(normalize_tests("all"), [])

    def test_valid_tests_are_preserved(self):
        self.assertEqual(normalize_tests("compute,cache-bandwidth"), ["compute", "cache-bandwidth"])

    def test_invalid_test_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_tests("compute,bad test")


class NormalizeDevices(unittest.TestCase):
    def test_empty_is_allowed(self):
        self.assertEqual(normalize_devices(""), "")

    def test_valid_devices_are_preserved(self):
        self.assertEqual(normalize_devices("cuda:0,opencl:1"), "cuda:0,opencl:1")

    def test_invalid_device_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_devices("cuda")


class PositiveIntegers(unittest.TestCase):
    def test_valid_value(self):
        self.assertEqual(normalize_positive_int("250", "max-time-gpu", 10, 60000), 250)

    def test_out_of_range_value(self):
        with self.assertRaises(ValueError):
            normalize_positive_int("1", "max-time-gpu", 10, 60000)


class GpuEnvironment(unittest.TestCase):
    def test_cuda_release_is_parsed(self):
        output = "Cuda compilation tools, release 12.8, V12.8.93"
        self.assertEqual(parse_cuda_release(output), "12.8")

    def test_driver_versions_are_parsed(self):
        self.assertEqual(
            parse_driver_versions("570.124.06\n575.51.03\n"),
            ["570.124.06", "575.51.03"],
        )

    def test_cuda_128_requires_minimum_driver(self):
        environment = {
            "cuda_toolkit_release": "12.8",
            "nvidia_smi_driver_versions": ["570.124.06"],
        }
        validate_nvidia_environment(environment)

        environment["nvidia_smi_driver_versions"] = ["570.26"]
        with self.assertRaisesRegex(ValueError, "requires NVIDIA Linux driver"):
            validate_nvidia_environment(environment)


class CommandBuilder(unittest.TestCase):
    def test_command_contains_output_and_budgets(self):
        command = build_clpeak_command(
            Path("/usr/local/bin/clpeak"),
            ["compute", "latency"],
            "cuda:0",
            250,
            2000,
            Path("clpeak-result.json"),
        )
        self.assertIn("--output", command)
        self.assertIn("clpeak-result.json", command)
        self.assertIn("--devices", command)
        self.assertIn("cuda:0", command)
        self.assertIn("--compute", command)
        self.assertIn("--latency", command)
        self.assertIn("--max-time-gpu", command)
        self.assertIn("250", command)
        self.assertIn("--max-time-cpu", command)
        self.assertIn("2000", command)


class ResultParser(unittest.TestCase):
    def test_fixture_is_parsed(self):
        parsed = parse_clpeak_result(FIXTURE.read_text())
        self.assertEqual(parsed["format_version"], 3)
        self.assertEqual(len(parsed["devices"]), 1)
        self.assertEqual(len(parsed["metrics"]), 3)
        self.assertEqual(parsed["metrics"][0]["backend"], "CUDA")
        self.assertEqual(parsed["metrics"][0]["test"], "single_precision_compute")

    def test_missing_metric_is_rejected(self):
        document = json.loads(FIXTURE.read_text())
        document["devices"][0]["tests"][0]["metrics"] = []
        with self.assertRaises(ValueError):
            parse_clpeak_result(json.dumps(document))

    def test_invalid_json_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_clpeak_result("{")

    def test_non_ok_metrics_are_skipped(self):
        document = json.loads(FIXTURE.read_text())
        document["devices"][0]["tests"][0]["metrics"].extend(
            [
                {"id": "skipped", "status": "skipped"},
                {"id": "failed", "status": "error", "reason": "device error"},
            ]
        )
        parsed = parse_clpeak_result(json.dumps(document))
        self.assertEqual(len(parsed["metrics"]), 3)
        self.assertNotIn("skipped", {metric["metric"] for metric in parsed["metrics"]})

    def test_unknown_metric_status_is_rejected(self):
        document = json.loads(FIXTURE.read_text())
        document["devices"][0]["tests"][0]["metrics"].append(
            {"id": "bad", "status": "unexpected"}
        )
        with self.assertRaises(ValueError):
            parse_clpeak_result(json.dumps(document))


class MetricMapping(unittest.TestCase):
    def test_units(self):
        self.assertEqual(metric_type("FLOPS"), "flops")
        self.assertEqual(metric_type("B/s"), "bytes-sec")
        self.assertEqual(metric_type("OPS"), "ops-sec")
        self.assertEqual(metric_type("Texel/s"), "texels-sec")
        self.assertEqual(metric_type("s"), "seconds")
        self.assertEqual(metric_type("ppm"), "ppm")

    def test_class_and_aggregation(self):
        self.assertEqual(metric_class("higher_is_better"), "throughput")
        self.assertEqual(metric_aggregation("higher_is_better"), "sum")
        self.assertEqual(metric_class("lower_is_better"), "latency")
        self.assertEqual(metric_aggregation("lower_is_better"), "avg")

    def test_cdm_names_include_breakouts(self):
        parsed = parse_clpeak_result(FIXTURE.read_text())
        names = cdm_names(parsed["metrics"][0])
        self.assertEqual(names["backend"], "CUDA")
        self.assertEqual(names["device"], "Test GPU")
        self.assertEqual(names["device_index"], "0")
        self.assertEqual(names["test"], "single_precision_compute")
        self.assertEqual(names["variant"], "AVX-512")
        self.assertEqual(names["metric"], "float")

    def test_metric_level_unit_and_direction_override_test_values(self):
        document = json.loads(FIXTURE.read_text())
        document["devices"][0]["tests"][0]["metrics"][0].update(
            {"unit": "OPS", "direction": "lower_is_better"}
        )
        parsed = parse_clpeak_result(json.dumps(document))
        metric = parsed["metrics"][0]
        self.assertEqual(metric["unit"], "OPS")
        self.assertEqual(metric["direction"], "lower_is_better")


if __name__ == "__main__":
    unittest.main()

