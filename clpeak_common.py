#!/usr/bin/env python3
"""Shared helpers for the clpeak Crucible integration."""

import json
import math
import re
from pathlib import Path


UPSTREAM_COMMIT = "b5da52485bb6873aef54ff30c3fe81037e523eac"
DEFAULT_BINARY = Path("/usr/local/bin/clpeak")
CUDA_12_8_UPDATE_1_MINIMUM_DRIVER = (570, 124, 6)
RESULT_FILE = Path("clpeak-result.json")
OUTPUT_LOG_FILE = Path("clpeak-client-stderrout.txt")
MANIFEST_FILE = Path("clpeak-run-manifest.json")
ERROR_FILE = Path("clpeak-error.txt")

ALLOWED_TEST_RE = re.compile(r"^[a-z0-9-]+$")
ALLOWED_DEVICE_RE = re.compile(r"^[a-z0-9]+:[0-9]+$")


def normalize_tests(tests):
    """Return validated clpeak test flags from a comma-separated string."""
    if tests is None:
        tests = "all"
    if isinstance(tests, (list, tuple)):
        tests = ",".join(str(test) for test in tests)
    if not isinstance(tests, str):
        raise ValueError("clpeak tests must be a string")
    tests = tests.strip()
    if not tests or tests == "all":
        return []
    result = []
    for test in tests.split(","):
        test = test.strip()
        if not test:
            raise ValueError("clpeak tests cannot contain an empty entry")
        if not ALLOWED_TEST_RE.fullmatch(test):
            raise ValueError("invalid clpeak test: %s" % test)
        result.append(test)
    return result


def normalize_devices(devices):
    """Return an optional validated clpeak device selector."""
    if devices is None:
        devices = ""
    if not isinstance(devices, str):
        raise ValueError("clpeak devices must be a string")
    devices = devices.strip()
    if not devices:
        return ""
    for device in devices.split(","):
        if not ALLOWED_DEVICE_RE.fullmatch(device.strip()):
            raise ValueError("invalid clpeak device selector: %s" % device)
    return devices


def normalize_positive_int(value, name, minimum, maximum):
    """Validate an integer within an inclusive range."""
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be an integer" % name) from exc
    if number < minimum or number > maximum:
        raise ValueError("%s must be between %d and %d" % (name, minimum, maximum))
    return number


def parse_cuda_release(nvcc_output):
    """Return the CUDA release advertised by nvcc."""
    match = re.search(r"release ([0-9]+\.[0-9]+)", nvcc_output)
    if not match:
        raise ValueError("nvcc output does not contain a CUDA release")
    return match.group(1)


def parse_driver_versions(nvidia_smi_output):
    """Return every driver version reported by nvidia-smi."""
    versions = []
    for line in nvidia_smi_output.splitlines():
        version = line.strip()
        if not version:
            continue
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", version):
            raise ValueError("nvidia-smi reported an invalid driver version: %s" % version)
        versions.append(version)
    if not versions:
        raise ValueError("nvidia-smi did not report a driver version")
    return versions


def _version_tuple(version):
    return tuple(int(part) for part in version.split("."))


def validate_nvidia_environment(environment):
    """Reject a CUDA 12.8.1 image paired with an older host driver."""
    if environment.get("cuda_toolkit_release") != "12.8":
        return
    driver_versions = environment.get("nvidia_smi_driver_versions")
    if not driver_versions:
        raise ValueError(
            "CUDA 12.8.1 requires nvidia-smi and an injected host driver"
        )
    minimum = CUDA_12_8_UPDATE_1_MINIMUM_DRIVER
    for driver_version in driver_versions:
        if _version_tuple(driver_version) < minimum:
            raise ValueError(
                "CUDA 12.8.1 requires NVIDIA Linux driver >= 570.124.06; found %s"
                % driver_version
            )


def build_clpeak_command(binary, tests, devices, max_time_gpu, max_time_cpu, output_path):
    """Build the clpeak command line from validated runtime parameters."""
    command = [str(binary), "--output", str(output_path)]
    if devices:
        command.extend(["--devices", devices])
    for test in normalize_tests(tests):
        command.append("--%s" % test)
    command.extend(["--max-time-gpu", str(max_time_gpu)])
    command.extend(["--max-time-cpu", str(max_time_cpu)])
    return command


def _require_mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError("clpeak %s must be an object" % name)
    return value


def _require_list(value, name):
    if not isinstance(value, list) or not value:
        raise ValueError("clpeak %s must be a nonempty array" % name)
    return value


def _required_string(mapping, key, name):
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("clpeak %s is missing %s" % (name, key))
    return value


def metric_type(unit):
    """Map an upstream unit to a stable CDM metric type."""
    normalized = unit.strip().lower()
    if normalized == "flops":
        return "flops"
    if normalized == "b/s":
        return "bytes-sec"
    if normalized == "ops":
        return "ops-sec"
    if normalized == "texel/s":
        return "texels-sec"
    if normalized == "s":
        return "seconds"
    if normalized == "ppm":
        return "ppm"
    raise ValueError("unsupported clpeak metric unit: %s" % unit)


def metric_class(direction):
    """Map an upstream direction to a CDM metric class."""
    if direction == "higher_is_better":
        return "throughput"
    if direction == "lower_is_better":
        return "latency"
    raise ValueError("unsupported clpeak metric direction: %s" % direction)


def metric_aggregation(direction):
    """Choose the default CDM aggregation for an upstream direction."""
    if direction == "higher_is_better":
        return "sum"
    if direction == "lower_is_better":
        return "avg"
    raise ValueError("unsupported clpeak metric direction: %s" % direction)


def parse_clpeak_result(json_text):
    """Parse and validate an upstream clpeak JSON result."""
    try:
        document = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError("clpeak result is not valid JSON: %s" % exc) from exc
    document = _require_mapping(document, "result")
    if document.get("format_version") != 3:
        raise ValueError("unsupported clpeak result format: %r" % document.get("format_version"))
    devices = _require_list(document.get("devices"), "devices")
    metrics = []
    for device_index, device in enumerate(devices):
        device = _require_mapping(device, "device")
        backend = _required_string(device, "backend", "device")
        device_name = _required_string(device, "name", "device")
        tests = _require_list(device.get("tests"), "device tests")
        for test in tests:
            test = _require_mapping(test, "test")
            test_id = _required_string(test, "id", "test")
            unit = _required_string(test, "unit", "test")
            direction = _required_string(test, "direction", "test")
            variant = test.get("variant", "default")
            if not isinstance(variant, str) or not variant.strip():
                raise ValueError("clpeak test variant must be a nonempty string")
            test_metrics = _require_list(test.get("metrics"), "test metrics")
            for metric in test_metrics:
                metric = _require_mapping(metric, "metric")
                metric_id = _required_string(metric, "id", "metric")
                status = metric.get("status", "ok")
                if status in {"unsupported", "skipped", "error"}:
                    continue
                if status != "ok":
                    raise ValueError("unsupported clpeak metric status: %s" % status)
                value = metric.get("value")
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("clpeak metric %s has a non-numeric value" % metric_id)
                if not math.isfinite(float(value)):
                    raise ValueError("clpeak metric %s is not finite" % metric_id)
                metrics.append(
                    {
                        "backend": backend,
                        "device": device_name,
                        "device_index": device_index,
                        "test": test_id,
                        "variant": variant,
                        "metric": metric_id,
                        "value": float(value),
                        "unit": metric.get("unit", unit),
                        "direction": metric.get("direction", direction),
                    }
                )
    if not metrics:
        raise ValueError("clpeak result contains no metrics")
    return {
        "format_version": document["format_version"],
        "clpeak_version": document.get("clpeak_version"),
        "devices": devices,
        "metrics": metrics,
    }


def cdm_names(metric):
    """Return CDM breakout names for one parsed clpeak metric."""
    return {
        "backend": metric["backend"],
        "device": metric["device"],
        "device_index": str(metric["device_index"]),
        "test": metric["test"],
        "variant": metric["variant"],
        "metric": metric["metric"],
        "unit": metric["unit"],
    }

