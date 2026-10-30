#!/usr/bin/env python3
"""Convert a validated clpeak result into Crucible CDM metrics."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from clpeak_common import (
    UPSTREAM_COMMIT,
    MANIFEST_FILE,
    RESULT_FILE,
    cdm_names,
    metric_aggregation,
    metric_class,
    metric_type,
    normalize_devices,
    normalize_positive_int,
    parse_clpeak_result,
)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Post-process a clpeak run")
    parser.add_argument("--tests", default="all")
    parser.add_argument("--devices", default="")
    parser.add_argument("--max-time-gpu", default="250")
    parser.add_argument("--max-time-cpu", default="2000")
    parser.add_argument("--deadline", default="900")
    return parser.parse_args(argv)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_toolbox_metrics():
    import os

    toolbox_home = os.environ.get("TOOLBOX_HOME")
    if not toolbox_home:
        raise RuntimeError("clpeak post-processing requires TOOLBOX_HOME")
    toolbox_python = Path(toolbox_home) / "python"
    if not toolbox_python.is_dir():
        raise RuntimeError("clpeak post-processing cannot find %s" % toolbox_python)
    sys.path.append(str(toolbox_python))
    from toolbox.metrics import finish_samples, log_sample

    return log_sample, finish_samples


def main(argv=None):
    output_file = Path("postprocess") / "post-process-data.json"
    output_file.unlink(missing_ok=True)
    if not MANIFEST_FILE.is_file() or not RESULT_FILE.is_file():
        raise RuntimeError("clpeak manifest or result file is missing")
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if not manifest.get("success"):
        raise RuntimeError("clpeak manifest does not record a successful run")
    if manifest.get("benchmark") != "clpeak":
        raise RuntimeError("clpeak manifest names a different benchmark")
    if manifest.get("upstream_commit") != UPSTREAM_COMMIT:
        raise RuntimeError("clpeak manifest upstream commit does not match the integration")
    if manifest.get("return_code") != 0 or manifest.get("timed_out"):
        raise RuntimeError("clpeak manifest does not record a clean process exit")
    expected_hash = manifest.get("result_sha256")
    actual_hash = sha256_file(RESULT_FILE)
    if expected_hash != actual_hash:
        raise RuntimeError("clpeak result hash does not match the run manifest")
    parsed = parse_clpeak_result(RESULT_FILE.read_text(encoding="utf-8"))
    if manifest.get("parsed_metric_count") != len(parsed["metrics"]):
        raise RuntimeError("clpeak manifest metric count does not match the result")

    args = parse_args(argv if argv is not None else sys.argv[1:])
    expected_parameters = {
        "tests": args.tests,
        "devices": normalize_devices(args.devices),
        "max_time_gpu": normalize_positive_int(
            args.max_time_gpu, "max-time-gpu", 10, 60000
        ),
        "max_time_cpu": normalize_positive_int(
            args.max_time_cpu, "max-time-cpu", 10, 60000
        ),
        "deadline": normalize_positive_int(args.deadline, "deadline", 30, 86400),
    }
    if manifest.get("parameters") != expected_parameters:
        raise RuntimeError("clpeak manifest parameters do not match the submitted parameters")

    begin_ns = manifest["launch_begin_epoch_ns"]
    end_ns = manifest["process_completion_epoch_ns"]
    if end_ns <= begin_ns:
        raise RuntimeError("clpeak process timestamps are not ordered")
    begin = begin_ns // 1_000_000
    end = end_ns // 1_000_000

    log_sample, finish_samples = load_toolbox_metrics()
    for metric in parsed["metrics"]:
        descriptor = {
            "source": "clpeak",
            "class": metric_class(metric["direction"]),
            "type": metric_type(metric["unit"]),
            "default-aggregation": metric_aggregation(metric["direction"]),
        }
        log_sample(
            "0",
            descriptor,
            cdm_names(metric),
            {
                "begin": begin,
                "end": end,
                "value": metric["value"],
            },
        )
    metric_file = finish_samples()

    output = {
        "rickshaw-bench-metric": {"schema": {"version": "2021.04.12"}},
        "benchmark": "clpeak",
        "primary-period": "measurement",
        "primary-metric": "flops",
        "periods": [{"name": "measurement", "metric-files": [metric_file]}],
        "clpeak-summary": {
            "device_count": len(parsed["devices"]),
            "metric_count": len(parsed["metrics"]),
            "backends": sorted({metric["backend"] for metric in parsed["metrics"]}),
            "tests": sorted({metric["test"] for metric in parsed["metrics"]}),
        },
    }
    output_file.parent.mkdir(exist_ok=True)
    output_file.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("clpeak post-processing failed: %s" % exc, file=sys.stderr)
        sys.exit(1)

