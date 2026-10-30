#!/usr/bin/env python3
"""Execute clpeak and preserve a validated run manifest."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from clpeak_common import (
    DEFAULT_BINARY,
    ERROR_FILE,
    MANIFEST_FILE,
    OUTPUT_LOG_FILE,
    RESULT_FILE,
    UPSTREAM_COMMIT,
    build_clpeak_command,
    normalize_devices,
    normalize_positive_int,
    normalize_tests,
    parse_clpeak_result,
    parse_cuda_release,
    parse_driver_versions,
    validate_nvidia_environment,
)


PROVENANCE_FILE = Path("/usr/local/share/clpeak/build-provenance.json")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run the clpeak Crucible benchmark")
    parser.add_argument("--tests", default="all")
    parser.add_argument("--devices", default="")
    parser.add_argument("--max-time-gpu", default="250")
    parser.add_argument("--max-time-cpu", default="2000")
    parser.add_argument("--deadline", default="900")
    return parser.parse_args(argv)


def write_error(message):
    ERROR_FILE.write_text(message.rstrip() + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_provenance():
    if not PROVENANCE_FILE.is_file():
        raise RuntimeError("clpeak build provenance is missing: %s" % PROVENANCE_FILE)
    provenance = json.loads(PROVENANCE_FILE.read_text(encoding="utf-8"))
    if provenance.get("commit") != UPSTREAM_COMMIT:
        raise RuntimeError(
            "installed clpeak commit %r does not match required %s"
            % (provenance.get("commit"), UPSTREAM_COMMIT)
        )
    return provenance


def discard_run_artifacts():
    """Remove artifacts from a previous sample before launching clpeak."""
    for path in (RESULT_FILE, OUTPUT_LOG_FILE, MANIFEST_FILE, ERROR_FILE):
        path.unlink(missing_ok=True)


def capture_command(command):
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError("could not execute %s: %s" % (command[0], exc)) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            "%s exited with status %d: %s"
            % (command[0], completed.returncode, completed.stdout.strip())
        )
    return completed.stdout


def collect_gpu_environment():
    environment = {"cuda_toolkit_release": None, "nvidia_smi_driver_versions": None}
    nvcc = shutil.which("nvcc")
    if nvcc:
        nvcc_output = capture_command([nvcc, "--version"])
        environment["cuda_toolkit_release"] = parse_cuda_release(nvcc_output)
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        driver_output = capture_command(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"]
        )
        environment["nvidia_smi_driver_versions"] = parse_driver_versions(driver_output)
    return environment


def execute_clpeak(command, deadline):
    """Run clpeak while forwarding termination signals to the native process."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    previous_term_handler = signal.getsignal(signal.SIGTERM)
    previous_int_handler = signal.getsignal(signal.SIGINT)
    received_signal = None

    def forward_signal(signum, _frame):
        nonlocal received_signal
        received_signal = signum
        try:
            process.send_signal(signum)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)
    timed_out = False
    try:
        native_output, _ = process.communicate(timeout=deadline)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        native_output, _ = process.communicate()
        native_output += "\nclpeak wrapper timeout after %d seconds\n" % deadline
    finally:
        signal.signal(signal.SIGTERM, previous_term_handler)
        signal.signal(signal.SIGINT, previous_int_handler)

    return process.returncode, native_output or "", timed_out, received_signal


def main(argv=None):
    args = parse_args(argv)
    discard_run_artifacts()
    try:
        tests = normalize_tests(args.tests)
        devices = normalize_devices(args.devices)
        max_time_gpu = normalize_positive_int(args.max_time_gpu, "max-time-gpu", 10, 60000)
        max_time_cpu = normalize_positive_int(args.max_time_cpu, "max-time-cpu", 10, 60000)
        deadline = normalize_positive_int(args.deadline, "deadline", 30, 86400)
        provenance = load_provenance()
        gpu_environment = collect_gpu_environment()
        validate_nvidia_environment(gpu_environment)
    except Exception as exc:
        write_error("clpeak parameter validation failed: %s" % exc)
        return 1

    binary = Path(os.environ.get("CLPEAK_BINARY", str(DEFAULT_BINARY)))
    if not binary.is_file() or not os.access(binary, os.X_OK):
        write_error("clpeak binary is missing or not executable: %s" % binary)
        return 1

    command = build_clpeak_command(
        binary,
        tests,
        devices,
        max_time_gpu,
        max_time_cpu,
        RESULT_FILE,
    )
    launch_ns = time.time_ns()
    return_code, native_output, timed_out, received_signal = execute_clpeak(
        command, deadline
    )
    completion_ns = time.time_ns()
    if received_signal is not None:
        native_output += "\nclpeak wrapper received signal %d\n" % received_signal

    OUTPUT_LOG_FILE.write_text(native_output, encoding="utf-8", errors="replace")
    success = False
    parsed = None
    error_message = None
    result_sha256 = None
    if timed_out:
        error_message = "clpeak timed out after %d seconds" % deadline
    elif received_signal is not None:
        error_message = "clpeak interrupted by signal %d" % received_signal
    elif return_code != 0:
        error_message = "clpeak exited with status %d" % return_code
    elif not RESULT_FILE.is_file():
        error_message = "clpeak did not create %s" % RESULT_FILE
    else:
        try:
            result_text = RESULT_FILE.read_text(encoding="utf-8")
            parsed = parse_clpeak_result(result_text)
            result_sha256 = sha256_file(RESULT_FILE)
            success = True
        except Exception as exc:
            error_message = "clpeak result validation failed: %s" % exc

    manifest = {
        "benchmark": "clpeak",
        "success": success,
        "return_code": return_code,
        "timed_out": timed_out,
        "command": command,
        "parameters": {
            "tests": args.tests,
            "devices": devices,
            "max_time_gpu": max_time_gpu,
            "max_time_cpu": max_time_cpu,
            "deadline": deadline,
        },
        "launch_begin_epoch_ns": launch_ns,
        "process_completion_epoch_ns": completion_ns,
        "launch_to_completion_seconds": (completion_ns - launch_ns) / 1_000_000_000,
        "upstream_commit": UPSTREAM_COMMIT,
        "build_provenance": provenance,
        "gpu_environment": gpu_environment,
        "result_file": str(RESULT_FILE),
        "result_sha256": result_sha256,
        "parsed_metric_count": len(parsed["metrics"]) if parsed else 0,
        "error": error_message,
    }
    if not success and RESULT_FILE.exists():
        RESULT_FILE.unlink()
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if success:
        return 0
    write_error(error_message or "clpeak run failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())

