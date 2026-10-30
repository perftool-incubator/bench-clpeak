#!/usr/bin/env python3
"""Build and install pinned clpeak in a Workshop image."""

import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


UPSTREAM_COMMIT = "b5da52485bb6873aef54ff30c3fe81037e523eac"
INSTALL_PREFIX = Path("/usr/local")
PROVENANCE_FILE = INSTALL_PREFIX / "share" / "clpeak" / "build-provenance.json"


def run(command, cwd=None):
    print("+ %s" % " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def command_output(command):
    executable = shutil.which(command[0])
    if not executable:
        return None
    return subprocess.run(
        command, capture_output=True, text=True, check=True
    ).stdout.strip()


def compiler_overrides():
    """Prefer Intel compilers when building in a oneAPI userenv."""
    if os.environ.get("CC") or os.environ.get("CXX"):
        return []
    if shutil.which("icx") and shutil.which("icpx"):
        return ["-DCMAKE_C_COMPILER=icx", "-DCMAKE_CXX_COMPILER=icpx"]
    return []


def configure_command(source_dir, build_dir):
    command = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCLPEAK_ENABLE_GUI=OFF",
        "-DCLPEAK_ENABLE_ONNX=OFF",
        "-DCLPEAK_ENABLE_LITERT=OFF",
        "-DCLPEAK_ENABLE_COREML=OFF",
        "-DCLPEAK_ENABLE_OPENCL=OFF",
        "-DCLPEAK_ENABLE_VULKAN=OFF",
        "-DCLPEAK_VERSION_OVERRIDE=%s" % UPSTREAM_COMMIT,
    ]
    command.extend(compiler_overrides())
    return command


def main():
    source_dir = Path.cwd()
    build_dir = source_dir / "build"
    configure = configure_command(source_dir, build_dir)
    run(configure)
    run(["cmake", "--build", str(build_dir), "--config", "Release", "--parallel"])
    run(["cmake", "--install", str(build_dir), "--prefix", str(INSTALL_PREFIX)])

    binary = INSTALL_PREFIX / "bin" / "clpeak"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError("installed clpeak binary is missing or not executable")

    cmake_version = subprocess.run(
        ["cmake", "--version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]
    nvcc_version = command_output(["nvcc", "--version"])
    provenance = {
        "upstream": "krrishnarraj/clpeak",
        "commit": UPSTREAM_COMMIT,
        "build_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cmake_version": cmake_version,
        "nvcc_version": nvcc_version,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "configure_command": configure,
        "install_prefix": str(INSTALL_PREFIX),
        "binary": str(binary),
    }
    PROVENANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE_FILE.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("clpeak Workshop build failed: %s" % exc, file=sys.stderr)
        sys.exit(1)

