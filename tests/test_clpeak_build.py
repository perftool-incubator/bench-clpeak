import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_build_module():
    spec = importlib.util.spec_from_file_location(
        "clpeak_build", REPO_ROOT / "clpeak-build.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConfigureCommand(unittest.TestCase):
    def setUp(self):
        self.build = load_build_module()

    def command(self):
        return self.build.configure_command(Path("/source"), Path("/build"))

    def test_optional_backends_are_disabled(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(self.build.shutil, "which", return_value=None):
                command = self.command()
        self.assertIn("-DCLPEAK_ENABLE_OPENCL=OFF", command)
        self.assertIn("-DCLPEAK_ENABLE_VULKAN=OFF", command)
        self.assertIn("-DCLPEAK_ENABLE_GUI=OFF", command)
        self.assertIn("-DCLPEAK_ENABLE_ONNX=OFF", command)

    def test_oneapi_compilers_are_preferred(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(self.build.shutil, "which", return_value="/usr/bin/tool"):
                command = self.command()
        self.assertIn("-DCMAKE_C_COMPILER=icx", command)
        self.assertIn("-DCMAKE_CXX_COMPILER=icpx", command)

    def test_explicit_compiler_environment_is_respected(self):
        environment = {"CC": "/usr/bin/clang", "CXX": "/usr/bin/clang++"}
        with mock.patch.dict(os.environ, environment, clear=True):
            with mock.patch.object(self.build.shutil, "which", return_value="/usr/bin/tool"):
                command = self.command()
        self.assertNotIn("-DCMAKE_C_COMPILER=icx", command)
        self.assertNotIn("-DCMAKE_CXX_COMPILER=icpx", command)


if __name__ == "__main__":
    unittest.main()
