import subprocess
import unittest
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class Entrypoints(unittest.TestCase):
    def test_python_sources_compile(self):
        sources = [
            REPO_ROOT / "clpeak_common.py",
            REPO_ROOT / "clpeak-build.py",
            REPO_ROOT / "clpeak-client.py",
            REPO_ROOT / "clpeak-post-process.py",
        ]
        subprocess.run(
            ["python3", "-m", "py_compile", *[str(path) for path in sources]],
            check=True,
        )

    def test_shell_entrypoints_are_executable(self):
        for name in ("clpeak-base", "clpeak-client", "clpeak-get-runtime"):
            path = REPO_ROOT / name
            self.assertTrue(path.is_file(), name)
            self.assertTrue(path.stat().st_mode & 0o111, name)

    def test_runtime_honors_deadline_parameter(self):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(REPO_ROOT)
        completed = subprocess.run(
            [str(REPO_ROOT / "clpeak-get-runtime"), "--deadline", "123"],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
        )
        self.assertEqual(completed.stdout.strip(), "123")


if __name__ == "__main__":
    unittest.main()

