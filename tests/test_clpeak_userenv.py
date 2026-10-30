import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class CudaUserenv(unittest.TestCase):
    def test_cuda_userenv_is_pinned(self):
        userenv = json.loads(
            (REPO_ROOT / "userenvs" / "cuda-12.8.json").read_text(encoding="utf-8")
        )
        workshop = json.loads((REPO_ROOT / "workshop.json").read_text(encoding="utf-8"))
        origin = userenv["userenv"]["origin"]

        self.assertEqual(userenv["userenv"]["name"], "cuda-12.8")
        self.assertEqual(origin["image"], "docker.io/nvidia/cuda")
        self.assertEqual(origin["tag"], "12.8.1-devel-rockylinux9")
        self.assertEqual(origin["update-policy"], "never")
        self.assertIn(
            "python3-pip",
            userenv["requirements"][0]["distro_info"]["packages"],
        )
        self.assertIn(
            "cuda-12.8", [userenv["name"] for userenv in workshop["userenvs"]]
        )
