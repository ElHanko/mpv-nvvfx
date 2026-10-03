#!/usr/bin/env python3
"""Regression checks against the built mpv, without NvVFX or a GPU call.

Usage: python3 test/nvvfx-vsr/check.py /absolute/path/to/test-video.mp4
Build with ./build.sh first. The default NvVFX directory must be empty.
"""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
VIDEO = Path(sys.argv.pop(1)).resolve()
BINARY = ROOT / "build/mpv"
ENV = dict(os.environ, MPV_NVVFX_LIBDIR=str(ROOT / "local/missing-nvvfx-test"))


def run(*arguments, env=ENV):
    return subprocess.run(
        [str(BINARY), "--no-config", *arguments],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=10,
    )


def filter_run(specification, env=ENV):
    return run("--vo=null", "--ao=null", "--hwdec=no", "--frames=1",
               f"--vf={specification}", str(VIDEO), env=env)


class VsrChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if (ROOT / "local/nvvfx/lib/libNVCVImage.so").exists():
            raise unittest.SkipTest("These checks require the default runtime to be absent")

    def test_start_without_runtime(self):
        version = run("--version")
        self.assertEqual(version.returncode, 0, version.stdout)
        self.assertIn("mpv v0.41.0", version.stdout)
        self.assertNotIn("VSR:", version.stdout)
        playback = run("--vo=null", "--ao=null", "--hwdec=no", "--frames=2", str(VIDEO))
        self.assertEqual(playback.returncode, 0, playback.stdout)
        self.assertIn("VO: [null]", playback.stdout)
        self.assertNotIn("VSR:", playback.stdout)

    def test_registration(self):
        result = run("--vf=help")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("NVIDIA NvVFX fixed-scale CUDA super resolution", result.stdout)
        builtins = result.stdout.split("Available libavfilter filters:")[0]
        names = [line.split()[0] for line in builtins.splitlines()[1:] if line.strip()]
        self.assertNotIn("hwup", names)
        self.assertNotIn("rife", names)

    def test_all_fixed_scales_and_qualities(self):
        for scale in ("4/3", "1.333333", "1.5", "2", "3", "4"):
            for quality in ("low", "medium", "high", "ultra"):
                with self.subTest(scale=scale, quality=quality):
                    result = filter_run(f"vsr:scale={scale}:quality={quality}")
                    self.assertGreater(result.returncode, 0, result.stdout)
                    self.assertIn("NvVFX runtime unavailable", result.stdout)
                    self.assertNotIn("Error parsing", result.stdout)
                    self.assertNotIn("scale must be", result.stdout)
                    self.assertNotIn("CUDA", result.stdout)

    def test_default_and_loading_order(self):
        result = filter_run("vsr")
        self.assertGreater(result.returncode, 0, result.stdout)
        first = result.stdout.index(str(ROOT / "local/missing-nvvfx-test"))
        second = result.stdout.index(str(ROOT / "local/nvvfx/lib"))
        self.assertLess(first, second)
        self.assertIn("NvVFX runtime unavailable", result.stdout)

    def test_reject_unsupported_scales(self):
        for scale in ("auto", "off", "1", "1.25", "2.5", "5", "nan", "inf", "4/0"):
            with self.subTest(scale=scale):
                result = filter_run(f"vsr:scale={scale}:quality=high")
                self.assertGreater(result.returncode, 0, result.stdout)
                self.assertNotIn("NvVFX runtime unavailable", result.stdout)
                self.assertTrue("Error parsing" in result.stdout or
                                "scale must be" in result.stdout, result.stdout)

    def test_reject_unknown_quality(self):
        result = filter_run("vsr:scale=2:quality=extreme")
        self.assertGreater(result.returncode, 0, result.stdout)
        self.assertIn("Error parsing", result.stdout)
        self.assertNotIn("NvVFX runtime unavailable", result.stdout)

    def test_no_relative_library_lookup(self):
        env = dict(ENV, MPV_NVVFX_LIBDIR=".")
        result = filter_run("vsr:scale=2:quality=high", env=env)
        self.assertGreater(result.returncode, 0, result.stdout)
        self.assertIn("MPV_NVVFX_LIBDIR must be an absolute path", result.stdout)
        self.assertNotIn("cannot load", result.stdout)

    def test_existing_filter(self):
        result = filter_run("format:fmt=rgba")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("rgba", result.stdout)
        self.assertNotIn("VSR:", result.stdout)

    def test_no_new_linked_runtime(self):
        result = subprocess.run(["readelf", "-d", str(BINARY)],
                                capture_output=True, text=True, check=True)
        needed = "\n".join(line for line in result.stdout.splitlines() if "NEEDED" in line)
        for name in ("libcuda", "libnvrtc", "libVideoFX", "libNVCVImage", "libnvinfer"):
            self.assertNotIn(name, needed)


if __name__ == "__main__":
    unittest.main()
