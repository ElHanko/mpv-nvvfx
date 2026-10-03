#!/usr/bin/env python3
"""Offline regression tests for prepare.sh; no downloads or Docker required."""
import contextlib
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / 'prepare.sh').read_text()
PYTHON = compile(SCRIPT.split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0],
                 'prepare.sh (container Python)', 'exec')
WHEEL = 'nvidia_vfx-0.2.0.0-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
LIBRARIES = [
    'libcudart-256e6409.so.12.9.79',
    'libnppc.so.12', 'libnppial.so.12', 'libnppicc.so.12', 'libnppidei.so.12',
    'libnppif.so.12', 'libnppig.so.12', 'libnppim.so.12', 'libnppist.so.12',
    'libnppitc.so.12', 'libcudnn.so.9', 'libNVCVImage.so', 'libnvngxruntime.so',
    'libVideoFXLocal.so', 'libVideoFX.so', 'libnvVFXVideoSuperRes.so',
    'libnvidia-ngx-vsr.so.1.8.3',
]


def make_wheel(missing=None):
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w') as archive:
        for name in LIBRARIES:
            if name != missing:
                directory = 'nvidia_vfx.libs/' if name.startswith('libcudart-') else 'nvvfx/libs/'
                archive.writestr(directory + name, b'fixture library ' + name.encode())
        archive.writestr('nvvfx/libs/libnvVFXTrueHDR.so', b'excluded')
        archive.writestr('nvvfx/libs/libnvVFXVideoFrameGeneration.so', b'excluded')
        archive.writestr('nvvfx/libs/libnvinfer.so', b'excluded')
    return buffer.getvalue()


class PreparationTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'local').mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='prepare-test-', dir=ROOT / 'local')
        self.addCleanup(self.temporary.cleanup)
        self.runtime = Path(self.temporary.name)
        (self.runtime / 'downloads').mkdir()
        (self.runtime / 'nvvfx/lib').mkdir(parents=True)
        self.wheel = self.runtime / 'downloads' / WHEEL
        self.data = make_wheel()

    def run_python(self, download=None):
        wheel_url = 'https://pypi.nvidia.com/packages/' + WHEEL
        index = ('<a href="nvidia_vfx-newer.whl">newer</a>'
                 f'<a href="{wheel_url}#sha256=ignored">pinned</a>').encode()

        def urlopen(url, timeout):
            if url == 'https://pypi.nvidia.com/nvidia-vfx/':
                return io.BytesIO(index)
            self.assertEqual(url, wheel_url + '#sha256=ignored')
            return io.BytesIO(download if download is not None else self.data)

        def runtime_path(path):
            return self.runtime / str(path).removeprefix('/runtime/')

        # The fixture's hash replaces the production hash only in argv.
        # Production code still computes and compares the real SHA256.
        argv = ['-', WHEEL, hashlib.sha256(self.data).hexdigest(), *LIBRARIES]
        with patch('pathlib.Path', side_effect=runtime_path), \
                patch('urllib.request.urlopen', side_effect=urlopen) as request, \
                patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
            exec(PYTHON, {})
        return request

    def test_exact_download_allowlist_and_repeat_without_network(self):
        request = self.run_python()
        self.assertEqual(request.call_count, 2)
        self.assertEqual(self.wheel.read_bytes(), self.data)
        libdir = self.runtime / 'nvvfx/lib'
        self.assertEqual({p.name for p in libdir.iterdir()}, set(LIBRARIES))
        before = {p.name: p.read_bytes() for p in libdir.iterdir()}
        self.assertEqual(self.run_python().call_count, 0)
        self.assertEqual({p.name: p.read_bytes() for p in libdir.iterdir()}, before)

    def test_bad_cached_hash_aborts_without_extraction(self):
        self.wheel.write_bytes(b'corrupt cached wheel')
        with self.assertRaisesRegex(SystemExit, 'SHA256 falsch'):
            self.run_python()
        self.assertEqual(list((self.runtime / 'nvvfx/lib').iterdir()), [])
        self.assertEqual(self.wheel.read_bytes(), b'corrupt cached wheel')

    def test_bad_download_hash_is_not_cached_or_extracted(self):
        with self.assertRaisesRegex(SystemExit, 'SHA256 falsch'):
            self.run_python(download=b'corrupt download')
        self.assertEqual(list((self.runtime / 'downloads').iterdir()), [])
        self.assertEqual(list((self.runtime / 'nvvfx/lib').iterdir()), [])

    def test_missing_library_aborts_before_extraction(self):
        self.data = make_wheel(missing='libVideoFX.so')
        self.wheel.write_bytes(self.data)
        with self.assertRaisesRegex(SystemExit, 'libVideoFX.so fehlt'):
            self.run_python()
        self.assertEqual(list((self.runtime / 'nvvfx/lib').iterdir()), [])

    def test_unignored_local_aborts_before_creating_runtime(self):
        fixture = self.runtime / 'repo'
        fixture.mkdir()
        subprocess.run(['git', 'init', '-q', str(fixture)], check=True)
        script = fixture / 'prepare.sh'
        script.write_text(SCRIPT)
        result = subprocess.run(['bash', str(script)], cwd=ROOT, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('local/ wird nicht von Git ignoriert', result.stderr)
        self.assertFalse((fixture / 'local').exists())

    def test_tracked_local_aborts_before_creating_runtime(self):
        # Simulate tracked proprietary files without staging any real files.
        fixture = self.runtime / 'repo'
        fixture.mkdir()
        script = fixture / 'prepare.sh'
        script.write_text(SCRIPT)
        bindir = self.runtime / 'bin'
        bindir.mkdir()
        git = bindir / 'git'
        git.write_text("#!/bin/sh\nprintf '%s\\n' local/runtime.so\n")
        git.chmod(0o755)
        env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'])
        result = subprocess.run(['bash', str(script)], env=env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('local/ enthält versionierte Dateien', result.stderr)
        self.assertFalse((fixture / 'local').exists())


if __name__ == '__main__':
    unittest.main()
