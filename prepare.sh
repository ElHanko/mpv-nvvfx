#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="mpv-nvvfx-build:debian13-cuda13.2"
WHEEL="nvidia_vfx-0.2.0.0-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
SHA256="0252866a7b798a6f92c8543443d88f716e8c5bbc6a7ddd77a2754728dada9fa8"
LIBRARIES=(
  libcudart-256e6409.so.12.9.79
  libnppc.so.12 libnppial.so.12 libnppicc.so.12 libnppidei.so.12
  libnppif.so.12 libnppig.so.12 libnppim.so.12 libnppist.so.12 libnppitc.so.12
  libcudnn.so.9 libNVCVImage.so libnvngxruntime.so libVideoFXLocal.so libVideoFX.so
  libnvVFXVideoSuperRes.so libnvidia-ngx-vsr.so.1.8.3
)
NVRTC=(libnvrtc.so.13 libnvrtc-builtins.so.13.2)

fail() { echo "prepare.sh: $*" >&2; exit 1; }
trap 'echo "prepare.sh: Fehler in Zeile $LINENO; Vorbereitung abgebrochen." >&2' ERR

check_git_safety() {
  [[ -z "$(git -C "$ROOT_DIR" ls-files -- local/)" ]] ||
    fail 'local/ enthält versionierte Dateien; proprietäre Runtime wird nicht vorbereitet.'
  git -C "$ROOT_DIR" check-ignore -q -- local/ ||
    fail 'local/ wird nicht von Git ignoriert; zuerst /local/ in .gitignore eintragen.'
}

check_git_safety
command -v docker >/dev/null || fail 'Docker wird benötigt; keine Host-Pakete werden installiert.'
mkdir -p "$ROOT_DIR/local/downloads" "$ROOT_DIR/local/nvvfx/lib"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "=== Build Docker image: $IMAGE ==="
  docker build -t "$IMAGE" -f "$ROOT_DIR/Dockerfile" "$ROOT_DIR"
fi

echo '=== NVIDIA VFX 0.2.0.0: Download, SHA256 und VSR-Extraktion ==='
docker run --rm -i --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$ROOT_DIR/local:/runtime" "$IMAGE" \
  python3 - "$WHEEL" "$SHA256" "${LIBRARIES[@]}" <<'PY'
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import shutil
import sys
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import urlopen
from zipfile import ZipFile

filename, expected_hash, *libraries = sys.argv[1:]
wheel = Path('/runtime/downloads') / filename


def verify(path):
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected_hash:
        sys.exit(f'prepare.sh: SHA256 falsch für {path.name}: '
                 f'{actual}, erwartet {expected_hash}; Datei wird nicht verwendet.')
    print(f'SHA256 verifiziert: {actual}', flush=True)


if wheel.exists():
    verify(wheel)
    print(f'Wheel wiederverwendet: {filename}', flush=True)
else:
    index_url = 'https://pypi.nvidia.com/nvidia-vfx/'

    class WheelLinks(HTMLParser):
        urls = []

        def handle_starttag(self, tag, attrs):
            href = dict(attrs).get('href')
            if tag == 'a' and href:
                url = urljoin(index_url, href)
                if (urlsplit(url).scheme == 'https' and
                        unquote(urlsplit(url).path).rsplit('/', 1)[-1] == filename):
                    self.urls.append(url)

    parser = WheelLinks()
    with urlopen(index_url, timeout=60) as response:
        parser.feed(response.read().decode('utf-8'))
    if not parser.urls:
        sys.exit(f'prepare.sh: Exakter Wheel {filename} fehlt auf {index_url}.')
    partial = wheel.with_name(filename + '.part')
    try:
        print(f'Download von NVIDIA: {filename}', flush=True)
        with urlopen(parser.urls[0], timeout=60) as response, partial.open('wb') as out:
            shutil.copyfileobj(response, out)
        verify(partial)
        partial.replace(wheel)
    finally:
        partial.unlink(missing_ok=True)

with ZipFile(wheel) as archive:
    members = {
        name: ('nvidia_vfx.libs/' if name.startswith('libcudart-') else 'nvvfx/libs/') + name
        for name in libraries
    }
    for name, member in members.items():
        if member not in archive.namelist():
            sys.exit(f'prepare.sh: Benötigte Library {name} fehlt im Wheel.')
    for name, member in members.items():
        target = Path('/runtime/nvvfx/lib') / name
        partial = target.with_name(name + '.tmp')
        try:
            with archive.open(member) as source, partial.open('wb') as out:
                shutil.copyfileobj(source, out)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
        print(f'Extrahiert: {name}', flush=True)
PY

echo '=== CUDA 13.2: NVRTC kopieren und Runtime prüfen ==='
docker run --rm -i --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$ROOT_DIR/local:/runtime" "$IMAGE" \
  bash -s -- "${LIBRARIES[@]}" "${NVRTC[@]}" <<'SH'
set -euo pipefail
cd /runtime/nvvfx/lib
trap 'rm -f libnvrtc.so.13.tmp libnvrtc-builtins.so.13.2.tmp' EXIT
for name in libnvrtc.so.13 libnvrtc-builtins.so.13.2; do
  cp -L "/usr/local/cuda-13.2/targets/x86_64-linux/lib/$name" "$name.tmp"
  mv -f "$name.tmp" "$name"
  echo "Kopiert aus CUDA 13.2: $name"
done
for name in "$@"; do
  if [[ ! -s "$name" || -L "$name" ]]; then
    echo "prepare.sh: Benötigte Runtime-Datei fehlt oder ist keine reale Datei: $name" >&2
    exit 1
  fi
done
for name in libnvrtc.so.13 libnvrtc-builtins.so.13.2; do
  soname="$(readelf -d "$name" | sed -n 's/.*(SONAME).*\[\(.*\)\].*/\1/p')"
  if [[ "$soname" != "$name" ]]; then
    echo "prepare.sh: Falscher SONAME für $name: '$soname' (erwartet '$name')." >&2
    exit 1
  fi
  echo "SONAME verifiziert: $name"
done
SH

check_git_safety
cat > "$ROOT_DIR/local/nvvfx/PREPARED.txt" <<EOF
NVIDIA_VFX_VERSION=0.2.0.0
NVIDIA_VFX_WHEEL=$WHEEL
NVIDIA_VFX_WHEEL_SHA256=$SHA256
CUDA_VERSION=13.2
PREPARED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
echo 'Runtime vorbereitet in local/nvvfx/lib; nächster Schritt: ./build.sh'
