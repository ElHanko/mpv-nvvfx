#!/usr/bin/env bash
set -euo pipefail

cd /src

echo '=== Environment ==='
gcc --version | head -n 1
meson --version
ninja --version
nvcc --version | tail -n 1
echo

rm -rf build

echo '=== Configure mpv ==='

meson setup build \
  -Dgpl=true \
  -Dcplayer=true \
  -Dlibmpv=false \
  -Dvulkan=enabled \
  -Dx11=enabled \
  -Dwayland=disabled \
  -Dcuda-hwaccel=enabled \
  -Dcuda-interop=enabled \
  -Dnvvfx-vsr=enabled \
  -Dpipewire=enabled \
  -Dalsa=enabled \
  -Dlua=lua5.2 \
  --buildtype=release

echo
echo '=== Build mpv ==='

ninja -C build

echo
echo '=== Result ==='

build/mpv --version

echo
echo 'Binary: /src/build/mpv'
