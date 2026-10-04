# Fixed-scale NvVFX VSR MVP

The `vsr` user filter targets upstream mpv v0.41.0 on Linux, with CUDA
NVDEC input and the existing gpu-next CUDA/Vulkan interop output. mpv remains
the player. No NVIDIA runtime or model files are distributed with this code.

## Fresh clone / reproducible setup

Docker and Git are required. Clone the public mpv-nvvfx fork (replace the URL
placeholder with its repository URL), then prepare the pinned runtime locally:

```sh
git clone <public-mpv-nvvfx-fork-url> mpv-nvvfx
cd mpv-nvvfx
./prepare.sh
./build.sh
```

The NVIDIA runtime is not redistributed in this repository. `prepare.sh`
downloads only NVIDIA VFX **0.2.0.0** from `https://pypi.nvidia.com`, without
installing the wheel or its dependencies. The exact wheel is
`nvidia_vfx-0.2.0.0-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`;
its SHA256 must match
`0252866a7b798a6f92c8543443d88f716e8c5bbc6a7ddd77a2754728dada9fa8`.
A cached wheel is verified and reused; a checksum mismatch aborts preparation.
Only the VSR libraries listed below are extracted, without TrueHDR, frame
generation, TensorRT or ONNX parser libraries.

Download and extraction use disposable containers running the same
`mpv-nvvfx-build:debian13-cuda13.2` image as `build.sh`; `prepare.sh` builds that
image from the Dockerfile if it is missing. CUDA/NVRTC 13.2 is copied locally
from this image, with symlinks dereferenced. Both NVRTC SONAMEs and all required
runtime files are checked. No CUDA toolkit or other packages are installed on
the host, and host pip/unzip/readelf are not required.

`local/`, `build/` and `output/` are Git-ignored and excluded from the Docker
build context. Preparation refuses to write runtime files unless `local/` is
ignored and contains no tracked files, and checks this again on completion.
Downloads live in `local/downloads`, runtime files in `local/nvvfx/lib`, and
version/checksum/time metadata in `local/nvvfx/PREPARED.txt`. Repeated runs
reuse the verified wheel and replace the required libraries cleanly. The script
can be invoked from any working directory.

Offline preparation regression tests (with an existing host Python 3) run via
`python3 test/nvvfx-vsr/check-prepare.py`; they cover exact-wheel selection,
checksum failures, the extraction allowlist, repeat runs and Git safety guards.

Test playback with an NVIDIA driver/GPU supporting the existing VSR path:

```sh
MPV_NVVFX_LIBDIR="$PWD/local/nvvfx/lib" \
./build/mpv \
  --vo=gpu-next \
  --gpu-api=vulkan \
  --hwdec=nvdec \
  --vf='vsr:scale=2:quality=ultra' \
  video.mp4
```

## Build and use

Build only through the existing Debian 13 / CUDA 13.2 Docker environment:

```sh
cd mpv-nvvfx
./build.sh
```

The Docker configuration explicitly enables `-Dnvvfx-vsr=enabled`. The Meson
feature defaults to disabled for upstream builds; enabled/auto require GPL,
Linux, CUDA interop, dlopen and the CUDA toolkit's NVRTC development files.
Meson discovers CUDA rather than using the reference's `/opt/cuda` paths.
Only toolkit include/compile flags are propagated: CUDA, NVRTC, NvVFX and
TensorRT are not added to mpv's link dependencies or toolkit rpath.

```sh
./build/mpv \
  --no-config \
  --vo=gpu-next \
  --gpu-api=vulkan \
  --hwdec=nvdec \
  --vf='vsr:scale=2:quality=high' \
  video.mp4
```

Defaults: `scale=2`, `quality=high`. Supported fixed scales are `4/3`,
`1.333333`, `1.5`, `2`, `3`, `4`. Factors within 0.000001 of a supported
factor are canonicalized, so `1.333333` and `4/3` produce the same dimensions.
Output dimensions are rounded to the nearest integer. The requested example
configures 1280×720 → 2560×1440. Unsupported factors, including `auto` and
`off`, fail. There is no automatic render-target scaling.

Supported qualities use the same `VideoSuperRes` effect in NVIDIA VFX 0.2.0.0:

| Quality | QualityLevel |
| --- | --- |
| `low` | 1 |
| `medium` | 2 |
| `high` | 3 |
| `ultra` | 4 |
| `highbitrate-low` | 16 |
| `highbitrate-medium` | 17 |
| `highbitrate-high` | 18 |
| `highbitrate-ultra` | 19 |

Standard modes 1–4 suit typically compressed video and also reduce compression
artifacts during upscaling. High-bitrate modes 16–19 target high-quality sources
with few compression artifacts. `highbitrate-ultra` is particularly useful for
high-quality Blu-ray/remux-like sources. The mpv filter does not select a mode
automatically; future Jellyfin/wrapper automation is outside this change.

```sh
MPV_NVVFX_LIBDIR="$PWD/local/nvvfx/lib" \
./build/mpv \
  --vo=gpu-next \
  --gpu-api=vulkan \
  --hwdec=nvdec \
  --vf='vsr:scale=2:quality=highbitrate-ultra' \
  video.mkv
```

## Runtime loading

NvVFX is loaded only when an actual `vsr` filter is created.
`MPV_NVVFX_LIBDIR` is required and must point to the absolute directory
containing the prepared NvVFX runtime. Relative paths are rejected. There is
no compiled-in user-specific fallback path, and the filter never searches the
working directory for NvVFX. The currently ELF-verified runtime layout is
NVIDIA VFX 0.2.0.0. Every library below is mandatory and loaded in this order,
using absolute paths and `RTLD_NOW | RTLD_GLOBAL`:

```text
libcudart-256e6409.so.12.9.79
libnppc.so.12
libnppial.so.12
libnppicc.so.12
libnppidei.so.12
libnppif.so.12
libnppig.so.12
libnppim.so.12
libnppist.so.12
libnppitc.so.12
libcudnn.so.9
libNVCVImage.so
libnvngxruntime.so
libVideoFXLocal.so
libVideoFX.so
libnvVFXVideoSuperRes.so
libnvidia-ngx-vsr.so.1.8.3
```

No TensorRT, libnvinfer or ONNX parser is required or loaded. The loader does
not rely on wheel-specific RPATH entries, modify `LD_LIBRARY_PATH`, add system
search paths or create symlinks. `libcuda.so.1` comes exclusively from the
NVIDIA host driver; it is never searched for in the local runtime directory.
Any dlopen failure reports the library name, complete attempted path and
dlerror, then releases the directory's loaded handles before failure.
All successful dlopen handles are kept per filter and closed in reverse order.

The runtime needs `libNVCVImage.so`, `libVideoFX.so` and the matching dependency,
effect and model files for `VideoSuperRes`. The filter checks required API
symbols and each NvVFX setup/run status. Missing libraries, symbols or models
produce an mpv error. The local runtime directory is ignored by Git.

The CUDA driver is loaded through mpv's existing ffnvcodec loader. NVRTC is
loaded when the first conversion kernel is needed: first
`<selected NvVFX directory>/libnvrtc.so.13`, then the system loader's
`libnvrtc.so.13`. Before loading NVRTC, the filter explicitly preloads
`<selected NvVFX directory>/libnvrtc-builtins.so.13.2` with
`RTLD_NOW | RTLD_GLOBAL`. Its verified SONAME is `libnvrtc-builtins.so.13.2`,
matching NVRTC's internal dlopen name; that lookup can reuse the preloaded
library without `LD_LIBRARY_PATH`. The builtins handle remains open until
after NVRTC is closed, including error cleanup. Preload failures report the
library name, full path and dlerror. Installing
CUDA on the host is not a prerequisite for normal mpv startup/playback; a
later real VSR test must supply NVRTC and the NvVFX runtime through authorized
runtime provisioning. This implementation downloads or installs neither.

## GPU data flow and lifetime

1. Receive `IMGFMT_CUDA` with NV12 or P010 planes from NVDEC.
2. Retain the input hardware-frames reference and borrow its CUDA context.
3. Create a filter stream; record a CUDA event on the decoder stream and wait
   for it on the filter stream before reading the decoded planes.
4. Convert YUV directly into NvVFX's GPU RGBA U8 input using an NVRTC kernel.
   BT.601, BT.709 and BT.2020 nonconstant luminance, limited/full range are
   supported. P010 samples are read as MSB-aligned 16-bit storage. The port
   fixes limited-range normalization to 219 luma / 224 chroma code values.
5. Run `VideoSuperRes` with `SrcImage0`, `DstImage0`, `QualityLevel` and
   `CudaStream` on that same borrowed context.
6. Copy output device-to-device into an FFmpeg CUDA/RGBA frame allocated from
   a separate frames pool using the original device reference. Each output
   buffer has an independent lifetime; NvVFX cannot overwrite an older frame.
7. Finish the filter stream before freeing input or handing output to mpv.
   gpu-next uses its unchanged CUDA/Vulkan GPU-memory interop.

There are no host pixel buffers, hardware downloads, software conversions or
uploads in this path. CPU stream waits do not copy pixels. Timestamps, frame
attributes and color transfer/primaries are retained, RGB full-range metadata
is set, crop coordinates are scaled and pixel aspect is adjusted for rounded
output dimensions. Seeking or input format/size/context changes rebuilds the
pipeline. Cleanup waits only for the filter's stream, not the VO's stream or
an entire CUDA context.

## Provenance and license

The six files in `video/filter/` are derived from the corresponding files in
`src/mpv/video/filter/` of `zhangmq/vsr-player`, reference commit
`f7941984e29c687bf2e318646728f06ee5e5cd0a`, read from
from a local checkout of that reference commit without executing its scripts.
The reference repository's `LICENSE` is GPL version 2. The six source files
carry GPL-2.0-only SPDX/provenance notices and refer to mpv's existing
`LICENSE.GPL`. Those reference files contained no individual copyright
notices to retain; existing upstream copyright notices are unchanged.

Specifically:

- `yuv_to_rgba.c/.h`: NVRTC kernel, matrix coefficients and launch layout are
  adapted; limited-range normalization, pitch indexing and error cleanup are
  corrected; NVRTC is loaded dynamically.
- `vsr_proc.c` and `vsr_internal.h`: NvCVImage C ABI layout, library dependency
  order, VideoSuperRes effect name, image descriptors and parameter names are
  adapted. Per-instance loading replaces the reference's global loader.
- `vf_vsr.c` and `cuda_shared.h`: borrowing decoder context/stream, GPU format
  normalization and CUDA/RGBA output lifetime concepts are adapted. FFmpeg
  manages the output frames on the existing device; the reference's custom
  context refcount/buffer destructors and secondary device creation are absent.
- Only `vf_vsr` is added to `filters/user_filters.c/.h`. The upstream double
  option parser already accepts ratios, so no custom option type is ported.
- Meson lists only these three C translation units. Existing player, VO,
  interop and decoder implementation files are unchanged.

Not ported: scale=auto, render-target/VOCTRL_GET_RENDER_TARGET_SIZE queries,
X11/Wayland patches, hwup, RIFE, TensorRT-RIFE, frame interpolation,
Qt/QML/libmpv frontend functions, RPC, browser integration, GUI, denoise mode,
filter commands, screenshot/status plumbing, hot quality/output changes,
random-data warmup or retirement queues. The NVIDIA VFX 0.2.0.0 loader has
no TensorRT dependencies.

## Verification (2026-10-03, initial MVP before runtime provisioning)

- `./build.sh`: successful, including the container's final `build/mpv --version`.
  The final build has no compiler warnings/errors. Dockerfile, build.sh,
  docker/build-mpv.sh and .dockerignore are retained; only the Meson enable
  flag was added to docker/build-mpv.sh. No host build or install was used.
- Host `./build/mpv --version`: successful with no NvVFX runtime present.
- Host normal playback, gpu-next/Vulkan/NVDEC, five frames: exit 0 and
  `Using hardware decoding (nvdec).`
  `VO: [gpu-next] 1280x720 cuda[nv12]`.
- Host requested VSR options with gpu-next/Vulkan/NVDEC, five-frame limit:
  controlled exit 2 with `VSR: NvVFX runtime unavailable`; no crash/abort/hang.
- `python3 test/nvvfx-vsr/check.py video.mp4`:
  nine tests pass. Tests exercise all 24 scale/quality combinations, invalid
  scales/quality, defaults, required/absolute runtime path handling, relative-path rejection,
  startup/playback without runtime, an existing format filter, registration
  and the absence of new CUDA/NVRTC/NvVFX/TensorRT ELF NEEDED entries.
- Full `--vf=help` output compared with the pre-change binary: identical after
  removing the one new built-in vsr row, including all libavfilter entries.
- `git diff --check`: clean. New files also checked for trailing whitespace.
- No host packages/driver changes, systemwide install, proprietary downloads,
  reference-repository writes or commits.

Local build/test logs are in `test/nvvfx-vsr/*.log` and are Git-ignored.

## Remaining validation and limits

NVIDIA VFX 0.2.0.0 libraries are now present in `local/nvvfx/lib`.
After the NVRTC builtins preload fix, the ten-second test video plays to EOF
with `MPV_NVVFX_LIBDIR` set, `--vf='vsr:scale=2:quality=high'` and no
`LD_LIBRARY_PATH`: NVDEC 1280×720 → VideoSuperRes (quality=3) → gpu-next/Vulkan
2560×1440 CUDA/RGBA, exit 0. The exact `--vf='vsr=2=high'` spelling fails at
option parsing in the current build; parsing is unchanged by this fix.
Image correctness, P010, seeking/reconfiguration and performance have not
been systematically validated.

The MVP accepts only CUDA NV12/P010 input; software frames fail. RGBA is U8,
so P010 precision is reduced to eight bits. There is no HDR tone mapping or
constant-luminance/Dolby Vision conversion. Each output dimension is limited
to 8192; the actual runtime may impose narrower limits. Processing synchronizes
once per frame and makes one GPU output copy for correct buffer ownership;
performance tuning is outside this MVP. Runtime library naming matches the
locally ELF-verified NVIDIA VFX 0.2.0.0 layout.
