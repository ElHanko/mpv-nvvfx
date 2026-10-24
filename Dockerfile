FROM debian:forky-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i \
      's/^Components: main$/Components: main contrib non-free non-free-firmware/' \
      /etc/apt/sources.list.d/debian.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      git \
      meson \
      ninja-build \
      pkgconf \
      python3 \
      libass-dev \
      libavcodec-dev \
      libavfilter-dev \
      libavformat-dev \
      libavutil-dev \
      libswresample-dev \
      libswscale-dev \
      libplacebo-dev \
      libarchive-dev \
      libasound2-dev \
      libdisplay-info-dev \
      libdrm-dev \
      libegl-dev \
      libffmpeg-nvenc-dev \
      libgbm-dev \
      libgl-dev \
      libjpeg-dev \
      liblcms2-dev \
      libpipewire-0.3-dev \
      libuchardet-dev \
      libvulkan-dev \
      libx11-dev \
      libxinerama-dev \
      libxkbcommon-dev \
      libxpresent-dev \
      libxrandr-dev \
      libxss-dev \
      liblua5.2-dev \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLo /tmp/cuda-keyring.deb \
      https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i /tmp/cuda-keyring.deb \
 && rm /tmp/cuda-keyring.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      cuda-minimal-build-13-2 \
      cuda-driver-dev-13-2 \
      cuda-nvrtc-13-2 \
      cuda-nvrtc-dev-13-2 \
 && rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda-13.2/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64

WORKDIR /src
