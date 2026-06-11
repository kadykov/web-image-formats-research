FROM mcr.microsoft.com/devcontainers/python:3.14-trixie

# Version variables for dependencies
ARG JPEG_XL_VERSION=v0.11.2
ARG DAV1D_VERSION=1.5.3
ARG LIBAOM_VERSION=v3.13.1
ARG LIBAVIF_VERSION=v1.3.0

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    just \
    build-essential \
    cmake \
    ninja-build \
    nasm \
    pkg-config \
    meson \
    git \
    ca-certificates \
    curl \
    p7zip-full \
    && rm -rf /var/lib/apt/lists/*

# Install Rust for building ssimulacra2
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install image encoding tools from Debian packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    # ImageMagick for general image manipulation
    imagemagick \
    # WebP tools (libwebp-dev includes cwebp and dwebp)
    libwebp-dev \
    webp \
    # HEIF/AVIF support (libheif includes heif-enc/heif-convert for AVIF)
    libheif-examples \
    libheif-plugin-dav1d \
    # JPEG tools
    libjpeg-turbo-progs \
    libjpeg-dev \
    # Brotli (required for libjxl) and other build deps
    libbrotli-dev \
    # General image processing libraries
    libpng-dev \
    libtiff-dev \
    libyuv-dev \
    && rm -rf /var/lib/apt/lists/*

# Build and install libjxl (JPEG XL) from source — fallback when prebuilt artifacts are unavailable
# Clones the tagged release, initializes submodules, builds tools, and installs them
RUN git clone --depth 1 --branch ${JPEG_XL_VERSION} https://github.com/libjxl/libjxl.git /tmp/libjxl && \
    cd /tmp/libjxl && \
    git submodule update --init --recursive --depth 1 || true && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_TESTING=OFF \
          -DJPEGXL_ENABLE_DEVTOOLS=ON \
          -DJPEGXL_ENABLE_TOOLS=ON \
          -GNinja \
          .. && \
    ninja && \
    cmake --install . && \
    ldconfig && \
    cd / && rm -rf /tmp/libjxl

# Build and install dav1d decoder
RUN git clone --depth 1 --branch ${DAV1D_VERSION} https://code.videolan.org/videolan/dav1d.git /tmp/dav1d && \
    cd /tmp/dav1d && \
    mkdir -p build && cd build && \
    meson setup --buildtype=release .. && \
    ninja && \
    ninja install && \
    ldconfig && \
    cd / && rm -rf /tmp/dav1d

# Build and install libaom encoder
RUN git clone --depth 1 --branch ${LIBAOM_VERSION} https://aomedia.googlesource.com/aom /tmp/aom && \
    cd /tmp/aom && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DENABLE_TESTS=OFF \
          -DENABLE_DOCS=OFF \
          -DENABLE_EXAMPLES=OFF \
          -GNinja \
          .. && \
    ninja && \
    ninja install && \
    ldconfig && \
    cd / && rm -rf /tmp/aom

# Build and install libavif with system dav1d and libaom
RUN git clone --depth 1 --branch ${LIBAVIF_VERSION} https://github.com/AOMediaCodec/libavif.git /tmp/libavif && \
    cd /tmp/libavif && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DAVIF_CODEC_DAV1D=SYSTEM \
          -DAVIF_CODEC_AOM=SYSTEM \
          -DAVIF_BUILD_APPS=ON \
          -DAVIF_BUILD_TESTS=OFF \
          -DAVIF_BUILD_EXAMPLES=OFF \
          -GNinja \
          .. && \
    ninja && \
    ninja install && \
    ldconfig && \
    cd / && rm -rf /tmp/libavif

# Install additional image quality metrics tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    # FFmpeg for VMAF and other metrics
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and npm for markdown tooling
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /workspace
