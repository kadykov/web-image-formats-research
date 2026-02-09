FROM mcr.microsoft.com/devcontainers/python:3-3.13-trixie

# Fix Yarn GPG key for Debian Trixie
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | gpg --dearmor -o /usr/share/keyrings/yarnkey.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/yarnkey.gpg] https://dl.yarnpkg.com/debian stable main" > /etc/apt/sources.list.d/yarn.list

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
    # General image processing libraries
    libpng-dev \
    libtiff-dev \
    libyuv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install JPEG XL tools from static binaries
# This includes cjxl, djxl, and other tools
RUN curl -L https://github.com/libjxl/libjxl/releases/download/v0.11.1/jxl-linux-x86_64-static-v0.11.1.tar.gz -o /tmp/jxl.tar.gz && \
    mkdir -p /tmp/jxl-extract && \
    tar -xzf /tmp/jxl.tar.gz -C /tmp/jxl-extract --strip-components=1 && \
    cp /tmp/jxl-extract/* /usr/local/bin/ && \
    rm -rf /tmp/jxl-extract /tmp/jxl.tar.gz && \
    chmod +x /usr/local/bin/*

# Build and install dav1d decoder
RUN git clone --depth 1 --branch 1.5.0 https://code.videolan.org/videolan/dav1d.git /tmp/dav1d && \
    cd /tmp/dav1d && \
    mkdir -p build && cd build && \
    meson setup --buildtype=release .. && \
    ninja && \
    ninja install && \
    ldconfig && \
    cd / && rm -rf /tmp/dav1d

# Build and install libaom encoder
RUN git clone --depth 1 --branch v3.11.0 https://aomedia.googlesource.com/aom /tmp/aom && \
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
RUN git clone --depth 1 --branch v1.1.1 https://github.com/AOMediaCodec/libavif.git /tmp/libavif && \
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


# Set up working directory
WORKDIR /workspace
