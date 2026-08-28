# llama-swap for AMD GPUs — ROCm + Vulkan in one image
#
# Extends the upstream unified Vulkan image (which already contains Vulkan
# builds of llama.cpp, whisper.cpp, stable-diffusion.cpp and audio.cpp, plus
# the llama-swap and vllm-wrapper binaries) with:
#
#   - the ROCm userspace runtime (HIP runtime, rocBLAS/hipBLAS + Tensile
#     kernels, hipBLASLt, rocminfo) from AMD's apt repository
#   - HIP rebuilds of llama.cpp, whisper.cpp and stable-diffusion.cpp,
#     installed as *-rocm binaries next to the Vulkan ones:
#       llama-server-rocm, llama-cli-rocm, llama-tts-rocm, llama-bench-rocm,
#       whisper-server-rocm, whisper-cli-rocm, sd-server-rocm, sd-cli-rocm
#
# The ROCm builds are pinned to the same project commits as the base image
# (parsed from its /versions.txt), so both backends of each binary run the
# same upstream revision. audio.cpp has no HIP backend upstream, so only its
# Vulkan build is present.
#
# Build:
#   docker buildx build -t llama-swap-amd .
#
# Run (container is root, so no --group-add is needed for device access):
#   docker run -it --rm --device /dev/kfd --device /dev/dri \
#     --security-opt seccomp=unconfined \
#     -p 8080:8080 -v $PWD/models:/models llama-swap-amd
#
# See README.md for build args, GPU support and runtime env vars.

# Must be the root variant (not *-rootless): ROCm packages are installed with
# apt in the final stage. Pin a dated tag or digest for reproducible builds.
ARG BASE_IMAGE=ghcr.io/mostlygeek/llama-swap:unified-vulkan

ARG ROCM_VERSION=7.2.1

# Fat build covering everything rocBLAS ships Tensile kernels for -- same list
# as llama.cpp's official ROCm image: CDNA1-3 (gfx908/90a/942), RDNA2 (gfx1030),
# RDNA3/3.5 (gfx1100/01/02, gfx1150/51), RDNA4 (gfx1200/01). GPUs not listed
# here can still use the Vulkan binaries.
ARG AMDGPU_TARGETS="gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201"

FROM ${BASE_IMAGE} AS vulkan-base

# ── ROCm builder base ──────────────────────────────────────────────────

FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION}-complete AS rocm-builder
ARG AMDGPU_TARGETS

ENV DEBIAN_FRONTEND=noninteractive
ENV AMDGPU_TARGETS=${AMDGPU_TARGETS}
ENV CCACHE_DIR=/ccache
ENV CCACHE_MAXSIZE=5G

# libav*-dev only for whisper.cpp's WHISPER_FFMPEG=ON; the base runtime already
# ships the matching Ubuntu 24.04 libav* runtime libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ccache curl ca-certificates \
        pkg-config libssl-dev \
        libavcodec-dev libavformat-dev libavutil-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# The commit of each project built into the base image, so the HIP rebuilds
# match the Vulkan binaries exactly.
COPY --from=vulkan-base /versions.txt /build/versions.txt

WORKDIR /build

# ── Build llama.cpp (HIP) ──────────────────────────────────────────────

FROM rocm-builder AS llama-rocm
# Leave empty to build the same commit as the base image's Vulkan binaries
ARG LLAMA_COMMIT=""
RUN --mount=type=cache,id=ccache-rocm,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

COMMIT="${LLAMA_COMMIT:-$(awk '$1=="llama.cpp:"{print $2}' /build/versions.txt)}"
[ -n "$COMMIT" ] || { echo "FATAL: no llama.cpp commit in versions.txt and no LLAMA_COMMIT given" >&2; exit 1; }

echo "=== Cloning llama.cpp at ${COMMIT} ==="
mkdir -p /src/llama.cpp && cd /src/llama.cpp
git init -q
git remote add origin https://github.com/ggml-org/llama.cpp.git
git fetch --depth=1 origin "${COMMIT}"
git checkout -q FETCH_HEAD

echo "=== Building llama.cpp (HIP) for ${AMDGPU_TARGETS} ==="
# POSITION_INDEPENDENT_CODE: ggml-hip's device-stub objects are non-PIC by
# default and fail to link into Ubuntu's default-PIE executables
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -B build \
    -DGGML_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLAMA_BUILD_TESTS=OFF \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="${AMDGPU_TARGETS}"
cmake --build build --config Release -j"$(nproc)" \
    --target llama-server llama-cli llama-tts llama-bench

mkdir -p /install/bin
for bin in llama-server llama-cli llama-tts llama-bench; do
    [ -f "build/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
    needed=$(readelf -d "build/bin/$bin" | grep NEEDED || true)
    grep -q 'libamdhip64\.so' <<<"$needed" || {
        echo "FATAL: $bin is not linked against the HIP runtime:" >&2
        echo "$needed" >&2; exit 1; }
    if grep -q 'libggml' <<<"$needed"; then
        echo "FATAL: $bin expects shared ggml libs; it would collide with the" >&2
        echo "       Vulkan libggml*.so already in the base image" >&2; exit 1; fi
    cp "build/bin/$bin" "/install/bin/${bin}-rocm"
done
BUILD

# ── Build whisper.cpp (HIP) ────────────────────────────────────────────

FROM rocm-builder AS whisper-rocm
ARG WHISPER_COMMIT=""
RUN --mount=type=cache,id=ccache-rocm,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

COMMIT="${WHISPER_COMMIT:-$(awk '$1=="whisper.cpp:"{print $2}' /build/versions.txt)}"
[ -n "$COMMIT" ] || { echo "FATAL: no whisper.cpp commit in versions.txt and no WHISPER_COMMIT given" >&2; exit 1; }

echo "=== Cloning whisper.cpp at ${COMMIT} ==="
mkdir -p /src/whisper.cpp && cd /src/whisper.cpp
git init -q
git remote add origin https://github.com/ggml-org/whisper.cpp.git
git fetch --depth=1 origin "${COMMIT}"
git checkout -q FETCH_HEAD

echo "=== Building whisper.cpp (HIP) for ${AMDGPU_TARGETS} ==="
# POSITION_INDEPENDENT_CODE: see llama.cpp stage
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -B build \
    -DGGML_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DWHISPER_FFMPEG=ON \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="${AMDGPU_TARGETS}"
cmake --build build --config Release -j"$(nproc)" \
    --target whisper-server whisper-cli

mkdir -p /install/bin
for bin in whisper-server whisper-cli; do
    [ -f "build/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
    needed=$(readelf -d "build/bin/$bin" | grep NEEDED || true)
    grep -q 'libamdhip64\.so' <<<"$needed" || {
        echo "FATAL: $bin is not linked against the HIP runtime:" >&2
        echo "$needed" >&2; exit 1; }
    if grep -q 'libggml' <<<"$needed"; then
        echo "FATAL: $bin expects shared ggml libs; it would collide with the" >&2
        echo "       Vulkan libggml*.so already in the base image" >&2; exit 1; fi
    cp "build/bin/$bin" "/install/bin/${bin}-rocm"
done
BUILD

# ── Build stable-diffusion.cpp (HIP) ───────────────────────────────────

FROM rocm-builder AS sd-rocm
ARG SD_COMMIT=""
RUN --mount=type=cache,id=ccache-rocm,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

COMMIT="${SD_COMMIT:-$(awk '$1=="stable-diffusion.cpp:"{print $2}' /build/versions.txt)}"
[ -n "$COMMIT" ] || { echo "FATAL: no stable-diffusion.cpp commit in versions.txt and no SD_COMMIT given" >&2; exit 1; }

echo "=== Cloning stable-diffusion.cpp at ${COMMIT} ==="
mkdir -p /src/stable-diffusion.cpp && cd /src/stable-diffusion.cpp
git init -q
git remote add origin https://github.com/leejet/stable-diffusion.cpp.git
git fetch --depth=1 origin "${COMMIT}"
git checkout -q FETCH_HEAD
git submodule update --init --recursive --depth=1

echo "=== Building stable-diffusion.cpp (HIP) for ${AMDGPU_TARGETS} ==="
# POSITION_INDEPENDENT_CODE: see llama.cpp stage (sd.cpp also sets it itself)
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -B build \
    -DGGML_NATIVE=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DSD_BUILD_EXAMPLES=ON \
    -DSD_HIPBLAS=ON \
    -DAMDGPU_TARGETS="${AMDGPU_TARGETS}"
cmake --build build --config Release -j"$(nproc)" \
    --target sd-server sd-cli

mkdir -p /install/bin
for bin in sd-server sd-cli; do
    [ -f "build/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
    needed=$(readelf -d "build/bin/$bin" | grep NEEDED || true)
    grep -q 'libamdhip64\.so' <<<"$needed" || {
        echo "FATAL: $bin is not linked against the HIP runtime:" >&2
        echo "$needed" >&2; exit 1; }
    if grep -q 'libggml' <<<"$needed"; then
        echo "FATAL: $bin expects shared ggml libs; it would collide with the" >&2
        echo "       Vulkan libggml*.so already in the base image" >&2; exit 1; fi
    cp "build/bin/$bin" "/install/bin/${bin}-rocm"
done
BUILD

# ── Final image: base + ROCm runtime + HIP binaries ────────────────────

FROM vulkan-base AS final
ARG ROCM_VERSION
ARG AMDGPU_TARGETS

LABEL org.opencontainers.image.source="https://github.com/SelfRef/llama-swap-docker-amd" \
      org.opencontainers.image.description="llama-swap unified image for AMD GPUs (ROCm + Vulkan)"

USER root
ENV DEBIAN_FRONTEND=noninteractive

# ROCm userspace from AMD's apt repo, matching the builder's ROCM_VERSION.
# hipblas/rocblas pull in the HIP runtime (libamdhip64), hsa-rocr, comgr etc.
# via package dependencies. hipblaslt is explicit: rocBLAS dlopens it on some
# architectures (gfx90a/gfx942/RDNA4), which no NEEDED-entry check can catch.
RUN apt-get update && apt-get install -y --no-install-recommends gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
        > /etc/apt/sources.list.d/rocm.list \
    && printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
        > /etc/apt/preferences.d/rocm-pin-600 \
    && apt-get update \
    && apt-get install -y --no-install-recommends hipblas rocblas hipblaslt rocminfo \
    && rm -rf /var/lib/apt/lists/* \
    && echo /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf \
    && ldconfig

ENV PATH="/opt/rocm/bin:${PATH}"

COPY --from=llama-rocm   /install/bin/ /usr/local/bin/
COPY --from=whisper-rocm /install/bin/ /usr/local/bin/
COPY --from=sd-rocm      /install/bin/ /usr/local/bin/

# Example config with both backends; override by mounting /etc/llama-swap/config
COPY config/config.yaml /etc/llama-swap/config/config.yaml

# Fail the build if any binary (new ROCm or inherited Vulkan) has unresolved
# shared libraries -- this is what catches a missing ROCm runtime package.
RUN <<'CHECK'
#!/bin/bash
set -euo pipefail
for bin in llama-server-rocm llama-cli-rocm llama-tts-rocm llama-bench-rocm \
           whisper-server-rocm whisper-cli-rocm sd-server-rocm sd-cli-rocm \
           llama-server whisper-server sd-server audiocpp_server; do
    out=$(ldd "$(command -v "$bin")")
    if grep -q 'not found' <<<"$out"; then
        echo "FATAL: $bin has unresolved libraries:" >&2
        grep 'not found' <<<"$out" >&2
        exit 1
    fi
done
echo "All binaries resolve their shared libraries."
CHECK

RUN { echo "rocm: ${ROCM_VERSION}"; \
      echo "amdgpu_targets: ${AMDGPU_TARGETS}"; } >> /versions.txt \
    && cat /versions.txt

# ENTRYPOINT, CMD, WORKDIR (/models) and ports are inherited from the base
# image: llama-swap -config /etc/llama-swap/config/config.yaml -listen :8080
