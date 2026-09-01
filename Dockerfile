# llama-swap for AMD GPUs — ROCm + Vulkan in one image
#
# Extends the upstream unified Vulkan image (llama-swap, vllm-wrapper, and
# Vulkan builds of llama.cpp, whisper.cpp, stable-diffusion.cpp, audio.cpp)
# with:
#
#   - the ROCm userspace runtime (HIP runtime, rocBLAS/hipBLAS + Tensile
#     kernels, hipBLASLt, rocminfo) from AMD's apt repository
#   - HIP rebuilds of llama.cpp, whisper.cpp and stable-diffusion.cpp,
#     installed as *-rocm binaries next to the Vulkan ones:
#       llama-server-rocm, llama-cli-rocm, llama-tts-rocm, llama-bench-rocm,
#       whisper-server-rocm, whisper-cli-rocm, sd-server-rocm, sd-cli-rocm
#   - EngramHalo.cpp (Aristo94's Strix Halo/qwen4exp fork of llama.cpp, HIP,
#     gfx1151 only) as *-engram binaries -- see the WITH_ENGRAM arg below
#   - REBUILT Vulkan binaries of llama.cpp, whisper.cpp and sd.cpp that
#     REPLACE the base image's (sd-server additionally gets its web UI
#     embedded -- upstream builds it without, see the sd-frontend stage). Why: upstream builds them on Ubuntu 24.04 with
#     its stock glslc (shaderc 2023.8 / glslang 14), which cannot compile the
#     GL_EXT_integer_dot_product and GL_EXT_bfloat16 shaders, so llama.cpp's
#     CMake silently drops those code paths (the device line at startup shows
#     "int dot: 0 | bf16: 0" even though RADV advertises both). The integer
#     dot path is the fast quantized prompt/matvec path on GPUs without
#     cooperative-matrix support (q8_1 MMQ for K-quants: ~2x pp on RDNA2 in
#     upstream's numbers, DP4A flash attention for q8_0/q4_0 KV caches, MMVQ
#     decode); on coopmat GPUs (RDNA3+) llama.cpp keeps its FP16 coopmat
#     matmul for prompts, so measured on an RX 7900 XTX the rebuild is
#     neutral for prompt speed (+~2% decode) -- there the big win is the
#     newer Mesa below. We build the same upstream commits on the same
#     Ubuntu 24.04 ABI, but with glslc taken from the Ubuntu 26.04 pocket
#     (see vulkan-builder), and verify at build time that the extensions are
#     compiled in, so nothing is silently left out for any GPU.
#   - llama.cpp (both backends) built with GGML_BACKEND_DL +
#     GGML_CPU_ALL_VARIANTS: the CPU backend is compiled once per x86 feature
#     level and the best one is picked at runtime, so a single image gets
#     AVX2 on Zen 3 (e.g. 5950X) and AVX-512/VNNI/BF16 on Zen 4/5 (e.g. Strix
#     Halo) for CPU-offloaded experts. Upstream ships one generic AVX2 build.
#   - llama.cpp ROCm built with GGML_CUDA_FA_ALL_QUANTS so flash attention has
#     kernels for every K/V cache type combination; without it only q8_0/q8_0
#     and q4_0/q4_0 stay on the GPU and e.g. q8_0/q4_0 falls back to the CPU
#     (upstream issue #27761: pp512 drops ~68%).
#
# whisper.cpp and sd.cpp are pinned to the commits recorded in the base
# image's /versions.txt. llama.cpp is built from LLAMA_COMMIT (default: current
# master) plus the upstream PRs in LLAMA_PATCHES, identically for Vulkan and
# ROCm, so both backends are always the same revision; the built commit is
# recorded in /versions.txt. audio.cpp has no HIP backend upstream and is left
# as shipped by the base image (Vulkan, its own ggml fork).
#
# Layout: llama.cpp is installed as self-contained directories
# /opt/llama-vulkan and /opt/llama-rocm (binaries + their shared libs, RPATH
# $ORIGIN, ggml backends discovered next to the executable) with symlinks in
# /usr/local/bin, so the two builds never share a libggml. whisper/sd binaries
# are static.
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

# Must be the root variant (not *-rootless): packages are installed with apt in
# the final stage. Pin a dated tag or digest for reproducible builds.
ARG BASE_IMAGE=ghcr.io/mostlygeek/llama-swap:unified-vulkan

ARG ROCM_VERSION=7.2.4

# Build the ROCm side at all? true = full image (Vulkan + ROCm runtime + *-rocm
# binaries); false = Vulkan-only image, the three HIP builder stages are not even
# started (BuildKit only builds stages the final one references). CI publishes
# the Vulkan-only image as `latest` and the full one as `rocm` on request.
ARG WITH_ROCM=true

# gfx architectures compiled into the HIP binaries: RDNA2 (gfx1030), RDNA3/3.5
# (gfx1100/01/02, gfx1150/51), RDNA4 (gfx1200/01) -- the consumer/APU cards this
# image is for. The CDNA data-center targets of llama.cpp's official ROCm image
# (gfx908;gfx90a;gfx942) are left out: they cost ~30% of an already long CI
# build (all-quant FA kernels x every target) and Instinct users have AMD's own
# containers; add them back here if needed. GPUs not listed can still use the
# Vulkan binaries.
ARG AMDGPU_TARGETS="gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201"

# Ubuntu release whose glslc/libshaderc1 are installed into the (24.04) Vulkan
# builder. Only those two packages come from it (per-package release selection
# + low pin), everything else stays 24.04 so the binaries run on the base
# image's glibc. 26.04 "resolute" ships shaderc 2026.1 / glslang 16.
ARG GLSLC_SUITE=resolute

# Compile flash-attention kernels for all K/V cache quant combinations in the
# ROCm llama.cpp build (see header). Costs build time and binary size; set to
# OFF to build faster.
ARG LLAMA_FA_ALL_QUANTS=ON

# EngramHalo.cpp: Aristo94's llama.cpp fork tuned for Qwen 3.8 Flash-Next on
# Strix Halo (gfx1151) — QSA sparse-gather attention, HIP wide top-k kernel,
# MTP draft-head speculative decoding, SSD-backed engram (PLE/n-gram) table
# via --tensor-read-lazy. Built as a FOURTH llama.cpp install
# (/opt/llama-engram, *-engram binaries) next to the Vulkan and ROCm ones,
# only when WITH_ROCM=true AND WITH_ENGRAM=true — the Vulkan-only image
# (`latest`) never builds it (the fork is ROCm/HIP-only; Vulkan is reported a
# net loss upstream). The fork's docs/strix-halo patches (#25992 iGPU
# host-buffer workaround, per-buffer mmap loader) are applied when they still
# fit the tree. ENGRAM_TARGETS is gfx1151 alone on purpose: the kernels are
# tuned for and only validated on Strix Halo.
ARG WITH_ENGRAM=true
ARG ENGRAM_REPO=https://github.com/Aristo94/EngramHalo.cpp.git
ARG ENGRAM_BRANCH=strix-halo-qwen4exp
ARG ENGRAM_TARGETS=gfx1151

# llama.cpp revision for BOTH llama.cpp builds (Vulkan and ROCm). Default:
# current master, so that the open PRs below apply and so the image carries the
# newest backend work. Set "" to build exactly the base image's commit (from
# /versions.txt), or pin a sha/tag. whisper.cpp and sd.cpp stay at the base
# image's commits (no patches there).
ARG LLAMA_COMMIT="master"

# Upstream llama.cpp pull requests to apply on top of LLAMA_COMMIT, as a
# space-separated list of PR numbers (fetched over git as refs/pull/N/head
# and merged; the .patch endpoint is rate-limited on CI runners). A PR that is
# closed on GitHub (merged or rejected) is skipped with a notice; one that no
# longer merges cleanly FAILS the build, so drift is never a silent no-op. Default: #27952 "vulkan: int8 coopmat1 matmul for AMD RDNA3/4"
# (0cc4m) -- measured on an RX 7900 XTX: pp512 +4.6% dense (Q4_K_XL),
# +18.5% MoE (Qwen3.6-35B-A3B Q4_K_M), decode unchanged. Remove it once merged
# (the build tells you). Also measured and NOT adopted: #25483 (MoE coopmat
# skip, +0.3%), #26284 + #26301 (HIP MMQ tuning / mmvdq: +2% pp, decode same,
# and #26284 carries RDNA4 changes its maintainer wants dropped), #22970
# (stale, conflicts with master).
ARG LLAMA_PATCHES="27952"

# Sources of the fixed Qwen chat templates shipped under
# /etc/llama-swap/templates/ (fetched at build time):
#   qwen-fixed.jinja -- froggeric's Qwen-Fixed-Chat-Templates (the base fix)
#   qwen-sharp.jinja -- peculiar-ragdoll's Qwen-Sharp-Chat-Templates: froggeric's
#                       template rebased with a terseness system prompt spliced in
#                       (opt out per request with chat_template_kwargs {"terse": false})
ARG QWEN_TEMPLATE_URL="https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja"
ARG QWEN_SHARP_TEMPLATE_URL="https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates/resolve/main/chat_template.jinja"

# Newer Mesa (RADV, the Vulkan driver) for the final image. Ubuntu 24.04's
# stock Mesa 25.2 is a year behind; kisak-mesa tracks the current stable
# release (26.1 at the time of writing). Measured on an RX 7900 XTX with the
# SAME Vulkan binaries: pp512 693 -> 856 t/s (+24%), decode unchanged, and the
# newer RADV exposes VK_VALVE_shader_mixed_float_dot_product (fp16 "dot2").
# Set to "" to keep the base image's stock Mesa.
ARG MESA_PPA="ppa:kisak/kisak-mesa"

FROM ${BASE_IMAGE} AS vulkan-base

# ── Vulkan builder: Ubuntu 24.04 ABI + modern glslc ────────────────────

FROM ubuntu:24.04 AS vulkan-builder
ARG GLSLC_SUITE

ENV DEBIAN_FRONTEND=noninteractive
ENV CCACHE_DIR=/ccache
ENV CCACHE_MAXSIZE=5G

# libav*-dev only for whisper.cpp's WHISPER_FFMPEG=ON; the base runtime already
# ships the matching Ubuntu 24.04 libav* runtime libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ccache curl ca-certificates \
        pkg-config libssl-dev \
        libvulkan-dev spirv-headers spirv-tools \
        libavcodec-dev libavformat-dev libavutil-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# glslc + libshaderc1 from the newer Ubuntu pocket, nothing else (pin 100 keeps
# apt from preferring that release; the pkg/suite syntax selects it explicitly).
# Their only dependencies are libc6 >= 2.38 / libstdc++6 >= 13.1, satisfied by
# 24.04. The three feature tests below are the ones llama.cpp's CMake runs;
# integer_dot and bfloat16 FAIL with 24.04's own glslc, which is the whole
# reason this stage exists -- so a regression here must fail the build.
RUN echo "deb http://archive.ubuntu.com/ubuntu ${GLSLC_SUITE} main universe" \
        > /etc/apt/sources.list.d/glslc.list \
    && printf 'Package: *\nPin: release n=%s\nPin-Priority: 100\n' "${GLSLC_SUITE}" \
        > /etc/apt/preferences.d/glslc-suite \
    && apt-get update \
    && apt-get install -y --no-install-recommends "glslc/${GLSLC_SUITE}" "libshaderc1/${GLSLC_SUITE}" \
    && rm -rf /var/lib/apt/lists/* \
    && glslc --version

# The commit of each project built into the base image, so the rebuilds match
# the upstream binaries exactly.
COPY --from=vulkan-base /versions.txt /build/versions.txt

WORKDIR /build

# ── Build llama.cpp (Vulkan) ───────────────────────────────────────────

FROM vulkan-builder AS llama-vulkan
ARG LLAMA_COMMIT
ARG LLAMA_PATCHES
RUN --mount=type=cache,id=ccache-vulkan,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

COMMIT="${LLAMA_COMMIT:-$(awk '$1=="llama.cpp:"{print $2}' /build/versions.txt)}"
[ -n "$COMMIT" ] || { echo "FATAL: no llama.cpp commit in versions.txt and no LLAMA_COMMIT given" >&2; exit 1; }

echo "=== Cloning llama.cpp at ${COMMIT} ==="
mkdir -p /src/llama.cpp && cd /src/llama.cpp
git init -q
git remote add origin https://github.com/ggml-org/llama.cpp.git
# Blobless (not shallow) fetch: full history so PR branches can be merged
# properly below; file contents are fetched lazily on checkout.
git fetch --filter=blob:none origin "${COMMIT}"
git checkout -q FETCH_HEAD
echo "llama.cpp at $(git rev-parse HEAD)"

# PRs are pulled over git (refs/pull/N/head) and MERGED, never via the
# github.com/.../pull/N.patch endpoint -- that one is rate-limited (HTTP 429)
# from shared CI runner IPs and made builds fail. GitHub keeps refs/pull/N/merge
# only while a PR is open, so a missing merge ref means the PR was closed
# (merged or rejected): skip it and say so. A conflicting merge is a real
# drift and fails the build.
for pr in ${LLAMA_PATCHES:-}; do
    echo "=== Merging upstream PR #${pr} ==="
    if ! git ls-remote --exit-code origin "refs/pull/${pr}/merge" >/dev/null 2>&1; then
        echo "PR #${pr} is closed on GitHub (merged or rejected) -- skipping; remove it from LLAMA_PATCHES"
        continue
    fi
    git fetch --filter=blob:none origin "refs/pull/${pr}/head"
    if git merge-base --is-ancestor FETCH_HEAD HEAD; then
        echo "PR #${pr} is already contained in ${COMMIT} -- skipping"
        continue
    fi
    git -c user.name=llama-swap-amd -c user.email=build@localhost \
        merge --no-edit --no-ff -m "merge upstream PR #${pr}" FETCH_HEAD \
        || { echo "FATAL: PR #${pr} does not merge cleanly into ${COMMIT} -- it has drifted, re-check it" >&2; exit 1; }
done

echo "=== glslc feature tests (llama.cpp's own) ==="
for t in integer_dot bfloat16 coopmat; do
    f="ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/$t.comp"
    [ -f "$f" ] || { echo "(no feature test $t in this revision, skipping)"; continue; }
    if glslc -o /dev/null -fshader-stage=compute --target-env=vulkan1.3 "$f" >/dev/null 2>&1; then
        echo "  $t: OK"
    else
        echo "FATAL: glslc cannot compile $f -- the Vulkan rebuild would lose that code path" >&2
        exit 1
    fi
done

echo "=== Building llama.cpp (Vulkan) ==="
# BACKEND_DL + CPU_ALL_VARIANTS need shared libs; building with the install
# RPATH ($ORIGIN, nothing else) makes binaries + libs relocatable as one
# directory (ggml also searches for its backend libs next to the executable).
cmake -B build \
    -DGGML_NATIVE=OFF \
    -DGGML_VULKAN=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_RPATH='$ORIGIN' \
    2>&1 | tee /tmp/configure.log
for ext in GL_EXT_integer_dot_product GL_EXT_bfloat16 GL_KHR_cooperative_matrix; do
    line=$(grep -i "$ext" /tmp/configure.log || true)
    echo "  cmake: ${line:-<no message for $ext>}"
    if grep -qi "not supported" <<<"$line"; then
        echo "FATAL: CMake reports $ext unsupported by glslc" >&2; exit 1; fi
done
# "all" so the per-feature-level ggml-cpu variants and the backend libs get
# built too (they are not link-time dependencies of the executables).
cmake --build build --config Release -j"$(nproc)"

echo "=== Collecting ==="
OUT=/install/llama-vulkan
mkdir -p "$OUT"
for bin in llama-server llama-cli llama-tts llama-bench; do
    [ -f "build/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
    cp "build/bin/$bin" "$OUT/"
done
cp -P build/bin/*.so* "$OUT/"
ls "$OUT"/libggml-cpu-*.so >/dev/null 2>&1 || { echo "FATAL: no ggml-cpu variants built" >&2; exit 1; }
ls "$OUT"/libggml-vulkan.so >/dev/null 2>&1 || { echo "FATAL: libggml-vulkan.so not built" >&2; exit 1; }
# Relocatable check: nothing may still point at the build tree.
# Relocatable check: every ELF's run path must start with $ORIGIN and must not
# point into the build tree (CMake may append toolchain lib dirs such as
# /opt/rocm-*/lib for the HIP backend -- those exist in the runtime image).
for f in "$OUT"/*; do
    [ -L "$f" ] && continue
    rp=$(readelf -d "$f" 2>/dev/null | awk '/RUNPATH|RPATH/ {gsub(/[\[\]]/,"",$NF); print $NF}')
    if [ -n "$rp" ] && { [[ "$rp" != '$ORIGIN'* ]] || [[ "$rp" == */src/* ]]; }; then
        echo "FATAL: $f has run path '$rp' (expected \$ORIGIN[:...])" >&2; exit 1; fi
    if ldd "$f" 2>/dev/null | grep -q "not found"; then
        echo "FATAL: $f has unresolved libraries" >&2; ldd "$f" | grep "not found" >&2; exit 1; fi
done
echo "llama_vulkan_commit: $(git rev-parse HEAD) (requested: ${LLAMA_COMMIT:-base image})" > "$OUT/.build-info"
echo "vulkan_glslc: $(glslc --version | head -1)" >> "$OUT/.build-info"
echo "llama_patches: ${LLAMA_PATCHES:-none}" >> "$OUT/.build-info"
BUILD

# ── Build whisper.cpp (Vulkan) ─────────────────────────────────────────

FROM vulkan-builder AS whisper-vulkan
ARG WHISPER_COMMIT=""
RUN --mount=type=cache,id=ccache-vulkan,target=/ccache <<'BUILD'
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

echo "=== Building whisper.cpp (Vulkan, static) ==="
# Static: the base image's whisper-server is the only user of the shared
# libggml*.so in /usr/local/lib; static binaries let the final stage drop them.
cmake -B build \
    -DGGML_NATIVE=OFF \
    -DGGML_VULKAN=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DWHISPER_FFMPEG=ON
cmake --build build --config Release -j"$(nproc)" \
    --target whisper-server whisper-cli

mkdir -p /install/bin
for bin in whisper-server whisper-cli; do
    [ -f "build/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
    if readelf -d "build/bin/$bin" | grep -q 'libggml\|libwhisper'; then
        echo "FATAL: $bin is not statically linked against ggml/whisper" >&2; exit 1; fi
    cp "build/bin/$bin" /install/bin/
done
BUILD

# ── sd-server web UI (built once, embedded into both sd-server builds) ──
# sd.cpp embeds its frontend (the sdcpp-webui submodule, pinned per revision)
# when CMake finds pnpm -- or a pre-generated frontend/dist/gen_index_html.h.
# Upstream's builder has neither, so its sd-server answers "/" with a text
# placeholder. Build the header here with Node and hand it to the C++ stages.

FROM node:22-alpine AS sd-frontend
RUN apk add --no-cache git && corepack enable
COPY --from=vulkan-base /versions.txt /build/versions.txt
ARG SD_COMMIT=""
RUN <<'BUILD'
#!/bin/sh
set -eu
COMMIT="${SD_COMMIT:-$(awk '$1=="stable-diffusion.cpp:"{print $2}' /build/versions.txt)}"
[ -n "$COMMIT" ] || { echo "FATAL: no stable-diffusion.cpp commit" >&2; exit 1; }
mkdir -p /src/sd && cd /src/sd
git init -q && git remote add origin https://github.com/leejet/stable-diffusion.cpp.git
git fetch --depth=1 origin "${COMMIT}" && git checkout -q FETCH_HEAD
git submodule update --init --depth=1 examples/server/frontend
cd examples/server/frontend
echo "sd_frontend: $(git rev-parse HEAD)" > /src/frontend-version
pnpm install --frozen-lockfile
pnpm run build
pnpm run build:header
[ -f dist/gen_index_html.h ] || { echo "FATAL: gen_index_html.h not produced" >&2; exit 1; }
BUILD

# ── Build stable-diffusion.cpp (Vulkan) ────────────────────────────────

FROM vulkan-builder AS sd-vulkan
ARG SD_COMMIT=""
COPY --from=sd-frontend /src/sd/examples/server/frontend/dist/gen_index_html.h /src/frontend-version /tmp/sd-frontend/
RUN --mount=type=cache,id=ccache-vulkan,target=/ccache <<'BUILD'
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
# Pre-built web UI header (see the sd-frontend stage) -> embedded frontend
mkdir -p examples/server/frontend/dist
cp /tmp/sd-frontend/gen_index_html.h examples/server/frontend/dist/

echo "=== Building stable-diffusion.cpp (Vulkan) ==="
mkdir -p build
cmake -B build \
    -DGGML_NATIVE=OFF \
    -DGGML_VULKAN=ON \
    -DSD_VULKAN=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DSD_BUILD_EXAMPLES=ON \
    2>&1 | tee /tmp/configure.log
grep -q "using pre-built frontend header" /tmp/configure.log \
    || { echo "FATAL: sd-server would be built WITHOUT its web UI (pre-built header not picked up)" >&2; exit 1; }
cmake --build build --config Release -j"$(nproc)" \
    --target sd-server sd-cli

mkdir -p /install/bin
cp /tmp/sd-frontend/frontend-version /install/sd-frontend-version
for bin in sd-server sd-cli; do
    [ -f "build/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
    if readelf -d "build/bin/$bin" | grep -q 'libggml\|libstable'; then
        echo "FATAL: $bin expects shared ggml/sd libs" >&2; exit 1; fi
    cp "build/bin/$bin" /install/bin/
done
BUILD

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
ARG LLAMA_COMMIT
ARG LLAMA_FA_ALL_QUANTS
ARG LLAMA_PATCHES
RUN --mount=type=cache,id=ccache-rocm,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

COMMIT="${LLAMA_COMMIT:-$(awk '$1=="llama.cpp:"{print $2}' /build/versions.txt)}"
[ -n "$COMMIT" ] || { echo "FATAL: no llama.cpp commit in versions.txt and no LLAMA_COMMIT given" >&2; exit 1; }

echo "=== Cloning llama.cpp at ${COMMIT} ==="
mkdir -p /src/llama.cpp && cd /src/llama.cpp
git init -q
git remote add origin https://github.com/ggml-org/llama.cpp.git
# Blobless (not shallow) fetch: full history so PR branches can be merged
# properly below; file contents are fetched lazily on checkout.
git fetch --filter=blob:none origin "${COMMIT}"
git checkout -q FETCH_HEAD
echo "llama.cpp at $(git rev-parse HEAD)"

# PRs are pulled over git (refs/pull/N/head) and MERGED, never via the
# github.com/.../pull/N.patch endpoint -- that one is rate-limited (HTTP 429)
# from shared CI runner IPs and made builds fail. GitHub keeps refs/pull/N/merge
# only while a PR is open, so a missing merge ref means the PR was closed
# (merged or rejected): skip it and say so. A conflicting merge is a real
# drift and fails the build.
for pr in ${LLAMA_PATCHES:-}; do
    echo "=== Merging upstream PR #${pr} ==="
    if ! git ls-remote --exit-code origin "refs/pull/${pr}/merge" >/dev/null 2>&1; then
        echo "PR #${pr} is closed on GitHub (merged or rejected) -- skipping; remove it from LLAMA_PATCHES"
        continue
    fi
    git fetch --filter=blob:none origin "refs/pull/${pr}/head"
    if git merge-base --is-ancestor FETCH_HEAD HEAD; then
        echo "PR #${pr} is already contained in ${COMMIT} -- skipping"
        continue
    fi
    git -c user.name=llama-swap-amd -c user.email=build@localhost \
        merge --no-edit --no-ff -m "merge upstream PR #${pr}" FETCH_HEAD \
        || { echo "FATAL: PR #${pr} does not merge cleanly into ${COMMIT} -- it has drifted, re-check it" >&2; exit 1; }
done

echo "=== Building llama.cpp (HIP) for ${AMDGPU_TARGETS}, FA_ALL_QUANTS=${LLAMA_FA_ALL_QUANTS} ==="
# Shared + BACKEND_DL + CPU_ALL_VARIANTS like llama.cpp's own ROCm image; the
# HIP backend lives in libggml-hip.so next to the binaries (RPATH $ORIGIN).
# POSITION_INDEPENDENT_CODE: ggml-hip's device-stub objects are non-PIC by
# default and fail to link into Ubuntu's default-PIE executables.
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -B build \
    -DGGML_NATIVE=OFF \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="${AMDGPU_TARGETS}" \
    -DGGML_CUDA_FA_ALL_QUANTS="${LLAMA_FA_ALL_QUANTS}" \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_RPATH='$ORIGIN'
cmake --build build --config Release -j"$(nproc)"

echo "=== Collecting ==="
OUT=/install/llama-rocm
mkdir -p "$OUT"
for bin in llama-server llama-cli llama-tts llama-bench; do
    [ -f "build/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
    cp "build/bin/$bin" "$OUT/${bin}-rocm"
done
cp -P build/bin/*.so* "$OUT/"
[ -f "$OUT/libggml-hip.so" ] || { echo "FATAL: libggml-hip.so not built" >&2; exit 1; }
readelf -d "$OUT/libggml-hip.so" | grep -q 'libamdhip64\.so' || {
    echo "FATAL: libggml-hip.so is not linked against the HIP runtime" >&2; exit 1; }
ls "$OUT"/libggml-cpu-*.so >/dev/null 2>&1 || { echo "FATAL: no ggml-cpu variants built" >&2; exit 1; }
# Relocatable check: every ELF's run path must start with $ORIGIN and must not
# point into the build tree (CMake may append toolchain lib dirs such as
# /opt/rocm-*/lib for the HIP backend -- those exist in the runtime image).
for f in "$OUT"/*; do
    [ -L "$f" ] && continue
    rp=$(readelf -d "$f" 2>/dev/null | awk '/RUNPATH|RPATH/ {gsub(/[\[\]]/,"",$NF); print $NF}')
    if [ -n "$rp" ] && { [[ "$rp" != '$ORIGIN'* ]] || [[ "$rp" == */src/* ]]; }; then
        echo "FATAL: $f has run path '$rp' (expected \$ORIGIN[:...])" >&2; exit 1; fi
    if ldd "$f" 2>/dev/null | grep -q "not found"; then
        echo "FATAL: $f has unresolved libraries" >&2; ldd "$f" | grep "not found" >&2; exit 1; fi
done
echo "llama_rocm_commit: $(git rev-parse HEAD) (requested: ${LLAMA_COMMIT:-base image})" > "$OUT/.build-info"
echo "rocm_fa_all_quants: ${LLAMA_FA_ALL_QUANTS}" >> "$OUT/.build-info"
echo "llama_patches: ${LLAMA_PATCHES:-none}" >> "$OUT/.build-info"
BUILD

# ── Build EngramHalo.cpp (HIP, Strix Halo only) ────────────────────────

FROM rocm-builder AS llama-engram
ARG ENGRAM_REPO
ARG ENGRAM_BRANCH
ARG ENGRAM_TARGETS
RUN --mount=type=cache,id=ccache-rocm,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

echo "=== Cloning EngramHalo.cpp (${ENGRAM_BRANCH}) ==="
git clone --single-branch --branch "${ENGRAM_BRANCH}" --depth=1 \
    --recurse-submodules --shallow-submodules "${ENGRAM_REPO}" /src/engram
cd /src/engram
echo "EngramHalo.cpp at $(git rev-parse HEAD)"

# The branch ships its Strix Halo patches in-tree under docs/strix-halo/.
# Same conditional logic as the fork's own Dockerfile.rocm-7.14: apply while
# they fit, treat reverse-applying as already-upstream, and only the
# correctness patch (#25992 multi-slot response mix-up on iGPUs) is fatal
# when it neither applies nor is present.
p=docs/strix-halo/llama-cpp-25992-rocm-host-buffer.patch
if git apply --check "$p" 2>/dev/null; then git apply "$p"; echo "applied: $p"
elif git apply --reverse --check "$p" 2>/dev/null; then echo "#25992 workaround already present upstream"
else echo "FATAL: #25992 host-buffer workaround no longer applies -- multi-slot serving would return wrong responses; re-check the branch" >&2; exit 1
fi
p=docs/strix-halo/llama-cpp-qwen38-per-buffer-mmap.patch
if git apply --check "$p" 2>/dev/null; then git apply "$p"; echo "applied: $p"
else echo "per-buffer mmap loader patch skipped as obsolete"
fi

echo "=== Building EngramHalo.cpp (HIP) for ${ENGRAM_TARGETS} ==="
# Same relocatable shared/BACKEND_DL layout as the llama-rocm stage. No
# FA_ALL_QUANTS: this binary serves one model (q8_0/q8_0 KV) and the default
# FA kernel set already covers q8_0/q8_0 and q4_0/q4_0.
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -B build \
    -DGGML_NATIVE=OFF \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="${ENGRAM_TARGETS}" \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_RPATH='$ORIGIN'
cmake --build build --config Release -j"$(nproc)"

echo "=== Collecting ==="
OUT=/install/llama-engram
mkdir -p "$OUT"
for bin in llama-server llama-cli llama-bench; do
    [ -f "build/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
    cp "build/bin/$bin" "$OUT/${bin}-engram"
done
cp -P build/bin/*.so* "$OUT/"
[ -f "$OUT/libggml-hip.so" ] || { echo "FATAL: libggml-hip.so not built" >&2; exit 1; }
readelf -d "$OUT/libggml-hip.so" | grep -q 'libamdhip64\.so' || {
    echo "FATAL: libggml-hip.so is not linked against the HIP runtime" >&2; exit 1; }
ls "$OUT"/libggml-cpu-*.so >/dev/null 2>&1 || { echo "FATAL: no ggml-cpu variants built" >&2; exit 1; }
for f in "$OUT"/*; do
    [ -L "$f" ] && continue
    rp=$(readelf -d "$f" 2>/dev/null | awk '/RUNPATH|RPATH/ {gsub(/[\[\]]/,"",$NF); print $NF}')
    if [ -n "$rp" ] && { [[ "$rp" != '$ORIGIN'* ]] || [[ "$rp" == */src/* ]]; }; then
        echo "FATAL: $f has run path '$rp' (expected \$ORIGIN[:...])" >&2; exit 1; fi
    if ldd "$f" 2>/dev/null | grep -q "not found"; then
        echo "FATAL: $f has unresolved libraries" >&2; ldd "$f" | grep "not found" >&2; exit 1; fi
done
{ echo "llama_engram_commit: $(git rev-parse HEAD) (${ENGRAM_REPO} @ ${ENGRAM_BRANCH})";
  echo "llama_engram_targets: ${ENGRAM_TARGETS}"; } > "$OUT/.build-info"
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
        echo "FATAL: $bin expects shared ggml libs" >&2; exit 1; fi
    cp "build/bin/$bin" "/install/bin/${bin}-rocm"
done
BUILD

# ── Build stable-diffusion.cpp (HIP) ───────────────────────────────────

FROM rocm-builder AS sd-rocm
ARG SD_COMMIT=""
COPY --from=sd-frontend /src/sd/examples/server/frontend/dist/gen_index_html.h /src/frontend-version /tmp/sd-frontend/
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
# Pre-built web UI header (see the sd-frontend stage) -> embedded frontend
mkdir -p examples/server/frontend/dist
cp /tmp/sd-frontend/gen_index_html.h examples/server/frontend/dist/

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
    -DAMDGPU_TARGETS="${AMDGPU_TARGETS}" \
    2>&1 | tee /tmp/configure.log
grep -q "using pre-built frontend header" /tmp/configure.log \
    || { echo "FATAL: sd-server would be built WITHOUT its web UI (pre-built header not picked up)" >&2; exit 1; }
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
        echo "FATAL: $bin expects shared ggml libs" >&2; exit 1; fi
    cp "build/bin/$bin" "/install/bin/${bin}-rocm"
done
BUILD

# ── ROCm stage selection (WITH_ROCM) ───────────────────────────────────
# Alias stages: the final stage copies from `<name>-sel`, which FROMs
# `<name>-${WITH_ROCM}`; with false that resolves to this empty stand-in and the
# HIP builders are never started.

FROM alpine:3 AS rocm-none
RUN mkdir -p /install/bin /install/llama-rocm /install/llama-engram

FROM llama-rocm   AS llama-rocm-true
FROM whisper-rocm AS whisper-rocm-true
FROM sd-rocm      AS sd-rocm-true
FROM rocm-none    AS llama-rocm-false
FROM rocm-none    AS whisper-rocm-false
FROM rocm-none    AS sd-rocm-false
# COPY --from cannot expand variables, FROM can: select here.
FROM llama-rocm-${WITH_ROCM}   AS llama-rocm-sel
FROM whisper-rocm-${WITH_ROCM} AS whisper-rocm-sel
FROM sd-rocm-${WITH_ROCM}      AS sd-rocm-sel

# Engram needs BOTH switches on (it links the ROCm runtime, which only the
# WITH_ROCM image installs), so the selection key is the concatenated pair.
FROM llama-engram AS llama-engram-true-true
FROM rocm-none    AS llama-engram-true-false
FROM rocm-none    AS llama-engram-false-true
FROM rocm-none    AS llama-engram-false-false
FROM llama-engram-${WITH_ROCM}-${WITH_ENGRAM} AS llama-engram-sel

# ── Final image: base (+ ROCm runtime) + rebuilt binaries ──────────────

FROM vulkan-base AS final
ARG ROCM_VERSION
ARG AMDGPU_TARGETS
ARG LLAMA_FA_ALL_QUANTS
ARG MESA_PPA
ARG QWEN_TEMPLATE_URL
ARG QWEN_SHARP_TEMPLATE_URL
ARG WITH_ROCM
ARG WITH_ENGRAM

LABEL org.opencontainers.image.source="https://github.com/SelfRef/llama-swap-docker-amd" \
      org.opencontainers.image.description="llama-swap unified image for AMD GPUs (ROCm + Vulkan)"

USER root
ENV DEBIAN_FRONTEND=noninteractive

# ROCm userspace from AMD's apt repo, matching the builder's ROCM_VERSION.
# hipblas/rocblas pull in the HIP runtime (libamdhip64), hsa-rocr, comgr etc.
# via package dependencies. hipblaslt is explicit: rocBLAS dlopens it on some
# architectures (gfx90a/gfx942/RDNA4), which no NEEDED-entry check can catch.
RUN if [ "${WITH_ROCM}" = "true" ]; then \
    apt-get update && apt-get install -y --no-install-recommends gnupg \
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
    && ldconfig; \
    fi

# Newer Mesa/RADV (Vulkan driver) for the Vulkan binaries, see MESA_PPA above.
# Only mesa-vulkan-drivers (+ its deps) is upgraded, not the whole GL stack.
RUN if [ -n "${MESA_PPA}" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends software-properties-common \
        && add-apt-repository -y "${MESA_PPA}" \
        && apt-get install -y --no-install-recommends --only-upgrade mesa-vulkan-drivers \
        && apt-get purge -y --auto-remove software-properties-common \
        && rm -rf /var/lib/apt/lists/*; \
    fi

ENV PATH="/opt/rocm/bin:${PATH}"

# Rebuilt Vulkan binaries replace the base image's (see header). The base's
# shared libggml*/libwhisper* in /usr/local/lib were only used by its
# whisper-server; ours are static, so they go too (a stale libggml-vulkan.so
# there would otherwise be a trap for anyone dlopen-ing "the" ggml).
RUN rm -f /usr/local/bin/llama-server /usr/local/bin/llama-cli \
          /usr/local/bin/llama-tts /usr/local/bin/llama-bench \
          /usr/local/lib/libggml*.so* /usr/local/lib/libwhisper*.so* \
    && ldconfig
COPY --from=llama-vulkan   /install/llama-vulkan/ /opt/llama-vulkan/
COPY --from=whisper-vulkan /install/bin/ /usr/local/bin/
COPY --from=sd-vulkan      /install/bin/ /usr/local/bin/
COPY --from=sd-vulkan      /install/sd-frontend-version /tmp/sd-frontend-version
COPY --from=llama-rocm-sel   /install/llama-rocm/ /opt/llama-rocm/
COPY --from=whisper-rocm-sel /install/bin/ /usr/local/bin/
COPY --from=sd-rocm-sel      /install/bin/ /usr/local/bin/
COPY --from=llama-engram-sel /install/llama-engram/ /opt/llama-engram/
RUN for bin in llama-server llama-cli llama-tts llama-bench; do \
        ln -sf "/opt/llama-vulkan/$bin" "/usr/local/bin/$bin"; \
        if [ "${WITH_ROCM}" = "true" ]; then \
            ln -sf "/opt/llama-rocm/$bin-rocm" "/usr/local/bin/$bin-rocm"; \
        fi; \
    done \
    && { [ "${WITH_ROCM}" = "true" ] || rmdir /opt/llama-rocm; } \
    && if [ "${WITH_ROCM}" = "true" ] && [ "${WITH_ENGRAM}" = "true" ]; then \
        for bin in llama-server llama-cli llama-bench; do \
            ln -sf "/opt/llama-engram/$bin-engram" "/usr/local/bin/$bin-engram"; \
        done; \
    else rmdir /opt/llama-engram; fi

# Example config with both backends; override by mounting /etc/llama-swap/config
COPY config/config.yaml /etc/llama-swap/config/config.yaml

# Fixed Qwen 3.5/3.6/3.8 chat templates for `--chat-template-file`:
#   qwen-fixed.jinja -- froggeric's (reasoning-depth default, enable_thinking=false,
#                       history <think> extraction, tool-call wire format -- see the
#                       model card)
#   qwen-sharp.jinja -- peculiar-ragdoll's Sharp variant: the same template with a
#                       force-appended terseness system prompt (fewer filler tokens,
#                       same kwargs; {"terse": false} in chat_template_kwargs drops it)
# ADD from the URL: BuildKit re-checks the remote file on every build
# (ETag/Last-Modified), so a rebuild picks up a new template version even when
# the layer would otherwise be cached. Paths are stable; the version strings
# are recorded in /versions.txt.
ADD --chmod=0644 ${QWEN_TEMPLATE_URL} /etc/llama-swap/templates/qwen-fixed.jinja
ADD --chmod=0644 ${QWEN_SHARP_TEMPLATE_URL} /etc/llama-swap/templates/qwen-sharp.jinja
# --chmod also applies to the directory ADD creates; make it traversable.
RUN chmod 755 /etc/llama-swap/templates

# Fail the build if any binary or backend library has unresolved shared
# libraries (catches a missing ROCm runtime package or a broken RPATH), and
# smoke-test that each llama-server starts, finds its ggml backends next to
# itself and lists devices (no GPU here, so the list is empty -- the point is
# that backend loading does not fail).
RUN <<'CHECK'
#!/bin/bash
set -euo pipefail
BINS="llama-server llama-cli llama-tts llama-bench whisper-server whisper-cli sd-server sd-cli audiocpp_server"
SERVERS="llama-server"
if [ "${WITH_ROCM}" = "true" ]; then
    BINS="$BINS llama-server-rocm llama-cli-rocm llama-tts-rocm llama-bench-rocm whisper-server-rocm whisper-cli-rocm sd-server-rocm sd-cli-rocm"
    SERVERS="$SERVERS llama-server-rocm"
fi
if [ "${WITH_ROCM}" = "true" ] && [ "${WITH_ENGRAM}" = "true" ]; then
    BINS="$BINS llama-server-engram llama-cli-engram llama-bench-engram"
    SERVERS="$SERVERS llama-server-engram"
fi
for bin in $BINS; do
    out=$(ldd "$(readlink -f "$(command -v "$bin")")")
    if grep -q 'not found' <<<"$out"; then
        echo "FATAL: $bin has unresolved libraries:" >&2
        grep 'not found' <<<"$out" >&2
        exit 1
    fi
done
for lib in /opt/llama-vulkan/*.so* $([ "${WITH_ROCM}" = "true" ] && echo /opt/llama-rocm/*.so*) $([ -d /opt/llama-engram ] && echo /opt/llama-engram/*.so*); do
    if ldd "$lib" | grep -q 'not found'; then
        echo "FATAL: $lib has unresolved libraries:" >&2; ldd "$lib" | grep 'not found' >&2; exit 1; fi
done
echo "All binaries and libraries resolve their shared libraries."
for bin in $SERVERS; do
    "$bin" --version
    out=$("$bin" --list-devices 2>&1 || true)
    if ! grep -q "Available devices" <<<"$out"; then
        echo "FATAL: $bin --list-devices did not run (backend libs not found?):" >&2
        echo "$out" >&2; exit 1
    fi
done
ls /opt/llama-vulkan/libggml-cpu-*.so | sed 's|.*/libggml-cpu-||; s|\.so||' | tr '\n' ' ' | sed 's/^/cpu variants: /; s/ $/\n/'
CHECK

RUN { echo "with_rocm: ${WITH_ROCM}"; \
      if [ "${WITH_ROCM}" = "true" ]; then echo "rocm: ${ROCM_VERSION}"; echo "amdgpu_targets: ${AMDGPU_TARGETS}"; fi; \
      echo "vulkan_rebuild: llama.cpp whisper.cpp stable-diffusion.cpp (base binaries replaced)"; \
      echo "sd_server_webui: embedded ($(cut -d' ' -f2 /tmp/sd-frontend-version))"; rm -f /tmp/sd-frontend-version; \
      cat /opt/llama-vulkan/.build-info; \
      if [ "${WITH_ROCM}" = "true" ]; then cat /opt/llama-rocm/.build-info; fi; \
      if [ -d /opt/llama-engram ]; then cat /opt/llama-engram/.build-info; fi; \
      echo "cpu_variants: $(ls /opt/llama-vulkan/libggml-cpu-*.so | sed 's|.*/libggml-cpu-||; s|\.so||' | tr '\n' ' ')"; \
      echo "mesa_ppa: ${MESA_PPA:-none}"; \
      echo "qwen_chat_template: $(grep -o 'template_version = "[^"]*"' /etc/llama-swap/templates/qwen-fixed.jinja | head -1 | cut -d'"' -f2) (${QWEN_TEMPLATE_URL})"; \
      echo "qwen_sharp_chat_template: $(grep -o 'template_version = "[^"]*"' /etc/llama-swap/templates/qwen-sharp.jinja | head -1 | cut -d'"' -f2) (${QWEN_SHARP_TEMPLATE_URL})"; } >> /versions.txt \
    && cat /versions.txt

# ENTRYPOINT, CMD, WORKDIR (/models) and ports are inherited from the base
# image: llama-swap -config /etc/llama-swap/config/config.yaml -listen :8080
