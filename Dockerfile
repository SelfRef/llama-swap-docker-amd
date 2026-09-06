# llama-swap for AMD GPUs — ROCm + Vulkan in one image, built from source
#
# The same shape as ghcr.io/mostlygeek/llama-swap:unified-vulkan (binary names
# in /usr/local/bin, config at /etc/llama-swap/config/config.yaml, models in
# /models, port 8080) so a config written for that image works here, but built
# from plain ubuntu:24.04 with no upstream image in the chain. Everything in it
# is compiled here from the projects' current default branches:
#
#   - llama-swap itself, from source (LLAMA_SWAP_COMMIT + the open upstream PRs
#     in LLAMA_SWAP_PATCHES, web UI embedded) and vllm-wrapper from the same
#     tree -- so an open llama-swap PR ships in THE llama-swap binary, not as a
#     side binary.
#   - llama.cpp from current master + the open upstream PRs in LLAMA_PATCHES
#     (Vulkan fusions, Qwen fixes, qwen4exp MTP, adaptive MTP, checkpoint
#     restore, ... -- the list with rationale is at the LLAMA_PATCHES arg), the
#     SAME tree for the Vulkan and the ROCm build. There is no un-patched
#     llama.cpp in the image any more: llama-server IS the patched build.
#   - the ROCm userspace runtime (HIP runtime, rocBLAS/hipBLAS + Tensile
#     kernels, hipBLASLt, rocminfo) from AMD's apt repository
#   - HIP builds of llama.cpp, whisper.cpp and stable-diffusion.cpp, installed
#     as *-rocm binaries next to the Vulkan ones:
#       llama-server-rocm, llama-cli-rocm, llama-tts-rocm, llama-bench-rocm,
#       whisper-server-rocm, whisper-cli-rocm, sd-server-rocm, sd-cli-rocm
#   - EngramHalo.cpp (Aristo94's Strix Halo/qwen4exp fork of llama.cpp, HIP,
#     gfx1151 only) as *-engram binaries -- see the WITH_ENGRAM arg below
#   - Vulkan builds of llama.cpp, whisper.cpp, sd.cpp and audio.cpp with a
#     MODERN glslc. Upstream builds them on Ubuntu 24.04 with its stock glslc
#     (shaderc 2023.8 / glslang 14), which cannot compile the
#     GL_EXT_integer_dot_product and GL_EXT_bfloat16 shaders, so llama.cpp's
#     CMake silently drops those code paths (the device line at startup shows
#     "int dot: 0 | bf16: 0" even though RADV advertises both). The integer
#     dot path is the fast quantized prompt/matvec path on GPUs without
#     cooperative-matrix support (q8_1 MMQ for K-quants: ~2x pp on RDNA2 in
#     upstream's numbers, DP4A flash attention for q8_0/q4_0 KV caches, MMVQ
#     decode); on coopmat GPUs (RDNA3+) llama.cpp keeps its FP16 coopmat
#     matmul for prompts, so measured on an RX 7900 XTX the rebuild is
#     neutral for prompt speed (+~2% decode) -- there the big win is the
#     newer Mesa below. We build on the Ubuntu 24.04 ABI but with glslc taken
#     from the Ubuntu 26.04 pocket (see vulkan-builder), and verify at build
#     time that the extensions are compiled in, so nothing is silently left
#     out for any GPU.
#   - llama.cpp (both backends) built with GGML_BACKEND_DL +
#     GGML_CPU_ALL_VARIANTS: the CPU backend is compiled once per x86 feature
#     level and the best one is picked at runtime, so a single image gets
#     AVX2 on Zen 3 (e.g. 5950X) and AVX-512/VNNI/BF16 on Zen 4/5 (e.g. Strix
#     Halo) for CPU-offloaded experts. Upstream ships one generic AVX2 build.
#   - llama.cpp ROCm built with GGML_CUDA_FA_ALL_QUANTS so flash attention has
#     kernels for every K/V cache type combination; without it only q8_0/q8_0
#     and q4_0/q4_0 stay on the GPU and e.g. q8_0/q4_0 falls back to the CPU
#     (upstream issue #27761: pp512 drops ~68%).
#   - sd-server with its web UI embedded (upstream builds it without, see the
#     sd-frontend stage).
#   - audio.cpp's server and CLI (Vulkan, as upstream) PLUS its GGUF converter
#     audiocpp_gguf (upstream ships no converter), all from one tree so the
#     converter's model-spec catalog matches the server that loads its output.
#   - a current Mesa/RADV from the kisak PPA instead of Ubuntu 24.04's.
#
# Why not FROM the upstream image any more: once llama-swap is built here the
# base contributed one apt line, two audio.cpp binaries and three commit
# hashes -- and cost ~650 MB per pull in binaries that were deleted in a child
# layer but still shipped in the parent layers, plus a daily base rebuild that
# invalidated every final-stage layer whether or not anything relevant changed.
# Nothing of upstream's docker/unified is vendored either: the entrypoint is
# llama-swap itself (defaults in CMD, so container arguments replace them).
#
# Versions: every project is built from the ref in its *_COMMIT arg (default:
# the current default branch). Because nothing in the build context changes
# between scheduled runs, CI resolves those refs to commits first
# (scripts/resolve-refs.sh) and passes them as build args -- a moved branch is
# then a cache miss for exactly the stages that use it. A local
# `docker buildx build .` resolves per stage at build time (fine on one
# machine; use `$(scripts/resolve-refs.sh --docker)` to pin). Everything that
# was built, with the PRs merged, is recorded in /versions.txt.
#
# Layout: llama.cpp is installed as self-contained directories
# /opt/llama-vulkan, /opt/llama-rocm and /opt/llama-engram (binaries + their
# shared libs, RPATH $ORIGIN, ggml backends discovered next to the executable)
# with symlinks in /usr/local/bin, so the builds never share a libggml.
# whisper/sd/audio.cpp binaries are static.
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

# ── llama-swap ─────────────────────────────────────────────────────────
# Revision of mostlygeek/llama-swap to build llama-swap and vllm-wrapper from
# (branch, tag such as v255, or sha). main: releases are cut from it every few
# days and the open PRs below are written against it.
ARG LLAMA_SWAP_COMMIT="main"

# Open upstream llama-swap PRs merged on top, same rules as LLAMA_PATCHES
# (closed PRs are skipped with a notice, conflicts fail the build):
#   #1099 ui/playground: live per-turn generation stats in the Chat tab
#         (prompt/thinking/answer tokens, speed, cache reuse, MTP acceptance,
#         TTFT, context use -- the first thing you want when comparing quants
#         or tuning flags)
ARG LLAMA_SWAP_PATCHES="1099"

# Cache key only: CI sets it to the PR heads' shas (scripts/resolve-refs.sh) so
# an updated PR rebuilds llama-swap even though the PR list did not change.
ARG LLAMA_SWAP_PATCHES_HEADS=""

# ── ROCm ───────────────────────────────────────────────────────────────
# ROCm source channel. AMD now ships ROCm through two repositories:
#   multiarch (default) — repo.amd.com/rocm/packages-multi-arch/ubuntu2404:
#     the current releases (ROCM_SERIES, e.g. 7.14 -> apt picks 7.14.1), split
#     into per-gfx packages, so the runtime image carries BLAS kernels only
#     for AMDGPU_TARGETS (~50 MB per arch) instead of the classic ~6 GB
#     all-arch rocBLAS/hipBLASLt Tensile blobs. Installs under
#     /opt/rocm/core-<series>/, symlinked back to the classic /opt/rocm
#     layout in the stages below.
#   classic — repo.radeon.com/rocm/apt + the rocm/dev-ubuntu-24.04 builder
#     images: tops out at 7.2.4 (checked 2026-09-01) and carries the HIP
#     graphs bug fixed in 7.13 (needs GGML_CUDA_DISABLE_GRAPHS=1, see
#     llama.cpp discussion #27950). Kept as a fallback.
ARG ROCM_CHANNEL=multiarch

# classic channel only: builder image tag + apt repo path.
ARG ROCM_VERSION=7.2.4

# multiarch channel only: the release series embedded in the package names
# (amdrocm-runtime7.14, amdrocm-blas7.14-gfx1151, ...); apt then resolves the
# newest point release of that series. Bump when AMD publishes the next one.
ARG ROCM_SERIES=7.14

# Build the ROCm side at all? true = full image (Vulkan + ROCm runtime + *-rocm
# binaries); false = Vulkan-only image, the HIP builder stages are not even
# started (BuildKit only builds stages the final one references). CI publishes
# the Vulkan-only image as :vulkan and the full one as :full / :latest.
ARG WITH_ROCM=true

# gfx architectures compiled into the HIP binaries: RDNA2 (gfx1030), RDNA3/3.5
# (gfx1100/01/02, gfx1150/51), RDNA4 (gfx1200/01) -- the consumer/APU cards this
# image is for. The CDNA data-center targets of llama.cpp's official ROCm image
# (gfx908;gfx90a;gfx942) are left out: they cost ~30% of an already long CI
# build (all-quant FA kernels x every target) and Instinct users have AMD's own
# containers; add them back here if needed. GPUs not listed can still use the
# Vulkan binaries.
ARG AMDGPU_TARGETS="gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201"

# Compile flash-attention kernels for all K/V cache quant combinations in the
# ROCm llama.cpp build (see header). Costs build time and binary size; set to
# OFF to build faster.
ARG LLAMA_FA_ALL_QUANTS=ON

# ── Vulkan ─────────────────────────────────────────────────────────────
# Ubuntu release whose glslc/libshaderc1 are installed into the (24.04) Vulkan
# builder. Only those two packages come from it (per-package release selection
# + low pin), everything else stays 24.04 so the binaries run on the runtime
# image's glibc. 26.04 "resolute" ships shaderc 2026.1 / glslang 16.
ARG GLSLC_SUITE=resolute

# Newer Mesa (RADV, the Vulkan driver) for the final image. Ubuntu 24.04's
# stock Mesa 25.2 is a year behind; kisak-mesa tracks the current stable
# release (26.1 at the time of writing). Measured on an RX 7900 XTX with the
# SAME Vulkan binaries: pp512 693 -> 856 t/s (+24%), decode unchanged, and the
# newer RADV exposes VK_VALVE_shader_mixed_float_dot_product (fp16 "dot2").
# Set to "" to keep Ubuntu's stock Mesa.
ARG MESA_PPA="ppa:kisak/kisak-mesa"

# ── llama.cpp ──────────────────────────────────────────────────────────
# llama.cpp revision for BOTH llama.cpp builds (Vulkan and ROCm): sha, tag,
# branch, or refs/pull/N/head. Default: current master, so that the open PRs
# below apply and the image carries the newest backend work.
ARG LLAMA_COMMIT="master"

# Upstream llama.cpp pull requests merged on top of LLAMA_COMMIT, for BOTH
# backends, as a space-separated list of PR numbers (fetched over git as
# refs/pull/N/head and merged in list order; the .patch endpoint is
# rate-limited on CI runners). A PR that is closed on GitHub (merged or
# rejected) is skipped with a notice; one that no longer merges cleanly FAILS
# the build, so drift is never a silent no-op. Dry-run the whole set with
# scripts/checkout-with-prs.sh on a local clone before changing it.
#
# This used to be two builds -- a "pure" llama-server with only #27952 and a
# llama-server-next with the full set -- merged into one on 2026-09-06: the
# full set had been the one actually serving models, and none of the PRs
# touches CUDA/HIP sources, so the ROCm build takes the same tree. Set on
# 2026-09-06 (all merged cleanly against master 9e0e2205; #28422 topk_moe
# fusion conflicts with #28024 and was left out):
#   Vulkan backend
#   #27952 int8 coopmat1 matmul for RDNA3/4 (0cc4m) -- measured on an RX 7900
#          XTX: pp512 +4.6% dense (Q4_K_XL), +18.5% MoE (Qwen3.6-35B-A3B
#          Q4_K_M), decode unchanged. Watch #25773 (mul_mm rewrite): when it
#          lands, this one needs a rebase.
#   #28024 rms_norm fusions (RMS_NORM+MUL+ADD, ROPE+VIEW+SET_ROWS) -- approved
#   #27220 fuse UNARY(silu/gelu/sigmoid)+MUL incl. MoE shared-expert gating
#          (2-3% on Qwen3.6 MoE upstream) -- approved
#   #28253 type-aligned quantized GET_ROWS (correctness on views) -- approved
#   #28457 small-M matmul tile selection for Qwen-shaped buckets (m=1/m=32)
#   Models
#   #28243 Qwen3.8-Flash-Next MTP draft head + draft-only sidecar loading
#          (unsloth's upstream PR; supersedes the local #27836/#28097 rebases)
#   #28068 gated-delta-net norm max->rsqrt (matches the Qwen reference) -- approved
#   #28265 keep Qwen3.5-family delta-net out-proj 2D (Strix Halo: +6-9% TG at
#          batch 4-8 = our --parallel 2 + MTP verify batches)
#   #28213 gather-based sparse attention for qwen4exp QSA decode (+50% tg
#          @130k upstream claim; measured 0 on RADV 2026-09-02, kept for depth)
#   #28136 direct pread()s for the lazy PLE/n-gram table (cold-start prefill;
#          throughput-neutral when the page cache is warm)
#   #28330 no V cache for the qwen4exp lightning indexer (pure VRAM win)
#   Speculative / server
#   #27210 `--spec-type draft-mtp-adaptive` (opt-in; R9700 Qwen3.8-27B code
#          53->72 t/s vs fixed n-max 3) -- to benchmark
#   #28333 zero the MTP carrier at sequence start (determinism across requests)
#   #25592 exact-position checkpoint restore for hybrid/recurrent models
#          (agentic multi-turn @130k: 35 s -> 1.3 s turn restore) -- to benchmark
# Measured and NOT adopted: #25483 (MoE coopmat skip, +0.3%), #26284 + #26301
# (HIP MMQ tuning / mmvdq: +2% pp, decode same, and #26284 carries RDNA4
# changes its maintainer wants dropped), #22970 (stale, conflicts with master).
# patches/*.patch (local rebased patches) apply after the merges to both
# backends; the directory is EMPTY since 2026-09-06 (see patches/README.md).
# Retire PRs from the list as they merge (the build says so).
ARG LLAMA_PATCHES="27952 28024 27220 28253 28457 28243 28068 28265 28213 28136 28330 27210 28333 25592"

# Cache key only (see LLAMA_SWAP_PATCHES_HEADS).
ARG LLAMA_PATCHES_HEADS=""

# ── EngramHalo.cpp ─────────────────────────────────────────────────────
# EngramHalo.cpp: Aristo94's llama.cpp fork tuned for Qwen 3.8 Flash-Next on
# Strix Halo (gfx1151) — QSA sparse-gather attention, HIP wide top-k kernel,
# MTP draft-head speculative decoding, SSD-backed engram (PLE/n-gram) table
# via --tensor-read-lazy. Built as a THIRD llama.cpp install
# (/opt/llama-engram, *-engram binaries) next to the Vulkan and ROCm ones,
# only when WITH_ROCM=true AND WITH_ENGRAM=true — the Vulkan-only image
# never builds it (the fork is ROCm/HIP-only; Vulkan is reported a net loss
# upstream). The fork's docs/strix-halo patches (#25992 iGPU host-buffer
# workaround, per-buffer mmap loader) are applied when they still fit the
# tree. ENGRAM_TARGETS is gfx1151 alone on purpose: the kernels are tuned for
# and only validated on Strix Halo. ENGRAM_COMMIT pins the branch to a sha
# (CI does; empty = branch tip).
ARG WITH_ENGRAM=true
ARG ENGRAM_REPO=https://github.com/Aristo94/EngramHalo.cpp.git
ARG ENGRAM_BRANCH=strix-halo-qwen4exp
ARG ENGRAM_COMMIT=""
ARG ENGRAM_TARGETS=gfx1151

# ── whisper.cpp, stable-diffusion.cpp, audio.cpp ───────────────────────
# Revisions (branch, tag or sha) of the other engines; their default branches.
ARG WHISPER_COMMIT="master"
ARG SD_COMMIT="master"
ARG AUDIOCPP_COMMIT="main"

# ── Chat templates ─────────────────────────────────────────────────────
# Sources of the fixed Qwen chat templates shipped under
# /etc/llama-swap/templates/ (fetched at build time):
#   qwen-fixed.jinja -- froggeric's Qwen-Fixed-Chat-Templates (the base fix)
#   qwen-sharp.jinja -- peculiar-ragdoll's Qwen-Sharp-Chat-Templates: froggeric's
#                       template rebased with a terseness system prompt spliced in
#                       (opt out per request with chat_template_kwargs {"terse": false})
ARG QWEN_TEMPLATE_URL="https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja"
ARG QWEN_SHARP_TEMPLATE_URL="https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates/resolve/main/chat_template.jinja"

# ── Final-stage cache key ──────────────────────────────────────────────
# Declared in the final stage right before its apt layers: CI passes the run's
# timestamp so Ubuntu updates, the PPA Mesa and the ROCm runtime are refreshed
# on every run (minutes). Locally, leave it empty and those layers stay cached.
ARG BUILD_DATE=""

# ══════════════════════════════════════════════════════════════════════
# ── Vulkan builder: Ubuntu 24.04 ABI + modern glslc ────────────────────

FROM ubuntu:24.04 AS vulkan-builder
ARG GLSLC_SUITE

ENV DEBIAN_FRONTEND=noninteractive
ENV CCACHE_DIR=/ccache
ENV CCACHE_MAXSIZE=5G

# libav*-dev only for whisper.cpp's WHISPER_FFMPEG=ON; the final stage installs
# the matching Ubuntu 24.04 libav* runtime libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ccache curl ca-certificates \
        pkg-config libssl-dev \
        libvulkan-dev spirv-headers spirv-tools \
        libavcodec-dev libavformat-dev libavutil-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# glslc + libshaderc1 from the newer Ubuntu pocket, nothing else (pin 100 keeps
# apt from preferring that release; the pkg/suite syntax selects it explicitly).
# Their only dependencies are libc6 >= 2.38 / libstdc++6 >= 13.1, satisfied by
# 24.04. The three feature tests in the llama.cpp stage are the ones llama.cpp's
# CMake runs; integer_dot and bfloat16 FAIL with 24.04's own glslc, which is the
# whole reason this stage exists -- so a regression there must fail the build.
RUN echo "deb http://archive.ubuntu.com/ubuntu ${GLSLC_SUITE} main universe" \
        > /etc/apt/sources.list.d/glslc.list \
    && printf 'Package: *\nPin: release n=%s\nPin-Priority: 100\n' "${GLSLC_SUITE}" \
        > /etc/apt/preferences.d/glslc-suite \
    && apt-get update \
    && apt-get install -y --no-install-recommends "glslc/${GLSLC_SUITE}" "libshaderc1/${GLSLC_SUITE}" \
    && rm -rf /var/lib/apt/lists/* \
    && glslc --version

COPY --chmod=0755 scripts/checkout-with-prs.sh /usr/local/bin/checkout-with-prs.sh
WORKDIR /build

# ── Build llama.cpp (Vulkan) ───────────────────────────────────────────

FROM vulkan-builder AS llama-vulkan
ARG LLAMA_COMMIT
ARG LLAMA_PATCHES
ARG LLAMA_PATCHES_HEADS
COPY patches/ /build/patches/
RUN --mount=type=cache,id=ccache-vulkan,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

# master + the open PRs + local patches; see scripts/checkout-with-prs.sh for
# the skip/fail rules.
LOCAL_PATCHES=/build/patches checkout-with-prs.sh \
    https://github.com/ggml-org/llama.cpp.git "${LLAMA_COMMIT}" /src/llama.cpp ${LLAMA_PATCHES:-}
cd /src/llama.cpp

echo "=== glslc feature tests (llama.cpp's own) ==="
for t in integer_dot bfloat16 coopmat; do
    f="ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/$t.comp"
    [ -f "$f" ] || { echo "(no feature test $t in this revision, skipping)"; continue; }
    if glslc -o /dev/null -fshader-stage=compute --target-env=vulkan1.3 "$f" >/dev/null 2>&1; then
        echo "  $t: OK"
    else
        echo "FATAL: glslc cannot compile $f -- the Vulkan build would lose that code path" >&2
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
mkdir -p "$OUT" /install/build-info
for bin in llama-server llama-cli llama-tts llama-bench; do
    [ -f "build/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
    cp "build/bin/$bin" "$OUT/"
done
cp -P build/bin/*.so* "$OUT/"
ls "$OUT"/libggml-cpu-*.so >/dev/null 2>&1 || { echo "FATAL: no ggml-cpu variants built" >&2; exit 1; }
ls "$OUT"/libggml-vulkan.so >/dev/null 2>&1 || { echo "FATAL: libggml-vulkan.so not built" >&2; exit 1; }
# Relocatable check: every ELF's run path must start with $ORIGIN and must not
# point into the build tree.
for f in "$OUT"/*; do
    [ -L "$f" ] && continue
    rp=$(readelf -d "$f" 2>/dev/null | awk '/RUNPATH|RPATH/ {gsub(/[\[\]]/,"",$NF); print $NF}')
    if [ -n "$rp" ] && { [[ "$rp" != '$ORIGIN'* ]] || [[ "$rp" == */src/* ]]; }; then
        echo "FATAL: $f has run path '$rp' (expected \$ORIGIN[:...])" >&2; exit 1; fi
    if ldd "$f" 2>/dev/null | grep -q "not found"; then
        echo "FATAL: $f has unresolved libraries" >&2; ldd "$f" | grep "not found" >&2; exit 1; fi
done
{ echo "llama_vulkan_commit: $(cat .base-commit) (requested: ${LLAMA_COMMIT}; merged tree $(git rev-parse --short HEAD))";
  echo "llama_patches: $(cat .merged-prs)";
  echo "llama_local_patches: $(cat .local-patches)";
  echo "vulkan_glslc: $(glslc --version | head -1)"; } > /install/build-info/llama-vulkan
BUILD

# ── Build whisper.cpp (Vulkan) ─────────────────────────────────────────

FROM vulkan-builder AS whisper-vulkan
ARG WHISPER_COMMIT
RUN --mount=type=cache,id=ccache-vulkan,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

echo "=== Cloning whisper.cpp at ${WHISPER_COMMIT} ==="
mkdir -p /src/whisper.cpp && cd /src/whisper.cpp
git init -q
git remote add origin https://github.com/ggml-org/whisper.cpp.git
git fetch --depth=1 origin "${WHISPER_COMMIT}"
git checkout -q FETCH_HEAD
echo "whisper.cpp at $(git rev-parse HEAD)"

echo "=== Building whisper.cpp (Vulkan, static) ==="
# Static: no shared libggml in /usr/local/lib to collide with anything.
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

mkdir -p /install/bin /install/build-info
for bin in whisper-server whisper-cli; do
    [ -f "build/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
    if readelf -d "build/bin/$bin" | grep -q 'libggml\|libwhisper'; then
        echo "FATAL: $bin is not statically linked against ggml/whisper" >&2; exit 1; fi
    cp "build/bin/$bin" /install/bin/
done
echo "whisper_vulkan_commit: $(git rev-parse HEAD) (requested: ${WHISPER_COMMIT})" > /install/build-info/whisper-vulkan
BUILD

# ── sd-server web UI (built once, embedded into both sd-server builds) ──
# sd.cpp embeds its frontend (the sdcpp-webui submodule, pinned per revision)
# when CMake finds pnpm -- or a pre-generated frontend/dist/gen_index_html.h.
# Upstream's builder has neither, so its sd-server answers "/" with a text
# placeholder. Build the header here with Node and hand it to the C++ stages.

FROM node:22-alpine AS sd-frontend
RUN apk add --no-cache git && corepack enable
ARG SD_COMMIT
RUN <<'BUILD'
#!/bin/sh
set -eu
mkdir -p /src/sd && cd /src/sd
git init -q && git remote add origin https://github.com/leejet/stable-diffusion.cpp.git
git fetch --depth=1 origin "${SD_COMMIT}" && git checkout -q FETCH_HEAD
git submodule update --init --depth=1 examples/server/frontend
cd examples/server/frontend
echo "sd_server_webui: embedded ($(git rev-parse HEAD))" > /src/frontend-version
pnpm install --frozen-lockfile
pnpm run build
pnpm run build:header
[ -f dist/gen_index_html.h ] || { echo "FATAL: gen_index_html.h not produced" >&2; exit 1; }
BUILD

# ── Build stable-diffusion.cpp (Vulkan) ────────────────────────────────

FROM vulkan-builder AS sd-vulkan
ARG SD_COMMIT
COPY --from=sd-frontend /src/sd/examples/server/frontend/dist/gen_index_html.h /src/frontend-version /tmp/sd-frontend/
RUN --mount=type=cache,id=ccache-vulkan,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

echo "=== Cloning stable-diffusion.cpp at ${SD_COMMIT} ==="
mkdir -p /src/stable-diffusion.cpp && cd /src/stable-diffusion.cpp
git init -q
git remote add origin https://github.com/leejet/stable-diffusion.cpp.git
git fetch --depth=1 origin "${SD_COMMIT}"
git checkout -q FETCH_HEAD
git submodule update --init --recursive --depth=1
echo "stable-diffusion.cpp at $(git rev-parse HEAD)"
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

mkdir -p /install/bin /install/build-info
for bin in sd-server sd-cli; do
    [ -f "build/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
    if readelf -d "build/bin/$bin" | grep -q 'libggml\|libstable'; then
        echo "FATAL: $bin expects shared ggml/sd libs" >&2; exit 1; fi
    cp "build/bin/$bin" /install/bin/
done
{ echo "sd_vulkan_commit: $(git rev-parse HEAD) (requested: ${SD_COMMIT})";
  cat /tmp/sd-frontend/frontend-version; } > /install/build-info/sd-vulkan
BUILD

# ── Build audio.cpp (Vulkan): audiocpp_server, audiocpp_cli, audiocpp_gguf ──
# Server and CLI exactly as upstream's install-audio.sh builds them for the
# vulkan flavour, plus the GGUF converter that upstream does not ship:
#   AUDIOCPP_DEPLOYMENT_BUILD=ON compiles the model_specs/*.json catalog into
#   the binaries (a bare binary otherwise fails with "model spec not found"
#   for anything that is not a GGUF with an embedded spec); the on-disk catalog
#   is installed too so --model-spec-override has a path to point at.
#   ENGINE_ENABLE_NATIVE_CPU=OFF: portable CPU kernels, and it keeps the build
#   static (audio.cpp only switches to shared libs under CPU_ALL_VARIANTS).
# audiocpp_gguf turns a HF safetensors checkpoint into an audio.cpp GGUF
# package (weights + embedded tokenizer/config sidecars + the family's spec
# from the compiled-in catalog):
#   audiocpp_gguf --input <ckpt>/model.safetensors --root <ckpt> \
#       --family audio8_asr --type q8_0 --output <out>/audio8-asr-0.1b-q8_0.gguf
# Needed for community models whose licence forbids redistributing converted
# weights (audio8_asr, CC-BY-NC-4.0), absent from audio-cpp/audio.cpp-gguf.
# One tree for all three, so the converter's catalog matches the server.

FROM vulkan-builder AS audiocpp
ARG AUDIOCPP_COMMIT
RUN --mount=type=cache,id=ccache-vulkan,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

echo "=== Cloning audio.cpp at ${AUDIOCPP_COMMIT} ==="
mkdir -p /src/audio.cpp && cd /src/audio.cpp
git init -q
git remote add origin https://github.com/0xShug0/audio.cpp.git
git fetch --depth=1 origin "${AUDIOCPP_COMMIT}"
git checkout -q FETCH_HEAD
echo "audio.cpp at $(git rev-parse HEAD)"

echo "=== Building audio.cpp (Vulkan, static) ==="
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DAUDIOCPP_DEPLOYMENT_BUILD=ON \
    -DAUDIOCPP_MODEL_SET=full \
    -DENGINE_ENABLE_NATIVE_CPU=OFF \
    -DENGINE_ENABLE_OPENMP=ON \
    -DENGINE_BUILD_EXAMPLES=OFF \
    -DENGINE_BUILD_TESTS=OFF \
    -DENGINE_BUILD_WARMBENCH=OFF \
    -DENGINE_ENABLE_CUDA=OFF \
    -DENGINE_ENABLE_HIP=OFF \
    -DENGINE_ENABLE_VULKAN=ON
cmake --build build --config Release -j"$(nproc)" \
    --target audiocpp_cli audiocpp_server audiocpp_gguf

mkdir -p /install/bin /install/share/audiocpp /install/build-info
for bin in audiocpp_cli audiocpp_server audiocpp_gguf; do
    [ -f "build/bin/$bin" ] || { echo "FATAL: $bin not built" >&2; exit 1; }
    needed=$(readelf -d "build/bin/$bin" | grep NEEDED || true)
    # A backend that silently failed to enable still produces working binaries
    # that fall back to CPU at runtime -- catch it here.
    if [ "$bin" != audiocpp_gguf ] && ! grep -q 'libvulkan\.so' <<<"$needed"; then
        echo "FATAL: $bin is not linked against libvulkan:" >&2; echo "$needed" >&2; exit 1; fi
    if grep -qE 'libggml|libengine' <<<"$needed"; then
        echo "FATAL: $bin expects audio.cpp shared libraries; only static builds are installed" >&2
        echo "$needed" >&2; exit 1; fi
    cp "build/bin/$bin" /install/bin/
done
cp -r model_specs /install/share/audiocpp/model_specs
# Usage exits non-zero; the point is that the converter loads and runs.
{ /install/bin/audiocpp_gguf 2>&1 || true; } | grep -q '^Usage: audiocpp_gguf' \
    || { echo "FATAL: audiocpp_gguf does not run" >&2; exit 1; }
echo "audiocpp_commit: $(git rev-parse HEAD) (requested: ${AUDIOCPP_COMMIT})" > /install/build-info/audiocpp
BUILD

# ── Build llama-swap + vllm-wrapper from source ────────────────────────
# Three stages: fetch + merge PRs (git), build the Svelte UI (node), build the
# Go binaries with the UI embedded (`-tags embed_ui`, see the upstream Makefile
# and internal/server/embed.go). vllm-wrapper is not in upstream's release
# archives either, so it comes from the same tree.

FROM golang:1.27-bookworm AS llama-swap-src
ARG LLAMA_SWAP_COMMIT
ARG LLAMA_SWAP_PATCHES
ARG LLAMA_SWAP_PATCHES_HEADS
COPY --chmod=0755 scripts/checkout-with-prs.sh /usr/local/bin/checkout-with-prs.sh
RUN <<'FETCH'
#!/bin/bash
set -euo pipefail
FETCH_TAGS=1 checkout-with-prs.sh \
    https://github.com/mostlygeek/llama-swap.git "${LLAMA_SWAP_COMMIT}" /src/llama-swap ${LLAMA_SWAP_PATCHES:-}
cd /src/llama-swap
# Version string as upstream's Makefile derives it (git describe on the base
# commit) plus a +prN suffix per merged PR, e.g. v255-2-g1a2b3c+pr1099.
BASE=$(cat .base-commit)
VERSION=$(git describe --tags --abbrev=6 "$BASE" 2>/dev/null || echo devel)
for pr in $(cat .merged-prs); do VERSION="${VERSION}+pr${pr}"; done
COMMIT=$(git rev-parse --short "$BASE")
[ -z "$(cat .merged-prs)" ] || COMMIT="${COMMIT}+"
{ echo "LS_VERSION=${VERSION}"; echo "LS_COMMIT=${COMMIT}"; } > .version
mkdir -p /install/build-info
{ echo "llama_swap_version: ${VERSION}";
  echo "llama_swap_commit: ${BASE} (requested: ${LLAMA_SWAP_COMMIT}; merged tree $(git rev-parse --short HEAD))";
  echo "llama_swap_patches: $(cat .merged-prs)"; } > /install/build-info/llama-swap
FETCH

FROM node:24-bookworm-slim AS llama-swap-ui
COPY --from=llama-swap-src /src/llama-swap/ui /src/ui
# vite.config.ts writes to ../internal/server/ui_dist
RUN --mount=type=cache,id=npm,target=/root/.npm \
    cd /src/ui && npm ci --no-audit --no-fund && npm run build \
    && test -f /src/internal/server/ui_dist/index.html

FROM golang:1.27-bookworm AS llama-swap-build
COPY --from=llama-swap-src /src/llama-swap /src/llama-swap
COPY --from=llama-swap-src /install/build-info /install/build-info
COPY --from=llama-swap-ui /src/internal/server/ui_dist /src/llama-swap/internal/server/ui_dist
RUN --mount=type=cache,id=go-build,target=/root/.cache/go-build \
    --mount=type=cache,id=go-mod,target=/go/pkg/mod <<'BUILD'
#!/bin/bash
set -euo pipefail
cd /src/llama-swap
. ./.version
DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== Building llama-swap ${LS_VERSION} (${LS_COMMIT}) ==="
mkdir -p /install/bin
CGO_ENABLED=0 go build -trimpath -tags embed_ui \
    -ldflags="-s -w -X main.version=${LS_VERSION} -X main.commit=${LS_COMMIT} -X main.date=${DATE}" \
    -o /install/bin/llama-swap .
echo "=== Building vllm-wrapper ==="
CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /install/bin/vllm-wrapper ./cmd/vllm-wrapper
/install/bin/llama-swap -version
# The UI must actually be inside the binary (the embed_ui tag with an empty
# ui_dist would build a server that 404s on /).
grep -q '<!doctype html' /install/bin/llama-swap || grep -qi '<!DOCTYPE html' /install/bin/llama-swap \
    || { echo "FATAL: llama-swap binary does not contain the embedded UI" >&2; exit 1; }
BUILD

# ══════════════════════════════════════════════════════════════════════
# ── ROCm toolchain (selected by ROCM_CHANNEL, see the arg) ─────────────

# classic: AMD's prebuilt dev image; the build scripts derive HIPCXX/HIP_PATH
# from its hipconfig.
FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION}-complete AS rocm-toolchain-classic

# multiarch: plain Ubuntu 24.04 + the dev packages from
# repo.amd.com/rocm/packages-multi-arch (AMD publishes no prebuilt dev image
# for these releases). Everything lands under /opt/rocm/core-<series>/; the
# classic layout is symlinked back so ROCM_PATH=/opt/rocm and the cmake
# configs keep working. amdrocm-runtime-dev pulls the HIP headers/cmake and
# the LLVM toolchain (amdclang++, device bitcode); blas-host/-dev +
# hipblas-common-dev provide librocblas/libhipblas(lt) with headers and cmake
# configs for ggml-hip's find_package(hipblas/rocblas); solver-host provides
# librocsolver, which libhipblas.so references since 7.14 (static links fail
# without it and the runtime needs it on the NEEDED chain); amdrocm-base has
# rocminfo/rocm_agent_enumerator. There is no hipconfig here — HIPCXX and
# HIP_PATH are exported instead and the build scripts prefer them when set.
FROM ubuntu:24.04 AS rocm-toolchain-multiarch
ARG ROCM_SERIES
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg \
        | gpg --dearmor -o /etc/apt/keyrings/rocm-multiarch.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm-multiarch.gpg] https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404 stable main" \
        > /etc/apt/sources.list.d/rocm-multiarch.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        "amdrocm-runtime-dev${ROCM_SERIES}" \
        "amdrocm-blas-host${ROCM_SERIES}" \
        "amdrocm-blas-dev${ROCM_SERIES}" \
        "amdrocm-hipblas-common-dev${ROCM_SERIES}" \
        "amdrocm-solver-host${ROCM_SERIES}" \
        "amdrocm-base${ROCM_SERIES}" \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s "core-${ROCM_SERIES}" /opt/rocm/core \
    && for d in bin include lib libexec share; do \
        ln -s "core-${ROCM_SERIES}/$d" "/opt/rocm/$d"; done \
    && ln -s "core-${ROCM_SERIES}/lib/llvm" /opt/rocm/llvm \
    && ln -s "core-${ROCM_SERIES}/lib/llvm/amdgcn" /opt/rocm/amdgcn \
    && test -x /opt/rocm/lib/llvm/bin/amdclang++ \
    && test -f /opt/rocm/lib/cmake/hip/hip-config.cmake
ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIPCXX=/opt/rocm/lib/llvm/bin/amdclang++ \
    PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/rocm/lib/rocm_sysdeps/lib:/opt/rocm/lib

# ── ROCm builder base ──────────────────────────────────────────────────

FROM rocm-toolchain-${ROCM_CHANNEL} AS rocm-builder
ARG AMDGPU_TARGETS

ENV DEBIAN_FRONTEND=noninteractive
ENV AMDGPU_TARGETS=${AMDGPU_TARGETS}
ENV CCACHE_DIR=/ccache
ENV CCACHE_MAXSIZE=5G

# libav*-dev only for whisper.cpp's WHISPER_FFMPEG=ON; the final stage installs
# the matching Ubuntu 24.04 libav* runtime libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ccache curl ca-certificates \
        pkg-config libssl-dev \
        libavcodec-dev libavformat-dev libavutil-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --chmod=0755 scripts/checkout-with-prs.sh /usr/local/bin/checkout-with-prs.sh
WORKDIR /build

# ── Build llama.cpp (HIP) ──────────────────────────────────────────────

FROM rocm-builder AS llama-rocm
ARG LLAMA_COMMIT
ARG LLAMA_FA_ALL_QUANTS
ARG LLAMA_PATCHES
ARG LLAMA_PATCHES_HEADS
COPY patches/ /build/patches/
RUN --mount=type=cache,id=ccache-rocm,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

# Same tree as the Vulkan build: master + the open PRs + local patches.
LOCAL_PATCHES=/build/patches checkout-with-prs.sh \
    https://github.com/ggml-org/llama.cpp.git "${LLAMA_COMMIT}" /src/llama.cpp ${LLAMA_PATCHES:-}
cd /src/llama.cpp

echo "=== Building llama.cpp (HIP) for ${AMDGPU_TARGETS}, FA_ALL_QUANTS=${LLAMA_FA_ALL_QUANTS} ==="
# Shared + BACKEND_DL + CPU_ALL_VARIANTS like llama.cpp's own ROCm image; the
# HIP backend lives in libggml-hip.so next to the binaries (RPATH $ORIGIN).
# POSITION_INDEPENDENT_CODE: ggml-hip's device-stub objects are non-PIC by
# default and fail to link into Ubuntu's default-PIE executables.
HIPCXX="${HIPCXX:-$(hipconfig -l)/clang}" HIP_PATH="${HIP_PATH:-$(hipconfig -R)}" \
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
mkdir -p "$OUT" /install/build-info
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
{ echo "llama_rocm_commit: $(cat .base-commit) (requested: ${LLAMA_COMMIT}; merged tree $(git rev-parse --short HEAD))";
  echo "llama_rocm_patches: $(cat .merged-prs)";
  echo "llama_rocm_local_patches: $(cat .local-patches)";
  echo "rocm_fa_all_quants: ${LLAMA_FA_ALL_QUANTS}"; } > /install/build-info/llama-rocm
BUILD

# ── Build EngramHalo.cpp (HIP, Strix Halo only) ────────────────────────

FROM rocm-builder AS llama-engram
ARG ENGRAM_REPO
ARG ENGRAM_BRANCH
ARG ENGRAM_COMMIT
ARG ENGRAM_TARGETS
RUN --mount=type=cache,id=ccache-rocm,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

REF="${ENGRAM_COMMIT:-${ENGRAM_BRANCH}}"
echo "=== Cloning EngramHalo.cpp (${ENGRAM_BRANCH} @ ${REF}) ==="
mkdir -p /src/engram && cd /src/engram
git init -q
git remote add origin "${ENGRAM_REPO}"
git fetch --depth=1 origin "${REF}"
git checkout -q FETCH_HEAD
git submodule update --init --recursive --depth=1
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
HIPCXX="${HIPCXX:-$(hipconfig -l)/clang}" HIP_PATH="${HIP_PATH:-$(hipconfig -R)}" \
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
mkdir -p "$OUT" /install/build-info
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
  echo "llama_engram_targets: ${ENGRAM_TARGETS}"; } > /install/build-info/llama-engram
BUILD

# ── Build whisper.cpp (HIP) ────────────────────────────────────────────

FROM rocm-builder AS whisper-rocm
ARG WHISPER_COMMIT
RUN --mount=type=cache,id=ccache-rocm,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

echo "=== Cloning whisper.cpp at ${WHISPER_COMMIT} ==="
mkdir -p /src/whisper.cpp && cd /src/whisper.cpp
git init -q
git remote add origin https://github.com/ggml-org/whisper.cpp.git
git fetch --depth=1 origin "${WHISPER_COMMIT}"
git checkout -q FETCH_HEAD
echo "whisper.cpp at $(git rev-parse HEAD)"

echo "=== Building whisper.cpp (HIP) for ${AMDGPU_TARGETS} ==="
# POSITION_INDEPENDENT_CODE: see llama.cpp stage
HIPCXX="${HIPCXX:-$(hipconfig -l)/clang}" HIP_PATH="${HIP_PATH:-$(hipconfig -R)}" \
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

mkdir -p /install/bin /install/build-info
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
echo "whisper_rocm_commit: $(git rev-parse HEAD) (requested: ${WHISPER_COMMIT})" > /install/build-info/whisper-rocm
BUILD

# ── Build stable-diffusion.cpp (HIP) ───────────────────────────────────

FROM rocm-builder AS sd-rocm
ARG SD_COMMIT
COPY --from=sd-frontend /src/sd/examples/server/frontend/dist/gen_index_html.h /src/frontend-version /tmp/sd-frontend/
RUN --mount=type=cache,id=ccache-rocm,target=/ccache <<'BUILD'
#!/bin/bash
set -euo pipefail

echo "=== Cloning stable-diffusion.cpp at ${SD_COMMIT} ==="
mkdir -p /src/stable-diffusion.cpp && cd /src/stable-diffusion.cpp
git init -q
git remote add origin https://github.com/leejet/stable-diffusion.cpp.git
git fetch --depth=1 origin "${SD_COMMIT}"
git checkout -q FETCH_HEAD
git submodule update --init --recursive --depth=1
echo "stable-diffusion.cpp at $(git rev-parse HEAD)"
# Pre-built web UI header (see the sd-frontend stage) -> embedded frontend
mkdir -p examples/server/frontend/dist
cp /tmp/sd-frontend/gen_index_html.h examples/server/frontend/dist/

echo "=== Building stable-diffusion.cpp (HIP) for ${AMDGPU_TARGETS} ==="
# POSITION_INDEPENDENT_CODE: see llama.cpp stage (sd.cpp also sets it itself)
HIPCXX="${HIPCXX:-$(hipconfig -l)/clang}" HIP_PATH="${HIP_PATH:-$(hipconfig -R)}" \
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

mkdir -p /install/bin /install/build-info
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
echo "sd_rocm_commit: $(git rev-parse HEAD) (requested: ${SD_COMMIT})" > /install/build-info/sd-rocm
BUILD

# ── ROCm stage selection (WITH_ROCM) ───────────────────────────────────
# Alias stages: the final stage copies from `<name>-sel`, which FROMs
# `<name>-${WITH_ROCM}`; with false that resolves to this empty stand-in and the
# HIP builders are never started.

FROM alpine:3 AS rocm-none
RUN mkdir -p /install/bin /install/llama-rocm /install/llama-engram /install/build-info

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

# ══════════════════════════════════════════════════════════════════════
# ── Final image: Ubuntu 24.04 runtime (+ ROCm) + everything built above ──

FROM ubuntu:24.04 AS final
ARG ROCM_CHANNEL
ARG ROCM_VERSION
ARG ROCM_SERIES
ARG AMDGPU_TARGETS
ARG MESA_PPA
ARG QWEN_TEMPLATE_URL
ARG QWEN_SHARP_TEMPLATE_URL
ARG WITH_ROCM
ARG WITH_ENGRAM

LABEL org.opencontainers.image.source="https://github.com/SelfRef/llama-swap-docker-amd" \
      org.opencontainers.image.description="llama-swap unified image for AMD GPUs (ROCm + Vulkan)"

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/bin:${PATH}"

# Cache key for everything below (see the arg's comment at the top).
ARG BUILD_DATE

# Runtime packages: Vulkan loader + RADV, libgomp for the CPU backends, libav*
# + ffmpeg for whisper-server's WHISPER_FFMPEG input decoding, rocm-smi for
# llama-swap's GPU monitor in its UI (sysfs-based, works without the ROCm
# runtime), curl for healthchecks, python3 + PyYAML for the bundled `benchmark`
# CLI. Mesa: only mesa-vulkan-drivers (+ deps) is taken from the PPA, not the
# whole GL stack; software-properties-common is only needed to add it and is
# purged again.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libvulkan1 mesa-vulkan-drivers \
        rocm-smi \
        curl ca-certificates \
        libavcodec60 libavformat60 libavutil58 libswresample4 \
        ffmpeg \
        python3 python3-yaml \
    && if [ -n "${MESA_PPA}" ]; then \
        apt-get install -y --no-install-recommends software-properties-common \
        && add-apt-repository -y "${MESA_PPA}" \
        && apt-get install -y --no-install-recommends --only-upgrade mesa-vulkan-drivers \
        && apt-get purge -y --auto-remove software-properties-common; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# uv/uvx (static binaries from the official image) for `uvx`-launched tools in
# config.yaml commands and maintenance scripts (e.g. `uvx --from huggingface_hub hf`).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# ROCm userspace matching the builder's channel (see ROCM_CHANNEL).
# classic: hipblas/rocblas pull in the HIP runtime (libamdhip64), hsa-rocr,
# comgr etc. via package dependencies; hipblaslt is explicit — rocBLAS dlopens
# it on some architectures (gfx90a/gfx942/RDNA4), which no NEEDED-entry check
# can catch. This is the all-arch variant (~6 GB of kernels).
# multiarch: HIP runtime + host-side BLAS libs + rocminfo (amdrocm-base) and
# ONLY the per-gfx kernel packages for AMDGPU_TARGETS (~50 MB per arch);
# layout is /opt/rocm/core-<series>/ symlinked to the classic paths, and
# rocm_sysdeps (AMD's vendored deps the libs link against) needs its own
# ld.so.conf entry.
RUN if [ "${WITH_ROCM}" = "true" ]; then \
    apt-get update && apt-get install -y --no-install-recommends gnupg \
    && mkdir -p /etc/apt/keyrings \
    && if [ "${ROCM_CHANNEL}" = "classic" ]; then \
        curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
            | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg \
        && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
            > /etc/apt/sources.list.d/rocm.list \
        && printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
            > /etc/apt/preferences.d/rocm-pin-600 \
        && apt-get update \
        && apt-get install -y --no-install-recommends hipblas rocblas hipblaslt rocminfo \
        && echo /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf; \
    else \
        curl -fsSL https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg \
            | gpg --dearmor -o /etc/apt/keyrings/rocm-multiarch.gpg \
        && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm-multiarch.gpg] https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404 stable main" \
            > /etc/apt/sources.list.d/rocm-multiarch.list \
        && apt-get update \
        && PKGS="amdrocm-runtime${ROCM_SERIES} amdrocm-blas-host${ROCM_SERIES} amdrocm-solver-host${ROCM_SERIES} amdrocm-base${ROCM_SERIES}" \
        && for t in $(echo "${AMDGPU_TARGETS}" | tr ';' ' '); do \
            PKGS="$PKGS amdrocm-blas${ROCM_SERIES}-${t}"; done \
        && apt-get install -y --no-install-recommends $PKGS \
        && ln -s "core-${ROCM_SERIES}" /opt/rocm/core \
        && for d in bin include lib libexec share; do \
            ln -s "core-${ROCM_SERIES}/$d" "/opt/rocm/$d"; done \
        && { echo /opt/rocm/lib; echo /opt/rocm/lib/rocm_sysdeps/lib; } \
            > /etc/ld.so.conf.d/rocm.conf; \
    fi \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig; \
    fi

ENV PATH="/opt/rocm/bin:${PATH}"

RUN mkdir -p /etc/llama-swap/config /models

# ── Binaries ──
COPY --from=llama-vulkan   /install/llama-vulkan/ /opt/llama-vulkan/
COPY --from=whisper-vulkan /install/bin/ /usr/local/bin/
COPY --from=sd-vulkan      /install/bin/ /usr/local/bin/
COPY --from=audiocpp       /install/bin/ /usr/local/bin/
COPY --from=audiocpp       /install/share/audiocpp/ /usr/local/share/audiocpp/
COPY --from=llama-swap-build /install/bin/ /usr/local/bin/
COPY --from=llama-rocm-sel   /install/llama-rocm/ /opt/llama-rocm/
COPY --from=whisper-rocm-sel /install/bin/ /usr/local/bin/
COPY --from=sd-rocm-sel      /install/bin/ /usr/local/bin/
COPY --from=llama-engram-sel /install/llama-engram/ /opt/llama-engram/
# build-info of every stage -> /versions.txt below
COPY --from=llama-vulkan     /install/build-info/ /tmp/build-info/
COPY --from=whisper-vulkan   /install/build-info/ /tmp/build-info/
COPY --from=sd-vulkan        /install/build-info/ /tmp/build-info/
COPY --from=audiocpp         /install/build-info/ /tmp/build-info/
COPY --from=llama-swap-build /install/build-info/ /tmp/build-info/
COPY --from=llama-rocm-sel   /install/build-info/ /tmp/build-info/
COPY --from=whisper-rocm-sel /install/build-info/ /tmp/build-info/
COPY --from=sd-rocm-sel      /install/build-info/ /tmp/build-info/
COPY --from=llama-engram-sel /install/build-info/ /tmp/build-info/
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
    else rmdir /opt/llama-engram; fi \
    && ldconfig

# Example config with both backends; override by mounting /etc/llama-swap/config
COPY config/config.yaml /etc/llama-swap/config/config.yaml

# `benchmark` CLI (scripts/benchmark): server-level (via llama-swap), kernel-level
# (llama-bench[-variant]) and standalone (llama-server-<variant>) benchmarks of the
# config.yaml text entries, one table. Pure python3 + PyYAML (apt: the image is
# PEP-668 externally managed).
COPY --chmod=0755 scripts/benchmark /usr/local/bin/benchmark

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
# that backend loading does not fail), and that llama-swap runs and accepts
# the bundled config.
RUN <<'CHECK'
#!/bin/bash
set -euo pipefail
BINS="llama-server llama-cli llama-tts llama-bench whisper-server whisper-cli sd-server sd-cli audiocpp_server audiocpp_cli audiocpp_gguf"
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
        echo "FATAL: $lib has unresolved libraries" >&2; ldd "$lib" | grep 'not found' >&2; exit 1; fi
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
# Vulkan feature line is checked at build time in the builder; here only that
# the driver is the PPA one when asked for.
if [ -n "${MESA_PPA}" ]; then
    dpkg-query -W -f '${Version}\n' mesa-vulkan-drivers | grep -q kisak \
        || { echo "FATAL: mesa-vulkan-drivers is not the ${MESA_PPA} build: $(dpkg-query -W -f '${Version}' mesa-vulkan-drivers)" >&2; exit 1; }
fi
llama-swap -version
vllm-wrapper --help >/dev/null 2>&1 || vllm-wrapper -h >/dev/null 2>&1 || true
llama-swap -config /etc/llama-swap/config/config.yaml -validate
uv --version && uvx --version
python3 -c "import yaml" || { echo "FATAL: PyYAML missing" >&2; exit 1; }
benchmark --list --config /etc/llama-swap/config/config.yaml >/dev/null \
    || { echo "FATAL: benchmark --list failed on the bundled config" >&2; exit 1; }
test -d /usr/local/share/audiocpp/model_specs
ls /opt/llama-vulkan/libggml-cpu-*.so | sed 's|.*/libggml-cpu-||; s|\.so||' | tr '\n' ' ' | sed 's/^/cpu variants: /; s/ $/\n/'
CHECK

# /versions.txt: one-line summary per project first, then every stage's full
# build-info (base commit, merged PRs, build options).
RUN <<'VERSIONS'
#!/bin/bash
set -euo pipefail
first() { awk -v k="$1" '$1==k {print $2; exit}' "/tmp/build-info/$2"; }
{
  echo "llama.cpp: $(first llama_vulkan_commit: llama-vulkan)"
  echo "whisper.cpp: $(first whisper_vulkan_commit: whisper-vulkan)"
  echo "stable-diffusion.cpp: $(first sd_vulkan_commit: sd-vulkan)"
  echo "audio.cpp: $(first audiocpp_commit: audiocpp)"
  echo "llama-swap: $(first llama_swap_version: llama-swap)"
  if [ "${WITH_ROCM}" = "true" ]; then echo "backend: vulkan rocm"; else echo "backend: vulkan"; fi
  echo "build_timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [ "${WITH_ROCM}" = "true" ]; then
    if [ "${ROCM_CHANNEL}" = "classic" ]; then echo "rocm: ${ROCM_VERSION} (classic)"
    else echo "rocm: $(dpkg-query -W -f '${Version}' "amdrocm-runtime${ROCM_SERIES}") (multiarch, series ${ROCM_SERIES})"; fi
    echo "amdgpu_targets: ${AMDGPU_TARGETS}"
  fi
  echo "mesa_vulkan_drivers: $(dpkg-query -W -f '${Version}' mesa-vulkan-drivers) (${MESA_PPA:-ubuntu})"
  echo "cpu_variants: $(ls /opt/llama-vulkan/libggml-cpu-*.so | sed 's|.*/libggml-cpu-||; s|\.so||' | tr '\n' ' ')"
  for f in llama-swap llama-vulkan llama-rocm llama-engram whisper-vulkan whisper-rocm sd-vulkan sd-rocm audiocpp; do
    [ -f "/tmp/build-info/$f" ] && cat "/tmp/build-info/$f"
  done
  echo "qwen_chat_template: $(grep -o 'template_version = "[^"]*"' /etc/llama-swap/templates/qwen-fixed.jinja | head -1 | cut -d'"' -f2) (${QWEN_TEMPLATE_URL})"
  echo "qwen_sharp_chat_template: $(grep -o 'template_version = "[^"]*"' /etc/llama-swap/templates/qwen-sharp.jinja | head -1 | cut -d'"' -f2) (${QWEN_SHARP_TEMPLATE_URL})"
} > /versions.txt
rm -rf /tmp/build-info
cat /versions.txt
VERSIONS

# Root (device access without --group-add), /models as the working directory.
# llama-swap is the entrypoint and its defaults live in CMD, so any container
# argument replaces them: `docker run <image> -version`, or
# `docker run <image> -config /models/my.yaml -listen 0.0.0.0:8080 -watch-config`.
WORKDIR /models
USER 0
ENTRYPOINT ["llama-swap"]
CMD ["-config", "/etc/llama-swap/config/config.yaml", "-listen", "0.0.0.0:8080", "-watch-config"]
