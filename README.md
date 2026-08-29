# llama-swap Docker image for AMD GPUs (ROCm + Vulkan)

[llama-swap](https://github.com/mostlygeek/llama-swap) publishes a `unified-vulkan` image with every inference engine built for Vulkan — which works on AMD, but leaves performance on the table. This repo extends that image into an AMD-focused one with **both GPU stacks**, both tuned:

- **Vulkan** (Mesa RADV) — works on practically any AMD GPU, including RDNA1/2, iGPUs/APUs and anything ROCm doesn't cover. The engines are **rebuilt here** at the base image's commits with a modern shader compiler (see [Why rebuild the Vulkan binaries](#why-rebuild-the-vulkan-binaries)) and the image ships a **current Mesa/RADV** instead of Ubuntu 24.04's.
- **ROCm 7.2** (HIP) — added here: the full ROCm userspace runtime (HIP, rocBLAS/hipBLAS + Tensile kernels, hipBLASLt, `rocminfo`) plus HIP rebuilds of the engines, with flash-attention kernels for every KV-cache quant.

`llama.cpp` is built from current master plus selected upstream PRs (`LLAMA_COMMIT`, `LLAMA_PATCHES`), and both backends of it are built with runtime CPU dispatch (`GGML_CPU_ALL_VARIANTS`), so one image gets AVX2 on Zen 3 and AVX-512/VNNI/BF16 on Zen 4/5 for CPU-offloaded layers. Every binary is compiled **from the exact same upstream commits** as the base image (parsed from its `/versions.txt`), so each engine ships as a matched Vulkan/ROCm pair — pick the backend per model in your llama-swap config.

## What's inside

| Engine | Vulkan (rebuilt here) | ROCm (built here) |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `llama-server`, `llama-cli`, `llama-tts`, `llama-bench` | `llama-server-rocm`, `llama-cli-rocm`, `llama-tts-rocm`, `llama-bench-rocm` |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | `whisper-server`, `whisper-cli` | `whisper-server-rocm`, `whisper-cli-rocm` |
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | `sd-server`, `sd-cli` | `sd-server-rocm`, `sd-cli-rocm` |
| [audio.cpp](https://github.com/0xShug0/audio.cpp) | `audiocpp_server`, `audiocpp_cli` (from base image) | — (no HIP backend upstream) |
| llama-swap + [vllm-wrapper](https://github.com/mostlygeek/llama-swap/tree/main/cmd/vllm-wrapper) | `llama-swap`, `vllm-wrapper` (backend-independent, from base image) | |

llama.cpp lives in self-contained directories `/opt/llama-vulkan` and `/opt/llama-rocm` (binaries, `libllama`/`libggml*` and the per-CPU-level `libggml-cpu-*.so` variants, RPATH `$ORIGIN`) with symlinks in `/usr/local/bin`; whisper/sd binaries are static. `ik-llama-server` from the CUDA image is not included (upstream builds it CUDA-only). Exact versions of everything — including the glslc used and the enabled build options — are recorded in `/versions.txt` inside the image.

## Why rebuild the Vulkan binaries

Upstream builds the Vulkan engines on Ubuntu 24.04 with its stock `glslc` (shaderc 2023.8 / glslang 14). That compiler cannot compile the `GL_EXT_integer_dot_product` and `GL_EXT_bfloat16` shaders, and llama.cpp's CMake then *silently* drops those code paths: at startup the device line reads `int dot: 0 | bf16: 0` even though RADV advertises both extensions. The integer-dot path is llama.cpp's fast quantized path on GPUs **without** cooperative-matrix support — q8_1 MMQ for K-quants (upstream measured ~2x prompt processing on an RX 6800 XT), DP4A flash attention for q8_0/q4_0 KV caches, MMVQ decode. On coopmat GPUs (RDNA3 and newer) llama.cpp keeps its FP16 coopmat matmul for prompts, so on those the rebuild changes little by itself (RX 7900 XTX: prompt speed identical, decode +2%); it still matters for RDNA1/2, older cards and iGPUs, for any future upstream work that needs those extensions, and it removes a silent build-dependent difference between GPUs.

This image rebuilds the same upstream commits on the same Ubuntu 24.04 ABI, but with `glslc`/`libshaderc1` taken from the Ubuntu 26.04 pocket (glslang 16; `GLSLC_SUITE`). The build fails if the feature tests do not pass, so the extensions cannot silently disappear again. Verify at runtime: the `ggml_vulkan: 0 = ...` line must show `int dot: 1`.

The larger measured win on current GPUs is the driver: the final image upgrades `mesa-vulkan-drivers` from the kisak-mesa PPA (`MESA_PPA`, Mesa 26.1 vs Ubuntu's 25.2). RX 7900 XTX, identical binaries, Qwen3.8-27B Q4_K_XL: pp512 693 → 862 t/s (+24%), decode 37.7 → 37.9 t/s; the newer RADV also exposes `VK_VALVE_shader_mixed_float_dot_product` (`fp16: dot2`).

## Get the image

Prebuilt by [GitHub Actions](.github/workflows/build.yml) on every change and weekly (to track the daily-rebuilt base image):

```bash
docker pull ghcr.io/selfref/llama-swap-docker-amd:latest
```

Or build locally (expect a couple of hours for the fat HIP builds):

```bash
docker buildx build -t llama-swap-amd .
```

Build args:

| Arg | Default | Purpose |
|---|---|---|
| `BASE_IMAGE` | `ghcr.io/mostlygeek/llama-swap:unified-vulkan` | Base image (must be the root variant, not `-rootless`). Pin a dated tag or digest for reproducibility. |
| `ROCM_VERSION` | `7.2.4` | ROCm version for both the builder image and the runtime apt packages |
| `AMDGPU_TARGETS` | `gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201` | gfx architectures compiled into the HIP binaries. Trim to just your GPU for a much faster build. |
| `LLAMA_COMMIT` | `master` | llama.cpp revision for both the Vulkan and the ROCm build (sha, tag, branch, or `refs/pull/N/head`). `""` = the base image's commit from `/versions.txt`. Defaults to master so the PRs in `LLAMA_PATCHES` apply and the newest backend work is in. |
| `WHISPER_COMMIT` / `SD_COMMIT` | *(empty)* | Override the project revision; empty means "same commit as the base image" |
| `GLSLC_SUITE` | `resolute` | Ubuntu release whose `glslc`/`libshaderc1` are used by the Vulkan builder (only those two packages; everything else stays 24.04) |
| `LLAMA_FA_ALL_QUANTS` | `ON` | ROCm llama.cpp: compile flash-attention kernels for all K/V cache quant combinations (without it only q8_0/q8_0 and q4_0/q4_0 stay on the GPU, see llama.cpp #27761). Set `OFF` for a faster build. |
| `MESA_PPA` | `ppa:kisak/kisak-mesa` | Newer Mesa/RADV for the final image; `""` keeps the base image's stock Mesa 25.2 |
| `LLAMA_PATCHES` | `27952` | Space-separated upstream llama.cpp PR numbers applied on top of `LLAMA_COMMIT` (both backends). A patch that no longer applies is skipped if the PR is merged, otherwise the build fails — never a silent no-op. See [Trying upstream PRs](#trying-upstream-prs). |

## Trying upstream PRs

`LLAMA_PATCHES="27952 26284"` fetches `github.com/ggml-org/llama.cpp/pull/<N>.patch` for each number and applies it to `LLAMA_COMMIT` before building (Vulkan and ROCm alike). If a patch no longer applies, the build asks GitHub whether the PR was merged: merged → skipped with a notice (drop it from the list), otherwise → build error (the PR moved; re-check it). Applied patches and the built llama.cpp commit are listed in `/versions.txt`.

Measured on an RX 7900 XTX with this image (Mesa 26.1, llama-bench, q8_0/q4_0 KV, ub 512; dense = Qwen3.8-27B UD-Q4_K_XL, MoE = Qwen3.6-35B-A3B UD-Q4_K_M):

| PR | what | dense pp512 | MoE pp512 | decode | verdict |
|---|---|---|---|---|---|
| [#27952](https://github.com/ggml-org/llama.cpp/pull/27952) | Vulkan int8 coopmat1 matmul for RDNA3/4 | 858 → **898** (+4.6%) | 3127 → **3704** (+18.5%) | unchanged | **default** |
| [#25483](https://github.com/ggml-org/llama.cpp/pull/25483) | Vulkan: skip unneeded MoE work in coopmat1 mul_mm | +0.3% | +0.2% | unchanged | not worth a patch |
| [#26284](https://github.com/ggml-org/llama.cpp/pull/26284) + [#26301](https://github.com/ggml-org/llama.cpp/pull/26301) | HIP: RDNA3 MMQ tuning + dequant-float matvec | ROCm 977 → 1000 (+2%, within noise) | +2% | unchanged | not adopted (#26284 still carries RDNA4 changes its reviewer wants removed; re-test when merged) |
| [#22970](https://github.com/ggml-org/llama.cpp/pull/22970) | Vulkan K-quant A-matrix transpose | – | – | – | stale, conflicts with master |

## Run

```bash
docker run -it --rm \
  --device /dev/kfd --device /dev/dri \
  --security-opt seccomp=unconfined \
  -p 8080:8080 \
  -v "$PWD/models:/models" \
  -v "$PWD/config:/etc/llama-swap/config" \
  ghcr.io/selfref/llama-swap-docker-amd:latest
```

Or `docker compose up` — see [compose.yml](compose.yml). The llama-swap UI is at http://localhost:8080. The host only needs the `amdgpu` kernel driver (ROCm userspace lives in the image); Vulkan-only use works without `/dev/kfd`.

Edit [config/config.yaml](config/config.yaml) to define your models — it shows the pattern: the same engine as `*-rocm` (ROCm) or plain (Vulkan), chosen per model. The container watches the config and reloads on change.

## Choosing ROCm vs Vulkan

| GPU | Recommendation |
|---|---|
| Instinct MI100–MI350 (gfx908/90a/942) | ROCm |
| RDNA3/3.5/4 — RX 7000/9000, Ryzen AI APUs (gfx11xx/12xx) | ROCm; Vulkan as fallback |
| RDNA2 — RX 6000 (gfx1030 covered, rest via override) | either; Vulkan often less fuss |
| RDNA1, Vega, older iGPUs | Vulkan |

The HIP binaries contain code for the `AMDGPU_TARGETS` listed above. A close-but-not-included consumer chip can often run with a spoofed architecture:

| GPU | Env var |
|---|---|
| RX 7000 series (gfx1100/1101/1102) | usually none needed |
| RX 6000 series below gfx1030 | `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| Ryzen AI MAX / Strix Halo (gfx1151) | usually none needed |

GPU selection on multi-GPU hosts: `HIP_VISIBLE_DEVICES=0` for `*-rocm` binaries, `GGML_VK_VISIBLE_DEVICES=0` for Vulkan ones.

## Verifying the stacks inside the container

```bash
# ROCm sees the GPU (needs /dev/kfd + /dev/dri)
docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
  --entrypoint rocminfo ghcr.io/selfref/llama-swap-docker-amd:latest

# Each backend's device list
docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
  --entrypoint llama-server-rocm ghcr.io/selfref/llama-swap-docker-amd:latest --list-devices
docker run --rm --device /dev/dri \
  --entrypoint llama-server ghcr.io/selfref/llama-swap-docker-amd:latest --list-devices

# Vulkan feature line -- must say "int dot: 1" (see "Why rebuild the Vulkan binaries")
docker run --rm --device /dev/dri --entrypoint llama-bench \
  ghcr.io/selfref/llama-swap-docker-amd:latest -m /dev/null 2>&1 | grep "ggml_vulkan: 0"

# Versions baked into the image
docker run --rm --entrypoint cat ghcr.io/selfref/llama-swap-docker-amd:latest /versions.txt
```

## Notes

- The image is large (~15–20 GB): the ROCm runtime with rocBLAS/hipBLASLt kernel libraries for 11 architectures accounts for most of it.
- The Vulkan engines are NOT the base image's binaries (see above); `audiocpp_server`, `llama-swap` and `vllm-wrapper` are. The base image's shared `libggml*.so`/`libwhisper*.so` in `/usr/local/lib` are removed (nothing uses them any more).
- The container runs as root (standard for ROCm images — device access works without any `--group-add`). Files created in `/models` will be root-owned on the host. If you run as a non-root user, pass the *numeric* host GIDs of your `video`/`render` groups (`--group-add $(getent group render | cut -d: -f3)`); the image has no `render` group, so adding it by name fails.
- Verified on a Ryzen AI MAX+ 395 / Radeon 8060S (Strix Halo, gfx1151): both `llama-server --list-devices` (Vulkan/RADV) and `llama-server-rocm --list-devices` (ROCm) see the GPU. A gfx1100-only build also works on it with `HSA_OVERRIDE_GFX_VERSION=11.0.0`.
- The base image is rebuilt daily by upstream; this image is rebuilt weekly by CI, so `latest` here can lag `unified-vulkan` by a few days. Trigger the workflow manually to sync sooner.

## Sources

- [llama-swap unified container docs](https://github.com/mostlygeek/llama-swap/tree/main/docker/unified)
- [llama.cpp ROCm Dockerfile](https://github.com/ggml-org/llama.cpp/blob/master/.devops/rocm.Dockerfile) (ROCm version + gfx target list followed here)
- [whisper.cpp ROCm build docs](https://github.com/ggml-org/whisper.cpp#amd-rocm-gpu-support)
- [ROCm apt installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager-index.html)
