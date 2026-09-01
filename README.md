# llama-swap Docker image for AMD GPUs (ROCm + Vulkan)

[llama-swap](https://github.com/mostlygeek/llama-swap) publishes a `unified-vulkan` image with every inference engine built for Vulkan — which works on AMD, but leaves performance on the table. This repo extends that image into an AMD-focused one with **both GPU stacks**, both tuned:

- **Vulkan** (Mesa RADV) — works on practically any AMD GPU, including RDNA1/2, iGPUs/APUs and anything ROCm doesn't cover. The engines are **rebuilt here** at the base image's commits with a modern shader compiler (see [Why rebuild the Vulkan binaries](#why-rebuild-the-vulkan-binaries)) and the image ships a **current Mesa/RADV** instead of Ubuntu 24.04's.
- **ROCm 7.2** (HIP) — added here: the full ROCm userspace runtime (HIP, rocBLAS/hipBLAS + Tensile kernels, hipBLASLt, `rocminfo`) plus HIP rebuilds of the engines, with flash-attention kernels for every KV-cache quant.
- **EngramHalo.cpp** (HIP, `:rocm` tag, gfx1151 only) — [Aristo94's llama.cpp fork](https://github.com/Aristo94/EngramHalo.cpp) tuned for Qwen 3.8 Flash-Next on Strix Halo, as a third `llama.cpp` install (`*-engram` binaries). See [EngramHalo.cpp for Strix Halo](#engramhalocpp-for-strix-halo).

`llama.cpp` is built from current master plus selected upstream PRs (`LLAMA_COMMIT`, `LLAMA_PATCHES`), and both backends of it are built with runtime CPU dispatch (`GGML_CPU_ALL_VARIANTS`), so one image gets AVX2 on Zen 3 and AVX-512/VNNI/BF16 on Zen 4/5 for CPU-offloaded layers. Every binary is compiled **from the exact same upstream commits** as the base image (parsed from its `/versions.txt`), so each engine ships as a matched Vulkan/ROCm pair — pick the backend per model in your llama-swap config.

## What's inside

| Engine | Vulkan (rebuilt here, all tags) | ROCm (built here, `:rocm` tag only) |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `llama-server`, `llama-cli`, `llama-tts`, `llama-bench` | `llama-server-rocm`, `llama-cli-rocm`, `llama-tts-rocm`, `llama-bench-rocm` |
| [EngramHalo.cpp](https://github.com/Aristo94/EngramHalo.cpp) (llama.cpp fork, Strix Halo/qwen4exp) | — (fork is ROCm/HIP-only) | `llama-server-engram`, `llama-cli-engram`, `llama-bench-engram` (gfx1151 only) |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | `whisper-server`, `whisper-cli` | `whisper-server-rocm`, `whisper-cli-rocm` |
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | `sd-server` (web UI embedded), `sd-cli` | `sd-server-rocm` (web UI embedded), `sd-cli-rocm` |
| [audio.cpp](https://github.com/0xShug0/audio.cpp) | `audiocpp_server`, `audiocpp_cli` (from base image) | — (no HIP backend upstream) |
| llama-swap + [vllm-wrapper](https://github.com/mostlygeek/llama-swap/tree/main/cmd/vllm-wrapper) | `llama-swap`, `vllm-wrapper` (backend-independent, from base image) | |

llama.cpp lives in self-contained directories `/opt/llama-vulkan`, `/opt/llama-rocm` and `/opt/llama-engram` (binaries, `libllama`/`libggml*` and the per-CPU-level `libggml-cpu-*.so` variants, RPATH `$ORIGIN`) with symlinks in `/usr/local/bin`; whisper/sd binaries are static. `ik-llama-server` from the CUDA image is not included (upstream builds it CUDA-only). Exact versions of everything — including the glslc used and the enabled build options — are recorded in `/versions.txt` inside the image.

## Why rebuild the Vulkan binaries

Upstream builds the Vulkan engines on Ubuntu 24.04 with its stock `glslc` (shaderc 2023.8 / glslang 14). That compiler cannot compile the `GL_EXT_integer_dot_product` and `GL_EXT_bfloat16` shaders, and llama.cpp's CMake then *silently* drops those code paths: at startup the device line reads `int dot: 0 | bf16: 0` even though RADV advertises both extensions. The integer-dot path is llama.cpp's fast quantized path on GPUs **without** cooperative-matrix support — q8_1 MMQ for K-quants (upstream measured ~2x prompt processing on an RX 6800 XT), DP4A flash attention for q8_0/q4_0 KV caches, MMVQ decode. On coopmat GPUs (RDNA3 and newer) llama.cpp keeps its FP16 coopmat matmul for prompts, so on those the rebuild changes little by itself (RX 7900 XTX: prompt speed identical, decode +2%); it still matters for RDNA1/2, older cards and iGPUs, for any future upstream work that needs those extensions, and it removes a silent build-dependent difference between GPUs.

This image rebuilds the same upstream commits on the same Ubuntu 24.04 ABI, but with `glslc`/`libshaderc1` taken from the Ubuntu 26.04 pocket (glslang 16; `GLSLC_SUITE`). The build fails if the feature tests do not pass, so the extensions cannot silently disappear again. Verify at runtime: the `ggml_vulkan: 0 = ...` line must show `int dot: 1`.

The larger measured win on current GPUs is the driver: the final image upgrades `mesa-vulkan-drivers` from the kisak-mesa PPA (`MESA_PPA`, Mesa 26.1 vs Ubuntu's 25.2). RX 7900 XTX, identical binaries, Qwen3.8-27B Q4_K_XL: pp512 693 → 862 t/s (+24%), decode 37.7 → 37.9 t/s; the newer RADV also exposes `VK_VALVE_shader_mixed_float_dot_product` (`fp16: dot2`).

## Get the image

Prebuilt by [GitHub Actions](.github/workflows/build.yml). Two tags from the same Dockerfile (`WITH_ROCM`):

```bash
docker pull ghcr.io/selfref/llama-swap-docker-amd:latest   # Vulkan only (~2 GB): every push + weekly rebuild
docker pull ghcr.io/selfref/llama-swap-docker-amd:rocm     # Vulkan + ROCm (~10 GB): manual runs with the "rocm" checkbox
```

The ROCm stages are fat multi-gfx HIP builds that take hours of runner time, so they are not part of the automatic builds: trigger the workflow by hand (Actions → Build image → Run workflow → tick **rocm**) whenever you want a fresh `:rocm`. The `*-rocm` binaries, the `*-engram` binaries (see [EngramHalo.cpp for Strix Halo](#engramhalocpp-for-strix-halo)), the ROCm runtime and `rocminfo` exist only in that tag.

Or build locally (expect a couple of hours for the fat HIP builds):

```bash
docker buildx build -t llama-swap-amd .
```

Build args:

| Arg | Default | Purpose |
|---|---|---|
| `BASE_IMAGE` | `ghcr.io/mostlygeek/llama-swap:unified-vulkan` | Base image (must be the root variant, not `-rootless`). Pin a dated tag or digest for reproducibility. |
| `WITH_ROCM` | `true` | `false` builds the Vulkan-only image (no HIP stages, no ROCm runtime); CI uses this for `:latest` |
| `ROCM_VERSION` | `7.2.4` | ROCm version for both the builder image and the runtime apt packages |
| `AMDGPU_TARGETS` | `gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201` | gfx architectures compiled into the HIP binaries (RDNA2/3/3.5/4). CDNA (`gfx908;gfx90a;gfx942`) is not included by default — add it if you run Instinct cards. Trim to just your GPU for a much faster build. |
| `LLAMA_COMMIT` | `master` | llama.cpp revision for both the Vulkan and the ROCm build (sha, tag, branch, or `refs/pull/N/head`). `""` = the base image's commit from `/versions.txt`. Defaults to master so the PRs in `LLAMA_PATCHES` apply and the newest backend work is in. |
| `WHISPER_COMMIT` / `SD_COMMIT` | *(empty)* | Override the project revision; empty means "same commit as the base image" |
| `GLSLC_SUITE` | `resolute` | Ubuntu release whose `glslc`/`libshaderc1` are used by the Vulkan builder (only those two packages; everything else stays 24.04) |
| `LLAMA_FA_ALL_QUANTS` | `ON` | ROCm llama.cpp: compile flash-attention kernels for all K/V cache quant combinations (without it only q8_0/q8_0 and q4_0/q4_0 stay on the GPU, see llama.cpp #27761). Set `OFF` for a faster build. |
| `WITH_ENGRAM` | `true` | Build [EngramHalo.cpp](https://github.com/Aristo94/EngramHalo.cpp) as `*-engram` binaries. Only takes effect together with `WITH_ROCM=true` (the fork is HIP-only), so `:latest` never contains it. `false` skips the stage. |
| `ENGRAM_REPO` / `ENGRAM_BRANCH` | Aristo94's repo, `strix-halo-qwen4exp` | Fork source. The branch rebases onto llama.cpp master and carries the Strix Halo patch series. |
| `ENGRAM_TARGETS` | `gfx1151` | gfx targets for the EngramHalo build. gfx1151 alone on purpose: the fork's kernels are tuned for and only validated on Strix Halo. |
| `MESA_PPA` | `ppa:kisak/kisak-mesa` | Newer Mesa/RADV for the final image; `""` keeps the base image's stock Mesa 25.2 |
| `QWEN_TEMPLATE_URL` | froggeric's `chat_template.jinja` | Source of the fixed Qwen chat template shipped at `/etc/llama-swap/templates/qwen-fixed.jinja` (see below) |
| `QWEN_SHARP_TEMPLATE_URL` | peculiar-ragdoll's `chat_template.jinja` | Source of the Sharp variant shipped at `/etc/llama-swap/templates/qwen-sharp.jinja` (see below) |
| `LLAMA_PATCHES` | `27952` | Space-separated upstream llama.cpp PR numbers merged on top of `LLAMA_COMMIT` (both backends), fetched over git as `refs/pull/N/head`. A PR that is closed on GitHub is skipped with a notice; one that no longer merges cleanly fails the build — never a silent no-op. See [Trying upstream PRs](#trying-upstream-prs). |

## Bundled chat templates

Two fixed Qwen 3.5/3.6/3.8 chat templates ship under `/etc/llama-swap/templates/`, both fetched at build time and refreshed on every rebuild (versions are in `/versions.txt` as `qwen_chat_template:` / `qwen_sharp_chat_template:`):

- `qwen-fixed.jinja` — [froggeric's Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates): sane reasoning-depth default, working `enable_thinking=false`, history `<think>` extraction, robust tool-call arguments — see its model card.
- `qwen-sharp.jinja` — [peculiar-ragdoll's Qwen-Sharp-Chat-Templates](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates): froggeric's template with a force-appended terseness system prompt (fewer filler/thinking tokens, same kwargs as above). Pass `{"terse": false}` in `chat_template_kwargs` to drop the appended prompt for a request.

Use one per model:

```yaml
cmd: >
  llama-server -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --port ${PORT}
    --jinja --chat-template-file /etc/llama-swap/templates/qwen-fixed.jinja
```

## EngramHalo.cpp for Strix Halo

The `:rocm` tag ships [EngramHalo.cpp](https://github.com/Aristo94/EngramHalo.cpp) (branch `strix-halo-qwen4exp`) as `llama-server-engram` / `llama-cli-engram` / `llama-bench-engram` — a llama.cpp fork tuned for **Qwen 3.8 Flash-Next on Strix Halo** (Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151): QSA sparse-gather attention, a HIP wide top-k kernel, a chunked GATED_DELTA_NET prefill kernel, an MTP draft head for speculative decoding, and the model's 26.8 GiB engram/PLE table SSD-backed via `--tensor-read-lazy` (~1 GiB resident). The fork's in-tree patches (`docs/strix-halo/`) are applied at build time; the [#25992](https://github.com/ggml-org/llama.cpp/issues/25992) iGPU host-buffer workaround is treated as required (the build fails if it stops applying), the per-buffer mmap loader patch is skipped once obsolete.

Measured on a 128 GB Strix Halo box (Qwen3.8-Flash-Next UD-Q4_K_XL, q8_0 KV, `--n-cpu-moe 24`, vs this image's stock `llama-server-rocm`): prompt processing 249 → 341–385 t/s at 11K, decode 11.3 → 15.5 t/s at 11K, and with the MTP sidecar 22–31 t/s on code at temp 0. Useful runtime env on gfx1151: `ROCBLAS_USE_HIPBLASLT=1`, `GGML_HIP_GDN_CHUNK=1`, `LLAMA_MMAP_DROP_BEHIND=1` (keeps the page cache warm behind a model-swapping proxy), and `LLAMA_QSA_GATHER=<n_kv threshold>` to tune when the sparse gather kicks in (default 16384). The MTP sidecar GGUF (draft weights, ~4 GB Q8_0) is at [EasiiX/Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF](https://huggingface.co/EasiiX/Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF); pass it with `-md` plus `--spec-type draft-mtp,ngram-mod --spec-draft-n-max 4 --spec-draft-p-min 0.75` (the fork defaults — they also won a local parameter sweep; tune speculative params at temperature 0, acceptance noise at higher temperatures misleads). MTP is validated up to a 164K slot — cap `--ctx-size 163840` when using `-md`, or drop MTP for the full 262144.

The binaries contain gfx1151 code only (`ENGRAM_TARGETS`) and exist only in the `:rocm` tag; on any other GPU, or for any other model, use `llama-server` / `llama-server-rocm`. The built fork commit is recorded in `/versions.txt` as `llama_engram_commit:`.

## Trying upstream PRs

`LLAMA_PATCHES="27952 26284"` fetches each PR's branch over git (`refs/pull/<N>/head`, blobless) and merges it into `LLAMA_COMMIT` before building (Vulkan and ROCm alike). The `.patch` HTTP endpoint is deliberately not used — GitHub rate-limits it (HTTP 429) from shared CI runner IPs. GitHub keeps `refs/pull/<N>/merge` only while a PR is open, so a closed PR (merged or rejected) is detected and skipped with a notice — drop it from the list; a PR that no longer merges cleanly fails the build (it drifted; re-check it). Merged PRs and the built llama.cpp commit are listed in `/versions.txt`.

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
| Instinct MI100–MI350 (gfx908/90a/942) | ROCm — add the targets to `AMDGPU_TARGETS` and build yourself (not in the published image) |
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

- The image is large (~13–15 GB): the ROCm runtime with rocBLAS/hipBLASLt kernel libraries accounts for most of it.
- `sd-server` (both backends) serves its web UI at `/` — upstream builds it without the frontend (no `pnpm` in its builder), so there `/` is a text placeholder and you need `--serve-html-path`. Here the pinned `sdcpp-webui` is built once (Node stage) and embedded; `/versions.txt` lists its commit as `sd_server_webui:`.
- The Vulkan engines are NOT the base image's binaries (see above); `audiocpp_server`, `llama-swap` and `vllm-wrapper` are. The base image's shared `libggml*.so`/`libwhisper*.so` in `/usr/local/lib` are removed (nothing uses them any more).
- The container runs as root (standard for ROCm images — device access works without any `--group-add`). Files created in `/models` will be root-owned on the host. If you run as a non-root user, pass the *numeric* host GIDs of your `video`/`render` groups (`--group-add $(getent group render | cut -d: -f3)`); the image has no `render` group, so adding it by name fails.
- Verified on a Ryzen AI MAX+ 395 / Radeon 8060S (Strix Halo, gfx1151): both `llama-server --list-devices` (Vulkan/RADV) and `llama-server-rocm --list-devices` (ROCm) see the GPU. A gfx1100-only build also works on it with `HSA_OVERRIDE_GFX_VERSION=11.0.0`.
- The base image is rebuilt daily by upstream; this image is rebuilt weekly by CI, so `latest` here can lag `unified-vulkan` by a few days. Trigger the workflow manually to sync sooner.

## Sources

- [llama-swap unified container docs](https://github.com/mostlygeek/llama-swap/tree/main/docker/unified)
- [llama.cpp ROCm Dockerfile](https://github.com/ggml-org/llama.cpp/blob/master/.devops/rocm.Dockerfile) (ROCm version + gfx target list followed here)
- [whisper.cpp ROCm build docs](https://github.com/ggml-org/whisper.cpp#amd-rocm-gpu-support)
- [EngramHalo.cpp Strix Halo docs](https://github.com/Aristo94/EngramHalo.cpp/blob/strix-halo-qwen4exp/docs/strix-halo/README.md) (fork background, benchmarks, MTP sidecar)
- [ROCm apt installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager-index.html)
