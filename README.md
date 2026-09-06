# llama-swap Docker image for AMD GPUs (ROCm + Vulkan)

The [llama-swap](https://github.com/mostlygeek/llama-swap) `unified-vulkan` image rebuilt from source for AMD hardware: the same binary names, config location and port, so a config written for upstream's image works here, but with **both GPU stacks**, everything at its **current upstream revision**, and the open PRs worth having merged in:

- **llama-swap itself built from source** — current `main` plus open upstream PRs (`LLAMA_SWAP_PATCHES`, today [#1099](https://github.com/mostlygeek/llama-swap/pull/1099): live per-turn generation stats in the Playground Chat), web UI embedded, `vllm-wrapper` from the same tree. It is *the* `llama-swap` binary, not a side binary.
- **llama.cpp from current master plus open upstream PRs** (`LLAMA_PATCHES`: Vulkan fusions and int8 coopmat matmul, Qwen 3.5/3.6/3.8 fixes, Qwen3.8-Flash-Next MTP, adaptive MTP, exact checkpoint restore — see [Upstream PRs in the llama.cpp build](#upstream-prs-in-the-llamacpp-build)), the **same tree for the Vulkan and the ROCm build**. There is no un-patched llama.cpp in the image: `llama-server` is the patched build.
- **Vulkan** (Mesa RADV) — works on practically any AMD GPU, including RDNA1/2, iGPUs/APUs and anything ROCm doesn't cover. Built with a modern shader compiler (see [Why build the Vulkan binaries ourselves](#why-build-the-vulkan-binaries-ourselves)) and shipped with a **current Mesa/RADV** instead of Ubuntu 24.04's.
- **ROCm 7.14** (HIP) — the full ROCm userspace runtime (HIP, rocBLAS/hipBLAS, hipBLASLt, `rocminfo`) plus HIP builds of the engines, with flash-attention kernels for every KV-cache quant. ROCm comes from AMD's per-gfx `packages-multi-arch` repository (`ROCM_CHANNEL=multiarch`), so the image carries BLAS kernels only for the gfx targets it is built for instead of ~6 GB of all-arch Tensile blobs; the classic `repo.radeon.com` channel (tops out at ROCm 7.2.4, which still has the HIP-graphs bug fixed in 7.13) remains available as `ROCM_CHANNEL=classic`.
- **EngramHalo.cpp** (HIP, `:full` tag, gfx1151 only) — [Aristo94's llama.cpp fork](https://github.com/Aristo94/EngramHalo.cpp) tuned for Qwen 3.8 Flash-Next on Strix Halo, as a third `llama.cpp` install (`*-engram` binaries). See [EngramHalo.cpp for Strix Halo](#engramhalocpp-for-strix-halo).

Both llama.cpp backends are built with runtime CPU dispatch (`GGML_CPU_ALL_VARIANTS`), so one image gets AVX2 on Zen 3 and AVX-512/VNNI/BF16 on Zen 4/5 for CPU-offloaded layers. Every engine is built once per backend from one resolved commit, so each ships as a matched Vulkan/ROCm pair — pick the backend per model in your llama-swap config.

## What's inside

| Engine | Vulkan (all tags) | ROCm (`:full`/`:latest` only) |
|---|---|---|
| [llama-swap](https://github.com/mostlygeek/llama-swap) `main` + PRs (`LLAMA_SWAP_PATCHES`) | `llama-swap` (UI embedded), `vllm-wrapper` | (backend-independent) |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) master + PRs (`LLAMA_PATCHES`) | `llama-server`, `llama-cli`, `llama-tts`, `llama-bench` | `llama-server-rocm`, `llama-cli-rocm`, `llama-tts-rocm`, `llama-bench-rocm` |
| [EngramHalo.cpp](https://github.com/Aristo94/EngramHalo.cpp) (llama.cpp fork, Strix Halo/qwen4exp) | — (fork is ROCm/HIP-only) | `llama-server-engram`, `llama-cli-engram`, `llama-bench-engram` (gfx1151 only) |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) master | `whisper-server`, `whisper-cli` | `whisper-server-rocm`, `whisper-cli-rocm` |
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) master | `sd-server` (web UI embedded), `sd-cli` | `sd-server-rocm` (web UI embedded), `sd-cli-rocm` |
| [audio.cpp](https://github.com/0xShug0/audio.cpp) main | `audiocpp_server`, `audiocpp_cli`, `audiocpp_gguf` (GGUF converter, upstream ships none) | — (no HIP backend upstream) |
| `benchmark` (this repo, `scripts/benchmark`) | one CLI for server-level / `llama-bench` / standalone-variant benchmarks of the config's text entries, see [Benchmarking](#benchmarking) | (uses the `*-rocm` / `*-engram` binaries via `--variant`) |

llama.cpp lives in self-contained directories `/opt/llama-vulkan`, `/opt/llama-rocm` and `/opt/llama-engram` (binaries, `libllama`/`libggml*` and the per-CPU-level `libggml-cpu-*.so` variants, RPATH `$ORIGIN`) with symlinks in `/usr/local/bin`; whisper/sd/audio.cpp binaries are static. Exact versions of everything — every commit, every merged PR, the glslc used and the enabled build options — are recorded in `/versions.txt` inside the image.

## Runtime layout

Built from plain `ubuntu:24.04` (see [Why not build on the upstream image](#why-not-build-on-the-upstream-image)); nothing from upstream's `docker/unified` is vendored.

- Binaries in `/usr/local/bin`; config at `/etc/llama-swap/config/config.yaml` (mount the directory or the file); models in `/models` (the working directory); audio.cpp's spec catalog at `/usr/local/share/audiocpp/model_specs`; `/versions.txt`.
- Entrypoint is `llama-swap` itself with `-config /etc/llama-swap/config/config.yaml -listen 0.0.0.0:8080 -watch-config` as the default `CMD`. Any argument passed to the container replaces those defaults (`docker run <image> -version`, `docker run <image> -config /models/my.yaml -listen 0.0.0.0:8080`).
- Also present: `rocm-smi` (llama-swap's GPU monitor in its UI reads it; sysfs-based, works without the ROCm runtime), `ffmpeg` + libav\* (whisper-server input decoding), `curl`, `python3` + PyYAML (for `benchmark`), `uv`/`uvx`.
- Runs as root, like upstream's root variant. There is no `-rootless` tag.

## Why build the Vulkan binaries ourselves

Upstream builds the Vulkan engines on Ubuntu 24.04 with its stock `glslc` (shaderc 2023.8 / glslang 14). That compiler cannot compile the `GL_EXT_integer_dot_product` and `GL_EXT_bfloat16` shaders, and llama.cpp's CMake then *silently* drops those code paths: at startup the device line reads `int dot: 0 | bf16: 0` even though RADV advertises both extensions. The integer-dot path is llama.cpp's fast quantized path on GPUs **without** cooperative-matrix support — q8_1 MMQ for K-quants (upstream measured ~2x prompt processing on an RX 6800 XT), DP4A flash attention for q8_0/q4_0 KV caches, MMVQ decode. On coopmat GPUs (RDNA3 and newer) llama.cpp keeps its FP16 coopmat matmul for prompts, so on those the rebuild changes little by itself (RX 7900 XTX: prompt speed identical, decode +2%); it still matters for RDNA1/2, older cards and iGPUs, for any future upstream work that needs those extensions, and it removes a silent build-dependent difference between GPUs.

This image builds on the Ubuntu 24.04 ABI, but with `glslc`/`libshaderc1` taken from the Ubuntu 26.04 pocket (glslang 16; `GLSLC_SUITE`). The build fails if the feature tests do not pass, so the extensions cannot silently disappear. Verify at runtime: the `ggml_vulkan: 0 = ...` line must show `int dot: 1`.

The larger measured win on current GPUs is the driver: the final image takes `mesa-vulkan-drivers` from the kisak-mesa PPA (`MESA_PPA`, Mesa 26.1 vs Ubuntu's 25.2). RX 7900 XTX, identical binaries, Qwen3.8-27B Q4_K_XL: pp512 693 → 862 t/s (+24%), decode 37.7 → 37.9 t/s; the newer RADV also exposes `VK_VALVE_shader_mixed_float_dot_product` (`fp16: dot2`).

## Why not build on the upstream image

Until 2026-09-06 this image was `FROM ghcr.io/mostlygeek/llama-swap:unified-vulkan` and replaced the engines. Once llama-swap itself had to be built from source (its open PRs are UI changes, so they cannot be applied to the release binary), the base contributed one apt line, two audio.cpp binaries and three commit hashes — and cost real things: deleting the base's engine binaries in a child layer does not remove them from the image, so every pull carried ~650 MB of unreachable files (the base's llama.cpp, sd and ggml builds plus the stock Mesa shadowed by the PPA one); the daily base rebuild invalidated every final layer whether or not anything relevant changed; and the tags could lag upstream by days. Building from `ubuntu:24.04` removes all of that, and every project is now taken directly from its default branch.

## Get the image

Prebuilt by [GitHub Actions](.github/workflows/build.yml). Two tags from the same Dockerfile (`WITH_ROCM`):

```bash
docker pull ghcr.io/selfref/llama-swap-docker-amd:vulkan   # Vulkan only (~1.5 GB)
docker pull ghcr.io/selfref/llama-swap-docker-amd:full     # Vulkan + ROCm
docker pull ghcr.io/selfref/llama-swap-docker-amd:latest   # alias for :full
```

Both tags are rebuilt on every push and every 3 days by the scheduled run, each time from the then-current default branches of llama-swap, llama.cpp, whisper.cpp, stable-diffusion.cpp, audio.cpp and EngramHalo.cpp and the current heads of the merged PRs. The ROCm stages are fat multi-gfx HIP builds that take hours of runner time, so on a PR or an ad-hoc run they only happen if you ask for them (Actions → Build image → Run workflow → tick **rocm**). The `*-rocm` binaries, the `*-engram` binaries (see [EngramHalo.cpp for Strix Halo](#engramhalocpp-for-strix-halo)), the ROCm runtime and `rocminfo` exist only in `:full`/`:latest`.

Or build locally (expect a couple of hours for the fat HIP builds; `WITH_ROCM=false` for the Vulkan-only image in well under an hour):

```bash
docker buildx build -t llama-swap-amd .
# pin every branch/PR to today's commit first (what CI does) -- see "How versions are pinned"
docker buildx build $(scripts/resolve-refs.sh --docker) -t llama-swap-amd .
```

Build args:

| Arg | Default | Purpose |
|---|---|---|
| `LLAMA_SWAP_COMMIT` | `main` | llama-swap revision to build `llama-swap` and `vllm-wrapper` from (sha, tag such as `v255`, or branch). |
| `LLAMA_SWAP_PATCHES` | `1099` | Space-separated open upstream llama-swap PR numbers merged on top; same drift rules as `LLAMA_PATCHES`. The version string becomes e.g. `v255-2-g1a2b3c+pr1099`. |
| `WITH_ROCM` | `true` | `false` builds the Vulkan-only image (no HIP stages, no ROCm runtime); CI uses this for `:vulkan` |
| `ROCM_CHANNEL` | `multiarch` | ROCm source: `multiarch` = repo.amd.com per-gfx packages (current releases, small runtime — kernels only for `AMDGPU_TARGETS`); `classic` = repo.radeon.com apt + `rocm/dev-ubuntu-24.04` builder (max 7.2.4, all-arch kernels, HIP-graphs bug — pair it with `GGML_CUDA_DISABLE_GRAPHS=1` at runtime) |
| `ROCM_SERIES` | `7.14` | multiarch channel: release series in the package names (`amdrocm-runtime7.14`, ...); apt resolves the newest point release of the series |
| `ROCM_VERSION` | `7.2.4` | classic channel: builder image tag and apt repo path |
| `AMDGPU_TARGETS` | `gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201` | gfx architectures compiled into the HIP binaries (RDNA2/3/3.5/4). CDNA (`gfx908;gfx90a;gfx942`) is not included by default — add it if you run Instinct cards. Trim to just your GPU for a much faster build. |
| `LLAMA_COMMIT` | `master` | llama.cpp revision for both the Vulkan and the ROCm build (sha, tag, branch, or `refs/pull/N/head`). |
| `LLAMA_PATCHES` | 13 PRs, see below | Space-separated upstream llama.cpp PR numbers merged on top of `LLAMA_COMMIT`, for both backends, fetched over git as `refs/pull/N/head`. A PR that is closed on GitHub is skipped with a notice; one that no longer merges cleanly fails the build — never a silent no-op. `patches/*.patch` (local rebased patches, empty since 2026-09-06) apply on top. See [Upstream PRs in the llama.cpp build](#upstream-prs-in-the-llamacpp-build). |
| `WHISPER_COMMIT` / `SD_COMMIT` / `AUDIOCPP_COMMIT` | `master` / `master` / `main` | Revision of the other engines (sha, tag or branch). |
| `GLSLC_SUITE` | `resolute` | Ubuntu release whose `glslc`/`libshaderc1` are used by the Vulkan builder (only those two packages; everything else stays 24.04) |
| `LLAMA_FA_ALL_QUANTS` | `ON` | ROCm llama.cpp: compile flash-attention kernels for all K/V cache quant combinations (without it only q8_0/q8_0 and q4_0/q4_0 stay on the GPU, see llama.cpp #27761). Set `OFF` for a faster build. |
| `WITH_ENGRAM` | `true` | Build [EngramHalo.cpp](https://github.com/Aristo94/EngramHalo.cpp) as `*-engram` binaries. Only takes effect together with `WITH_ROCM=true` (the fork is HIP-only), so `:vulkan` never contains it. `false` skips the stage. |
| `ENGRAM_REPO` / `ENGRAM_BRANCH` / `ENGRAM_COMMIT` | Aristo94's repo, `strix-halo-qwen4exp`, *(empty)* | Fork source. The branch rebases onto llama.cpp master and carries the Strix Halo patch series; `ENGRAM_COMMIT` pins it to a sha (CI does), empty = branch tip. |
| `ENGRAM_TARGETS` | `gfx1151` | gfx targets for the EngramHalo build. gfx1151 alone on purpose: the fork's kernels are tuned for and only validated on Strix Halo. |
| `MESA_PPA` | `ppa:kisak/kisak-mesa` | Newer Mesa/RADV for the final image; `""` keeps Ubuntu 24.04's stock Mesa 25.2 |
| `QWEN_TEMPLATE_URL` | froggeric's `chat_template.jinja` | Source of the fixed Qwen chat template shipped at `/etc/llama-swap/templates/qwen-fixed.jinja` (see below) |
| `QWEN_SHARP_TEMPLATE_URL` | peculiar-ragdoll's `chat_template.jinja` | Source of the Sharp variant shipped at `/etc/llama-swap/templates/qwen-sharp.jinja` (see below) |
| `LLAMA_PATCHES_HEADS` / `LLAMA_SWAP_PATCHES_HEADS` / `BUILD_DATE` | *(empty)* | Cache keys only, set by CI from `scripts/resolve-refs.sh` (see below). |

### How versions are pinned

Nothing in the build context changes between two scheduled runs, and "master" in a build arg is the same string today and tomorrow, so on its own BuildKit would serve every cached stage forever. `scripts/resolve-refs.sh` therefore resolves every branch, tag and PR head to a commit *once* per run (a 4-second `git ls-remote` per repository) and CI passes the results as build args: a moved branch or an updated PR is a cache miss for exactly the stages that use it, while the ROCm stages (hours of compile) stay cached as long as nothing they build from moved. It is also what guarantees that the Vulkan and ROCm builds of an engine, compiled in different jobs hours apart, are the same revision. `BUILD_DATE` does the same for the final stage's apt layers (Ubuntu updates, the PPA Mesa, ROCm runtime point releases): they are redone on every run, minutes not hours.

A local `docker buildx build .` resolves per stage at build time, which is fine on one machine — but a *second* local build gets the cached stages, not new commits. Prefix it with `$(scripts/resolve-refs.sh --docker)` to build today's revisions, or pass a `*_COMMIT` build arg to pin one.

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

The `:full` tag ships [EngramHalo.cpp](https://github.com/Aristo94/EngramHalo.cpp) (branch `strix-halo-qwen4exp`) as `llama-server-engram` / `llama-cli-engram` / `llama-bench-engram` — a llama.cpp fork tuned for **Qwen 3.8 Flash-Next on Strix Halo** (Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151): QSA sparse-gather attention, a HIP wide top-k kernel, a chunked GATED_DELTA_NET prefill kernel, an MTP draft head for speculative decoding, and the model's 26.8 GiB engram/PLE table SSD-backed via `--lazy-mode on` (~1 GiB resident; the flag was `--tensor-read-lazy` before the fork rebased onto upstream #27794). The fork's in-tree patches (`docs/strix-halo/`) are applied at build time; the [#25992](https://github.com/ggml-org/llama.cpp/issues/25992) iGPU host-buffer workaround is treated as required (the build fails if it stops applying), the per-buffer mmap loader patch is skipped once obsolete.

Measured on a 128 GB Strix Halo box (Qwen3.8-Flash-Next UD-Q4_K_XL, q8_0 KV, `--n-cpu-moe 24`, vs this image's `llama-server-rocm` before the qwen4exp PRs were merged into it): prompt processing 249 → 341–385 t/s at 11K, decode 11.3 → 15.5 t/s at 11K, and with the MTP sidecar 22–31 t/s on code at temp 0. Useful runtime env on gfx1151: `ROCBLAS_USE_HIPBLASLT=1`, `GGML_HIP_GDN_CHUNK=1`, `LLAMA_MMAP_DROP_BEHIND=1` (keeps the page cache warm behind a model-swapping proxy), and `LLAMA_QSA_GATHER=<n_kv threshold>` to tune when the sparse gather kicks in (default 16384). The MTP sidecar GGUF (draft weights, ~4 GB Q8_0) is at [EasiiX/Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF](https://huggingface.co/EasiiX/Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF); pass it with `-md` plus `--spec-type draft-mtp,ngram-mod --spec-draft-n-max 4 --spec-draft-p-min 0.75` (the fork defaults — they also won a local parameter sweep; tune speculative params at temperature 0, acceptance noise at higher temperatures misleads). MTP is validated up to a 164K slot — cap `--ctx-size 163840` when using `-md`, or drop MTP for the full 262144.

The binaries contain gfx1151 code only (`ENGRAM_TARGETS`) and exist only in the `:full`/`:latest` tag; on any other GPU, or for any other model, use `llama-server` / `llama-server-rocm` (which carry upstream's own qwen4exp MTP/QSA PRs, see below). The built fork commit is recorded in `/versions.txt` as `llama_engram_commit:`.

## Upstream PRs in the llama.cpp build

`LLAMA_PATCHES` fetches each PR's branch over git (`refs/pull/<N>/head`, blobless) and merges it into `LLAMA_COMMIT` before building — the same tree for Vulkan and ROCm (`scripts/checkout-with-prs.sh`). The `.patch` HTTP endpoint is deliberately not used — GitHub rate-limits it (HTTP 429) from shared CI runner IPs. GitHub keeps `refs/pull/<N>/merge` only while a PR is open, so a closed PR (merged or rejected) is detected and skipped with a notice — drop it from the list; a PR that no longer merges cleanly fails the build (it drifted; re-check it). Merged PRs and the base llama.cpp commit are listed in `/versions.txt` (`llama_patches:`, `llama.cpp:`). Dry-run a changed set before committing it:

```bash
scripts/checkout-with-prs.sh https://github.com/ggml-org/llama.cpp.git master /tmp/llama.cpp 27952 28024 ...
```

Until 2026-09-06 there were two Vulkan builds — a "pure" `llama-server` with only #27952 and a `llama-server-next` with the full set. The full set had become the one actually serving models, and none of the PRs touches CUDA/HIP sources, so they were merged into one build that the ROCm side now shares. The current set (all merged cleanly against master `9e0e2205` on 2026-09-06; `#28422` topk_moe fusion was dropped for conflicting with `#28024`; `#28068` GDN norm max→rsqrt was retired the same day when it merged upstream):

| PR | area | what | status upstream |
|---|---|---|---|
| [#27952](https://github.com/ggml-org/llama.cpp/pull/27952) | Vulkan | int8 coopmat1 matmul (RDNA3/4) | open, measured (table below) |
| [#28024](https://github.com/ggml-org/llama.cpp/pull/28024) | Vulkan | rms_norm / rope fusions | approved by 0cc4m |
| [#27220](https://github.com/ggml-org/llama.cpp/pull/27220) | Vulkan | UNARY+MUL fusion (MoE shared-expert gating) | approved by jeffbolznv |
| [#28253](https://github.com/ggml-org/llama.cpp/pull/28253) | Vulkan | type-aligned quantized GET_ROWS | approved ×4 |
| [#28457](https://github.com/ggml-org/llama.cpp/pull/28457) | Vulkan | small-M matmul tiles for Qwen buckets | new |
| [#28243](https://github.com/ggml-org/llama.cpp/pull/28243) | models | Qwen3.8-Flash-Next MTP + draft-only sidecar (unsloth) | draft, ggerganov reviewing |
| [#28265](https://github.com/ggml-org/llama.cpp/pull/28265) | models | Qwen3.5-family delta-net out-proj 2D (+6-9% batched TG on Strix Halo) | open |
| [#28213](https://github.com/ggml-org/llama.cpp/pull/28213) | models | qwen4exp QSA gather decode | open |
| [#28136](https://github.com/ggml-org/llama.cpp/pull/28136) | loading | direct reads for the lazy PLE table | approved by pwilkin |
| [#28330](https://github.com/ggml-org/llama.cpp/pull/28330) | memory | no V cache for the qwen4exp indexer | approved by pwilkin |
| [#27210](https://github.com/ggml-org/llama.cpp/pull/27210) | spec | `--spec-type draft-mtp-adaptive` (opt-in) | open |
| [#28333](https://github.com/ggml-org/llama.cpp/pull/28333) | spec | zero MTP carrier at sequence start | open |
| [#25592](https://github.com/ggml-org/llama.cpp/pull/25592) | server | exact-position checkpoint restore (hybrid/recurrent) | open |

Measured on an RX 7900 XTX with this image (Mesa 26.1, llama-bench, q8_0/q4_0 KV, ub 512; dense = Qwen3.8-27B UD-Q4_K_XL, MoE = Qwen3.6-35B-A3B UD-Q4_K_M):

| PR | what | dense pp512 | MoE pp512 | decode | verdict |
|---|---|---|---|---|---|
| [#27952](https://github.com/ggml-org/llama.cpp/pull/27952) | Vulkan int8 coopmat1 matmul for RDNA3/4 | 858 → **898** (+4.6%) | 3127 → **3704** (+18.5%) | unchanged | **in the set** |
| [#25483](https://github.com/ggml-org/llama.cpp/pull/25483) | Vulkan: skip unneeded MoE work in coopmat1 mul_mm | +0.3% | +0.2% | unchanged | not worth a patch |
| [#26284](https://github.com/ggml-org/llama.cpp/pull/26284) + [#26301](https://github.com/ggml-org/llama.cpp/pull/26301) | HIP: RDNA3 MMQ tuning + dequant-float matvec | ROCm 977 → 1000 (+2%, within noise) | +2% | unchanged | not adopted (#26284 still carries RDNA4 changes its reviewer wants removed; re-test when merged) |
| [#22970](https://github.com/ggml-org/llama.cpp/pull/22970) | Vulkan K-quant A-matrix transpose | – | – | – | stale, conflicts with master |

The same mechanism serves llama-swap: `LLAMA_SWAP_PATCHES` (default [#1099](https://github.com/mostlygeek/llama-swap/pull/1099), live generation stats in the Playground Chat) is merged into `LLAMA_SWAP_COMMIT` and the result is compiled with the UI embedded, exactly as upstream's release build does (`make linux-amd64`). Retire PRs from either list as they merge — the build says so.

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

Edit [config/config.yaml](config/config.yaml) to define your models — it shows the pattern: the same engine as `*-rocm` (ROCm) or plain (Vulkan), chosen per model. The container watches the config and reloads on change. To use a config elsewhere, pass the flags as the container command (they replace the defaults, see [Runtime layout](#runtime-layout)): `... ghcr.io/selfref/llama-swap-docker-amd:latest -config /models/my.yaml -listen 0.0.0.0:8080 -watch-config`.

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

# Vulkan feature line -- must say "int dot: 1" (see "Why build the Vulkan binaries ourselves")
docker run --rm --device /dev/dri --entrypoint llama-bench \
  ghcr.io/selfref/llama-swap-docker-amd:latest -m /dev/null 2>&1 | grep "ggml_vulkan: 0"

# Versions baked into the image (every commit, merged PR and build option)
docker run --rm --entrypoint cat ghcr.io/selfref/llama-swap-docker-amd:latest /versions.txt

# llama-swap version (e.g. v255-2-g1a2b3c+pr1099: base release + merged PRs)
docker run --rm ghcr.io/selfref/llama-swap-docker-amd:latest -version
```

## Benchmarking

`benchmark` (in `/usr/local/bin`, source `scripts/benchmark`) measures the llama.cpp text entries of
the mounted llama-swap `config.yaml` and prints one table: model, the parameters that matter
(quant, context, KV types, speculative type / draft length, placement, batch sizes) and the numbers.

```sh
docker compose exec llama-swap benchmark                         # every listed text entry, all presets
docker compose exec llama-swap benchmark qwen38 qwen38-hh        # server-level, via llama-swap
docker compose exec llama-swap benchmark --prompt prose,prefill --prefill-tokens 65536 qwen38-fl
docker compose exec llama-swap benchmark --kernel --std --unload qwen38      # llama-bench, community-comparable line
docker compose exec llama-swap benchmark --standalone --variant rocm --unload qwen38   # this entry on the ROCm build
docker compose exec llama-swap benchmark --standalone --variant engram --unload qwen38-fl  # ... on EngramHalo (:full)
docker compose exec llama-swap benchmark --sampled qwen38:l      # production sampling + filters (not comparable)
docker compose exec llama-swap benchmark --list                  # parsed entries, no requests
```

| Mode | What runs | What it tells you |
|---|---|---|
| server (default) | greedy requests through `/upstream/<model>/` (real entry flags: MTP, mmproj, `--parallel`, KV types; llama-swap filters bypassed) | what users get |
| `--kernel` | `llama-bench[-<variant>]` on the entry's cached GGUF with the entry's KV/batch/placement flags, or `--std` for `-fa 1 -ctk q8_0 -ctv q4_0 -b 2048 -ub 512 -p 512 -n 128 -d 0,8192` | hardware / driver / build (no speculative decoding, no mmproj) |
| `--standalone --variant V` | spawns `llama-server-<V>` with the entry's exact cmd (host/port substituted), benches it like server mode, kills it | binary A/Bs without a temporary config entry |
| `--sampled` | llama-swap's `/v1/chat/completions` with the entry's filters (aliases like `:l` allowed) | production sampling; hash blank, not comparable with greedy |

Variants are `vulkan` (plain `llama-server`), `rocm` and `engram`. Presets (`--prompt`, default `all`): `prose` (free text, speculative worst case), `json` (structured
output, best case), `refactor` (copy-heavy code edit, ~2k-token module), `agent` (tool-calling
transcript, 6 tools, ~2.5k prompt tokens; column `call`), `tools` (single tool call, natural stop;
`call`/`finish`), `reasoning` (`enable_thinking` on; `think`), `prefill` (pure big-context prefill,
`--prefill-tokens`, default 32768 capped to the entry's context; `tok` is the real prompt length,
`ttft` the prompt time). `--format md` prints rows for a markdown log, `--format json` for diffing.

Notes: the tool holds the box while it runs -- a watcher thread polls llama-swap's `/running` and
immediately unloads any model that is not the one under test (other services' requests fail for
that moment; rows that saw an intruder are flagged `EVICTED:<n>`; `--no-guard` disables it and a
stranger then only gets `CONTAMINATED`). Concurrent requests to the *same* model cannot be blocked
and are detected through llama-server's `/metrics` counters (`SHARED`). `--kernel` and
`--standalone` refuse to start while models are resident unless `--unload` is given. `--fit on` entries map to `-fitt 1024` in kernel mode (llama-server's margin), `-ngl all`
to llama-bench's default. `gtt` growth above `--spill-threshold` (default 1.5 GiB) on a
VRAM-resident entry flags `SPILL`; on UMA GPUs (Strix Halo) the check is off. The `hash` is the
sha256 of the first measured output and is reproducible for the same request sequence across
loads and builds. Needs `LLAMA_SWAP_API_KEY` in the environment when llama-swap has `apiKeys`.
`-rocm`/`-engram` variants exist only in the `:full`/`:latest` tag.

## Notes

- The full image is large: the ROCm runtime with rocBLAS/hipBLASLt kernel libraries accounts for most of it. The Vulkan-only image is about 650 MB smaller than it was when it extended the upstream image (nothing unreachable is shipped any more).
- `sd-server` (both backends) serves its web UI at `/` — upstream builds it without the frontend (no `pnpm` in its builder), so there `/` is a text placeholder and you need `--serve-html-path`. Here the pinned `sdcpp-webui` is built once (Node stage) and embedded; `/versions.txt` lists its commit as `sd_server_webui:`.
- `audiocpp_gguf` is audio.cpp's GGUF converter, which upstream's image does not ship. It is built from the same tree as `audiocpp_server` (so its model-spec catalog matches the server that loads its output; `/versions.txt`: `audiocpp_commit:`) for the community models whose licence forbids redistributing converted weights — e.g. [audio8_asr](https://github.com/0xShug0/audio.cpp/blob/main/docs/community_models/audio8_asr.md) (Audio8-ASR-0.1B, CC-BY-NC-4.0) is absent from `audio-cpp/audio.cpp-gguf` and has to be converted from the HF checkpoint you download yourself:

  ```sh
  hf download Audio8/Audio8-ASR-0.1B --local-dir /models/Audio8-ASR-0.1B-hf
  audiocpp_gguf --input /models/Audio8-ASR-0.1B-hf/model.safetensors --root /models/Audio8-ASR-0.1B-hf \
      --family audio8_asr --type q8_0 --output /models/Audio8-ASR-0.1B-GGUF/audio8-asr-0.1b-q8_0.gguf
  ```

  The output is self-contained (tokenizer/config sidecars and the model spec are embedded); point the `path` of an `audio8_asr` entry in your audio.cpp server config at the `.gguf` file.
- The container runs as root (standard for ROCm images — device access works without any `--group-add`). Files created in `/models` will be root-owned on the host. If you run as a non-root user, pass the *numeric* host GIDs of your `video`/`render` groups (`--group-add $(getent group render | cut -d: -f3)`); the image has no `render` group, so adding it by name fails.
- Verified on a Ryzen AI MAX+ 395 / Radeon 8060S (Strix Halo, gfx1151): both `llama-server --list-devices` (Vulkan/RADV) and `llama-server-rocm --list-devices` (ROCm) see the GPU. A gfx1100-only build also works on it with `HSA_OVERRIDE_GFX_VERSION=11.0.0`.
- `llama-swap` is built from `main`, not from the latest release archive: releases are cut from `main` every few days and the merged PRs are written against it. Set `LLAMA_SWAP_COMMIT=v255` (any tag) to build a release instead.

## Sources

- [llama-swap unified container docs](https://github.com/mostlygeek/llama-swap/tree/main/docker/unified) (the image this one mirrors in layout)
- [llama.cpp ROCm Dockerfile](https://github.com/ggml-org/llama.cpp/blob/master/.devops/rocm.Dockerfile) (ROCm version + gfx target list followed here)
- [whisper.cpp ROCm build docs](https://github.com/ggml-org/whisper.cpp#amd-rocm-gpu-support)
- [EngramHalo.cpp Strix Halo docs](https://github.com/Aristo94/EngramHalo.cpp/blob/strix-halo-qwen4exp/docs/strix-halo/README.md) (fork background, benchmarks, MTP sidecar)
- [ROCm apt installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager-index.html)
