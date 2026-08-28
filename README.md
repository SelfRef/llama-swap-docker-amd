# llama-swap Docker image for AMD GPUs (ROCm + Vulkan)

[llama-swap](https://github.com/mostlygeek/llama-swap) publishes a `unified-vulkan` image with every inference engine built for Vulkan — which works on AMD, but leaves ROCm performance on the table. This repo extends that image into an AMD-focused one with **both GPU stacks**:

- **Vulkan** (Mesa RADV) — inherited from the base image: works on practically any AMD GPU, including RDNA1/2, iGPUs/APUs and anything ROCm doesn't cover.
- **ROCm 7.2** (HIP) — added here: the full ROCm userspace runtime (HIP, rocBLAS/hipBLAS + Tensile kernels, hipBLASLt, `rocminfo`) plus HIP rebuilds of the engines, typically faster than Vulkan on supported GPUs.

The ROCm binaries are compiled **from the exact same upstream commits** as the Vulkan ones in the base image (parsed from its `/versions.txt`), so each engine ships as a matched pair — pick the backend per model in your llama-swap config.

## What's inside

| Engine | Vulkan (from base image) | ROCm (built here) |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `llama-server`, `llama-cli`, `llama-tts`, `llama-bench` | `llama-server-rocm`, `llama-cli-rocm`, `llama-tts-rocm`, `llama-bench-rocm` |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | `whisper-server`, `whisper-cli` | `whisper-server-rocm`, `whisper-cli-rocm` |
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | `sd-server`, `sd-cli` | `sd-server-rocm`, `sd-cli-rocm` |
| [audio.cpp](https://github.com/0xShug0/audio.cpp) | `audiocpp_server`, `audiocpp_cli` | — (no HIP backend upstream) |
| llama-swap + [vllm-wrapper](https://github.com/mostlygeek/llama-swap/tree/main/cmd/vllm-wrapper) | `llama-swap`, `vllm-wrapper` (backend-independent) | |

`ik-llama-server` from the CUDA image is not included (upstream builds it CUDA-only). Exact versions of everything are recorded in `/versions.txt` inside the image.

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
| `ROCM_VERSION` | `7.2.1` | ROCm version for both the builder image and the runtime apt packages |
| `AMDGPU_TARGETS` | `gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201` | gfx architectures compiled into the HIP binaries. Trim to just your GPU for a much faster build. |
| `LLAMA_COMMIT` / `WHISPER_COMMIT` / `SD_COMMIT` | *(empty)* | Override the project revision; empty means "same commit as the base image's Vulkan build" |

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

# Versions baked into the image
docker run --rm --entrypoint cat ghcr.io/selfref/llama-swap-docker-amd:latest /versions.txt
```

## Notes

- The image is large (~15–20 GB): the ROCm runtime with rocBLAS/hipBLASLt kernel libraries for 11 architectures accounts for most of it.
- The container runs as root (standard for ROCm images — device access works without any `--group-add`). Files created in `/models` will be root-owned on the host. If you run as a non-root user, pass the *numeric* host GIDs of your `video`/`render` groups (`--group-add $(getent group render | cut -d: -f3)`); the image has no `render` group, so adding it by name fails.
- Verified on a Ryzen AI MAX+ 395 / Radeon 8060S (Strix Halo, gfx1151): both `llama-server --list-devices` (Vulkan/RADV) and `llama-server-rocm --list-devices` (ROCm) see the GPU. A gfx1100-only build also works on it with `HSA_OVERRIDE_GFX_VERSION=11.0.0`.
- The base image is rebuilt daily by upstream; this image is rebuilt weekly by CI, so `latest` here can lag `unified-vulkan` by a few days. Trigger the workflow manually to sync sooner.

## Sources

- [llama-swap unified container docs](https://github.com/mostlygeek/llama-swap/tree/main/docker/unified)
- [llama.cpp ROCm Dockerfile](https://github.com/ggml-org/llama.cpp/blob/master/.devops/rocm.Dockerfile) (ROCm version + gfx target list followed here)
- [whisper.cpp ROCm build docs](https://github.com/ggml-org/whisper.cpp#amd-rocm-gpu-support)
- [ROCm apt installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager-index.html)
