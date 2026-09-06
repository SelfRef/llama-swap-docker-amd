# Vendored upstream files (drop-in compatibility)

This image used to be built `FROM ghcr.io/mostlygeek/llama-swap:unified-vulkan`
and inherited the runtime contract from it. It is now built from plain
`ubuntu:24.04`, so the pieces of upstream's
[`docker/unified`](https://github.com/mostlygeek/llama-swap/tree/main/docker/unified)
that define that contract are copied here verbatim and installed by the final
stage of the [Dockerfile](../Dockerfile):

| file | used as |
|---|---|
| `run.sh` | the container entrypoint: maps `LLAMA_SWAP_*` environment variables to llama-swap flags, defaults `-listen 0.0.0.0:8080 -watch-config -config /etc/llama-swap/config/config.yaml`; any container argument replaces all of that (the pre-`run.sh` behaviour) |
| `audiocpp-server.example.json` | `/etc/llama-swap/audiocpp-server.example.json` with `__BACKEND__` replaced by `vulkan`, as upstream does |
| `runtime.Dockerfile` | reference only, not built: the apt package set, paths, user and entrypoint of the final stage mirror it |

Do not edit these files; re-copy them from upstream. The `prepare` job of the
[build workflow](../.github/workflows/build.yml) fetches the current upstream
versions on every run and emits a warning with the diff when one has changed,
so a new entrypoint variable or runtime package upstream adds shows up in the
next run's annotations instead of silently drifting.

Vendored from mostlygeek/llama-swap `main` @ `7761aa13360ea379cb89366d07c2d08aa9f1ed10` (2026-09-06).
