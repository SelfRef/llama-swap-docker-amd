#!/bin/bash
# resolve-refs.sh -- pin every moving ref the Dockerfile builds from.
#
# Prints the resolved values as docker build args. Why this exists: the image
# is built from plain ubuntu:24.04, so between two scheduled CI runs nothing in
# the build context changes and BuildKit would keep serving every cached stage
# forever -- "master" in an ARG is the same string today and tomorrow, and the
# registry cache does not know the branch moved. Resolving branches, tags and
# PR heads to commits HERE and passing them as build args turns a moved branch
# or an updated PR into a cache miss for exactly the stages that build from it
# (the ROCm stages, hours of compile, stay cached while nothing they use moved).
# BUILD_DATE does the same for the final stage's apt layers (Ubuntu updates,
# Mesa from the PPA, ROCm runtime point releases): it is declared right before
# them, so they are redone on every run -- minutes, not hours.
#
#   scripts/resolve-refs.sh            KEY=VALUE per line (CI: >> $GITHUB_OUTPUT)
#   scripts/resolve-refs.sh --docker   --build-arg=KEY=VALUE per line, for a local
#                                      build that picks up today's commits:
#                                        docker buildx build $(scripts/resolve-refs.sh --docker) -t llama-swap-amd .
#
# Every input defaults to the Dockerfile's ARG default and can be overridden
# from the environment: LLAMA_COMMIT LLAMA_PATCHES WHISPER_COMMIT SD_COMMIT
# AUDIOCPP_COMMIT LLAMA_SWAP_COMMIT LLAMA_SWAP_PATCHES ENGRAM_REPO ENGRAM_BRANCH
# FPX_REPO FPX_BRANCH.
# A ref may be a branch, a tag, a full sha or refs/pull/N/head.
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
DOCKERFILE="$HERE/Dockerfile"
MODE=${1:-plain}

LLAMA_REPO=https://github.com/ggml-org/llama.cpp.git
WHISPER_REPO=https://github.com/ggml-org/whisper.cpp.git
SD_REPO=https://github.com/leejet/stable-diffusion.cpp.git
AUDIOCPP_REPO=https://github.com/0xShug0/audio.cpp.git
LLAMA_SWAP_REPO=https://github.com/mostlygeek/llama-swap.git

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

# Dockerfile ARG default, quotes stripped.
dflt() { sed -n "s/^ARG $1=\"\{0,1\}\([^\"]*\)\"\{0,1\}\$/\1/p" "$DOCKERFILE" | head -1; }
# Environment override, else the Dockerfile default.
val() { local v="${!1:-}"; if [ -n "$v" ]; then echo "$v"; else dflt "$1"; fi; }

# One `git ls-remote` per repository, reused for the ref and its PR heads.
refs() {  # <repo> -> file with all remote refs
    local f="$TMP/$(echo "$1" | tr -c 'A-Za-z0-9' _)"
    [ -s "$f" ] || git ls-remote "$1" > "$f" 2>/dev/null || { echo "ERROR: git ls-remote $1 failed" >&2; exit 1; }
    echo "$f"
}
resolve() {  # <repo> <ref> -> full commit sha
    local repo=$1 ref=$2 f sha
    if [[ "$ref" =~ ^[0-9a-f]{40}$ ]]; then echo "$ref"; return; fi
    f=$(refs "$repo")
    # annotated tag -> peeled commit; else lightweight tag, branch, or the ref as given
    sha=$(awk -v r="$ref" '
        $2=="refs/tags/"r"^{}" {p=$1} $2=="refs/tags/"r {t=$1}
        $2=="refs/heads/"r {h=$1} $2==r {x=$1}
        END {print (p!="")?p:(t!="")?t:(h!="")?h:x}' "$f")
    [ -n "$sha" ] || { echo "ERROR: cannot resolve '$ref' in $repo" >&2; exit 1; }
    echo "$sha"
}
pr_heads() {  # <repo> <pr list> -> "N:sha" per open PR, "N:closed" otherwise (comma-separated)
    local repo=$1 list=$2 f out="" pr h
    [ -n "$list" ] || return 0
    f=$(refs "$repo")
    for pr in $list; do
        h=$(awk -v pr="$pr" '$2=="refs/pull/"pr"/head"{h=$1} $2=="refs/pull/"pr"/merge"{m=1} END{print (m&&h!="")?h:"closed"}' "$f")
        out="${out:+$out,}${pr}:${h}"
    done
    echo "$out"
}

emit() {  # <key> <value>
    case "$MODE" in
        --docker) echo "--build-arg=$1=$2" ;;
        *)        echo "$1=$2" ;;
    esac
}

LLAMA_PATCHES=$(val LLAMA_PATCHES)
LLAMA_SWAP_PATCHES=$(val LLAMA_SWAP_PATCHES)

emit LLAMA_COMMIT            "$(resolve "$LLAMA_REPO" "$(val LLAMA_COMMIT)")"
emit LLAMA_PATCHES_HEADS     "$(pr_heads "$LLAMA_REPO" "$LLAMA_PATCHES")"
emit WHISPER_COMMIT          "$(resolve "$WHISPER_REPO" "$(val WHISPER_COMMIT)")"
emit SD_COMMIT               "$(resolve "$SD_REPO" "$(val SD_COMMIT)")"
emit AUDIOCPP_COMMIT         "$(resolve "$AUDIOCPP_REPO" "$(val AUDIOCPP_COMMIT)")"
emit LLAMA_SWAP_COMMIT       "$(resolve "$LLAMA_SWAP_REPO" "$(val LLAMA_SWAP_COMMIT)")"
emit LLAMA_SWAP_PATCHES_HEADS "$(pr_heads "$LLAMA_SWAP_REPO" "$LLAMA_SWAP_PATCHES")"
emit ENGRAM_COMMIT           "$(resolve "$(val ENGRAM_REPO)" "$(val ENGRAM_BRANCH)")"
emit FPX_COMMIT              "$(resolve "$(val FPX_REPO)" "$(val FPX_BRANCH)")"
emit BUILD_DATE              "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# The PR lists themselves contain spaces; in --docker mode (unquoted $(...) in a
# shell) they would split into separate words, so they are only printed in
# plain mode. A local build that overrides a list passes it by hand:
#   docker buildx build $(scripts/resolve-refs.sh --docker) --build-arg 'LLAMA_PATCHES=27952 28024' .
if [ "$MODE" != "--docker" ]; then
    emit LLAMA_PATCHES "$LLAMA_PATCHES"
    emit LLAMA_SWAP_PATCHES "$LLAMA_SWAP_PATCHES"
fi
