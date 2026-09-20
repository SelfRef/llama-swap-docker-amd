#!/bin/bash
# checkout-with-prs.sh -- fetch a git ref and merge open upstream PRs on top.
#
#   checkout-with-prs.sh <repo-url> <ref> <dest-dir> [<pr-number>...]
#
# <ref> may be a branch, a tag, a full sha or refs/pull/N/head. The fetch is
# blobless (--filter=blob:none): full history, so the PR merges below are real
# merges, while file contents are fetched lazily on checkout.
#
# PRs are pulled over git as refs/pull/N/head and MERGED -- never via the
# github.com/.../pull/N.patch endpoint, which is rate-limited (HTTP 429) from
# shared CI runner IPs and made builds fail. GitHub keeps refs/pull/N/merge only
# while a PR is open, so a missing merge ref means the PR was closed (merged or
# rejected): it is skipped with a notice -- drop it from the list. A PR whose
# head is already an ancestor of <ref> is skipped as well. A merge conflict is
# real drift and FAILS the build, so a stale list is never a silent no-op.
#
# Environment:
#   LOCAL_PATCHES=<dir>  apply <dir>/*.patch (git apply) after the merges, in
#                        glob order. A patch that reverse-applies is treated as
#                        already upstream and skipped; one that fits neither way
#                        fails the build (rebase it against the merged tree).
#   FETCH_TAGS=1         also fetch the repository's tags (for `git describe`).
#
# Leaves three files in <dest-dir> for the caller's build-info:
#   .base-commit    sha <ref> resolved to
#   .merged-prs     space-separated PR numbers that were actually merged
#   .local-patches  space-separated names of the local patches applied
set -euo pipefail

[ $# -ge 3 ] || { echo "usage: $0 <repo-url> <ref> <dest-dir> [pr...]" >&2; exit 2; }
REPO=$1 REF=$2 DEST=$3
shift 3

echo "=== Fetching ${REPO} @ ${REF} ==="
mkdir -p "$DEST" && cd "$DEST"
git init -q
git remote add origin "$REPO"
git fetch --filter=blob:none origin "$REF"
git checkout -q FETCH_HEAD
BASE=$(git rev-parse HEAD)
echo "$BASE" > .base-commit
echo "$(basename "$REPO" .git) at ${BASE}"
if [ "${FETCH_TAGS:-0}" = 1 ]; then
    git fetch -q --filter=blob:none --tags origin
fi

MERGED=""
for pr in "$@"; do
    echo "=== Merging upstream PR #${pr} ==="
    if ! git ls-remote --exit-code origin "refs/pull/${pr}/merge" >/dev/null 2>&1; then
        echo "PR #${pr} is closed on GitHub (merged or rejected) -- skipping; remove it from the patch list"
        continue
    fi
    git fetch --filter=blob:none origin "refs/pull/${pr}/head"
    if git merge-base --is-ancestor FETCH_HEAD HEAD; then
        echo "PR #${pr} is already contained in ${REF} -- skipping"
        continue
    fi
    git -c user.name=llama-swap-rdna -c user.email=build@localhost \
        merge --no-edit --no-ff -m "merge upstream PR #${pr}" FETCH_HEAD \
        || { echo "FATAL: PR #${pr} does not merge cleanly into ${REF} -- it has drifted, re-check it" >&2; exit 1; }
    MERGED="${MERGED} ${pr}"
done
echo "${MERGED# }" > .merged-prs

APPLIED=""
if [ -n "${LOCAL_PATCHES:-}" ]; then
    for p in "${LOCAL_PATCHES}"/*.patch; do
        [ -e "$p" ] || continue
        name=$(basename "$p")
        if git apply --check "$p" 2>/dev/null; then
            git apply "$p"
            echo "applied local patch: ${name}"
            APPLIED="${APPLIED} ${name}"
        elif git apply --reverse --check "$p" 2>/dev/null; then
            echo "local patch already upstream: ${name}"
        else
            echo "FATAL: local patch ${name} no longer applies -- rebase it against the merged tree" >&2
            exit 1
        fi
    done
fi
echo "${APPLIED# }" > .local-patches

echo "=== ${DEST}: base ${BASE}, merged PRs:${MERGED:- none}, local patches:${APPLIED:- none}, tree $(git rev-parse HEAD) ==="
