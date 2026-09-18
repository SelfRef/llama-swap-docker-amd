# Local patches for the llama.cpp build

Every `*.patch` here is applied (`git apply`) on top of `LLAMA_COMMIT` + the merged
`LLAMA_PATCHES` PRs, in glob order, to BOTH llama.cpp builds (Vulkan and ROCm) --
see `scripts/checkout-with-prs.sh`. A patch that reverse-applies is treated as
"already upstream" and skipped; one that no longer applies FAILS the build.

Use this only for an upstream PR that has drifted out of mergeability and is
worth carrying anyway: merge it by hand on top of the other merges and
`git diff <tree-without-it> HEAD > patches/<pr>-rebased.patch`. Generate the
diff against the tree that has ALL the other `LLAMA_PATCHES` merges in it --
these patches are applied last, so a diff taken against plain master fails on
any file a later PR also touches.

## 25666-rebased.patch

Upstream PR #25666 (Vulkan: do not enable MMVQ for speculative-decode steps on
AMD), rebased on 2026-09-18 onto master + the PRs in `LLAMA_PATCHES`
(verified by a local `--target llama-vulkan` build on master `ec9281505`).

The PR has not changed since 2026-08-26 -- master drifted under it, and GitHub
marks it CONFLICTING upstream too. The conflict is placement, not substance:
the PR's first hunk inserts its `MMVQ_MAX_DECODE_LIKE_N` constant directly
after the `// Device tuning` comment above `ggml_vk_should_use_mmvq()`, and
master deleted that comment, so git had no anchor for the insertion. The other
three hunks (the `is_k_quant` predicate, `n > MMVQ_MAX_DECODE_LIKE_N`, and the
`!is_k_quant || k >= 4096` default) still auto-merge untouched.

Resolving it means taking the PR side of a conflict whose other side is empty,
so this patch is byte-for-byte the PR's own four-hunk diff with the constant
re-anchored -- no hunk was reinterpreted, and the `// Device tuning` comment
comes back with it (verified: it appears exactly once in the merged tree).

This one is carried on a weaker justification than 28243: the +12.9 % TG and
the acceptance gain are **gfx1151 numbers from upstream, never measured on this
box**. Run `benchmark` against the stock-Vulkan MTP entries (`qwen38-bart`,
`-hau`, `-hui`) on a quiet GPU; if the gain is not there, delete this patch
rather than carrying it. Delete it and put `25666` back in `LLAMA_PATCHES` the
moment the author rebases.

## Retired

### 28243-rebased.patch (removed 2026-09-18)

Carried from 2026-09-15 because upstream PR #28243 (Qwen3.8-Flash-Next MTP
draft head + draft-only sidecar GGUFs) had stopped merging: master reshaped the
qwen4exp grouped-norm gammas to `{ n_embd, hc }` + `TENSOR_ALLOW_RESHAPE` while
the PR rewrote the same `create_tensor` lines to pass MTP-aware flags.

The author rebased the PR on 2026-09-18 and GitHub now reports it mergeable, so
`28243` went back into `LLAMA_PATCHES` (in its original slot, right after
`27952`) and the patch was deleted -- exactly the exit condition this file
described. Verified the same day by a local `--target llama-vulkan` build on
master `ec9281505`. The local patch had in fact already gone stale --
its `src/llama-arch.h` hunk no longer applied -- so this was not optional
tidying; keeping it would have failed the build a second time.
