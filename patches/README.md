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

## 28243-rebased.patch

Upstream PR #28243 (Qwen3.8-Flash-Next MTP draft head + draft-only sidecar
GGUFs), rebased on 2026-09-15 onto master 38a5b42d + the seven other PRs in
`LLAMA_PATCHES`. It is the MTP draft head `qwen38-flash` runs on, so it is
carried rather than dropped.

The PR stopped merging when master reshaped the qwen4exp grouped-norm gammas:
`hc_head_norm`, `hc_attn_norm`, `hc_ffn_norm` and `ple_norm_*` now load as
`{ n_embd, hc }` with `TENSOR_ALLOW_RESHAPE`, because `build_hc_mix()` scales
the `[n_embd, hc, n_tokens]` stream directly instead of reshaping it to
`[hc_dim, n_tokens]` first. The PR rewrote the same `create_tensor` lines to
pass MTP-aware flags (`trunk_flags` / `flags`) so a draft-only GGUF may omit
the trunk. Three conflicting hunks in `src/models/qwen4exp.cpp`, nothing else
in the set; the resolution keeps master's shapes and the PR's flags
(`flags | TENSOR_ALLOW_RESHAPE`), and also gives the PR's own
`nextn.hc_head_norm` the `{ n_embd, hc }` shape, since the MTP graph feeds it
to the same `build_hc_mix()`.

Delete this patch and put `28243` back in `LLAMA_PATCHES` as soon as the author
rebases the PR (it is open, marked CONFLICTING against master upstream too).

Until 2026-09-06 this directory carried the rebased #27836 (qwen4exp MTP draft
head) and #28097 (draft-only sidecar GGUFs), which fed a separate
`llama-server-next` build; #28243 carries both features and that build is now
*the* `llama-server`.
