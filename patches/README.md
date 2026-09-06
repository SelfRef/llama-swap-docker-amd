# Local patches for the llama.cpp build

Every `*.patch` here is applied (`git apply`) on top of `LLAMA_COMMIT` + the merged
`LLAMA_PATCHES` PRs, in glob order, to BOTH llama.cpp builds (Vulkan and ROCm) --
see `scripts/checkout-with-prs.sh`. A patch that reverse-applies is treated as
"already upstream" and skipped; one that no longer applies FAILS the build.

Use this only for an upstream PR that has drifted out of mergeability and is
worth carrying anyway: merge it by hand on top of the other merges and
`git diff <tree-without-it> HEAD > patches/<pr>-rebased.patch`.

Empty since 2026-09-06: the rebased #27836 (qwen4exp MTP draft head) and #28097
(draft-only sidecar GGUFs) were replaced by upstream PR #28243, which carries
both features and merges cleanly. (Until 2026-09-06 these patches fed a separate
`llama-server-next` build; that build is now *the* `llama-server`.)
