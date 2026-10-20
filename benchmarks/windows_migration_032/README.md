# Windows source migration to vLLM 0.32.0

This branch rebases all 12 Windows commits from `edge-infer/windows-0.29.0`
(`afdcfb7c2`) onto the official `v0.32.0` tag. The old branch remains intact.
The worktree directory name still contains `029`; it does not describe HEAD.

Conflict adaptations retain the new upstream quantization paths, FP8 reduction,
Marlin generator, and optional CUDA components. Windows guards and CPU markers
remain conditional. The hash-routing kernel dispatch now uses a templated lambda
so CUDA/HIP preprocessor directives are outside macro arguments. Windows HTTP
workers select asyncio locally; other platforms continue to use uvloop.

Validation is recorded in [results.json](results.json) and adjacent hook logs:
Python syntax on changed sources, Ruff check/format, clang-format, SPDX,
root lazy imports, filenames, and the MoE macro-directive scanner passed.
Full pre-commit could not initialize markdownlint: repeated nodejs.org archive
transfers ended with `IncompleteRead`. Selected source hooks ran independently;
this is not a claim that the complete pre-commit suite passed.

**Native Windows build and model execution are pending.** The installed Windows
`omni029venv` and exported 0.29 source directories were not replaced. Rebased
source is not a qualified 0.32 Windows wheel. Next run the native CUDA/MSVC build,
then kernel checks, control-plane checks and same-checkpoint Spark/TTS regression.
Historical 0.29 measurements do not establish 0.32 performance.

The paired Omni branch is `codex/omni-windows-0.32.0`, based on `v0.32.0rc1`
(the available Omni prerelease, not a final Omni 0.32.0 tag).