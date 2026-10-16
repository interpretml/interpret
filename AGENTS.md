# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

InterpretML is a multi-language project. The repo holds several artifacts that build/ship together:

- `shared/libebm/` — C++11 native library `libebm` (the EBM training/scoring engine). Built into platform-specific `.dll` / `.so` / `.dylib` shared libraries placed in `bld/lib/`.
- `shared/vis/` — JavaScript/TypeScript visualization (`interpret-inline.js`) built with `npm` from `shared/vis/`.
- `python/interpret-core/` — main Python package (`interpret-core`). Contains the Python implementation of EBM, all explainers, the visual dashboard, and the ctypes bridge to `libebm`.
- `python/interpret/` — thin meta-package (`interpret`) that just depends on `interpret-core` plus extras.
- `python/powerlift/` — separate `powerlift` package for benchmarking (has its own pyproject and tests).
- `R/` — R package `interpret`, vendoring the same libebm sources.
- `docs/`, `scripts/`, `bld/` — documentation source, top-level Makefile/Docker orchestration, and build outputs.

Wheels do **not** build `libebm` from source — CI builds the shared library on each OS/arch and bundles the artifacts into the wheel. The C++ build is only triggered by an `sdist` install (see `python/interpret-core/setup.py` `BuildCommand` / `SDistCommand`) or by running `build.sh` / `build.bat` directly.

## Common commands

Working-directory cheat sheet (commands assume these cwds unless stated otherwise):
- Repo root — `./build.sh` / `./build.bat` and `clang-format-16` whole-tree (both reference `shared/libebm/` as a path argument; no `cd` into it).
- `python/` — `ruff check` and `ruff format` whole-tree (matches CI; keeps docs/JS/etc. out of scope; ruff still walks up to the repo-root `ruff.toml` for config).
- `python/interpret-core/` — Python install, `pytest`, `python -m mypy`.
- `shared/vis/` — npm commands for the visualization bundle.

### Native library (`libebm`)
- Linux/macOS: `./build.sh` (debug + release, default arch). Flags: `-release_64`, `-debug_64`, `-release_arm`, `-debug_arm`, `-release_32`, `-debug_32`, `-asm` (emit assembly), `-asan`, `-extra_debugging` (`-g`), `-conda` (use environment `CXX`/`CXXFLAGS`/`LDFLAGS`).
- Windows: `./build.bat` with the same `-release_64` / `-debug_64` / `-release_32` / `-debug_32` flags, plus `-analysis` for clang-tidy. Requires Visual Studio 2022 (`vcvars64.bat`).
- Output goes to `bld/lib/` (named `libebm[_<os>_<arch>][_debug].{dll,so,dylib}`).

### Python (run from `python/interpret-core/`)
- Install dev: `pip install -e ".[debug,notebook,plotly,lime,sensitivity,shap,linear,treeinterpreter,aplr,dash,skoperules,excel,testing]"`.
- Run all tests in parallel: `python -m pytest -vv -n auto`.
- Run a single test file: `python -m pytest -vv tests/glassbox/test_ebm.py`.
- Run a single test: `python -m pytest -vv tests/glassbox/test_ebm.py::test_name`.
- Pytest markers (defined in `pytest.ini`): `slow` and `selenium`. CI normally skips selenium; the Makefile passes `--runslow` (note: only honored if a corresponding conftest fixture is present). Filter with `-m "not slow"` or `-m "not selenium"` to skip.
- Coverage (matches CI): `python -m pytest -vv -n auto --cov=interpret --cov-report=xml`.

### JS visualization bundle (`shared/vis/`)
- Build (matches CI, Node 22): `cd shared/vis && npm install && npm run build-prod`. Output: `shared/vis/dist/interpret-inline.js`. The Python package re-bundles this file as package data, so Python users don't need npm unless they're editing the visualization.
- Dev variants: `npm run build-dev` (unminified), `npm start` (webpack-dev-server).

### Lint
- Whole tree: `cd python && ruff check` — scoped to `python/` (matches the same cwd convention as `ruff format`; ruff walks up to the repo-root `ruff.toml`). Single file: `ruff check path/to/file.py` from any cwd.
- Configuration is in `ruff.toml` (target `py310`, broad rule set including `B`, `I`, `PL`, `RUF`, `PERF`, `NPY`, `PD`, `UP`, etc.). `tests/**` waives `T20` (print). Project-wide Python floor is 3.10 — keep `ruff.toml`'s `target-version` and `pyproject.toml`'s `[tool.mypy] python_version` aligned if the floor is bumped.

### Type check (mypy)
- Config: `[tool.mypy]` in `python/interpret-core/pyproject.toml`.
- Run (matches CI): `cd python/interpret-core && python -m mypy`. CI runs with `continue-on-error: true`, so a failing mypy job won't break the build — but new strict modules should land clean.

### Format
After editing, format only the touched files. The whole-tree commands below are what CI runs.
- Python single file: `ruff format path/to/file.py` (works from any cwd; ruff walks up to `ruff.toml`/`pyproject.toml`). Check without rewriting: `ruff format --check path/to/file.py`. Whole tree (CI): `cd python && ruff format` — scoped to `python/` so docs and other non-package text aren't touched.
- C++ (always run in bash, even on Windows, so the versioned `clang-format-16` binary resolves and matches CI; `-style=file` picks up the repo's `.clang-format`):
  - Single file: `clang-format-16 -i -style=file path/to/file.cpp`.
  - Whole tree (CI): `find shared/libebm \( -iname "*.cpp" -o -iname "*.h" -o -iname "*.hpp" \) | xargs clang-format-16 -i -style=file`.

## C++ architecture (read before editing `shared/libebm/`)

Two long-form design docs in `shared/libebm/` are required reading before non-trivial C++ changes:
- `IMPORTANT.md` — the **zone** model and One Definition Rule (ODR) discipline.
- `CODING_STYLE.md` — formatting, naming, and the deliberate avoidance of exceptions/RAII for the core library.

Key consequences for editing:

1. **Zone model.** Source is partitioned into zones with strict rules about what may be shared between translation units that are compiled with **different** flags (notably the per-SIMD compute units):
   - `zone_main` — root C++ files in `shared/libebm/`. Free to use full C++ but must not leak class definitions out of the library; everything wrapped in the `EbmMain` namespace.
   - `zone_separate` — high-performance compute kernels under `shared/libebm/compute/` compiled multiple times with different flags (`cpu_ebm/`, `avx2_ebm/`, `avx512f_ebm/`, `cuda_ebm/`). These translation units **must** live inside their own namespace; do not share C++ class definitions across SIMD variants.
   - `zone_c_interface` — `shared/libebm/inc/` (the public C ABI; `extern "C"` only, POD types). This is also the boundary the Python/R bridges call into via ctypes (`python/interpret-core/interpret/utils/_native.py`).
   - `zone_cpp_interface` — POD/templated headers shared between zones. No nested includes; namespace wrapping must happen at include site.
   - `zone_shared` — utilities shared across zones; must live in a unique namespace.
   - `zone_safe` — pure-C files/headers (`unzoned/`) safely shareable everywhere.
   ODR violations across these boundaries are silent and dangerous; preserve the existing zone walls when adding code.

2. **SIMD/CUDA variants are added by registering objectives**, not by per-call dispatch. New objectives go in `shared/libebm/compute/objectives/` as a header (e.g. `ExampleRegressionObjective.hpp`) and are registered in `objective_registrations.hpp`; metric counterparts live in `compute/metrics/`. Each compute backend (`cpu_ebm`, `avx2_ebm`, `avx512f_ebm`, `cuda_ebm`) is a thin wrapper that compiles the same kernels with different flags. The `BRIDGE_AVX2_32` / `BRIDGE_AVX512F_32` defines (set by `build.sh` / `build.bat`) gate which SIMD bridges are linked.

3. **No exceptions, mostly no RAII** in the core library. The library uses `malloc`/`free` consistently (mixing `new`/`free` was the bug that motivated this), POD-with-struct-hack data layouts (e.g. `Bin`, `Tensor`, `TreeNode`), and explicit error codes across the C boundary. Exceptions are tolerated only when interfacing with STL or in user-extensible code (custom `Objective` classes). Keep this discipline when editing.

4. **C++ style highlights** (see `CODING_STYLE.md`): 3-space indent, max 120 cols, K&R braces required even for one-liners, constants on the LHS of `==`, only `<` / `<=` in comparisons, `auto` only for genuinely unwieldy types. Naming uses the `p`/`a`/`s`/`i`/`c`/`b` prefixes and `m_`/`s_`/`g_`/`k_` qualifiers throughout — match the existing style.

5. **Symbol visibility**: shared library is built `-fvisibility=hidden` and the public surface is gated by `shared/libebm/libebm_exports.txt` (Linux/macOS) and `libebm_exports.def` (Windows). New public C entry points must be added to those export lists.

## Python architecture (`python/interpret-core/interpret/`)

- `glassbox/` — interpretable models. `_ebm.py` is the sklearn-facing `EBM{Classifier,Regressor}` and `_ebm/` holds the implementation pieces; `_ebm_core/` houses `_boost.py`, `_bin.py`, `_tensor.py`, `_multiclass.py`, `_merge_ebms.py`, `_excel.py`, `_json.py` — the plumbing called from `_ebm.py`. Other glassbox models: `_decisiontree.py`, `_linear.py`, `_aplr.py`, `_skoperules.py`.
- `privacy/_dpebm.py` — Differentially Private EBM (`DPEBM{Classifier,Regressor}`).
- `blackbox/`, `greybox/`, `perf/`, `data/` — wrappers around third-party explainers (SHAP, LIME, Morris, PDP, etc.) using the `core/base.py` and `core/sklearn.py` mixins.
- `utils/_native.py` — ctypes bindings to `libebm`. `utils/_preprocessor.py`, `_clean_x.py`, `_unify_data.py`, `_unify_predict.py` handle pandas/numpy normalization. `_purify.py`, `_measure_interactions.py`, `_rank_interactions.py`, `_compressed_dataset.py`, `_shared_dataset.py` are the higher-level pieces called by EBM training.
- `visual/` and `provider/` — the explanation visualization stack; `from interpret import show` is wired through `visual/_interactive.py`. The browser bundle `interpret-inline.js` is shipped as package data.
- `ext/` and the `entry_points` in `setup.py` define the extension/plugin system (`interpret_ext_blackbox`, `interpret_ext_glassbox`, etc.).
- The `interpret/root/` directory is populated by `setup.py._copy_native_code_to_setup` for sdist builds — it is a snapshot, not source of truth. Edit `shared/libebm/` and `build.{sh,bat}` at the repo root.

## Generated / vendored files (do not hand-edit)

These trees are populated from `shared/libebm/` by build scripts. Hand-edits will be silently overwritten:
- `python/interpret-core/interpret/root/` — copied by `python/interpret-core/setup.py._copy_native_code_to_setup` during sdist builds.
- `R/src/libebm/` — copied by `R/build.R` (see `copy_code`) when packaging the R library; the R package vendors the same C++ sources rather than re-using the shared one.
- `shared/vis/dist/interpret-inline.js` — webpack output from `npm run build-prod`; edit sources under `shared/vis/src/` instead.

Source of truth for C++ is always `shared/libebm/`; for the visualization, `shared/vis/src/`.

## CI and platforms

`.github/workflows/ci.yml` is the source of truth for supported build matrices. CI:
1. Builds `libebm` on Linux x64 (manylinux2010 docker), Linux ARM (`ubuntu-22.04-arm`), macOS x64 + ARM (`macos-14`), Windows x64 (`windows-2022`). CUDA toolkit is installed only on Windows.
2. Builds the JS bundle with Node 22 and the R package on Ubuntu 22.04.
3. Builds `sdist` from `python/interpret-core` and `python/interpret`, then runs pytest against the sdist install across multiple Python versions on each OS.
4. Pytest invocation in CI is `python -m pytest -vv -n auto --cov=interpret --cov-report=xml` (no `--runslow`); slow tests are skipped in CI but available locally via the marker.

When adding a new platform/SIMD target, update both `build.sh` and `build.bat`, the `package_data` list in `python/interpret-core/setup.py`, and the matrix in `.github/workflows/ci.yml`.
