# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Working safely

- Preserve unrelated edits, including work from other agents. Before editing or committing, inspect the worktree and index; stage only your own paths or hunks. Never reset, overwrite, or commit another agent's changes.
- Start with the smallest tests covering the change. Format only touched files; use whole-tree checks and platform builds when the scope warrants them.
- Run applicable format, lint, and type checks locally. CI tolerates failures in the formatting, mypy, docs-build, and R-check steps, so a green workflow does not prove those checks passed.

## Generated / vendored files (do not hand-edit)

| Generated output | Source of truth / regeneration |
| --- | --- |
| `python/interpret-core/interpret/root/` | Packaging tree, including native sources copied from `shared/libebm/` by `setup.py._copy_native_code_to_setup` and bundled build artifacts. Created by local `sdist` / `pip install .` builds (editable installs skip the custom build command). It is gitignored, so a stale copy never shows in `git status`; delete it to make Python use the repository `bld/lib/` and `shared/vis/dist/` again. |
| `R/src/libebm/` | Native sources copied from `shared/libebm/` by `R/build.R` (`copy_code`). |
| `shared/vis/dist/interpret-inline.js` | Webpack output; edit `shared/vis/src/` and run `npm run build-prod`. |
| `docs/requirements.txt` | Compiled from `docs/requirements.in`; regenerate from `docs/` with `pip-compile --strip-extras requirements.in`, using the Python version recorded in the lockfile header. |

## Repository layout

- `shared/libebm/` — C++11 EBM training/scoring engine; shared libraries are built into `bld/lib/`.
- `shared/vis/` — JavaScript/TypeScript visualization, bundled for Python users.
- `python/interpret-core/` — models, explainers, dashboard, and ctypes bridge to `libebm`.
- `python/interpret/` — thin meta-package depending on `interpret-core` plus extras.
- `python/powerlift/` — separate benchmarking package with its own configuration and tests.
- `R/` — R package, vendoring the native library.
- `docs/` — jupyter-book documentation; notebooks live under `docs/interpret/`.
- `scripts/` — Docker/release orchestration. `scripts/Makefile` is legacy and out of date (for example it passes `--runslow`); prefer the commands below.
- `bld/` — build outputs.

## Common commands

Paths in the table are relative to the indicated working directory.

| Task | Working directory | Command |
| --- | --- | --- |
| Build native library, Linux/macOS x64 | Repo root | `sh ./build.sh -release_64` |
| Build native library, Windows x64 | Repo root | `.\build.bat -release_64` |
| Build and run native tests, Linux/macOS x64 | Repo root | `sh ./shared/libebm/tests/libebm_test.sh -release_64` |
| Build and run native tests, Windows x64 | Repo root | `.\shared\libebm\tests\libebm_test.bat -release_64` |
| Editable install with test dependencies | `python/interpret-core/` | `python -m pip install -e ".[testing]"` |
| Test EBM | `python/interpret-core/` | `python -m pytest -vv tests/glassbox/ebm/test_ebm.py` |
| Test one case | `python/interpret-core/` | `python -m pytest -vv tests/glassbox/ebm/test_ebm.py::test_name` (replace `test_name`) |
| Parallel tests, excluding slow/browser tests | `python/interpret-core/` | `python -m pytest -vv -n auto -m "not slow and not selenium"` |
| Full tests with coverage (CI invocation) | `python/interpret-core/` | `python -m pytest -vv -n auto --cov=interpret --cov-report=xml` |
| Check / format touched Python files | Any | `ruff check <files>` / `ruff format <files>` |
| Whole-tree Python lint / format check | `python/` | `ruff check` / `ruff format --check` |
| Python type check | `python/interpret-core/` | `python -m mypy` |
| Test powerlift (CI invocation) | `python/powerlift/` | `python -m pytest -vv -n auto --cov=powerlift --cov-report=xml` |
| Format touched C++ files | Repo root, bash (also on Windows) | `clang-format-16 -i -style=file <files>` |
| Whole-tree C++ format (CI invocation) | Repo root, bash (also on Windows) | `find shared/libebm \( -iname "*.cpp" -o -iname "*.h" -o -iname "*.hpp" \) \| xargs clang-format-16 -i -style=file` |
| Install JS dependencies / build production bundle | `shared/vis/` | `npm install` / `npm run build-prod` |
| JS development build / server | `shared/vis/` | `npm run build-dev` / `npm start` |
| Build and check R package (Linux; tarball in `bld/R/`, check log in `bld/tmp/R/interpret.Rcheck/`) | `R/` | `Rscript build.R` |
| Check R package (CI invocation) | `bld/R/` | `R CMD check --as-cran interpret_<version>.tar.gz` |
| Build docs | `docs/` | `sh ./build.sh` (`.\build.bat` on Windows) after `python -m pip install -r requirements.txt` and installing the package/extras described below |

- Libraries are written to `bld/lib/`. With no flags, `build.sh` builds only a release `libebm.{so,dylib}` for the host, while `build.bat` builds debug and release for x64 and x86. Explicit configuration flags produce `libebm_<os>_<arch>[_debug].{dll,so,dylib}`.
- Native test runners build the selected library and test executable, then run the tests. To reuse an already-built library, add the matching `-existing_*` flag (e.g. `-release_64 -existing_release_64`); the Windows runner still rebuilds the library after testing. The shell runner supports macOS ARM (`-release_arm` / `-debug_arm`), `-asan`, and `-valgrind`; see the scripts for other targets.
- On Windows, use a Visual Studio developer shell with `MSBuild.exe` available. Keep any `vcvars` setup and build/test invocation in the same process. `.\build.bat -release_64 -analysis` runs clang-tidy through MSBuild for the release x64 configuration. With Visual Studio 2026, set `PreferredToolArchitecture=x64` in the environment before running analysis (`$env:PreferredToolArchitecture = "x64"` in PowerShell), as CI does; the default x86-hosted clang-tidy is no longer shipped.
- Python's `testing` extra is a starting point, not the complete optional-explainer/UI environment. Add extras for the code under test; `setup.py` defines them and `.github/workflows/ci.yml` records the full CI install.
- `pytest.ini` registers `slow` and `selenium`; markers do not skip tests automatically. CI's invocation above has no marker exclusion. `interpret-core` has no `--runslow` hook.
- `python/powerlift/` has its own extras; `.github/workflows/ci.yml` records the set CI installs before testing it.
- Ruff configuration is in `ruff.toml`; mypy configuration is in `python/interpret-core/pyproject.toml`. The mypy baseline is deliberately permissive; strict checking is opted into per module through `[[tool.mypy.overrides]]`, so add modules there rather than loosening the baseline. Keep Ruff's target and mypy's Python version aligned with the supported Python floor. CI runs `ruff format --check` only, never `ruff check`, so lint findings surface only locally.
- `.pre-commit-config.yaml` runs ruff format plus a `disallow-caps` pygrep hook that rejects miscapitalized tool names in any file, including docs and comments. Write `NumPy`, `GitHub`, `pytest`, `CMake`, `ccache`, and `pybind` as their projects spell them.
- The docs build sets `INTERPRET_DOCS_MODE=docs` and runs jupyter-book with warnings as errors. `docs/requirements.txt` installs documentation tooling, not `interpret-core` or its optional explainer dependencies; install those separately using the docs CI step as a reference. Notebooks without saved outputs execute during the build, so the explainer extras they import must be installed. Inspect the underlying Sphinx / myst-nb log for warnings, errors, and tracebacks rather than relying on CI's summary warning.

## Native binaries, JS bundle, and Python testing

Wheels bundle prebuilt `libebm` binaries; they do not normally compile the library. Source-distribution installs build the copied native sources through `setup.py`'s `BuildCommand` / `SDistCommand`. For checkout development, build the library explicitly.

Before testing C++ changes through Python, verify the binary selected by `interpret/utils/_native.py`. If `interpret/root/` exists, the loader only searches `interpret/root/bld/lib/` and never consults the repository `bld/lib/`, even when `root/` contains no library. In each location it tries `libebm.<ext>` before the platform-specific name. Rebuilding the repository library does not guarantee Python will load it.

From `python/interpret-core/`, inspect the selected release library with `python -c "from interpret.utils._native import Native; print(Native._get_ebm_lib_path())"` (pass `debug=True` for the debug library). Ensure it is the artifact you rebuilt; do not hand-edit generated source copies to work around a stale binary. Use a fresh Python process after rebuilding: the native singleton retains the library already loaded in that process.

The default local visualization bundle follows the same rule in `interpret/visual/_inline.py`: with `interpret/root/` present it loads `root/bld/lib/interpret-inline.js`, otherwise it falls back to `shared/vis/dist/interpret-inline.js`. After editing `shared/vis/src/`, run `npm run build-prod`. If Python uses the packaged location, refresh `interpret-inline.js` and its `.LICENSE.txt` companion there from `shared/vis/dist/` (`setup.py`'s `build_vis_if_needed` leaves an existing bundle untouched), or remove `interpret/root/` as described above. Restart the notebook kernel and rerun the visualization so its initialization JavaScript is injected again.

## C++ architecture (read before editing `shared/libebm/`)

Two long-form design docs in `shared/libebm/` are required reading before non-trivial C++ changes:
- `IMPORTANT.md` — the **zone** model and One Definition Rule (ODR) discipline.
- `CODING_STYLE.md` — formatting, naming, and the deliberate avoidance of exceptions/RAII for the core library.

Key consequences for editing:

1. **Zone model.** Source is partitioned into zones with strict rules about what may be shared between translation units that are compiled with **different** flags (notably the per-SIMD compute units):
   - `zone_main` — root C++ files in `shared/libebm/`. Free to use full C++ but must not leak class definitions out of the library; everything wrapped in the `EbmMain` namespace.
   - `zone_separate` — high-performance compute kernels under `shared/libebm/compute/` compiled multiple times with different flags (`cpu_ebm/`, `avx2_ebm/`, `avx512f_ebm/`, `cuda_ebm/`). These translation units **must** live inside their own namespace; do not share C++ class definitions across SIMD variants.
   - `zone_c_interface` — `shared/libebm/inc/` (the public C ABI; `extern "C"` only, POD types). Python calls this boundary via ctypes (`python/interpret-core/interpret/utils/_native.py`); R calls it through `R/src/interpret_R.cpp`.
   - `zone_cpp_interface` — POD/templated headers shared between zones. No nested includes; namespace wrapping must happen at include site.
   - `zone_shared` — utilities shared across zones; must live in a unique namespace.
   - `zone_safe` — pure-C files/headers (`unzoned/`) safely shareable everywhere.
   ODR violations across these boundaries are silent and dangerous; preserve the existing zone walls when adding code.

2. **Objectives and backends are separate extension points.** New objectives go in `shared/libebm/compute/objectives/` and are registered in `objective_registrations.hpp`; metric counterparts live in `compute/metrics/`. Compute backends (`cpu_ebm`, `avx2_ebm`, `avx512f_ebm`, `cuda_ebm`) compile shared kernels with backend-specific flags and namespaces. Adding a backend requires its build/bridge integration, not just an objective registration. Preserve the zone boundaries and existing dispatch mechanism; `BRIDGE_AVX2_32` / `BRIDGE_AVX512F_32` gate the corresponding SIMD bridges.

3. **No exceptions, mostly no RAII** in the core library. The library uses `malloc`/`free` consistently (mixing `new`/`free` was the bug that motivated this), POD-with-struct-hack data layouts (e.g. `Bin`, `Tensor`, `TreeNode`), and explicit error codes across the C boundary. Exceptions are tolerated only when interfacing with STL or in user-extensible code (custom `Objective` classes). Keep this discipline when editing.

4. **C++ style highlights** (see `CODING_STYLE.md`): 3-space indent, max 120 cols, K&R braces required even for one-liners, constants on the LHS of `==`, only `<` / `<=` in comparisons, `auto` only for genuinely unwieldy types. Naming uses the `p`/`a`/`s`/`i`/`c`/`b` prefixes and `m_`/`s_`/`g_`/`k_` qualifiers throughout — match the existing style.

5. **Symbol visibility**: shared library is built `-fvisibility=hidden` and the public surface is gated by `shared/libebm/libebm_exports.txt` (Linux/macOS) and `libebm_exports.def` (Windows). New public C entry points must be added to those export lists.

6. **Source-file lists**: when adding C++ files, update the relevant Windows `.vcxproj` source lists and R `OBJECTS` lists as applicable; `build.sh` discovers `.cpp` files within its selected source directories, so a successful shell build does not prove the other builds include them. Keep `R/src/Makevars` and `R/src/Makevars.interpret` identical except for `NATIVEDIR`, which points to checkout sources in the former and vendored sources in the latter.

## Python architecture (`python/interpret-core/interpret/`)

- `glassbox/_ebm.py` — sklearn-facing `ExplainableBoostingClassifier` / `ExplainableBoostingRegressor`; `glassbox/_ebm_core/` contains training, binning, tensor operations, preprocessing (`_preprocessor.py`), interaction measurement/ranking, native dataset preparation, merging, callbacks, and export helpers. Other glassbox models (`_decisiontree.py`, `_linear.py`, `_aplr.py`, `_skoperules.py`) sit beside `_ebm.py`.
- `privacy/_dpebm.py` — differentially private EBM estimators.
- `blackbox/`, `greybox/`, `perf/`, `data/` — explainers and evaluation/data utilities; shared interfaces live in `core/base.py` and `core/_sklearn.py`.
- `utils/_native.py` — ctypes bindings to `libebm`; neighboring utilities handle input normalization (`_clean_x.py`, `_unify_data.py`, `_unify_predict.py`), purification, link functions, privacy helpers, and seeding.
- `visual/` and `provider/` — the explanation visualization stack; `from interpret import show` is wired through `visual/_interactive.py`. The browser bundle `interpret-inline.js` is shipped as package data.
- `ext/` and the `entry_points` in `setup.py` define the extension/plugin system (`interpret_ext_blackbox`, `interpret_ext_glassbox`, etc.).

## CI and platforms

`.github/workflows/ci.yml` is the source of truth for supported platforms, Python/Node versions, tool setup, and full validation commands. It builds native libraries and the JS/R artifacts, then tests Python distributions across its matrix. Consult it rather than assuming local editable-install tests cover packaging.

The workflow also runs nightly on a `schedule`. Source-distribution tests (`testS`) and docs builds run only nightly; wheel tests (`testB`) cover only the oldest and newest Python versions in regular CI and all configured versions nightly. Manual runs with `run_as_scheduled` enabled use the same nightly coverage. Nightly runs also pass extra flags to the native tests: `-valgrind` on Linux and `-analysis` (clang-tidy) on the Windows release configurations. To reproduce a nightly failure, trigger the workflow manually with the `run_as_scheduled` input, or run the native test script locally with the same configuration and existing-library flags as CI, plus the scheduled flag. On Windows, the test runner forwards `-analysis` only to the rebuild selected by an `-existing_*` flag; for analysis without testing an existing library, use `.\build.bat -release_64 -analysis` directly.

When adding a platform or SIMD target, update the relevant build/bridge integration in `build.sh` and `build.bat`, the package-data list in `python/interpret-core/setup.py` when new artifacts are shipped, and CI coverage.
