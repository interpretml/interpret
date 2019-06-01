# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and the versioning is mostly derived from [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.6] - 2019-05-31
### Added
- Integer indexing for preserve method.
- Public-facing CI build added. Important for pull requests.
### Changed
- Visual-related imports are now loaded when visualize is called for explanations.

## [v0.1.5] - 2019-05-30
### Added
- Added preserve method. Can now save visuals into notebook/file - does not work with decision trees.
- Added status_show_server method. Acts as a check for server reachability.
- Exposed init_show_server method. Can adjust address, base_url, etc.
- Added print_debug_info method in develop module. Important for troubleshooting/bug-reports.
### Fixed
- Various internal C++ fixes.
- Minor clean up on example notebooks.
### Changed
- Additional dependency required: psutil.
- Test refactoring.

## [v0.1.4] - 2019-05-23
### Added
- Root path for show server now has a light monitor page.
- Logging registration can now print to both standard streams and files.
### Fixed
- Error handling for non-existent links fixed for visualization backend.
- In some circumstances, Python process will hang. Resolved with new threading.
### Changed
- Unpinning scipy version, upstream dependencies now compatible with latest.
- Show server is now run by a thread directly, not via executor pools.
- Re-enabled notebook/show tests, new threading resolves hangs.
- Small clean-up of setup.py and Azure pipelines config.

## [v0.1.3] - 2019-05-21
### Added
- Model fit can now support lists of lists as instance data.
- Model fit can now support lists for label data.
### Fixed
- Various internal C++ fixes.
### Changed
- Removed hypothesis as public test dependency.
- C++ logging introduced (no public access).

## [v0.1.2] - 2019-05-17
### Added
- EBM can now disable early stopping with run length set to -1.
- EBM tracking of final episodes per base estimator.
### Fixed
- Pinning scipy, until upstream dependencies are compatible.
### Changed
- Clean-up of EBM logging for training.
- Temporary disable of notebook/show tests until CI environment is fixed.

## [v0.1.1] - 2019-05-16
### Added
- Added server shutdown call for 'show' method.
### Fixed
- Axis titles now included in performance explainer.
- Fixed hang on testing interface.

## [v0.1.0] - 2019-05-14
### Added
- Added port number assignments for 'show' method.
- Includes codebase of v0.0.6.
### Changed
- Native code build scripts hardened.
- Libraries are statically linked where possible.
- Code now conforms to Python Black and its associated flake8.

[v0.1.6]: https://github.com/microsoft/interpret/releases/tag/v0.1.6
[v0.1.5]: https://github.com/microsoft/interpret/releases/tag/v0.1.5
[v0.1.4]: https://github.com/microsoft/interpret/releases/tag/v0.1.4
[v0.1.3]: https://github.com/microsoft/interpret/releases/tag/v0.1.3
[v0.1.2]: https://github.com/microsoft/interpret/releases/tag/v0.1.2
[v0.1.1]: https://github.com/microsoft/interpret/releases/tag/v0.1.1
[v0.1.0]: https://github.com/microsoft/interpret/releases/tag/v0.1.0
