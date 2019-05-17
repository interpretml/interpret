# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and the versioning is mostly derived from [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.2] - 2019-05-17
### Added
- EBM can now disable early stopping with run length set to -1.
### Fixed
- Pinning scipy, until upstream dependencies are compatible.
### Changed
- Clean-up of EBM logging for training.

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

[v0.1.2]: https://github.com/microsoft/interpret/releases/tag/v0.1.2
[v0.1.1]: https://github.com/microsoft/interpret/releases/tag/v0.1.1
[v0.1.0]: https://github.com/microsoft/interpret/releases/tag/v0.1.0
