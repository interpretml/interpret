# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and the versioning is mostly derived from [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.1.7 - 2024-06-18
### Fixed
- Raise exception keyword previously not honored for execution runner.

## v0.1.6 - 2024-06-18
### Fixed
- Adjuments to dockerized containers where executor is called multiple times.

## v0.1.5 - 2024-06-13
### Added
- Retrieval of catboost datasets that are less than 50k instances.
### Fixed
- Minor API fixes for docker/ACI.

## v0.1.4 - 2024-05-22
### Fixed
- Docker executor was using wrong arguments (after API change).

## v0.1.3 - 2024-05-20
### Fixed
- Typing fix for python 3.8.

## v0.1.2 - 2024-05-20
### Fixed
- Fix for binary streams with fastparquet that breaks on some systems.

## v0.1.1 - 2024-05-20
### Added
- `exist_ok` parameter added for `populate_with_datasets`

## v0.1.0 - 2024-05-15
### Changed
- Numerous changes across the board. This version has multiple breaking changes. Re-read the README for the new API.
