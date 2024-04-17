<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.0a0] - 2024-07-XX

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

### Dependencies

## [1.5.0] - 2024-04-17

### Added

- Added reservoir examples using GenAI and CCUS workflows.

### Security

- Update OpenCV and Pillow versions to fix security

## [1.4.0] - 2024-01-25

### Fixed

- Fix bug for `ConvFullyConnectedArch`.
- Updating `Activation.SILU` test to conform with updated nvFuser kernel generation.

## [1.3.0] - 2023-11-20

### Added

- Added instructions for docker build on ARM architecture.
- Added domain decomposition examples using X-PINN and FB-PINN style.

### Changed

- Integrated the network architecture layers into Modulus-Core.

### Fixed

- Fixed Gradient Aggregation bug.

### Security

- Upgrade Pillow and Sympy to their latest versions.
- Upgrade Scikit-Learn version.

### Dependencies

- Updated base PyTorch container to 23.10 and Optix version to 7.3.0

## [1.2.0] - 2023-09-21

### Added

- Example for reservoir modeling using PINO and FNO

## [1.1.0] - 2023-08-10

### Added

- Added a CHANGELOG.md

### Removed

- Accompanying licenses (will provide in the Modulus docker image).

### Fixed

- Arch `from_config` bug for literal params.
- Fixed fused SiLU activation test.
- Update `np.bool` to `np.bool_`.
- Added a workaround fix for the CUDA graphs error in multi-node runs

### Dependencies

- Updated the base container to latest PyTorch base container which is based on torch 2.0
- Container now supports CUDA 12, Python 3.10
- Updated symengine to >=0.10.0 and vtk to >=9.2.6

## [1.0.0] - 2023-05-08

### Added

- Initial public release.
