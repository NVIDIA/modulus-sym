# Modulus Symbolic (Beta)

<!-- markdownlint-disable -->
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![GitHub](https://img.shields.io/github/license/NVIDIA/modulus-sym)](https://github.com/NVIDIA/modulus-sym/blob/master/LICENSE.txt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<!-- markdownlint-enable -->

Modulus Symbolic (Modulus Sym) provides an abstraction layer for using PDE-based symbolic
loss functions. Additional information can be found in the [Modulus documentation](https://docs.nvidia.com/modulus/index.html#sym).
Users of Modulus versions older than 23.05 can refer to the [migration guide](https://docs.nvidia.com/deeplearning/modulus/migration-guide/index.html)
for updating to the latest version.

## Modulus Packages

- [Modulus (Beta)](https://github.com/NVIDIA/modulus)
- [Modulus Launch (Beta)](https://github.com/NVIDIA/modulus-launch)
- [Modulus Symbolic (Beta)](https://github.com/NVIDIA/modulus-sym)
- [Modulus Tool-Chain (Beta)](https://github.com/NVIDIA/modulus-toolchain)

## Installation

### PyPi

The recommended method for installing the latest version of Modulus Symbolic is using PyPi:

```bash
pip install nvidia-modulus.sym
```

### Container

The recommended Modulus docker image can be pulled from the [NVIDIA Container Registry](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/containers/modulus):

```bash
docker pull nvcr.io/nvidia/modulus/modulus:23.05
```

## From Source

### Package

For a local build of the Modulus Symbolic Python package from source use:

```Bash
git clone git@github.com:NVIDIA/modulus-sym.git && cd modulus-sym

pip install --upgrade pip
pip install .
```

### Source Container

To build release image, you will need to do the below preliminary steps:

Clone this repo, and download the Optix SDK from
<https://developer.nvidia.com/designworks/optix/downloads/legacy>.

```bash
git clone https://github.com/NVIDIA/modulus-sym.git
cd modulus-sym/ && mkdir deps
```

Currently Modulus supports v7.0. Place the Optix file in the deps directory and make it
executable. Also clone the pysdf library in the deps folder (NVIDIA Internal)

```bash
chmod +x deps/NVIDIA-OptiX-SDK-7.0.0-linux64.sh 
git clone <internal pysdf repo>
```

Then to build the image, insert next tag and run below:

```bash
docker build -t modulus-sym:deploy -f Dockerfile --target deploy .
```

Alternatively, if you want to skip pysdf installation, you can run the following:

```bash
docker build -t modulus-sym:deploy -f Dockerfile --target no-pysdf .
```

## Contributing

For guidance on making a contribution to Modulus, see the [contributing guidelines](https://github.com/NVIDIA/modulus-sym/blob/main/CONTRIBUTING.md).

## Communication

- Github Discussions: Discuss architectures, implementations, Physics-ML research, etc.
- GitHub Issues: Bug reports, feature requests, install issues, etc.
- Modulus Forum: The [Modulus Forum](https://forums.developer.nvidia.com/c/physics-simulation/modulus-physics-ml-model-framework)
hosts an audience of new to moderate level users and developers for general chat, online
discussions, collaboration, etc.

## License

Modulus Symbolic is provided under the Apache License 2.0, please see
[LICENSE.txt](./LICENSE.txt) for full license text.
