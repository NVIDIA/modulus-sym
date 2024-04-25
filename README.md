<!-- markdownlint-disable -->
# Modulus Symbolic (Beta)


[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![GitHub](https://img.shields.io/github/license/NVIDIA/modulus-sym)](https://github.com/NVIDIA/modulus-sym/blob/master/LICENSE.txt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Modulus Symbolic (Modulus Sym) provides pythonic APIs, algorithms and utilities to be used with Modulus core, to explicitly physics inform the model training. This includes symbolic APIs for PDEs, domain sampling and PDE-based residuals.  

It also provides higher level abstraction to compose a training loop from specification of the geometry, PDEs and constraints like boundary conditions using simple symbolic APIs. 
Please refer to the [Lid Driven cavity](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/basics/lid_driven_cavity_flow.html) that illustrates the concept.
Additional information can be found in the [Modulus documentation](https://docs.nvidia.com/modulus/index.html#sym).

Users of Modulus versions older than 23.05 can refer to the [migration guide](https://docs.nvidia.com/deeplearning/modulus/migration-guide/index.html)
for updating to the latest version.

## Modulus Packages

- [Modulus (Beta)](https://github.com/NVIDIA/modulus): Open-source deep-learning framework for building, training, and fine-tuning deep learning models using state-of-the-art Physics-ML methods.
- [Modulus Symbolic (Beta)](https://github.com/NVIDIA/modulus-sym): Framework providing pythonic APIs, algorithms and utilities to be used with Modulus core to physics inform model training as well as higher level abstraction for domain experts.

### Domain Specific Packages

- [Earth-2 MIP (Beta)](https://github.com/NVIDIA/earth2mip): Python framework to enable climate researchers and scientists to explore and experiment with AI models for weather and climate.

## Installation

### PyPi

The recommended method for installing the latest version of Modulus Symbolic is using PyPi:

```bash
pip install nvidia-modulus.sym
```

Note, the above method only works for x86/amd64 based architectures. For installing Modulus Sym on Arm based systems using pip, 
Install VTK from source as shown [here](https://gitlab.kitware.com/vtk/vtk/-/blob/v9.2.6/Documentation/dev/build.md?ref_type=tags#python-wheels) and then install Modulus-Sym and other dependencies

```bash
pip install nvidia-modulus.sym --no-deps
pip install "hydra-core>=1.2.0" "termcolor>=2.1.1" "chaospy>=4.3.7" "Cython==0.29.28" "numpy-stl==2.16.3" "opencv-python==4.5.5.64" \
    "scikit-learn==1.0.2" "symengine>=0.10.0" "sympy==1.12" "timm==0.5.4" "torch-optimizer==0.3.0" "transforms3d==0.3.1" \
    "typing==3.7.4.3" "pillow==10.0.1" "notebook==6.4.12" "mistune==2.0.3" "pint==0.19.2" "tensorboard>=2.8.0"
```

### Container

The recommended Modulus docker image can be pulled from the [NVIDIA Container Registry](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/containers/modulus):

```bash
docker pull nvcr.io/nvidia/modulus/modulus:24.04
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
docker build -t modulus-sym:deploy \
    --build-arg TARGETPLATFORM=linux/amd64 --target deploy -f Dockerfile .
```

Alternatively, if you want to skip pysdf installation, you can run the following:

```bash
docker build -t modulus-sym:deploy \
    --build-arg TARGETPLATFORM=linux/amd64 --target no-pysdf -f Dockerfile .
```

Currently only `linux/amd64` and `linux/arm64` platforms are supported.

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
