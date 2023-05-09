# Modulus Sym (Beta)

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![GitHub](https://img.shields.io/github/license/NVIDIA/modulus-sym)](https://github.com/NVIDIA/modulus-sym/blob/master/LICENSE.txt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Modulus-Sym is a repo with an abstracted workflow comparable to existing numerical solvers for training data-free and data-driven neural models. Additional information can be found in the [Modulus documentation](https://docs.nvidia.com/modulus/index.html#sym).


## Modulus Packages

- [Modulus (Beta)](https://github.com/NVIDIA/modulus)
- [Modulus Launch (Beta)](https://github.com/NVIDIA/modulus-launch)
- [Modulus Symbolic (Beta)](https://github.com/NVIDIA/modulus-sym)

## Installation 

### PyPi

The recommend method for installing the latest version of Modulus is using PyPi:
```Bash
pip install nvidia-modulus.sym
```

### Container

The recommended Modulus docker image can be pulled from the [NVIDIA Container Registry](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/containers/modulus):
```Bash
docker pull nvcr.io/nvidia/modulus/modulus:23.05
```

## From Source

For a local build of the Modulus Symbolic Python package from source use:

```Bash
git clone git@github.com:NVIDIA/modulus-sym.git && cd modulus-sym

pip install --upgrade pip
pip install .
```

### Container

To build release image, you will need to do the below preliminary steps:

Clone this repo, and download the Optix SDK from https://developer.nvidia.com/designworks/optix/downloads/legacy. 
```
git clone https://github.com/NVIDIA/modulus-sym.git
cd modulus-sym/ && mkdir deps
```
Currently Modulus supports v7.0. Place the Optix file in the deps directory and make it executable. Also clone the pysdf library in the deps folder (NVIDIA Internal)
```
chmod +x deps/NVIDIA-OptiX-SDK-7.0.0-linux64.sh 
git clone <internal pysdf repo>
```

Then to build the image, insert next tag and run below:
```
docker build -t modulus-sym:deploy -f Dockerfile --target with-pysdf .
```

Alternatively, if you want to skip pysdf installation, you can run the following:
```
docker build -t modulus-sym:deploy -f Dockerfile --target no-pysdf .
```

## Contributing

Modulus is in an open-source beta. We are not accepting external contributions at this time.

## Contact

Reach out to Modulus team members and user community on the [NVIDIA developer forums](https://forums.developer.nvidia.com/c/physics-simulation/modulus-physics-ml-model-framework).

## License
Modulus Symbolic is provided under the Apache License 2.0, please see [LICENSE.txt](./LICENSE.txt) for full license text
