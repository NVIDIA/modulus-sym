# Modulus Symbolic

<!-- markdownlint-disable -->
[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![GitHub](https://img.shields.io/github/license/NVIDIA/modulus)](https://github.com/NVIDIA/modulus/blob/master/LICENSE.txt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<!-- markdownlint-enable -->
[**Getting Started**](#getting-started)
| [**Install guide**](#installation)
| [**Contributing Guidelines**](#contributing-to-modulus)
| [**Resources**](#resources)
| [**Communication**](#communication)

## What is Modulus Symbolic?

Modulus Symbolic (Modulus Sym) repository is part of Modulus SDK and it provides
algorithms and utilities to be used with Modulus core, to explicitly physics inform the
model training. This includes utilities for explicitly integrating symbolic PDEs,
domain sampling and computing PDE-based residuals using various gradient computing schemes.

It also provides higher level abstraction to compose a training loop from specification
of the geometry, PDEs and constraints like boundary conditions using simple symbolic APIs.
Please refer to the
[Lid Driven cavity](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/basics/lid_driven_cavity_flow.html)
that illustrates the concept.

Additional information can be found in the
[Modulus documentation](https://docs.nvidia.com/modulus/index.html#sym).

Please refer to the [Modulus SDK](https://github.com/NVIDIA/modulus/blob/main/README.md)
to learn more about the full stack.

### Hello world

You can run below example to start using the geometry module from Modulus-Sym as shown
below:

```python
>>> import numpy as np
>>> from modulus.sym.geometry.primitives_3d import Box
>>> from modulus.sym.utils.io.vtk import var_to_polyvtk
>>> nr_points = 100000
>>> box = Box(point_1=(-1, -1, -1), point_2=(1, 1, 1))
>>> s = box.sample_boundary(nr_points=nr_points)
>>> var_to_polyvtk(s, "boundary")
>>> print("Surface Area: {:.3f}".format(np.sum(s["area"])))
Surface Area: 24.000
```

To use the PDE module from Modulus-Sym, you can run the below example:

```python
>>> from modulus.sym.eq.pdes.navier_stokes import NavierStokes
>>> ns = NavierStokes(nu=0.01, rho=1, dim=2)
>>> ns.pprint()
continuity: u__x + v__y
momentum_x: u*u__x + v*u__y + p__x + u__t - 0.01*u__x__x - 0.01*u__y__y
momentum_y: u*v__x + v*v__y + p__y + v__t - 0.01*v__x__x - 0.01*v__y__y
```

To use the computational graph builder from Modulus Sym:

```python
>>> import torch
>>> from sympy import Symbol
>>> from modulus.sym.graph import Graph
>>> from modulus.sym.node import Node
>>> from modulus.sym.key import Key
>>> from modulus.sym.eq.pdes.diffusion import Diffusion
>>> from modulus.sym.models.fully_connected import FullyConnectedArch
>>> net = FullyConnectedArch(input_keys=[Key("x")], output_keys=[Key("u")], nr_layers=3, layer_size=32)
>>> diff = Diffusion(T="u", time=False, dim=1, D=0.1, Q=1.0)
>>> nodes = [net.make_node(name="net")] + diffusion.make_nodes()
>>> graph = Graph(nodes, [Key("x")], [Key("diffusion_u")])
>>> graph.forward({"x": torch.tensor([1.0, 2.0]).requires_grad_(True).reshape(-1, 1)})
{'diffusion_u': tensor([[-0.9956],
        [-1.0161]], grad_fn=<SubBackward0>)}
```

Please refer [Introductory Example](https://github.com/NVIDIA/modulus/tree/main/examples/cfd/darcy_physics_informed)
for usage of the physics utils in custom training loops and
[Lid Driven cavity](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/basics/lid_driven_cavity_flow.html)
for an end-to-end PINN workflow.

Users of Modulus versions older than 23.05 can refer to the
[migration guide](https://docs.nvidia.com/deeplearning/modulus/migration-guide/index.html)
for updating to the latest version.

## Getting started

The following resources will help you in learning how to use Modulus. The best way is to
start with a reference sample and then update it for your own use case.

- [Using Modulus Sym with your PyTorch model](https://github.com/NVIDIA/modulus/tree/main/examples/cfd/darcy_physics_informed)
- [Using Modulus Sym to construct computational graph](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/basics/modulus_overview.html)
- [Reference Samples](https://github.com/NVIDIA/modulus-sym/blob/main/examples/README.md)
- [User guide Documentation](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/index.html)

## Resources

- [Getting started Webinar](https://www.nvidia.com/en-us/on-demand/session/gtc24-dlit61460/?playlistId=playList-bd07f4dc-1397-4783-a959-65cec79aa985)
- [AI4Science Modulus Bootcamp](https://github.com/openhackathons-org/End-to-End-AI-for-Science)

## Installation

### PyPi

The recommended method for installing the latest version of Modulus Symbolic is using PyPi:

```bash
pip install "pint==0.19.2"
pip install nvidia-modulus.sym --no-build-isolation
```

Note, the above method only works for x86/amd64 based architectures. For installing
Modulus Sym on Arm based systems using pip,
Install VTK from source as shown
[here](https://gitlab.kitware.com/vtk/vtk/-/blob/v9.2.6/Documentation/dev/build.md?ref_type=tags#python-wheels)
and then install Modulus-Sym and other dependencies.

```bash
pip install nvidia-modulus.sym --no-deps
pip install "hydra-core>=1.2.0" "termcolor>=2.1.1" "chaospy>=4.3.7" "Cython==0.29.28" "numpy-stl==2.16.3" "opencv-python==4.5.5.64" \
    "scikit-learn==1.0.2" "symengine>=0.10.0" "sympy==1.12" "timm>=1.0.3" "torch-optimizer==0.3.0" "transforms3d==0.3.1" \
    "typing==3.7.4.3" "pillow==10.0.1" "notebook==6.4.12" "mistune==2.0.3" "pint==0.19.2" "tensorboard>=2.8.0"
```

### Container

The recommended Modulus docker image can be pulled from the
[NVIDIA Container Registry](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/containers/modulus):

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

To build release image insert next tag and run below:

```bash
docker build -t modulus-sym:deploy \
    --build-arg TARGETPLATFORM=linux/amd64 --target deploy -f Dockerfile .
```

Currently only `linux/amd64` and `linux/arm64` platforms are supported.

## Contributing to Modulus

Modulus is an open source collaboration and its success is rooted in community
contribution to further the field of Physics-ML. Thank you for contributing to the
project so others can build on top of your contribution.

For guidance on contributing to Modulus, please refer to the
[contributing guidelines](CONTRIBUTING.md).

## Cite Modulus

If Modulus helped your research and you would like to cite it, please refer to the
[guidelines](https://github.com/NVIDIA/modulus/blob/main/CITATION.cff)

## Communication

- Github Discussions: Discuss new architectures, implementations, Physics-ML research, etc.
- GitHub Issues: Bug reports, feature requests, install issues, etc.
- Modulus Forum: The [Modulus Forum](https://forums.developer.nvidia.com/c/physics-simulation/modulus-physics-ml-model-framework)
hosts an audience of new to moderate-level users and developers for general chat, online
discussions, collaboration, etc.

## Feedback

Want to suggest some improvements to Modulus? Use our feedback form
[here](https://docs.google.com/forms/d/e/1FAIpQLSfX4zZ0Lp7MMxzi3xqvzX4IQDdWbkNh5H_a_clzIhclE2oSBQ/viewform?usp=sf_link).

## License

Modulus is provided under the Apache License 2.0, please see [LICENSE.txt](./LICENSE.txt)
for full license text.
