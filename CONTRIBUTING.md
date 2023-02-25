# Contribute to NVIDIA Modulus

## Introduction
![NVIDIA Modulus Overview](/Modulus_overview.png)

NVIDIA Modulus is a neural network training and inference platform that blends the power of physics in the form of governing partial differential equations (PDEs) with data to build high-fidelity, parameterized surrogate models with near-real-time latency. 
With NVIDIA Modulus, we aim to provide researchers and industry specialists, various tools that will help accelerate your development of such models for the scientific discipline of your need. 

Modulus User Guide comes in with several reference examples to help you jumpstart your development of AI driven models. 
The purpose of this document is to briefly describe the architecture of the Modulus so that partners can extend the Modulus platform by applying Modulus to build AI models for other application areas and domains of scientific computation.

## Architecture

The code location of the Modulus package can be found in the `/modulus` folder.
Inside are various submodules that contain larger features of Modulus including:

- *dataset*: Dataset and dataloading features
- *distributed*: Distributed training features
- *domain*: Training domain features including constraints, inferencers and validators
- *eq*: Physics-informed / equation related features
- *geometry*: Geometry features
- *hydra*: Hydra configuration
- *loss*: Loss functions
- *models*: Built-in neural network models
- *solver*: Training solvers
- *test*: Unit tests
- *utils*: Miscellaneous utilities

Please refer to the [user guide](https://docs.nvidia.com/deeplearning/modulus/user_guide/getting_started/toc.html) for additional information.

## License

The NVIDIA Modulus and its components are licensed under the [Modulus EULA license](/LICENSE.txt).

## Artifacts

NVIDIA Modulus has the following artifacts as part of the product release:
- Source Code
- Container Images
- Examples

Releases of Modulus include container images that are currently available on [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/containers/modulus).
The following are the container images (and tag format) that are released:
├── nvidia/
│   ├── xx
 
## Contributions
NVIDIA is willing to work with partners for extending and adding functionality to the Modulus platform. Modulus and its components are licensed under the [Modulus EULA license](/LICENSE.txt).

To get started developing with the Modulus code base we suggest mounting it as a volume inside the current docker container.
E.g. for version `22.09` we suggest the following:

```shell
$ mkdir modulus_dev && cd modulus_dev
$ git clone https://gitlab.com/nvidia/Modulus/Modulus.git
$ docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
           --runtime nvidia -v ${PWD}:/examples/ -it modulus:22.09 bash
$ cd modulus
$ python setup.py develop
$ cd ..
```
Now anything developed inside of the Modulus source `modulus_dev/modulus` and working folder `modulus_dev` can be ran inside the docker container.

 
### Signing your work
Want to hack on NVIDIA Modulus? Awesome! We only require you to sign your work, the below section describes this!
The sign-off is a simple line at the end of the explanation for the patch. Your signature certifies that you wrote the patch or otherwise have the right to pass it on as a patch in accordance with the Modulus EULA (Link). 
 
Then you just add a line to every git commit message:
```
Signed-off-by: Joe Smith <joe.smith@email.com>
```

Use your real name (sorry, no pseudonyms or anonymous contributions.)
If you set your user.name and user.email git configs, you can sign your commit automatically with git commit -s.
