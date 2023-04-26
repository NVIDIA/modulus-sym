Installation
===================================

.. _system_requirements:

System Requirements
-------------------

- **Operating System** 

   -  Ubuntu 20.04 or Linux 5.13 kernel

- **Driver and GPU Requirements** 

   -  ``pip``: NVIDIA driver that is compatible with local PyTorch installation.
   
   -  Docker container: Modulus container is based on CUDA 11.8, which requires NVIDIA which requires NVIDIA Driver release 520 or later. However, if you are running on a data center GPU (for example, T4 or any other data center GPU), you can use NVIDIA driver release 450.51 (or later R450), 470.57 (or later R470), 510.47 (or later R510), or 515.65 (or later R515).The CUDA driver's compatibility package only supports particular drivers. Thus, users should upgrade from all R418, R440, and R460 drivers, which are not forward-compatible with CUDA 11.8. 3Driver release 515 or later. However, if you are running on a data center GPU (for example, T4 or any other data center GPU), you can use NVIDIA driver release 450.51 (or later R450), 470.57 (or later R470), or 510.47 (or later R510). However, any drivers older than 465 will not support the SDF library. For additional support details, see `PyTorch NVIDIA Container <https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-12.html#rel-22-12>`_.
    
- **Required installations for pip install** 

   -  Python 3.8
   
- **Recommended Hardware** 

   -  64-bit x86
 
   - `NVIDIA GPUs <https://developer.nvidia.com/cuda-gpus>`_:

      -  NVIDIA Ampere GPUs - A100, A30, A4000

      -  Volta GPUs - V100

      -  Turing GPUs - T4 

   - Other Supported GPUs:
      
      - NVIDIA Ampere GPUs - RTX 30xx

      - Volta GPUs - Titan V, Quadro GV100

   - For others, please reach us out at `Modulus Forums <https://forums.developer.nvidia.com/t/welcome-to-the-modulus-physics-ml-model-framework-forum>`_ 

**All studies in the User Guide are done using V100 on DGX-1. A100 has also been tested.**

.. note::
 To get the benefits of all the performance improvements (e.g. AMP, multi-GPU scaling, etc.), use the NVIDIA container for Modulus Sym. This container comes with all the prerequisites and dependencies and allows you to get started efficiently with Modulus Sym.

.. _install_modulus_docker:

Modulus Sym with Docker Image (Recommended)
-------------------------------------------

Install the Docker Engine
^^^^^^^^^^^^^^^^^^^^^^^^^   

To start working with Modulus Sym, ensure that you have `Docker Engine <https://docs.docker.com/engine/install/ubuntu/>`_ installed. 

You will also need to install the `NVIDIA docker toolkit <https://github.com/NVIDIA/nvidia-docker>`_. This should work on most debian based systems: 

.. code-block:: bash
   
   sudo apt-get install nvidia-docker2 
       
Running Modulus Sym in the docker image while using SDF library may require NVIDIA container toolkit version greater or equal to 1.0.4.

To run the docker commands without :code:`sudo`, add yourself to the docker group by following the steps 1-4 found in `Manage Docker as a non-root user <https://docs.docker.com/engine/install/linux-postinstall/>`_ . 

Install Modulus Sym
^^^^^^^^^^^^^^^^^^^  

Download the Modulus Sym docker container from NGC using:

.. code-block::
   
   docker pull nvcr.io/nvidia/modulus/modulus:<tag>


Using the Modulus Sym examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the docker container using: 

.. note::
   All examples can be found in the ``examples/`` directory from the `GitHub Repo <https://github.com/NVIDIA/modulus-sym/>`_. 

.. code-block::
   
   docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \  
              --runtime nvidia -v ${PWD}/examples:/examples \              
              -it --rm modulus:xx.xx bash                                      
.. warning::
   The modulus-sym repository has Git LFS enabled. You will need to have Git LFS installed for the clone to work correctly. 
   More information about Git LFS can be found `here <https://git-lfs.github.com/>`_ .

To verify the installation has been done correctly, run these commands: 

.. code-block:: bash
   
   cd helmholtz/                                                           
   python helmholtz.py                                                     

If you see the ``outputs/`` directory created after the execution of the command (~5 min), the installation is successful.

.. note:: 
    If you intend to use the quadrature functionality of Modulus Sym :ref:`variational-example` please install the ``quadpy``, ``orthopy``, and ``ndim`` packages inside the container. Similarly, if you plan to use the Neural operators within Modulus Sym and wish to download some of the example data, install the ``gdown`` package. Both these packages can easily be installed inside the container using ``pip install <package-name>``.

.. _install_modulus_bare_metal:

Modulus Sym ``pip`` Install
----------------------------

While NVIDIA recommends using the docker image provided to run Modulus Sym, installation instructions for Ubuntu 20.04 are also provided. Currently the ``pip`` installation does not support the tesselated geometry module in Modulus Sym. If this is required please use the docker image provided. 
Modulus Sym requires CUDA to be installed. 
For compatibility with PyTorch >=1.12, use CUDA 11.6 or later. Modulus Sym requires Python 3.8 or later. 

Modulus Sym can then be installed using 

.. code-block::

   pip install nvidia-modulus-sym

.. warning:: Depending on the version of PyTorch, you would need a specific version of functorch. The best recommended way is to use latest version for both PyTorch and functorch.

.. warning:: Add packages for ``quadpy``, ``orthopy``, ``ndim`` and ``gdown`` if you intend to use the quadrature functionality of Modulus Sym :ref:`variational-example` or wish to download the example data for the Neural Operator training.

To verify the installation, run these commands: 

.. code-block:: bash

   cd examples/helmholtz/                                                                      
   python helmholtz.py                                                           

If you see ``outputs/`` directory created after the execution of the command (~5 min), the installation is successful. 

Modulus Sym on Public Cloud instances
-------------------------------------

Modulus Sym can be used on public cloud instances like AWS and GCP. To install and run Modulus Sym, 

#. Get your GPU instance on AWS or GCP. (Please see :ref:`system_requirements` for recommended hardware platform)
#. Use the `NVIDIA GPU-Optimized VMI <https://aws.amazon.com/marketplace/pp/prodview-7ikjtg3um26wq?sr=0-3&ref_=beagle&applicationId=AWSMPContessa>`_ on the cloud instance. For detailed instructions on setting up VMI refer `NGC Certified Public Clouds <https://docs.nvidia.com/ngc/ngc-deploy-public-cloud/index.html#ngc-certified-public-clouds>`_.
#. Once the instance spins up, follow the :ref:`install_modulus_docker` to load the Modulus Sym Docker container and the examples. 
