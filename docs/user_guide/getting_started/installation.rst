Installation
===================================

.. _system_requirements:

System Requirements
-------------------

- **Operating System** 

   -  Ubuntu 20.04 or Linux 5.13 kernel

- **Driver and GPU Requirements** 

   -  Bare Metal version: NVIDIA driver that is compatible with local PyTorch installation.
   
   -  Docker container: Modulus container is based on CUDA 11.7, which requires NVIDIA Driver release 515 or later. However, if you are running on a data center GPU (for example, T4 or any other data center GPU), you can use NVIDIA driver release 450.51 (or later R450), 470.57 (or later R470), or 510.47 (or later R510). However, any drivers older than 465 will not support the SDF library. For additional support details, see `PyTorch NVIDIA Container <https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-05.html#rel_22-05>`_.
    
- **Required installations for Bare Metal version** 

   -  Python 3.8
   
   -  PyTorch 1.12

- **Recommended Hardware** 

   -  64-bit x86
 
   - `NVIDIA GPUs <https://developer.nvidia.com/cuda-gpus>`_:

      -  NVIDIA Ampere GPUs - A100, A30, A4000

      -  Volta GPUs - V100

      -  Turing GPUs - T1 

   - Other Supported GPUs:
      
      - NVIDIA Ampere GPUs - RTX 30xx

      - Volta GPUs - Titan V, Quadro GV100

   - For others, please email us at : `modulus-team@exchange.nvidia.com <modulus-team@exchange.nvidia.com>`_

**All studies in the User Guide are done using V100 on DGX-1. A100 has also been tested.**

.. note::
 To get the benefits of all the performance improvements (e.g. AMP, multi-GPU scaling, etc.), use the NVIDIA container for Modulus. This container comes with all the prerequisites and dependencies and allows you to get started efficiently with Modulus.

.. _install_modulus_docker:

Modulus with Docker Image (Recommended)
---------------------------------------

Install the Docker Engine
^^^^^^^^^^^^^^^^^^^^^^^^^   

To start working with Modulus, ensure that you have `Docker Engine <https://docs.docker.com/engine/install/ubuntu/>`_ installed. 

You will also need to install the `NVIDIA docker toolkit <https://github.com/NVIDIA/nvidia-docker>`_. This should work on most debian based systems: 

.. code-block:: bash
   
   sudo apt-get install nvidia-docker2 
       
Running Modulus in the docker image while using SDF library may require NVIDIA container toolkit version greater or equal to 1.0.4.

To run the docker commands without :code:`sudo`, add yourself to the docker group by following the steps 1-4 found in `Manage Docker as a non-root user <https://docs.docker.com/engine/install/linux-postinstall/>`_ . 

Install Modulus
^^^^^^^^^^^^^^^  

Download the Modulus docker container. 
Once downloaded, load the Modulus container into docker using the following command (This may take several minutes): 

Replace the ``xx.xx`` with the release version you are using. The latest release is ``22.09`` 

.. code-block::
   
   docker load -i modulus_image_vxx.xx.tar.gz

Once complete, ``Loaded image: modulus:xx.xx`` will get printed in the console.


Using the Modulus examples
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
   All examples can be found in the `Modulus GitLab repository <https://gitlab.com/nvidia/modulus>`_. To get access to the GitLab repo, visit 
   the `NVIDIA Modulus Downloads page <https://developer.nvidia.com/modulus-downloads>`_ and sign up 
   for the `Modulus GitLab Repository Access <https://developer.nvidia.com/modulus-gitlab-repository-access>`_ .

.. note:: 
   NVIDIA Modulus recommends using SSH to clone the GitLab repos. Information on adding SSH keys to your GitLab account can be found on `GitLab SSH Tutorial <https://docs.gitlab.com/ee/user/ssh.html>`_.
   The basic steps to create and add a SSH key is below:
   
   #. Generate SSH key:  ``ssh-keygen -t ed25519 -C "<comment>"`` (`More Info <https://docs.gitlab.com/ee/user/ssh.html#generate-an-ssh-key-pair-for-a-fidou2f-hardware-security-key>`__)
   
   #. Copy the public key: ``xclip -sel clip < ~/.ssh/id_ed25519.pub`` (`More Info <https://docs.gitlab.com/ee/user/ssh.html#add-an-ssh-key-to-your-gitlab-account>`__)
   
   #. Paste public key and set expiration date at https://gitlab.com/-/profile/keys    
   
   #. Verify ssh set up with ``ssh -T git@gitlab.example.com`` (`More Info  <https://docs.gitlab.com/ee/user/ssh.html#verify-that-you-can-connect>`__)


You can clone the examples repository using:

.. code-block::

   git clone git@gitlab.com:nvidia/modulus/examples.git

Once the repository is cloned, you can run the docker image and mount the Modulus examples using: 

Replace the ``xx.xx`` with the release version you are using. The latest release is ``22.09``

.. code-block::
   
   docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \  
              --runtime nvidia -v ${PWD}/examples:/examples \              
              -it --rm modulus:xx.xx bash                                      
.. warning::
   The examples repository contains several validation data files that are stored as LFS objects. You will need to have Git LFS installed for the all the examples to work correctly. 
   More information about Git LFS can be found `here <https://git-lfs.github.com/>`_ .

To verify the installation has been done correctly, run these commands: 

.. code-block:: bash
   
   cd helmholtz/                                                           
   python helmholtz.py                                                     


If you see the ``outputs/`` directory created after the execution of the command (~5 min), the installation is successful. For some of the examples, we have trained checkpoints for reference contained here, ``https://gitlab.com/nvidia/modulus/checkpoints.git`` . We will continue to add checkpoints for more examples in the future. 

.. note:: 
    If you intend to use the quadrature functionality of Modulus :ref:`variational-example` please install the ``quadpy``, ``orthopy``, and ``ndim`` packages inside the container. Similarly, if you plan to use the Neural operators within Modulus and wish to download some of the example data, install the ``gdown`` package. Both these packages can easily be installed inside the container using ``pip install <package-name>``.

.. _install_modulus_bare_metal:

Modulus Bare Metal Install
--------------------------

While NVIDIA recommends using the docker image provided to run Modulus, installation instructions for Ubuntu 20.04 are also provided. Currently the bare metal installation does not support the tesselated geometry module in Modulus. If this is required please use the docker image provided. 
Modulus requires CUDA to be installed. 
For compatibility with PyTorch 1.12, use CUDA 11.6 or later. Modulus requires Python 3.8 or later. 

Other dependencies can be installed using: 

.. code-block::

   pip3 install matplotlib transforms3d future typing numpy quadpy\    
         	numpy-stl==2.16.3 h5py sympy==1.5.1 termcolor psutil\            
          	symengine==0.6.1 numba Cython chaospy torch_optimizer\
                vtk chaospy termcolor omegaconf hydra-core==1.1.1 einops\
                timm tensorboard pandas orthopy ndim functorch pint

.. warning:: Depending on the version of PyTorch, you would need a specific version of functorch. The best recommended way is to use latest version for both PyTorch and functorch.

.. warning:: Currently, Modulus has only been tested for ``numpy-stl`` 2.16.3, ``sympy`` 1.5.1, ``symengine`` 0.6.1 and ``hydra-core`` 1.1.1 versions. 
   Using other versions for these packages might give errors. 
   Add packages for ``quadpy``, ``orthopy``, ``ndim`` and ``gdown`` if you intend to use the quadrature functionality of Modulus :ref:`variational-example` or wish to download the example data for the Neural Operator training.


Once all dependencies are installed, the Modulus source code can be downloaded from Modulus GitLab repository. Modulus can be installed from the Modulus repository using: 

.. code-block:: bash

   git clone git@gitlab.com:nvidia/modulus/modulus.git
   cd ./Modulus/                                                                 
   python setup.py install                                                       


Using the Modulus examples
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
   All examples can be found in the `Modulus GitLab repository <https://gitlab.com/nvidia/modulus>`_. To get access to the GitLab repo, visit 
   the `NVIDIA Modulus Downloads page <https://developer.nvidia.com/modulus-downloads>`_ and sign up 
   for the `Modulus GitLab Repository Access <https://developer.nvidia.com/modulus-gitlab-repository-access>`_ .

You can clone the examples repository using:

.. code-block::

   git clone git@gitlab.com:nvidia/modulus/examples.git

.. warning::
   The examples repository contains several validation data files that are stored as LFS objects. You will need to have Git LFS installed for the all the examples to work correctly. 
   More information about Git LFS can be found `here <https://git-lfs.github.com/>`_ .


To verify the installation has been done correctly, run these commands: 

.. code-block:: bash

   cd examples/helmholtz/                                                                      
   python helmholtz.py                                                           


If you see ``outputs/`` directory created after the execution of the command (~5 min), the installation is successful. For some of the examples, we have trained checkpoints for reference contained here, ``https://gitlab.com/nvidia/modulus/checkpoints.git`` . We will continue to add checkpoints for more examples in the future.

Modulus on Public Cloud instances
---------------------------------

Modulus can be used on public cloud instances like AWS and GCP. To install and run Modulus, 

#. Get your GPU instance on AWS or GCP. (Please see :ref:`system_requirements` for recommended hardware platform)
#. Use the `NVIDIA GPU-Optimized VMI <https://aws.amazon.com/marketplace/pp/prodview-7ikjtg3um26wq?sr=0-3&ref_=beagle&applicationId=AWSMPContessa>`_ on the cloud instance. For detailed instructions on setting up VMI refer `NGC Certified Public Clouds <https://docs.nvidia.com/ngc/ngc-deploy-public-cloud/index.html#ngc-certified-public-clouds>`_.
#. Once the instance spins up, follow the :ref:`install_modulus_docker` to load the Modulus Docker container and the examples. 
