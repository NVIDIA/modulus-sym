.. _whatsnew:

Release Notes
=============

New features/Highlights v22.09
-------------------------------

New Network Architectures
^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Generalized Neural Operators**: Extended Fourier Neural Operator (FNO) and DeepONet to support compatibility with other built in Modulus networks. FNO can now use any point wise network inside of Modulus for its decoder. DeepONet can now accept any branch/trunk net.

* Model parallelism has been introduced as a beta feature with model-parallel AFNO. This allows for parallelizing the model across multiple GPUs along the channel dimension.

* Support for the self-scalable tanh (Stan) activation function is now available.

Training features
^^^^^^^^^^^^^^^^^^

* **Criteria based training termination**: APIs to terminate training based on the convergence criteria like total loss or individual loss terms.

* **Utilities for Nondimensionalization**: Nondimensionalization tools are now provided in Modulus to help users properly scale their system’s units for physics informed training.

* **Causal weighting scheme**: Causal weighting scheme by reformulating the losses for the residual and initial conditions for better convergence in case of transient problems.

* **Selective Equations Term Suppression**: Allows creation of different instances of the same PDE and freeze different terms to improve convergence on stiff PDEs in physics informed training.

Performance Enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^

* **FuncTorch Integration**: Modulus now supports FuncTorch gradient calculations (A Jax like paradigm) for faster gradient calculations in physics-informed training. 

Documentation Enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* More example-guided workflows for beginners and Jupyter notebook based getting started example.

* Enhancements to Modulus Features section to serve as a user guide.


New features/Highlights v22.07
-------------------------------

New Network Architectures 
^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Generalized DeepONet architecture**: DeepONet in Modulus is restructured so that it can easily be applied to data-informed and physics-informed 1D/2D problems with any arbitrary network architectures as the backbone.

* **FourCastNet**: FourCastNet, short for **Four**\ier Fore\ **\Cast**\ing Neural **Net**\work, is a global data-driven weather forecasting model that provides accurate short to medium range global predictions at :math:`0.25^{\circ}` resolution. In the current iteration, FourCastNet forecasts 20 atmospheric variables. (`Paper <https://arxiv.org/abs/2202.11214>`_)
  
Training features
^^^^^^^^^^^^^^^^^^ 

* **L2-L1 Loss Decaying**: A L2 to L1 loss decay is now supported. This feature allows users to slowly transition between a L2  loss and L1 loss during training. This can improve training accuracy since decaying to an L1 loss can help reduce the impact of outlier training points with unstable loss values. This can be particularly useful for problems with singularities and sharp gradient interfaces.


Performance Enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^
* **Meshless Finite Differentiation**: Modulus now includes a new approximate differentiation approach for physics-informed problems based on finite difference calculations. This new method allows for the computational complexity of training to be dramatically decrease compared to the standard automatic differentiation approach. For some examples this can yield upto 4x speed up in training time with minimal impact on accuracy. This feature is in beta and subject to change with improvements in the future. 

* **Dataset Refactor**: Both map style PyTorch datasets and iterable style datasets are supported inside of Modulus for both physics based and data-driven problems. This includes built in functionality for multithreading workers and data parallel training in multi-GPU / multi-node environments. 

* **Tiny CUDA NN**: Modulus now offers several Tiny CUDA NN architectures which are fully fused neural networks. These models provide a lightweight, heavily optimized implementation which can improve computation performance. Tiny Cuda NN combined with meshless finite derivatives can yield significant speed up over vanilla physics-informed implementations. 

* **CUDA Graphs**: Modulus now leverages `CUDA graphs <https://developer.nvidia.com/blog/cuda-graphs/>`_ to record the series of CUDA kernels used during a training iteration and save it as a single graph that can then be replayed on the GPU as opposed to individual launches reducing CPU launch latency bottlenecks.

* **Geometry Module Refactor**: The geometry module inside of Modulus has been refactored to improve point sampling performance for both continuous and tessellated geometries. This greatly reduces the initial overhead of creating training/testing datasets from complex geometries.

 
New features/Highlights v22.03
-------------------------------

New Network Architectures 
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :ref:`fno`: Physics inspired Neural Network model that uses global convolutions in spectral space as an inductive bias for training Neural Network models of physical systems. It incorporates important spatial and temporal correlations, which strongly govern the dynamics of many physical systems that obey PDE laws. 

* :ref:`pino`: PINO is the explicitly physics-informed version of the FNO. PINO combines the operator learning and function optimization frameworks. In the operator learning phase, PINO learns the solution operator over multiple instances of the parametric PDE family. 

* :ref:`afno`: An adaptive FNO for scaling self-attention to high resolution images in vision transformers by establishing a link between operator learning and token mixing. AFNO is based on FNO which allows framing token mixing as a continuous global convolution without any dependence on the input resolution. The resulting model is highly parallel with a quasi-linear complexity and has linear memory in the sequence size. 

* :ref:`deeponet_theory`: A DeepONet consists of two sub-networks, one for encoding the input function and another for encoding the locations and then merged to compute the output. Using inductive bias, DeepONets are shown to reduce the generalization error compared to the fully connected networks. 

 
Modeling Enhancements
^^^^^^^^^^^^^^^^^^^^^^

* **Two equation turbulence**: Solution to two equation turbulence (k-epsilon & k-omega) models on a fully developed turbulent flow in a 2D channel case using wall functions. Two types of wall functions (standard and Launder-Spalding) have been tested and demonstrated on the above example problem. 

* **Exact boundary condition imposition**: A new algorithm based on the theory of R-functions and transfinite interpolation is implemented to exactly impose the Dirichlet boundary conditions on 2D geometries. In this algorithm, the neural network solution to a given PDE is constrained to a boundary condition aware and geometry aware ansatz, and a loss function based on the first-order formulation of the PDE is minimized to train a solution that exactly satisfies the boundary conditions. 


Training features
^^^^^^^^^^^^^^^^^^ 

* **Support for new optimizers**: Modulus now supports 30+ optimizers including the built-in PyTorch optimizers and the optimizers in the `torch-optimizer`` library. Includes support for AdaHessian, a second-order stochastic optimizer that approximates an exponential moving average of the Hessian diagonal for adaptive preconditioning of the gradient vector.  

* **New algorithms for loss balancing**: Three new loss balancing algorithms, namely Grad Norm, ReLoBRaLo (Relative Loss Balancing with Random Lookback), and Soft Adapt are implemented. These algorithms dynamically tune the loss weights based on the relative training rates of different losses. Also, Neural Tangent Kernel (NTK) analysis is implemented. NTK is a neural network analysis tool that indicates the convergent speed of each component. It will provide an explainable choice for the weights for different loss terms. Grouping the MSE of the loss allows computation of NTK dynamically. 

* **Sobolev (gradient-enhanced) training**: Sobolev training of neural networks solvers incorporate derivative information of the PDE residuals into the loss function.

* **Hydra Configs**: A big part of model development is hyperparameter tuning that requires performing multiple training runs with different configurations. Usage of Hydra within Modulus allows for more extensibility and configurability. Certain components of the training pipeline can now be switched out for other variants with no code change. Hydra multi-run also allows for better training workflows and running a hyperparameter sweep with a single command. 

* **Post-processing**: Modulus now supports new Tensorboard and VTK features that will allow better visualizations of the Model outputs during and after training. 
  

Feature Summary
---------------

* Improved stability in multi-GPU/multi-Node implementations using linear-exponential learning rate and utilization of TF32 precision for A100 GPUs
* Physics types:
  
  * Linear Elasticity (plane stress, plane strain and 3D)
  * Fluid Mechanics
  * Heat Transfer
  * Coupled Fluid Thermal
  * Electromagnetics
  * 2D wave propagation
  * 2 Equation Turbulence Model for channel flow

* Solution of differential equations:
  
  * Ordinary Differential Equations
  * Partial Differential Equations
    
    * Differential (strong) form
    * Integral (weak) form

* Several Neural Network architectures to choose from:
  
  * Fully Connected Network
  * Fourier Feature Network
  * Sinusoidal Representation Network
  * Modified Fourier Network
  * Deep Galerkin Method Network
  * Modified Highway Network
  * Multiplicative Filter Network
  * Multi-scale Fourier Networks
  * Spatio-temporal Fourier Feature Networks
  * Hash Encoding Network
  * Super Resolution Net

* Neural Operators
  
  * Fourier Neural Operator (FNO)
  * Physics Informed Neural Operator (PINO)
  * Adaptive Fourier Neural Operator (AFNO)
  * DeepONet 

* Other Features include:
  
  * Global mass balance constraints
  * SDF (Signed Distance Function) weighting for PDEs in flow problems for rapid convergence
  * Exact mass balance constraints
  * Exact boundary condition imposition
  * Global and local learning rate annealing
  * Global adaptive activation functions
  * Halton sequences for low discrepancy point cloud generation
  * Gradient accumulation
  * Time stepping schemes for transient problems
  * Temporal loss weighting and time marching for continuous time approach
  * Importance Sampling
  * Homoscedastic task uncertainty quantification for loss weighting
  * Exact boundary condition imposition
  * Sobolev (gradient-enhanced) training
  * Criteria based training termination
  * Utilities for Nondimensionalization
  * Causal weighting scheme
  * Selective Equation Term Suppression
  * FuncTorch Integration
  * L2-L1 loss norm decay
  * Meshless Finite Differentiation
  * CUDA Graphs Integration
  * Loss balancing schemes:
    
    * Grad Norm
    * ReLoBRaLo
    * Soft Adapt
    * NTK
  
  * Parameterized system representation for solving several configurations concurrently
  * Transfer learning for efficient surrogate based parameterizations
  * Polynomial chaos expansion method for accessing how the model input uncertainties manifest in its output
  * APIs to automatically generate point clouds from boolean compositions of geometry primitives or import point clouds for complex geometry (STL files)
  * STL point cloud generation from superfast ray tracing method with uniformly emanating rays using Fibonacci sphere. Points categorized as inside, outside and on the surface, SDF, and its derivative calculation
  * Logically separate APIs for physics, boundary conditions and geometry consistent with traditional solver datasets
  * Support for optimizers: Modulus supports 30+ optimizers including the built-in PyTorch optimizers and optimizers from the `torch-optimizer` library. Support for AdaHessian optimizer 
  * Hydra configs to allow for easy customization, improved accessibility and hyperparameter tuning
  * Tensorboard plots to easily visualize the outputs, histograms, etc. during training


Known Issues
------------

* The Modulus team is aware of `CVE-2021-29063 <https://nvd.nist.gov/vuln/detail/CVE-2021-29063#range-8144236>`_ in the ``mpmath`` library. This flaw in the regex parsing could DoS the container process if untrusted users are allowed to send crafted regex input. As soon as the released fix is available, the Modulus team will update this image. 
* Tiny CUDA NN models are only supported on Ampere or newer GPU architectures using the Docker container.
* Multi-GPU training not supported for all use cases of Sequential Solver.
