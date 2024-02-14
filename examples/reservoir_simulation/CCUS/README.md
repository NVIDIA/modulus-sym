# C02 -Brine surrogate computed with a Physics Informed Neural Operator (PINO) 
![alt text](Visuals/All1.png)


## An AI enabled Automatic History Matching Workflow with a PINO based forward solver:
Calibration of subsurface structures is an important step to forecast fluid dynamics and behaviours in a plethora of geoenvironments such as petroleum and geothermal reservoirs. History matching is an ill-posed inverse process to find reservoir model parameters honouring observations by integration of static (e.g., core, logging, and seismic) and dynamic data (e.g., oil and gas rate, water cut, bottomhole pressure, and subsidence/uplift) .In recent developments, the ensemble Kalman filter (EnKF) (Evensen 1994, Law et al 2012, Stuart 2010), ensemble smoother (ES) (van Leeuwen & Evensen 1996) ES with multiple data assimilation (ES-MDA) (Emerick & Reynolds 2013) and Iterative variants of ES (Chen & Oliver 2011, Iglesias 2013) have all been utilised for history matching/data assimilation problems. However, the ensemble-based data assimilation approaches have limited capability in preserving non-Gaussian distributions of model parameters such as facies (Aanonsen et al.,2009, Villegas et al, 2018, Lorentzen et al, 2013). In the ensemble-based data assimilation techniques, model parameters lose the non- Gaussianity of their original distributions (generated from a geostatistical software) that are initially constrained and constructed to available hard data and the distributions of the model parameters tend towards Gaussian ones (Evensen & Eikrem 2018, Kim et al 2018). Some methods utilised for this parametrisation technique in reservoir characterization literature is, the discrete cosine transform (DCT) (Jafarpour & McLaughlin, 2007)  (Liu & Jafarpour, 2013), level set (Moreno & Aanonsen 2007, Dorn & Villegas 2008, Villegas et al 2018, Chang et al 2010, Lorentzen et al., 2013) and sparse geologic dictionaries (Etienam et al, 2019, Kim et al, 2018, Khaninezhad et al, 2012). In particular, Fourier transform-based methods such as DCT are capable of capturing essential traits such as main shapes and patterns of a facies channel reservoir (Khaninezhad, et al., 2012) but reveal a deficiency in describing a crisp contrast among different facies because of data loss from inverse transformation (Kim et al 2018, Khaninezhad et al, 2012, Tarrahi & Afra,2016).

Reservoir model calibration is applicable and relevant for locating new hydrocarbon deposits and for CCUS stratigraphic trapping initiatives in many energy companies. Energy companies are aiming to accelerate their forward simulation codes for precise and detailed subsurface/dynamic/Petro-physical mappings of their assets/plays. A fascinating and easy to implement meshless approximation to solve reservoir simulation forward problems using physics constrained/informed deep neural networks show promising results.In this project, a physics informed neural operators (PINOs) is developed for surrogating a Two phase flow black oil model and a recently developed weighted adaptive regularised ensemble kalman inversion method is used for solving the inverse problem.


The aim of this project is to develop an integrated workflow, where the finite volume fully/adaptive implicit black oil reservoir simulator is replaced by a phyiscs informed neural operator. This developed PINO surrogate is now used in an inverse problem methodology. Two methods are developed for the inverse problem, a newly developed adaptive regularised ensemble alman inversion method (with various forms of exotic priors). This approach is well suited for forward and inverse uncertainty quantifiication and a gradient based conjugate gradient method with line-search methodlogy and armijio conditions being adhered to.


## Methods for the forward problem (In the weeds):




### Forward problem
**CO2-Brine model**;


### Forward problem
**Black oil model**;

 - Our simplified model for three-phase flow in porous media for reservoir simulation is given as [7].
```math
\begin{equation}
\nabla \cdot \left( \frac{{\rho_w}}{{B_w}} u_w \right) - Q_w = -\frac{{\partial}}{{\partial t}} \left( \frac{{\varphi \rho_w}}{{B_w}} S_w \right) 
\end{equation}
```
```math
\begin{equation}
\nabla \cdot \left( \frac{{\rho_o}}{{B_o}} u_o \right) - Q_o = -\frac{{\partial}}{{\partial t}} \left( \frac{{\varphi \rho_o}}{{B_o}} S_o \right)    
\end{equation}
```

```math
\begin{equation}
\nabla \cdot \left( \frac{{\rho_g}}{{B_g}} u_g + \frac{{R_{so} \rho_g}}{{B_o}} u_o \right) - Q_g = -\frac{{\partial}}{{\partial t}} \left[ \varphi \left( \frac{{\rho_g}}{{B_g}} S_g + \frac{{R_{so} \rho_g}}{{B_o}} S_o \right) \right]
\end{equation}
```



$`φ(x)`$ stands for the porosity,subscript $`w`$ stands for water, subscript $`o`$ stands for oil and subscript $`g`$ stands for gas . $`T_{o},T_{w},T_{g},T`$ stands for the transmissibilities, which are known functions of the permeability $`K`$ the water saturation $`S_w`$ and gas saturation  $`S_g`$. $`u_{\gamma}`$ is the Darcy velocity of each phase.

```math
\begin{equation}
u_\gamma = -\frac{{k_{r\gamma}}}{{\mu_\gamma}} K (\nabla p_\gamma - \rho_\gamma \mathbf{g} \nabla z)
\end{equation}
```

Where:
- $`u_γ`$ is the Darcy velocity of phase γ.
- $`k_{rγ}`$ is the relative permeability of phase γ.
- $`μ_(γ)`$ is the viscosity of phase γ.
- $`B_(γ)`$ is the formation volume factor of phase γ.
- $`K`$ is the permeability.
- $`∇p_{γ}`$ is the gradient of pressure for phase γ.
- $`ρ_(γ)`$ is the density of phase γ.
- $`ĝ`$ is the gravitational vector.
- $`∇z`$ is the gradient of elevation.
- $`R_{so}`$ is the solution gas oil ratio.


```math
\begin{equation}
S_o + S_w + S_g = 1; P_{cwo} = p_o- p_w  ;P_{cog} = p_g- p_o  
\end{equation}
```


This gives six unknowns
```math
\begin{equation}
p_{o}, p_{w},p_{g}, S_{w}, S_{o}, S_{g}
\end{equation}
``` 
Gravity effects are considered by the terms 
```math
\begin{equation}
ρ_{w}gk,ρ_{o}gk\quad Ω \subset R^{n}(n = 2, 3) 
\end{equation}
``` 
The subsequent water, oil ,gas and overall transmissibilities is given by,
```math
\begin{equation} 
T_w = \frac{K(x) K_{rw}S_w}{μ_w}  ;T_o = \frac{K(x) K_{ro}S_o}{μ_o};T_g = \frac{K(x) K_{rg}S_g}{μ_g}   ;T= T_w + T_o + T_g           
\end{equation}
``` 
The relative permeabilities $`K_{rw} (S_w ),K_{ro} (S_w ),K_{rg} (S_o )`$ are available as tabulated functions, and $`μ_w,μ_o,μ_g`$ denote the viscosities of each phase. 

We define the oil flow, the water flow,gas flow and the total flow, respectively, which are measured at the well position as; 
```math
\begin{equation} 
Q = Q_{o}  + Q_{w } + Q_{g }
\end{equation}
``` 

The final pressure , water saturation and gas saturation equations for a three-phase oil-water flow is
```math
\begin{equation} 
- ∇ .[T∇p]=Q   
\end{equation}
```  
```math
\begin{equation}                                               
φ\frac{∂S_w}{∂t} - \nabla .[T_{w} (\nabla.p_{w}]= Q_{w} 
\end{equation}
``` 
```math
\begin{equation}
\nabla \cdot \left( \frac{{\rho_g}}{{B_g}} u_g + \frac{{R_{so} \rho_g}}{{B_o}} u_o \right) - Q_g = -\frac{{\partial}}{{\partial t}} \left[ \varphi \left( \frac{{\rho_g}}{{B_g}} S_g + \frac{{R_{so} \rho_g}}{{B_o}} S_o \right) \right]
\end{equation}
``` 
### Surrogate Forward modelling

**Fourier Neural operator based machine infused with physics constraint from black oil model ansatz**

An FNO model architecture, introduced in [54], is shown below.
![Nvidia-Energy](https://zongyi-li.github.io/assets/img/fourier_layer.png)

The goal is to replace the Finite volume simulator with an FNO surrogate.

For the PINO reservoir modelling [2], we are interested in predicting the pressure, saturation and fluxes given any input of the absolute permeability & porosity field for the pressure and saturation equation. We will introduce an additional (vector) variable, namely flux, F, which turns Eq. (3a) into a system of equations below. 

```math
\begin{equation} 
u=∇p ;    F = T∇p ;   - ∇ · F = Q
\end{equation}
``` 

Using a mixed residual loss formulation, the pressure equation loss ansatz is expressed as,
```math
\begin{equation} 
V(F,u;T)=\int_{Ω}[(F-T∇u)^2 + (- ∇ ·F-Q)^2 ]
\end{equation}
``` 
Ω ⊂ R^n  (n = 2, 3). The discretised pressure, water saturation and gas saturation equation loss then becomes.

```math
\begin{equation} 
V(F,u;T)_{p} ≈ \frac{1}{n_{s}}  (‖F-T⨀∇u‖_{2}^{2} + ‖- ∇ ·F-Q‖_{2}^{2} )       
\end{equation}
``` 
```math
\begin{equation} 
V(u,S_w;t)_{S_{w}} = \frac{1}{n_s}  ‖(φ \frac{∂S_w}{∂t}- ∇ .[T_{w} (∇u)])-Q_w ‖_{2}^{2}     
\end{equation}
``` 

```math
\begin{equation} 
V(u,S_g,S_o;t)_{S_{g}} = \frac{1}{n_s}  ‖ \nabla \cdot \left( \frac{{\rho_g}}{{B_g}} u_g + \frac{{R_{so} \rho_g}}{{B_o}} u_o \right) - Q_g + \frac{{\partial}}{{\partial t}} \left[ \varphi \left( \frac{{\rho_g}}{{B_g}} S_g + \frac{{R_{so} \rho_g}}{{B_o}} S_o \right) \right] ‖_{2}^{2}     
\end{equation}
``` 

```math
\begin{equation} 
Loss_{cfd} =V(F,u;T)_{p} + V(u,S_w;t)_{S_{w}} + V(u,S_g,S_o;t)_{S_{g}}      
\end{equation}
``` 


## Important Dependencies & Prerequisites:
- Nvidia's Modulus symbolic v23.09 :[link](https://github.com/NVIDIA/modulus-sym)
- CUDA 11.8 : [link](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- CuPy : [link](https://github.com/cupy/cupy.git)
- Python 3.8 upwards

## Getting Started:
- These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
- The code is developed in a Linux enviroment.

## Installation - Bare metal / Docker

- From terminal create a new conda enviroment named **MDLO** (check if conda is installed) .
```bash
conda create --name MDLO python=3.8
```

Clone this code base repository in a dedicated **work folder**.
```bash
cd **work folder**
conda activate MDLO
git lfs clone https://github.com/NVIDIA/modulus-sym.git
```
### Bare-metal
- From terminal do these sequence of operations to install Modulus v23.09: [link](https://github.com/NVIDIA/modulus-sym.git)
```bash
pip install nvidia-modulus.sym
             
```
- From terminal, install (missing) dependencies in 'requirements.txt' in the conda enviroment **MDLO**
- Follow instructions to install CuPy from : [link](https://github.com/cupy/cupy.git)



### Docker (Recommended)
- You may need to temporarily deactivate any **vpn** for the docker installation & run
- Note, NVIDIA Container Toolkit must be installed first. This extension enables Docker daemon to treat GPUs properly.

- Please follow the installation instructions in [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

If you encounter a **Conflicting values set for option Signed** error when running apt update as shown below,

```bash
sudo apt-get update
E: Conflicting values set for option Signed-By regarding source https://nvidia.github.io/libnvidia-container/stable/ubuntu18.04/amd64/ /: /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg !=
E: The list of sources could not be read.
```
Do the following
```bash
grep "nvidia.github.io" /etc/apt/sources.list.d/*
grep -l "nvidia.github.io" /etc/apt/sources.list.d/* | grep -vE "/nvidia-container-toolkit.list\$"
```
Delete the file(s) that will be shown from running the command above
```bash
sudo rm -f FILENAME
```
where FILENAME is the name of the file(s) shown above

More Troubleshooting can be found at [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/troubleshooting.html)

```bash
# enable shell script execution.
sudo chmod +x ./scripts/docker/docker-build.sh
sudo chmod +x set_env.sh
# Build docker image
./scripts/docker/docker-build.sh

# enable shell script execution.
sudo chmod +x ./scripts/docker/docker-run.sh
# Run docker container (also enables X server for docker)
./scripts/docker/docker-run.sh
```


### Run
**OPM Flow** is a fully CPU based Black oil reservoir simulator.

[link](https://opm-project.org/?page_id=19 )

#### Forward problem


- Navigate to the code base root directory - **work folder** via terminal.

##### Bare Metal alone
```bash
cd **work folder**
```
- where **work folder** is the location you downloaded the code base to.

- Download the supplemental material.

- Run the Forward Problem surrogation with PINO  via the **src** folder.

##### Bare Metal
```bash
conda activate MDLO 
wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/modulus/modulus_reservoir_simulation_supplemental_material/versions/latest/zip -O modulus_reservoir_simulation_supplemental_material_latest.zip
unzip modulus_reservoir_simulation_supplemental_material_latest.zip
unzip modulus_reservoir_simulation_norne_supplemental_material.zip
cp -r modulus_reservoir_simulation_norne_supplemental_material/* .
cd src
python Forward_problem_PINO.py
python Learn_CCR.py
cd ..
conda deactivate
```

##### Docker
```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/modulus/modulus_reservoir_simulation_supplemental_material/versions/latest/zip -O modulus_reservoir_simulation_supplemental_material_latest.zip
unzip modulus_reservoir_simulation_supplemental_material_latest.zip
unzip modulus_reservoir_simulation_norne_supplemental_material.zip
cp -r modulus_reservoir_simulation_norne_supplemental_material/* .
cd src
python Forward_problem_PINO.py
python Learn_CCR.py
cd ..
```


- Forward problem solution results are found in the root directory folder **outputs**

- Compare the surrogate solution from **PINO** with the finite volume reservoir simulator (**NVRS**) from the **src** folder.



##### Bare Metal
```bash
conda activate MDLO
wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/modulus/modulus_reservoir_simulation_supplemental_material/versions/latest/zip -O modulus_reservoir_simulation_supplemental_material_latest.zip
unzip modulus_reservoir_simulation_supplemental_material_latest.zip
unzip modulus_reservoir_simulation_norne_supplemental_material.zip
cp -r modulus_reservoir_simulation_norne_supplemental_material/* .
cd src
python Compare_FVM_Surrogate.py
cd ..
conda deactivate
```

##### Docker
```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/modulus/modulus_reservoir_simulation_supplemental_material/versions/latest/zip -O modulus_reservoir_simulation_supplemental_material_latest.zip
unzip modulus_reservoir_simulation_supplemental_material_latest.zip
unzip modulus_reservoir_simulation_norne_supplemental_material.zip
cp -r modulus_reservoir_simulation_norne_supplemental_material/* .
cd src
python Compare_FVM_Surrogate.py
cd ..
```

- Results for the comparison are found in the root directory folder **COMPARE_RESULTS**

#### Inverse problem


##### Bare Metal
```bash
conda activate MDLO
wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/modulus/modulus_reservoir_simulation_supplemental_material/versions/latest/zip -O modulus_reservoir_simulation_supplemental_material_latest.zip
unzip modulus_reservoir_simulation_supplemental_material_latest.zip
unzip modulus_reservoir_simulation_norne_supplemental_material.zip
cp -r modulus_reservoir_simulation_norne_supplemental_material/* .
cd src
python Inverse_problem.py
cd ..
conda deactivate
```

##### Docker
```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/modulus/modulus_reservoir_simulation_supplemental_material/versions/latest/zip -O modulus_reservoir_simulation_supplemental_material_latest.zip
unzip modulus_reservoir_simulation_supplemental_material_latest.zip
unzip modulus_reservoir_simulation_norne_supplemental_material.zip
cp -r modulus_reservoir_simulation_norne_supplemental_material/* .
cd src
python Inverse_problem.py
cd ..
```




## Release Notes
**23.03.01**
* 3D implementation with 3D FNO neural architecture
* Gaussian reservoir with variogram analysis
* Multiple point statistics with SNESIM/FILTERSIM
* Bug Fixes

**23.03**
* 2D implementation with 2D FNO neural architecture
* Increase degree of freedom in the number of injector and producer well configuration.
* Weighted scheme during the Kalman gain computation for the aREKI update.
* Diffusion model adjusted and implemented.
* 3D well plots with injection and producer wells animations/movie.
* Variation convolution autoencoder better implemented.
* 2 new PINO implementation accounting for the Overall Flux and water specific flux during the pde loss computation.
* Experimental gradient based method implementd with Adam & LBFGS for comparison purposes only to the aREKI scheme.
* Sensible initial ensemble initialisation from MPS and 6 new training images (TI's).
* Constrained presure residual method for the fully/adaptive implict finitie volume numerical solver implemented. This method has a 2 stage V cyclec AMG, with the coarsening, aggregation, colouring & level-scheduling implemented during the restrcition operation. SOR, Gauss-Seidel, Jacobi implemented during the smoothing operation. The solution from the pressure solve serves as an initialiser for the saturation solver, which is a left-preconditioned GMRES with an ILU(0) preconditoner. 
* Bug Fixes

**23.02**
* Bug Fixes

**23.01**
* First release 

## End User License Agreement (EULA)
Refer to the included Energy SDK License Agreement in **Energy_SDK_License_Agreement.pdf** for guidance.

## Author:
- Clement Etienam- Solution Architect-Energy @Nvidia  Email: cetienam@nvidia.com

## Contributors:
- Kaustubh - Nvidia



## References:
[1] Aanonsen, S., Oliver, D., Reynolds, A. & Valles, B., 2009. The Ensemble Kalman Filter in Reservoir Engineering--a Review. SPE Journal, 14(3), pp. 393-412.

[2] Aharon, M., Elad, M. & Bruckstein, A., 2006. K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation. IEEE Transactions on Signal Processing, 54(11), pp. 4311-4322.

[3] Bishop. Christopher M 2006. Pattern Recognition and Machine Learning (Information Science and Statistics). Springer-Verlag, Berlin, Heidelberg.

[4] Candes, E. J., Romberg, J. & Tao, T., 2006. Robust uncertainty principles: Exact signal reconstruction from high incomplete frequency information. Information Theory, IEEE Transactions on, 52(2), pp. 489-509.

[5] Chang, H., Zhang, D. & Lu, Z., 2010. History matching of facies distribution with the EnKF and level set parameterization. Journal of Computational Physics, 229(1), pp. 8011-8030.

[6] Chen, Y. and Oliver,D., .2011 Ensemble randomized maximum likelihood method as an iterative ensemble smoother, Mathematical Geosciences, Online First.


[7] Dorn, O. & Villegas, R., 2008. History matching of petroleum reservoirs using a level set technique. Inverse problems, p. 24.


[8] Elsheikh, A., Wheeler, M. & Hoteit, I., 2013. Sparse calibration of subsurface flow models using nonlinear orthogonal matching pursuit and an iterative stochastic ensemble method. Advances in Water Resources, 56(1), pp. 14-26.

[9] Emerick, A. A. and Reynolds A. C., 2013 Ensemble smoother with multiple data assimilation, Computers & Geosciences.

[10] Etienam, C., Mahmood, I. & Villegas, R., 2017. History matching of Reservoirs by Updating Fault Properties Using 4D Seismic Results and Ensemble Kalman Filter. Paris, SPE.

[11] Etienam, C., Villegas, R., Babaei, M & Dorn,O 2018. An Improved Reservoir Model Calibration through Sparsity Promoting ES-MDA. ECMI conference proceeding. 20th European Conference on Mathematics for Industry (ECMI) 18-22 June 2018, Budapest, Hungary 

[12] Etienam C. 4D Seismic History Matching Incorporating Unsupervised Learning, Society of Petroleum Engineers B (2019 June), 10.2118/195500-MS

[13] Etienam C., Law, J.H L., Wade, S., Ultra-fast Deep Mixtures of Gaussian Process Experts. https://arxiv.org/abs/2006.13309.

[14] Evensen, G., 2003. The Ensemble Kalman Filter: Theoretical formulation and practical implementation. Ocean Dynamics, 53(4), pp. 343-367.

[15] Evensen, G.,2009. The ensemble Kalman filter for combined state and parameter estimation, IEEE Control Systems Magazine, pp. 83-104.

[16] Evensen, G., Eikrem, K, S 2018. Conditioning reservoir models on rate data using ensemble smoothers.Computational Geosciences,22(5), pp.1251-1270.

[17] Haibin, C. & Zhang, D., 2015. Jointly updating the mean size and spatial distribution of facies in reservoir history matching. Computational Geosciences.

[18] Hanke, M.,1997. A regularizing Levenberg-Marquardt scheme, with applications to inverse groundwater filtration problems. Inverse problems 13(1), pp. 79–95 

[19] Iglesias, M.A., Dawson, C.2013. The regularizing Levenberg-Marquardt scheme for history matching of petroleum reservoirs. Computational Geosciences 17, pp.1033–1053 

[20] Jafarpour, B., 2011. Wavelet reconstruction of geologic facies from nonlinear dynamic flow measurements. Geosciences and Remote sensing, IEEE Transactions, 49(5), pp. 1520-1535.

[21] Jafarpour, B. & McLaughlin, D. B., 2007. History matching with an ensemble Kalman filter and discrete cosine parametrization. Anaheim, California, SPE.

[22] Khaninezhad, M. M., Jafarpour, B. & Li, L., 2012. Sparse geologic dictionaries for subsurface flow model calibration: Part 1, Inversion formulation. Advances in Water Resources, 39(1), pp. 106-121.

[23] Kim S, Min B, Lee K,& Jeong H, 2018 Integration of an Iterative Update of Sparse Geologic Dictionaries with ES-MDA for History Matching of Channelized Reservoirs. Geofluids, Volume 2018, Article ID 1532868, 21 pages.

[24] Law K J H & Stuart A M, 2012 Evaluating Data Assimilation Algorithms Mon.  Weather  Rev 140 37-57

[25] Liu, E. & Jafarpour, B., 2013. Learning sparse geologic dictionaries from low-rank representations of facies connectivity for flow model calibration. Water resources, Volume 49, pp. 7088-7101.


[26] Lorentzen, R. J., Flornes, M. K. & Naevdal, G., 2012. History matching Channelized Reservoirs Using the Ensemble Kalman Filter. Society of Petroleum Engineers, 17(1).


[27] Luo X, Bhakta T, Jakobsen M, Naevdal G, 2018. Efficient big data assimilation through sparse representation: A 3D benchmark case study in petroleum engineering. PLoS ONE 13(7): e0198586. https://doi.org/10.1371/journal.pone.0198586


[28] Moreno, D. L., 2011. Continuous Facies Updating Using the Ensemble Kalman Filter and the Level set method. s.l., Mathematical Geosciences.

[29] Nocedal, J. and Wright, S.J, 1999.Numerical Optimization,  Springer, New York.

[30] Oliver, D. S. & Chen, Y., 2010. Recent progress on reservoir history matching: a review. s.l.: Computational Geoscience - Springer Science.

[31] Oliver, D. S., Reynolds, A. C. & Liu, N., 2008. Inverse Theory for Petroleum Reservoir Characterization and History Matching. s.l.:Cambridge University Press.


[32] Sana, F, Katterbauer, K , Al-Naffouri, T.Y and Hoteit, I.,2016 Orthogonal matching pursuit for enhanced recovery of sparse geological structures with the ensemble Kalman filter, IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 9, no. 4, pp. 1710–1724

[33] Sarma, P., L. J. Durlofsky, and K. Aziz,.2008. Kernel principal component analysis for efficient differentiable parameterization of multipoint geostatistics, Mathematical Geosciences, 40, 3-32, 2008.


[34] Sherman, J., Morrison, W.J.1950. Adjustment of an inverse matrix corresponding to a change in one element of a given matrix. The Annals of Mathematical Statistics 21, 124–127.


[35] Stuart A M ,2010 Inverse problems:  A Bayesian perspective Acta Numerica 19 451-559.

[36] Tarantola, 2005. Inverse problem theory and methods for model parameter estimation. 1 ed. Philadelphia: SIAM.

[37] Tarrahi, M. & Afra, S., 2016. Improved Geological model calibration through Sparsity-promoting Ensemble Kalman Filter. Offshore Technology Conference.

[38] Tropp, J. A. & Gilbert, C. A., 2007. Signal recovery from random measurements via orthogonal matching pursuit. Information Theory, IEEE Transactions on, 53(12), pp. 4655-4666.


[39] Villegas, R., Etienam, C., Dorn, O., & Babaei, M.,2018. The shape and Distributed Parameter Estimation for History Matching using a Modified Ensemble Kalman Filter and Level Sets. ;  Inverse problems Science and Engineering.


[40] William P, Mitchell J,1993 JPEG: Still Image Data Compression Standard, Van Nostrand Reinhold, 

[41] Wu, J., Boucher, A. & Journel, G. A., 2006. A 3D code for mp simulation of continuous and categorical variables: FILTERSIM. SPE.

[42] Zhou, H,   Li, L and. Gómez-Hernández, J.,2012, “Characterizing curvilinear features using the localized normal-score ensemble Kalman filter,” Abstract and Applied Analysis, vol. 2012, Article ID 805707, 18 pages.

[43] L. Yang, D. Zhang, G. E. Karniadakis, Physics-informed generative adversarial networks for stochastic differential equations, arXiv preprint arXiv:1811.02033

[43] J. Adler, O. Oktem, solving ill-posed inverse problems using iterative ¨ deep neural networks, Inverse Problems 33 (12) (2017) 124007. URL http://stacks.iop.org/0266-5611/33/i=12/a=124007

[44] J.-Y. Zhu, R. Zhang, D. Pathak, T. Darrell, A. A. Efros, O. Wang, E. Shechtman, Toward multimodal image-to-image translation, in Advances in Neural Information Processing Systems, 2017, pp. 465–476.

[45] S. Rojas, J. Koplik, Nonlinear flow in porous media, Phys. Rev. E 58 (1998) 4776–4782.doi:10.1103/PhysRevE.58.4776.URL,https://link.aps.org/doi/10.1103/PhysRevE.58.4776

[46] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, Y. Bengio, Generative adversarial nets, in: Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, K. Q. Weinberger (Eds.), Advances in Neural Information Processing Systems 27, Curran Associates, Inc., 2014, pp. 2672–2680. URL http://papers.nips.cc/paper/5423-generative-adversarial-nets. pdf

[47] M. Raissi, P. Perdikaris, G. Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational Physics 378 (2019) 686 – 707. doi:https://doi.org/10.1016/j.jcp.2018.10.045.URL,http://www.sciencedirect.com/science/article/pii/ S0021999118307125

[48] M. Raissi, Forward-backward stochastic neural networks: Deep learning of high-dimensional partial differential equations, arXiv preprint arXiv:1804.07010

[49] M. Raissi, P. Perdikaris, G. E. Karniadakis, Physics Informed Deep Learning (Part I): Data-driven solutions of nonlinear partial differential equations, arXiv preprint arXiv:1711.10561

[50] Bishop. Christopher M 2006. Pattern Recognition and Machine Learning (Information Science and Statistics). Springer-Verlag, Berlin, Heidelberg.

[51] Dorn O & Villegas R, 2008 History matching of petroleum reservoirs using a level set technique, Inverse problems Volume: 24   Issue: 3 Article Number: 035015 

[52] Hansen, T.M., Vu. L.T., and Bach, T. 2016. MPSLIB: A C++ class for sequential simulation of multiple-point statistical models, in Softwar X, doi:10.1016/j.softx.2016.07.001. [pdf,www].

[53] David E. Bernholdt, Mark R. Cianciosa, David L. Green, Jin M. Park, Kody J. H. Law, and Clement Etienam. Cluster, classify, regress: A general method for learning discontinuous functions. Foundations of Data Science, 1(2639-8001-2019-4-491):491, 2019.

[54] Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar. Fourier Neural Operator for Parametric Partial Differential Equations. https://doi.org/10.48550/arXiv.2010.08895

[55] Zongyi Li, Hongkai Zheng, Nikola Kovachki, David Jin, Haoxuan Chen, Burigede Liu, Kamyar Azizzadenesheli, Anima Anandkumar.Physics-Informed Neural Operator for Learning Partial Differential Equations. https://arxiv.org/pdf/2111.03794.pdf

