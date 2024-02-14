# C02 -Brine surrogate computed with a Physics Informed Neural Operator (PINO) 
![alt text](Visuals/All1.png)


## An AI enabled Automatic History Matching Workflow with a PINO based forward solver:
Calibration of subsurface structures is an important step to forecast fluid dynamics and behaviours in a plethora of geoenvironments such as petroleum and geothermal reservoirs. History matching is an ill-posed inverse process to find reservoir model parameters honouring observations by integration of static (e.g., core, logging, and seismic) and dynamic data (e.g., oil and gas rate, water cut, bottomhole pressure, and subsidence/uplift) .In recent developments, the ensemble Kalman filter (EnKF) (Evensen 1994, Law et al 2012, Stuart 2010), ensemble smoother (ES) (van Leeuwen & Evensen 1996) ES with multiple data assimilation (ES-MDA) (Emerick & Reynolds 2013) and Iterative variants of ES (Chen & Oliver 2011, Iglesias 2013) have all been utilised for history matching/data assimilation problems. However, the ensemble-based data assimilation approaches have limited capability in preserving non-Gaussian distributions of model parameters such as facies (Aanonsen et al.,2009, Villegas et al, 2018, Lorentzen et al, 2013). In the ensemble-based data assimilation techniques, model parameters lose the non- Gaussianity of their original distributions (generated from a geostatistical software) that are initially constrained and constructed to available hard data and the distributions of the model parameters tend towards Gaussian ones (Evensen & Eikrem 2018, Kim et al 2018). Some methods utilised for this parametrisation technique in reservoir characterization literature is, the discrete cosine transform (DCT) (Jafarpour & McLaughlin, 2007)  (Liu & Jafarpour, 2013), level set (Moreno & Aanonsen 2007, Dorn & Villegas 2008, Villegas et al 2018, Chang et al 2010, Lorentzen et al., 2013) and sparse geologic dictionaries (Etienam et al, 2019, Kim et al, 2018, Khaninezhad et al, 2012). In particular, Fourier transform-based methods such as DCT are capable of capturing essential traits such as main shapes and patterns of a facies channel reservoir (Khaninezhad, et al., 2012) but reveal a deficiency in describing a crisp contrast among different facies because of data loss from inverse transformation (Kim et al 2018, Khaninezhad et al, 2012, Tarrahi & Afra,2016).

Reservoir model calibration is applicable and relevant for locating new hydrocarbon deposits and for CCUS stratigraphic trapping initiatives in many energy companies. Energy companies are aiming to accelerate their forward simulation codes for precise and detailed subsurface/dynamic/Petro-physical mappings of their assets/plays. A fascinating and easy to implement meshless approximation to solve reservoir simulation forward problems using physics constrained/informed deep neural networks show promising results.In this project, a physics informed neural operators (PINOs) is developed for surrogating a Two phase flow black oil model and a recently developed weighted adaptive regularised ensemble kalman inversion method is used for solving the inverse problem.


The aim of this project is to develop an integrated workflow, where the finite volume fully/adaptive implicit black oil reservoir simulator is replaced by a phyiscs informed neural operator. This developed PINO surrogate is now used in an inverse problem methodology. Two methods are developed for the inverse problem, a newly developed adaptive regularised ensemble alman inversion method (with various forms of exotic priors). This approach is well suited for forward and inverse uncertainty quantifiication and a gradient based conjugate gradient method with line-search methodlogy and armijio conditions being adhered to.


## Methods for the forward problem (In the weeds):




### Forward problem
**CO2-Brine model**;


.. _CO2-EOS:

##################################################################################
CO2-brine model 
##################################################################################


Summary
=======

The CO2-brine model implemented in GEOS includes two components (CO2 and H2O) that are transported by one or two fluid phases (the brine phase and the CO2 phase).
We refer to the brine phase with the subscript :math:`\ell` and to the CO2 phase with the subscript :math:`g` (although the CO2 phase can be in supercritical, liquid, or gas state).
The water component is only present in the brine phase, while the CO2 component can be present in the CO2 phase as well as in the brine phase.
Thus, considering the molar phase component fractions, :math:`y_{c,p}` (i.e., the fraction of the molar mass of phase :math:`p` represented by component :math:`c`) the following partition matrix determines the component distribution within the two phases:

.. math::
    \begin{bmatrix}
    y_{H2O,\ell} & y_{CO2,\ell} \\
         0 & 1            \\
    \end{bmatrix}

The update of the fluid properties is done in two steps:

1) The phase fractions (:math:`\nu_p`) and phase component fractions (:math:`y_{c,p}`) are computed as a function of pressure (:math:`p`), temperature (:math:`T`), component fractions (:math:`z_c`), and a constant salinity.

2) The phase densities (:math:`\rho_p`) and phase viscosities (:math:`\mu_p`) are computed as a function of pressure, temperature, the updated phase component fractions, and a constant salinity.

Once the phase fractions, phase component fractions, phase densities, phase viscosities--and their derivatives with respect to pressure, temperature, and component fractions--have been computed, the :ref:`CompositionalMultiphaseFlow` proceeds to the assembly of the accumulation and flux terms.
Note that the current implementation of the flow solver is isothermal and that the derivatives with respect to temperature are therefore discarded.

The models that are used in steps 1) and 2) are reviewed in more details below.

Step 1: Computation of the phase fractions and phase component fractions (flash)
================================================================================

At initialization, GEOS performs a preprocessing step to construct a two-dimensional table storing the values of CO2 solubility in brine as a function of pressure, temperature, and a constant salinity.
The user can parameterize the construction of the table by specifying the salinity and by defining the pressure (:math:`p`) and temperature (:math:`T`) axis of the table in the form:

+------------+---------------+-----------------+-----------------+------------------+-----------------+-----------------+------------------+----------+
| FlashModel | CO2Solubility | :math:`p_{min}` | :math:`p_{max}` | :math:`\Delta p` | :math:`T_{min}` | :math:`T_{max}` | :math:`\Delta T` | Salinity | 
+------------+---------------+-----------------+-----------------+------------------+-----------------+-----------------+------------------+----------+

Note that the pressures are in Pascal, temperatures are in Kelvin, and the salinity is a molality (moles of NaCl per kg of brine). 
The temperature must be between 283.15 and 623.15 Kelvin.
The table is populated using the model of Duan and Sun (2003).
Specifically, we solve the following nonlinear CO2 equation of state (equation (A1) in Duan and Sun, 2003) for each pair :math:`(p,T)` to obtain the reduced volume, :math:`V_r`.

.. math::
   \frac{p_r V_r}{T_r} &= 1 + \frac{a_1 + a_2/T^2_r + a_3/T^3_r}{V_r} 
   + \frac{a_4 + a_5/T^2_r + a_6/T^3_r}{V^2_r} + \frac{a_7 + a_8/T^2_r + a_9/T^3_r}{V^4_r} \\
   &+ \frac{a_{10} + a_{11}/T^2_r + a_{12}/T^3_r}{V^5_r} 
   + \frac{a_{13}}{T^3_r V^2_r} \big( a_{14} + \frac{a_{15}}{V^2_r} \big) \exp( - \frac{a_{15}}{V^2_r} )

where :math:`p_r = p / p_{crit}` and :math:`T_r = T / T_{crit}` are respectively the reduced pressure and the reduced temperature.
We refer the reader to Table (A1) in Duan and Sun (2003) for the definition of the coefficients :math:`a_i` involved in the previous equation. 
Using the reduced volume, :math:`V_r`, we compute the fugacity coefficient of CO2, :math:`\ln_{\phi}(p,T)`, using equation (A6) of Duan and Sun (2003).
To conclude this preprocessing step, we use the fugacity coefficient of CO2 to compute and store the solubility of CO2 in brine, :math:`s_{CO2}`, using equation (6) of Duan and Sun (2003):

.. math::
   \ln \frac{ y_{CO2} P }{ s_{CO2} } = \frac{\Phi_{CO2}}{RT} - \ln_{\phi}(p,T) + \sum_c 2 \lambda_c m + \sum_a 2 \lambda_a m + \sum_{a,c} \zeta_{a,c} m^2

where :math:`\Phi_{CO2}` is the chemical potential of the CO2 component, :math:`R` is the gas constant, and :math:`m` is the salinity.
The mole fraction of CO2 in the vapor phase, :math:`y_{CO2}`, is computed with equation (4) of Duan and Sun (2003).
Note that the first, third, fourth, and fifth terms in the equation written above are approximated using equation (7) of Duan and Sun (2003) as recommended by the authors.

During the simulation, Step 1 starts with a look-up in the precomputed table to get the CO2 solubility, :math:`s_{CO2}`, as a function of pressure and temperature.
Then, we compute the phase fractions as:

.. math::
   \nu_{\ell} &= \frac{1 + s_{CO2}}{1 + z_{CO2} / ( 1 - z_{CO2} ) } \\
   \nu_{g} &= 1 - \nu_{\ell}

We conclude Step 1 by computing the phase component fractions as:

.. math::
   y_{CO2,\ell} &= \frac{ s_{CO2} }{ 1 + s_{CO2} } \\
   y_{H2O,\ell} &= 1 - y_{CO2,\ell} \\
   y_{CO2,g} &= 1 \\
   y_{H2O,g} &= 0 
    
   
Step 2: Computation of the phase densities and phase viscosities
================================================================

CO2 phase density and viscosity
-------------------------------

In GEOS, the computation of the CO2 phase density and viscosity  is entirely based on look-up in precomputed tables.
The user defines the pressure (in Pascal) and temperature (in Kelvin) axis of the density table in the form:

+------------+----------------------+-----------------+-----------------+------------------+-----------------+-----------------+------------------+
| DensityFun | SpanWagnerCO2Density | :math:`p_{min}` | :math:`p_{max}` | :math:`\Delta p` | :math:`T_{min}` | :math:`T_{max}` | :math:`\Delta T` |
+------------+----------------------+-----------------+-----------------+------------------+-----------------+-----------------+------------------+

This correlation is valid for pressures less than :math:`8 \times 10^8` Pascal and temperatures less than 1073.15 Kelvin.  
Using these parameters, GEOS internally constructs a two-dimensional table storing the values of density as a function of pressure and temperature.
This table is populated as explained in the work of Span and Wagner (1996) by solving the following nonlinear Helmholtz energy equation for each pair :math:`(p,T)` to obtain the value of density, :math:`\rho_{g}`:

.. math::
   \frac{p}{RT\rho_{g}} = 1 + \delta \phi^r_{\delta}( \delta, \tau )

where :math:`R` is the gas constant, :math:`\delta := \rho_{g} / \rho_{crit}` is the reduced CO2 phase density, and :math:`\tau := T_{crit} / T` is the inverse of the reduced temperature.
The definition of the residual part of the energy equation, denoted by :math:`\phi^r_{\delta}`, can be found in equation (6.5), page 1544 of Span and Wagner (1996).
The coefficients involved in the computation of :math:`\phi^r_{\delta}` are listed in Table (31), page 1544 of Span and Wagner (1996).   
These calculations are done in a preprocessing step.

The pressure and temperature axis of the viscosity table can be parameterized in a similar fashion using the format:

+--------------+----------------------+-----------------+-----------------+------------------+-----------------+-----------------+------------------+
| ViscosityFun | FenghourCO2Viscosity | :math:`p_{min}` | :math:`p_{max}` | :math:`\Delta p` | :math:`T_{min}` | :math:`T_{max}` | :math:`\Delta T` |
+--------------+----------------------+-----------------+-----------------+------------------+-----------------+-----------------+------------------+

This correlation is valid for pressures less than :math:`3 \times 10^8` Pascal and temperatures less than 1493.15 Kelvin.  
This table is populated as explained in the work of Fenghour and Wakeham (1998) by computing the CO2 phase viscosity, :math:`\mu_g`, as follows:

.. math::
   \mu_{g} = \mu_{0}(T) + \mu_{excess}( \rho_{g}, T ) + \mu_{crit}( \rho_{g}, T )  
   
The "zero-density limit" viscosity, :math:`\mu_{0}(T)`, is computed as a function of temperature using equations (3), (4), and (5), as well as Table (1) of Fenghour and Wakeham (1998).
The excess viscosity, :math:`\mu_{excess}( \rho_{g}, T )`, is computed as a function of temperature and CO2 phase density (computed as explained above) using equation (8) and Table (3) of Fenghour and Wakeham (1998).
We currently neglect the critical viscosity, :math:`\mu_{crit}`.
These calculations are done in a preprocessing step.

During the simulation, the update of CO2 phase density and viscosity is simply done with a look-up in the precomputed tables. 

Brine density and viscosity using Phillips correlation
-------------------------------------------------------

The computation of the brine density involves a tabulated correlation presented in Phillips et al. (1981). 
The user specifies the (constant) salinity and defines the pressure and temperature axis of the brine density table in the form:

+------------+----------------------+-----------------+-----------------+------------------+-----------------+-----------------+------------------+----------+
| DensityFun | PhillipsBrineDensity | :math:`p_{min}` | :math:`p_{max}` | :math:`\Delta p` | :math:`T_{min}` | :math:`T_{max}` | :math:`\Delta T` | Salinity | 
+------------+----------------------+-----------------+-----------------+------------------+-----------------+-----------------+------------------+----------+

The pressure must be in Pascal and must be less than :math:`5 \times 10^7` Pascal.
The temperature must be in Kelvin and must be between 283.15 and 623.15 Kelvin.
The salinity is a molality (moles of NaCl per kg of brine).
Using these parameters, GEOS performs a preprocessing step to construct a two-dimensional table storing the brine density, :math:`\rho_{\ell,table}` for the specified salinity as a function of pressure and temperature using the expression:

.. math::
 
   \rho_{\ell,table} &= A + B x + C x^2 + D x^3 \\
   x &= c_1 \exp( a_1 m ) + c_2 \exp( a_2 T ) + c_3 \exp( a_3 P )

We refer the reader to Phillips et al. (1981), equations (4) and (5), pages 14 and 15 for the definition of the coefficients involved in the previous equation.
This concludes the preprocessing step.

Then, during the simulation, the brine density update proceeds in two steps.
First, a table look-up is performed to retrieve the value of density, :math:`\rho_{\ell,table}`.
Then, in a second step, the density is modified using the method of Garcia (2001) to account for the presence of CO2 dissolved in brine as follows:

.. math::

   \rho_{\ell} = \rho_{\ell,table} + M_{CO2} c_{CO2} - c_{CO2} \rho_{\ell,table} V_{\phi}

where :math:`M_{CO2}` is the molecular weight of CO2, :math:`c_{CO2}` is the concentration of CO2 in brine, and :math:`V_{\phi}` is the apparent molar volume of dissolved CO2.
The CO2 concentration in brine is obtained as:

.. math::

   c_{CO2} = \frac{y_{CO2,\ell} \rho_{\ell,table}}{M_{H2O}(1-y_{CO2,\ell})} 

where :math:`M_{H2O}` is the molecular weight of water. 
The apparent molar volume of dissolved CO2 is computed as a function of temperature using the expression:

.. math::

   V_{\phi} = 37.51 - 9.585 \times 10^{-2} T + 8.740 \times 10^{-4} T^2 - 5.044 \times 10^{-7} T^3

The brine viscosity is controlled by a salinity parameter provided by the user in the form:

+--------------+------------------------+----------+
| ViscosityFun | PhillipsBrineViscosity | Salinity |
+--------------+------------------------+----------+

During the simulation, the brine viscosity is updated as a function of temperature using the analytical relationship of Phillips et al. (1981):

.. math::
   \mu_{\ell} = a T + b

where the coefficients :math:`a` and :math:`b` are defined as:

.. math::
   a &= \mu_{w}(T) \times 0.000629 (1.0 - \exp( -0.7 m ) ) \\
   b &= \mu_{w}(T) (1.0 + 0.0816 m + 0.0122 m^2 + 0.000128 m^3) 
   
where :math:`\mu_{w}` is the pure water viscosity computed as a function of temperature,
and :math:`m` is the user-defined salinity (in moles of NaCl per kg of brine).


Brine density and viscosity using Ezrokhi correlation
-------------------------------------------------------

Brine density :math:`\rho_l` is computed from pure water density :math:`\rho_w` at specified pressure and temperature corrected by Ezrokhi correlation presented in Zaytsev and Aseyev (1993):

.. math::
   log_{10}(\rho_l) &= log_{10}(\rho_w(P, T)) + A(T) x_{CO2,\ell} \\
   A(T) &= a_0 + a_1T +  a_2T^2,

where :math:`a_0, a_1, a_2` are correlation coefficients defined by user:

+------------+----------------------+-------------+-------------+-------------+
| DensityFun | EzrokhiBrineDensity  | :math:`a_0` | :math:`a_1` | :math:`a_2` |
+------------+----------------------+-------------+-------------+-------------+

While :math:`x_{CO2,\ell}` is mass fraction of CO2 component in brine, computed from molar fractions as

.. math::
   x_{CO2,\ell} = \frac{M_{CO2}y_{CO2,\ell}}{M_{CO2}y_{CO2,\ell} + M_{H2O}y_{H2O,\ell}},

Pure water density is computed according to:

.. math::
   \rho_w = \rho_{w,sat}(T) e^{c_w * (P-P_{w,sat}(T))},

where :math:`c_w` is water compressibility defined as a constant :math:`4.5 \times 10^{-10} Pa^{-1}`, while :math:`\rho_{w,sat}(T)` and :math:`P_{w,sat}(T)` are density and pressure of saturated water at a given temperature.
Both are obtained through internally constructed tables tabulated as functions of temperature and filled with the steam table data from Engineering ToolBox (2003, 2004).

Brine viscosity :math:`\mu_{\ell}` is computed from pure water viscosity :math:`\mu_w` similarly:

.. math::
   log_{10}(\mu_l) &= log_{10}(\mu_w(P, T)) + B(T) x_{CO2,\ell} \\
   B(T) &= b_0 + b_1T +  b_2T^2,

where :math:`b_0, b_1, b_2` are correlation coefficients defined by user:

+--------------+------------------------+-------------+-------------+-------------+
| ViscosityFun | EzrokhiBrineViscosity  | :math:`b_0` | :math:`b_1` | :math:`b_2` |
+--------------+------------------------+-------------+-------------+-------------+

Mass fraction of CO2 component in brine :math:`x_{CO2,\ell}` is exactly as in density calculation. The dependency of pure water viscosity from pressure is ignored, and it is approximated as saturated pure water viscosity:

.. math::
   \mu_w(P, T) = \mu_{w, sat} (T),

which is tabulated using internal table as a function of temperature based on steam table data Engineering ToolBox (2004).
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

