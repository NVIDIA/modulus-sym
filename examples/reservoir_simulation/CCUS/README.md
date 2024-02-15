<!-- markdownlint-disable -->
# <b>A Modulus based CO<sub>2</sub>-Brine Physics constrained Neural Operator Forward Model</b>

Clement Etienam\*<sup>[^1]</sup> 
# <a name="_toc132706283"></a>**1. Forward Problem.**
## <a name="_toc132706284"></a>**1.1 Governing Equations** 
The governing equations for a CO<sub>2</sub>-Brine system is given by. 

```math
\begin{equation}
\phi \frac{\partial}{\partial t} \left( \sum_{l} \rho_l y_{cl} S_l \right) - \nabla \cdot \mathbf{k} \left( \sum_{l} \rho_l y_{cl} \lambda_l u_l \right) - \sum_{l} \rho_l y_{cl} q_l = 0

\end{equation}
```

```math
\begin{equation}
u_l = -k\lambda_l \nabla\Theta_l = -k\lambda_l \left( \nabla(p - P_{cl}) - \rho_l g\nabla z \right)
\end{equation}
```

```math
\begin{equation}
\lambda_l = \frac{K_{rl}}{\mu_l}
\end{equation}
```

```math
\begin{equation}
\phi \frac{\partial}{\partial t} \left( \sum_{l} \rho_l y_{cl} S_l \right) - \nabla \cdot k \left( \sum_{l} \rho_l y_{cl} \lambda_l \nabla\Theta_l \right) - \sum_{l} \rho_l y_{cl} q_l = 0
\end{equation}
```

Where: $`l`$ is the phase (brine/gas), $`k`$ is the rock absolute permeability, $`\lambda_l`$ is the phase mobility ratio, $`\mu_l`$ is the phase viscosity, $`K_{rl}`$ is the phase relative permeability, $`S_l`$ is the phase saturation, $`u_l`$ is the phase Darcy velocity, $`g`$ is the acceleration due to gravity, $`z`$ is the depth, $`y_{c,l}`$ is the mass fraction of component $`c`$ in phase $`l`$, $t`$ is time, and $`p`$ is the pressure.

The system is closed by assuming,
```math
\begin{equation}
\sum_{l} S_l = 1, \quad Z_c = \frac{\rho_c}{\rho_T}, \quad \rho_T = \sum_{c} \rho_c
\end{equation}
```


```math
\begin{equation}
S_l = v_l \frac{\rho_T}{\rho_l}, \quad S_g = v_g \frac{\rho_T}{\rho_g}
\end{equation}
```

## **1.2 Thermodynamic Equations** 
The CO$`_2`$-brine model includes two components (CO$`_2`$ and H$`_2`$O) that are transported by one or two fluid phases (the brine phase and the CO$`_2`$ phase). We refer to the brine phase with the subscript $`l`$ and to the CO$`_2`$ phase with the subscript $`g`$ (although the CO$`_2`$ phase can be in supercritical, liquid, or gas state). The water component is only present in the brine phase, while the CO$`_2`$ component can be present in the CO$`_2`$ phase as well as in the brine phase. Thus, considering the molar phase component fractions, $`y_{c,p}`$ (i.e., the fraction of the molar mass of phase $`p`$ represented by component $`c`$).

The update of the fluid properties is done in two steps:

1) The phase fractions ($`v_p`$) and phase component fractions ($`y_{c,p}`$) are computed as a function of pressure ($`p`$), temperature ($`T`$), component fractions ($`z_c`$), and a constant salinity.
2) The phase densities ($`\rho_p`$) and phase viscosities ($`\mu_p`$) are computed as a function of pressure, temperature, the updated phase component fractions, and a constant salinity.

Once the phase fractions, phase component fractions, phase densities, phase viscosities--and their derivatives to pressure, temperature, and component fractions--have been computed, the.

Note that the current implementation of the flow solver is isothermal and that the derivatives to temperature are therefore discarded.

The models that are used in steps 1) and 2) are reviewed in more detail below.


### **1.2.1 Computation of the phase fractions and phase component fractions (flash)** 
We compute the values of CO$`_2`$ solubility in brine as a function of pressure, temperature, and a constant salinity. We define the pressure ($`p`$) and temperature ($`T`$):

Note that the pressures are in Pascal, temperatures are in Kelvin, and the salinity is a molality (moles of NaCl per kg of brine). The temperature must be between 283.15 and 623.15 Kelvin. The table is populated using the model of Duan and Sun (2003). Specifically, we solve the following nonlinear CO$`_2$ equation of state (Duan and Sun, 2003) for each pair to obtain the reduced volume,

```math
\begin{equation}
Z = \left( \frac{p_r V_r}{T_r} \right) = 1 + \frac{ (a_1 + \frac{a_2}{T_r^2} + \frac{a_3}{T_r^3})}{V_r} + \frac{(a_4 + \frac{a_5}{T_r^2} + \frac{a_6}{T_r^3})}{(V_r^2)} + \frac{(a_7 + \frac{a_8}{T_r^2} + \frac{a_9}{T_r^3})}{(V_r^4)} + \frac{(a_{10} + \frac{a_{11}}{T_r^2} + \frac{a_{12}}{T_r^3})}{(V_r^5)} + \frac{a_{13}}{(T_r^3 V_r^2)} \left( a_{14} + a_{15} \left( \frac{1}{V_r^2} \right) \right) \exp \left( -a_{15} \left( \frac{1}{V_r^2} \right) \right)
\end{equation}
```

Where $`p_r = \frac{p}{p_{crit}}`$ is the reduced pressure and the reduced temperature $`T_r = \frac{T}{T_{crit}}`$. The coefficients $`a_1, a_2, \ldots, a_{15}`$ are given as:

$`a_1 = 8.99288497 \times 10^{-2}`$, $`a_2 = -4.94783127 \times 10^{-1}`$, $`a_3 = 4.77922245 \times 10^{-2}`$, $`a_4 = 1.03808883 \times 10^{-2}`$, $`a_5 = -2.82516861 \times 10^{-2}`$, $`a_6 = 9.49887563 \times 10^{-2}`$, $`a_7 = 5.20600880 \times 10^{-4}`$, $`a_8 = -2.93540971 \times 10^{-4}`$, $`a_9 = -1.77265112 \times 10^{-3}`$, $`a_{10} = -2.51101973 \times 10^{-5}`$, $`a_{11} = 8.93353441 \times 10^{-5}`$, $`a_{12} = 7.88998563 \times 10^{-5}`$, $`a_{13} = -1.66727022 \times 10^{-2}`$, $`a_{14} = 1.39800000`$, $`a_{15} = 2.96000000 \times 10^{-2}`$.

Using the reduced volume, $`V_r`$, we compute the fugacity coefficient of CO$`_2`$,

```math
\begin{equation}
\ln \phi (T,P) = Z - 1 - \ln Z + \frac{(a_1 + \frac{a_2}{T_r^2} + \frac{a_3}{T_r^3})}{V_r} + \frac{(a_4 + \frac{a_5}{T_r^2} + \frac{a_6}{T_r^3})}{(2V_r^2)} + \frac{(a_7 + \frac{a_8}{T_r^2} + \frac{a_9}{T_r^3})}{(4V_r^4)} + \frac{(a_{10} + \frac{a_{11}}{T_r^2} + \frac{a_{12}}{T_r^3})}{(5V_r^5)} + \frac{a_{13}}{(2T_r^3 V_r^2)} \left[ a_{14} + 1 - \left( a_{14} + 1 + \frac{a_{15}}{(V_r^2)} \right) \right] \exp \left( -\frac{a_{15}}{(V_r^2)} \right)
\end{equation}
```

To conclude, we use the fugacity coefficient of CO$`_2`$ to compute and store the solubility of CO$`_2`$ in brine, $`s_{\text{CO}_2}`$,

```math
\begin{equation}
\ln \frac{y_{\text{CO}_2}}{s_{\text{CO}_2}} P = \frac{\Phi_{\text{CO}_2}}{RT} - \ln \phi (T,P) + \sum_{c} 2\lambda_c m + \sum_{a} 2\lambda_a m + \sum_{(a,c)} \xi_{(a,c)} m^2
\end{equation}
```

Where $`\Phi_{\text{CO}_2}`$ is the chemical potential of the CO$`_2`$ component, $`R`$ is the gas constant, and $`m`$ is the salinity. The mole fraction of CO$`_2`$ in the vapor phase, $`y_{\text{CO}_2} = \left( \frac{p - p_{\text{H}_2\text{O}}}{p} \right)`$.

Then, we compute the phase fractions as:
```math
\begin{equation}
v_l = \frac{(1 + s_{\text{CO}_2})}{\left( 1 + \frac{z_{\text{CO}_2}}{(1-z_{\text{CO}_2})} \right)}

\quad v_g = 1 - v_l

\end{equation}
```

We conclude by computing the phase component fractions as:
```math
\begin{equation}

y_{\text{CO}_2,l} = \frac{s_{\text{CO}_2}}{(1 + s_{\text{CO}_2})}, \quad y_{\text{H}_2\text{O},l} = 1 - y_{\text{CO}_2,l}

\quad y_{\text{CO}_2,g} = 1, \quad y_{\text{H}_2\text{O},g} = 0

\end{equation}
```


### **1.2.2 Computation of the phase densities and phase viscosities**

**<u>CO<sub>2</sub></u> phase density and viscosity**


The nonlinear Helmholtz energy equation yields, 

```math
\begin{equation}
\frac{P}{(RT\rho_g)} = 1 + \psi \phi_\psi^r (\psi,\tau)
\end{equation}
```

Where, 
```math
\begin{equation}
\psi = \frac{\rho_g}{\rho_{crit}}, \quad \tau = \frac{T_{crit}}{T}
\end{equation}
```

```math
\begin{equation}
\mu_g = \mu_o (T) + \mu_{excess} (\rho_g,T)
\end{equation}
```

```math
\begin{equation}
\mu_{excess} (\rho_g,T) = d_1 \rho_g + d_2 \rho_g^2 + \frac{d_3 \rho_g^6}{T^3} + d_4 \rho_g^8 + \frac{d_5 \rho_g^8}{T}
\end{equation}
```
Where,
$`d_1 = 0.4071119 \times 10^{-2}`$ , $`d_2 = 0.7198037 \times 10^{-4}`$ , $`d_3 = 0.2411697 \times 10^{-16}`$ , $`d_4 = 0.2971072 \times 10^{-22}`$ , $`d_5 = -0.1627888 \times 10^{-22}`$ 

```math
\begin{equation}
\mu_o (T) = \frac{1.00697T^{0.5}}{B^*(T^*)}, \quad \ln B^*(T^*) = \sum_{i=0}^{4} x_i (\ln T^*)^i, \quad T^* = \omega T, \quad \omega = \frac{1}{251.196 \text{ K}}
\end{equation}
```
Where,
$`x_0 = 0.235156`$ , $`x_1 = -0.491266`$ , $`x_2 = 5.211155 \times 10^{-2}`$ , $`x_3 = 5.347906 \times 10^{-2}`$ , $`x_4 = -1.537102 \times 10^{-2}`$ 

**<u>Brine phase density and viscosity</u>**


```math
\begin{equation}
\rho_{(l,table)} = A_1 + A_2 x + A_3 x^2 + A_4 x^3
\end{equation}
```

```math
\begin{equation}
x = c_1 \exp(a_1 m) + c_2 \exp(a_2 T) + c_3 \exp(a_3 P)
\end{equation}
```

```math
\begin{equation}
\rho_l = \rho_{(l,table)} + M_{\text{CO}_2} C_{\text{CO}_2} - C_{\text{CO}_2} \rho_{(l,table)} V_\phi
\end{equation}
```

```math
\begin{equation}
C_{\text{CO}_2} = \frac{y_{\text{CO}_2,l} \rho_{(l,table)}}{M_{\text{H}_2\text{O}} (1-y_{\text{CO}_2,l})}
\end{equation}
```

```math
\begin{equation}
V_\phi = 37.51 - (T \times 9.585 \times 10^{-2}) + (T^2 \times 8.740 \times 10^{-4}) - (T^3 \times 5.044 \times 10^{-7})
\end{equation}
```

```math
\begin{equation}
\mu_l = a_z T + b_z
\end{equation}
```

```math
\begin{equation}
a_z = \mu_w (T) \times 0.000629(1 - \exp(-0.7m))
\end{equation}
```

```math
\begin{equation}
b_z = \mu_w (T)(1 + 0.0816m + 0.0122m^2 + 0.000128m^3)
\end{equation}
```



# <b>2. Physics Constrained Neural operator for the CO<sub>2</sub>-Brine case</b>

## **2.1 Overall discretized equations loss**

The physics loss *ansatz* is then,

The physics loss ansatz is then,
```math
\begin{equation}
V(q_g,p;\lambda_g)_{\text{pressure equation},\text{CO}_2,g} = \frac{1}{n_s} \left\| \nabla \cdot k(\rho_g y_{\text{CO}_2,g} \lambda_g \nabla(p - P_{\text{CO}_2,g})) - \rho_g y_{\text{CO}_2,g} q_g \right\|_2^2
\end{equation}
```

```math
\begin{equation}
V(q_l,p;\lambda_l)_{\text{pressure equation},\text{CO}_2,l} = \frac{1}{n_s} \left\| \nabla \cdot k(\rho_l y_{\text{CO}_2,l} \lambda_l \nabla(p - P_{\text{CO}_2,l})) - \rho_l y_{\text{CO}_2,l} q_l \right\|_2^2
\end{equation}
```

```math
\begin{equation}
V(q_l,p;\lambda_l)_{\text{pressure equation},\text{H}_2\text{O},l} = \frac{1}{n_s} \left\| \nabla \cdot k(\rho_l y_{\text{H}_2\text{O},l} \lambda_l \nabla(p - P_{\text{H}_2\text{O},l})) - \rho_l y_{\text{H}_2\text{O},l} q_l \right\|_2^2
\end{equation}
```

```math
\begin{equation}
V(p,S_g;t)_{\text{gas equation},\text{CO}_2,g} = \frac{1}{n_s} \left\| \phi \frac{\partial}{\partial t} (\rho_g y_{\text{CO}_2,g} S_g) -
\nabla \cdot k(\rho_g y_{\text{CO}_2,g} \lambda_g \nabla(p - P_{\text{CO}_2,g})) - \rho_g y_{\text{CO}_2,g} q_g \right\|_2^2
\end{equation}
```

```math
\begin{equation}
V(p,S_g;t)_{\text{gas equation},\text{CO}_2,l} = \frac{1}{n_s} \left\| \phi \frac{\partial}{\partial t} (\rho_l y_{\text{CO}_2,l} S_l) -
 \nabla \cdot k(\rho_l y_{\text{CO}_2,l} \lambda_g \nabla(p - P_{\text{CO}_2,l})) - \rho_l y_{\text{CO}_2,l} q_l \right\|_2^2
\end{equation}
```

```math
\begin{equation}
V(p,S_l;t)_{\text{brine equation},\text{H}_2\text{O},l} = \frac{1}{n_s} \left\| \phi \frac{\partial}{\partial t} (\rho_l y_{\text{H}_2\text{O},l} S_l) -
 \nabla \cdot k(\rho_l y_{\text{H}_2\text{O},l} \lambda_g \nabla(p - P_{\text{H}_2\text{O},l})) - \rho_l y_{\text{H}_2\text{O},l} q_l \right\|_2^2
\end{equation}
```


```math
\begin{equation}
\phi_{\text{cfd}} = V(q_g,p;\lambda_g)_{\text{pressure equation},\text{CO}_2,g} + V(q_l,p;\lambda_l)_{\text{pressure equation},\text{CO}_2,l} + V(q_l,p;\lambda_l)_{\text{pressure equation},\text{H}_2\text{O},l} + V(p,S_g;t)_{\text{gas equation},\text{CO}_2,g} + V(p,S_g;t)_{\text{gas equation},\text{CO}_2,l} + V(p,S_w;t)_{\text{brine equation},\text{H}_2\text{O},l}
\end{equation}
```

```math
\begin{equation}
\phi = \phi_{\text{cfd}} + \phi_{\text{data}}
\end{equation}
```

```math
\begin{equation}
\theta = [\theta_p,\theta_s,\theta_g]^T
\end{equation}
```

```math
\begin{equation}
\theta^{(j+1)} = \theta^j - \epsilon \nabla \phi_{\theta}^j
\end{equation}
```




## **2.2 Pseudocode**

```math
|***Algorithm 1***: PINO CO<sub>2</sub>-Brine Reservoir simulator |
\begin{algorithm}
\caption{PINO CO$_2$-Brine Reservoir simulator}
\begin{algorithmic}
\State \textbf{Input:} $X1=K, \varphi \in \mathbb{R}^{B_0 \times 1 \times D \times W \times H}, XN1=ql,qg,dt \in \mathbb{R}^{B_0 \times T \times D \times W \times H}$
\State $Ypt$, --labelled pressure
\State $Ylt$, -- labelled water saturation
\State $Ygt$, -- labelled gas saturation
\State $f1:, \theta_p$, 
\State $f2:, \theta_l$, 
\State $f3:, \theta_g$, 
\State $T=$ Time 
\State epoch, tol, $w1,w2,w3,w4,w5, w6, \epsilon$
\State $j = 0$ 
\While{$j \leq \text{epoch}$ or $(\phi \leq \text{tol})$}
\State $Y0p=f1X1;\theta_p$, $Y0s=f2X1;\theta_l$, $Y0g=f3X1;\theta_g$
\State Compute: $vl,vg, y_{CO2,l},y_{CO2,g},y_{H2O,l},y_{H2O,g}$ \textbf{using Eqn. (5-9)}
\State Compute: $\rho_g,\rho_l,\mu_g,\mu_l$ \textbf{using Eqn. (10-14)}
\State Compute: $Zc= \frac{\rho_c}{\rho_T}$
\State \textbf{Compute:} $\rho_T= c \rho_c$
\State Compute: $Sl^*= vl\frac{\rho_T}{\rho_l}$, $Sg^*= vg\frac{\rho_T}{\rho_g}$
\State Compute : $\phi_l^*= Y1st,-Sl^*22$
\State Compute : $\phi_g^*= Y1gt,-Sg^*22$
\State Compute: 
\State \hspace{1cm} $\text{V}(q_g,p;\lambda_g)\text{pressure equation,CO}_2,g=\frac{1}{n_s} \nabla \cdot k\rho_gy_{CO2,g}\lambda_g\nabla p- P_{CO2,g}-\rho_gy_{CO2,g}q_g22$
\State \hspace{1cm} $\text{V}(q_l,p;\lambda_l)\text{pressure equation,CO}_2,l=\frac{1}{n_s} \nabla \cdot k\rho_ly_{CO2,l}\lambda_l\nabla p- P_{CO2,l}-\rho_ly_{CO2,l}q_l22$
\State \hspace{1cm} $\text{V}(q_l,p;\lambda_l)\text{pressure equation,H}_2\text{O},l=\frac{1}{n_s} \nabla \cdot k\rho_ly_{H2O,l}\lambda_l\nabla p- P_{H2O,l}-\rho_ly_{H2O,l}q_l22$
\State \hspace{1cm} $\text{V}(p,Sg;t)\text{gas equation,CO}_2,g=\frac{1}{n_s} \phi\frac{\partial}{\partial t}\rho_gy_{CO2,g}Sg- \nabla \cdot k\rho_gy_{CO2,g}\lambda_g\nabla p- P_{CO2,g}-\rho_gy_{CO2,g}q_g22$
\State \hspace{1cm} $\text{V}(p,Sg;t)\text{gas equation,CO}_2,l=\frac{1}{n_s} \phi\frac{\partial}{\partial t}\rho_ly_{CO2,l}Sl- \nabla \cdot k\rho_ly_{CO2,l}\lambda_g\nabla p- P_{CO2,l}-\rho_ly_{CO2,l}q_l22$
\State \hspace{1cm} $\text{V}(p,Sl;t)\text{brine equation,H}_2\text{O},l=\frac{1}{n_s} \phi\frac{\partial}{\partial t}\rho_ly_{H2O,l}Sl- \nabla \cdot k\rho_ly_{H2O,l}\lambda_g\nabla p- P_{H2O,l}-\rho_ly_{H2O,l}q_l22$
\State $\phi_p= Ypt,-f1X1;\theta_p22$
\State $\phi_s= Ylt,-f2X1;\theta_l22$
\State $\phi_g= Ygt,-f3X1;\theta_g22$
\State $\phi= w1\phi_p+ w2\phi_s + w3\phi_g+ w4\phi_l^*+ w5\phi_g^*+ w6\text{V}(q_g,p;\lambda_g)\text{pressure equation,CO}_2,g+w7\text{V}(q_l,p;\lambda_l)\text{pressure equation,CO}_2,l+ w8\text{V}(p,Sg;t)\text{gas equation,CO}_2,g+w9\text{V}(q_l,p;\lambda_l)\text{pressure equation,H}_2\text{O},l+ w10\text{V}(p,Sg;t)\text{gas equation,CO}_2,l+ \text{V}(p,Sl;t)\text{brine equation,H}_2\text{O},l$
\State \textbf{Update models:}
\State $\theta= \theta_p,\theta_l,\theta_g^T$
\State $\theta^{(j+1)}=\theta^j-\epsilon \nabla \phi_{\theta}^j$
\State $j \leftarrow j+ 1$
\EndWhile
\State \textbf{Output:} $f1:,\theta_p,f2:,\theta_l, f3:,\theta_g$
\end{algorithmic}
\end{algorithm}
```





w1,…w10 are the weights associated to the loss functions associated to the 10 terms. X0=k,φ∈RB0×1×W×H are the dictionary inputs. epoch,tol are the number or epochs and the tolerance level for the loss function f1:,θp,f2:,θl, f3:,θgare the *FNO* models for the pressure, brine saturation and gas saturations respectively.

# **3. Results for Physics Constrained Black Oil Model (***Norne* Field)**.**
Below we show the application of a physics constrained neural operator to solve the black oil model of a real field.

![](Aspose.Words.19495806-8bc0-4e3d-951a-290e1d8e2574.001.png)

***Figure 1(a):** Forwarding of the Norne Field.* Nx=46 ,  Ny=112 , Nz=22*. At Time = 8 days. Dynamic properties comparison between the pressure, water saturation, oil saturation and gas saturation field between Nvidia Modulus’s PINO surrogate (left column), Flow reservoir simulator (middle column) and the difference between both approaches (last column). They are 22 oil/water/gas producers (green), 9 water injectors (blue) and 4 gas injectors) red. We can see good concordance between the surrogate’s prediction and the numerical reservoir simulator (Flow)*



![](Aspose.Words.19495806-8bc0-4e3d-951a-290e1d8e2574.002.png)

***Figure 1(b):** Forwarding of the Norne Field.* Nx=46 ,  Ny=112 , Nz=22*. At Time = 968 days. Dynamic properties comparison between the pressure, water saturation, oil saturation and gas saturation field between Nvidia Modulus’s PINO surrogate (left column), Flow reservoir simulator (middle column) and the difference between both approaches (last column). They are 22 oil/water/gas producers (green), 9 water injectors (blue) and 4 gas injectors) red. We can see good concordance between the surrogate’s prediction and the numerical reservoir simulator (Flow)*





![](Aspose.Words.19495806-8bc0-4e3d-951a-290e1d8e2574.003.png)

***Figure 1(c):** Forwarding of the Norne Field.* Nx=46 ,  Ny=112 , Nz=22*. At Time = 2104 days. Dynamic properties comparison between the pressure, water saturation, oil saturation and gas saturation field between Nvidia Modulus’s PINO surrogate (left column), Flow reservoir simulator (middle column) and the difference between both approaches (last column). They are 22 oil/water/gas producers (green), 9 water injectors (blue) and 4 gas injectors) red. We can see good concordance between the surrogate’s prediction and the numerical reservoir simulator (Flow)*



![](Aspose.Words.19495806-8bc0-4e3d-951a-290e1d8e2574.004.png)

***Figure 1(d):** Forwarding of the Norne Field.* Nx=46 ,  Ny=112 , Nz=22*. At Time = 3298 days. Dynamic properties comparison between the pressure, water saturation, oil saturation and gas saturation field between Nvidia Modulus’s PINO surrogate (left column), Flow reservoir simulator (middle column) and the difference between both approaches (last column). They are 22 oil/water/gas producers (green), 9 water injectors (blue) and 4 gas injectors) red. We can see good concordance between the surrogate’s prediction and the numerical reservoir simulator (Flow)*






# **References**
1. Z. Duan and R. Sun, An improved model calculating CO2 solubility in pure water and aqueous NaCl solutions from 273 to 533 K and from 0 to 2000 bar., Chemical Geology,vol. 193.3-4, pp. 257-271, 2003.
1. R. Span and W. Wagner, A new equation of state for carbon dioxide covering the fluid region from the triple-point temperature to 1100 K at pressure up to 800 MPa, J. Phys.Chem. Ref. Data, vol. 25, pp. 1509-1596, 1996.
1. Fenghour and W. A. Wakeham, The viscosity of carbon dioxide, J. Phys. Chem. Ref.Data, vol. 27, pp. 31-44, 1998.
1. S. L. Phillips et al., A technical data book for geothermal energy utilization, Lawrence Berkeley Laboratory report, 1981.
1. J. E. Garcia, Density of aqueous solutions of CO2. No. LBNL-49023. Lawrence Berkeley National Laboratory, Berkeley, CA, 2001.
1. Zaytsev, I.D. and Aseyev, G.G. Properties of Aqueous Solutions of Electrolytes, BocaRaton, Florida, USA CRC Press, 1993.
1. Engineering ToolBox, Water - Density, Specific Weight and Thermal Expansion Coefficients, 2003
1. Engineering Tool Box, Water - Dynamic (Absolute) and Kinematic Viscosity, 2004
1. Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar. Fourier Neural Operator for Parametric Partial Differential Equations. https://doi.org/10.48550/arXiv.2010.08895
1. Zongyi Li, Hongkai Zheng, Nikola Kovachki, David Jin, Haoxuan Chen, Burigede Liu, Kamyar Azizzadenesheli, Anima Anandkumar.Physics-Informed Neural Operator for Learning Partial Differential Equations. https://arxiv.org/pdf/2111.03794.pdf

**2 **|** Page


[^1]: `  `Nvidia Corporation

    \* Corresponding author
