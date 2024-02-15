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

Where: $l$ is the phase (brine/gas), $k$ is the rock absolute permeability, $\lambda_l$ is the phase mobility ratio, $\mu_l$ is the phase viscosity, $K_{rl}$ is the phase relative permeability, $S_l$ is the phase saturation, $u_l$ is the phase Darcy velocity, $g$ is the acceleration due to gravity, $z$ is the depth, $y_{c,l}$ is the mass fraction of component $c$ in phase $l$, $t$ is time, and $p$ is the pressure.

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
\begin{enumerate}
    \item The phase fractions ($`v_p`$) and phase component fractions ($`y_{c,p}`$) are computed as a function of pressure ($`p$), temperature ($`T`$), component fractions ($`z_c`$), and a constant salinity.
    \item The phase densities ($`\rho_p`$) and phase viscosities ($`\mu_p`$) are computed as a function of pressure, temperature, the updated phase component fractions, and a constant salinity.
\end{enumerate}

Once the phase fractions, phase component fractions, phase densities, phase viscosities--and their derivatives with respect to pressure, temperature, and component fractions--have been computed, the.

Note that the current implementation of the flow solver is isothermal and that the derivatives to temperature are therefore discarded.

The models that are used in steps 1) and 2) are reviewed in more detail below.




### ***1.2.1 Computation of the phase fractions and phase component fractions (flash)***
We compute the values of CO<sub>2</sub> solubility in brine as a function of pressure, temperature, and a constant salinity. we define the pressure (p) and temperature (T):

**Note that the pressures are in Pascal, temperatures are in Kelvin, and the salinity is a molality** (moles of NaCl per kg of brine). The temperature must be between 283.15 and 623.15 Kelvin. The table is populated using the model of Duan and Sun (2003). Specifically, we solve the following nonlinear CO<sub>2</sub> equation of state (Duan and Sun, 2003) for each pair to obtain the reduced volume,

Z=prVrTr=  1+ a1+a2Tr2+ a3Tr3Vr+ a4+a5Tr2+ a6Tr3Vr2+ a7+a8Tr2+ a9Tr3Vr4+ a10+a11Tr2+ a12Tr3Vr5+ a13Tr3Vr2a14+ a15Vr2exp-a15Vr2
Eqn. 5

.

Where pr= ppcrit   is the reduced pressure and the reduced temperature Tr= TTcrit

|*a1 = 8.99288497e-2, a2 = -4.94783127e-1, a3 = 4.77922245e-2, a4 = 1.03808883e-2, a5 = -2.82516861e-2, a6 = 9.49887563e-2, a7 = 5.20600880e-4, a8 = -2.93540971e-4, a9 = -1.77265112e-3, a10 = -2.51101973e-5, a11 = 8.93353441e-5, a12 = 7.88998563e-5, a13 = -1.66727022e-2, a14 = 1.39800000e0, a15 = 2.96000000e-2*|
| - |

Using the reduced volume, Vr, we compute the fugacity coefficient of CO<sub>2</sub>,

InϕT,P=Z-1-InZ+ a1+a2Tr2+ a3Tr3Vr+ a4+a5Tr2+ a6Tr32Vr2+ a7+a8Tr2+ a9Tr34Vr4+ a10+a11Tr2+ a12Tr35Vr5a132Tr3Vr2a14+1- a14+1+a15Vr2exp-a15Vr2⁡
Eqn. 6

To conclude, we use the fugacity coefficient of CO<sub>2</sub> to compute and store the solubility of CO<sub>2</sub> in brine,sCO2

InyCO2sCO2P= ΦCO2RT- InϕT,P+ c2λcm+ a2λam+ a,cςa,cm2
Eqn.7

Where ΦCO2 is the chemical potential of the CO<sub>2</sub> component, R is the gas constant, and m is the salinity. The mole fraction of CO<sub>2</sub> in the vapor phase, yCO2= p- pH2Op

Then, we compute the phase fractions as:

vl= 1+ sCO21+ zCO21-zCO2
Eqn.8(a)

vg=1- vl
Eqn. 8(b)

We conclude by computing the phase component fractions as:

yCO2,l=sCO21+ sCO2 , yH2O,l=1- yCO2,l
Eqn. 9(a)

yCO2,g=1, yH2O,g=0
Eqn. 9(b)

### ***1.2.2 Computation of the phase densities and phase viscosities***

<b>CO<sub>2</sub> phase density and viscosity</b>

The nonlinear Helmholtz energy equation yields, 

PRTρg=1+ ψϕψrψ,τ
Eqn. 10(a)

Were, 

ψ= ρgρcrit , τ= TcritT
Eqn. 10(b)

μg= μoT+ μexcessρg,T
Eqn. 10(c)

μexcessρg,T= d1ρg+d2ρg2+ d3ρg6T3+ d4ρg8+d5ρg8T 
Eqn. 10(d)

|d1=0.4071119e-2, d2=0.7198037e-4,d3=0.2411697e-16,d4=0.2971072e-22,d5=-0.1627888e-22|
| - |

μoT= 1.00697T0.5B\*T\*,  InB\*T\*=i=04xiInT\*i , T\*=ωT, ω=1251.196K
Eqn. 11

|x0=0.235156,   x1=-0.491266,  x2=5.211155e-2,  x3=5.347906e-2,  x4=-1.537102e-2|
| - |




**Brine phase density and viscosity**

ρl,table= A1+ A2x+ A3x2+ A4x3
Eqn. 12(a)

x= c1expa1m+c2expa2T+c3exp⁡a3P
Eqn. 12(b)

ρl= ρl,table+ MCO2CCO2-CCO2ρl,tableVϕ
Eqn. 13(a)

CCO2= yCO2,lρl,tableMH2O1-yCO2,l
Eqn. 13(b)

Vϕ=37.51-T×9.585e-2+T2×8.740e-4-(T3×5.044e-7)
Eqn. 13(c)

μl=azT+bz
Eqn. 14(a)

az= μw(T)×0.0006291-exp⁡(-0.7m) 
Eqn. 14(b)

bz= μw(T)1+0.0816m+0.0122m2+0.000128m3
Eqn. 14(c)
# <b>2. Physics Constrained Neural operator for the CO<sub>2</sub>-Brine case</b>

## **2.1 Overall discretized equations loss**

The physics loss *ansatz* is then,

Vqg,p;λgpressure equation,CO2,g=1ns ∇.kρgyCO2,gλg∇p- PCO2,g-ρgyCO2,gqg22

Eqn. (15a)

Vql,p;λlpressure equation,CO2,l=1ns ∇.kρlyCO2,lλl∇p- PCO2,l-ρlyCO2,lql22

Eqn. (15b)

Vql,p;λlpressure equation,H2O,l=1ns ∇.kρlyH2O,lλl∇p- PH2O,l-ρlyH2O,lql22

Eqn. (15c)

Vp,Sg;tgas equation,CO2,g=1nsφ∂∂tρgyCO2,gSg- ∇.kρgyCO2,gλg∇p- PCO2,g-ρgyCO2,gqg22

Eqn. (16a)

Vp,Sg;tgas equation,CO2,l=1nsφ∂∂tρlyCO2,lSl- ∇.kρlyCO2,lλg∇p- PCO2,l-ρlyCO2,lql22

Eqn. (16b)


Vp,Sl;tbrine equation,H2O,l=1nsφ∂∂tρlyH2O,lSl- ∇.kρlyH2O,lλg∇p- PH2O,l-ρlyH2O,lql22
Eqn. (16c)

ϕcfd=Vqg,p;λgpressure equation,CO2,g+ Vql,p;λlpressure equation,CO2,l+ Vql,p;λlpressure equation,H2O,l+ Vp,Sg;tgas equation,CO2,g+Vp,Sg;tgas equation,CO2,l+ Vp,Sw;tbrine equation,H2O,l  

Eqn. (17)

ϕ= ϕcfd+ϕdata

Eqn. (18)

θ= θp,θs,θgT

θj+1=θj-ϵ∇ϕθj
## **2.2 Pseudocode**

|***Algorithm 1***: PINO CO<sub>2</sub>-Brine Reservoir simulator |
| - |
|<p>***Input:***    X1=K, φ∈RB0×1×D×W×H, XN1=ql,qg,dt ∈RB0×T×D×W×H </p><p>Ypt, --labelled pressure</p><p>Ylt, -- labelled water saturation</p><p>Ygt, -- labelled gas saturation</p><p>f1:,θp, </p><p>f2:,θl, </p><p>f3:,θg, </p><p>T= -- Time </p><p>epoch, tol, w1,w2,w3,w4,w5, w6, ϵ </p><p></p><p>j = 0 </p><p>while  j≤epoch or (ϕ≤tol) do </p><p>1. Y0p=f1X1;θp, Y0s=f2X1;θl , Y0g=f3X1;θg  </p><p>2. Compute: vl,vg, yCO2,l,yCO2,g,yH2O,l,yH2O,g **using Eqn. (5-9)**</p><p>3. Compute: ρg,ρl,μg,μl **using Eqn. (10-14)**</p><p>4. Compute: Zc= ρcρT </p><p>5. ***Compute:*** ρT= cρc</p><p>6. Compute: Sl\*= vlρTρl , Sg\*= vgρTρg </p><p>7. Compute : ϕl\*= Y1st,-Sl\*22 </p><p>8. Compute : ϕg\*= Y1gt,-Sg\*22 </p><p>&emsp;</p><p>9. Compute:*** <br>   Vqg,p;λgpressure equation,CO2,g=1ns ∇.kρgyCO2,gλg∇p- PCO2,g-ρgyCO2,gqg22</p><p>10. ***Compute :**** <br>    Vql,p;λlpressure equation,CO2,l=1ns ∇.kρlyCO2,lλl∇p- PCO2,l-ρlyCO2,lql22</p><p>11. Compute:* <br>    Vql,p;λlpressure equation,H2O,l=1ns ∇.kρlyH2O,lλl∇p- PH2O,l-ρlyH2O,lql22</p><p>12. Compute:* <br>    Vp,Sg;tgas equation,CO2,g=1nsφ∂∂tρgyCO2,gSg- ∇.kρgyCO2,gλg∇p- PCO2,g-ρgyCO2,gqg22</p><p>13. Compute:* <br>    Vp,Sg;tgas equation,CO2,l=1nsφ∂∂tρlyCO2,lSl- ∇.kρlyCO2,lλg∇p- PCO2,l-ρlyCO2,lql22</p><p></p><p>14. Compute:* <br>    Vp,Sl;tbrine equation,H2O,l=1nsφ∂∂tρlyH2O,lSl- ∇.kρlyH2O,lλg∇p- PH2O,l-ρlyH2O,lql22</p><p>&emsp;</p><p>15. ϕp= Ypt,-f1X1;θp22  </p><p>16. ϕs= Ylt,-f2X1;θl22  </p><p>17. ϕg= Ygt,-f3X1;θg22          </p><p>18. ϕ= w1ϕp+ w2ϕs + w3ϕg+ w4ϕl\*+ w5ϕg\*+ w6Vqg,p;λgpressure equation,CO2,g+w7Vql,p;λlpressure equation,CO2,l+ w8Vp,Sg;tgas equation,CO2,g+w9Vql,p;λlpressure equation,H2O,l+ w10Vp,Sg;tgas equation,CO2,l+ Vp,Sl;tbrine equation,H2O,l</p><p>19. Update models:</p><p>&emsp;θ= θp,θl,θgT</p><p>&emsp;θj+1=θj-ϵ∇ϕθj</p><p>`               `j ← j+ 1 </p><p>***Output***:f1:,θp,f2:,θl, f3:,θg</p>|




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
