# C02 - Brine surrogate computed with a Physics Informed Neural Operator (PINO) 

![Visualization](Visuals/All1.png)

## 1. Forward Problem

### 1.1 Governing Equations 

The governing equations for a CO<sub>2</sub>-Brine system are given by:

```latex
\begin{equation}
\varphi \frac{\partial }{\partial t}\left(\sum_{\ell }{{\rho }_{\ell }}y_{c\ell }S_{\ell }\right)-\ \nabla \cdot k\left(\sum_{\ell }{{\rho }_{\ell }y_{c\ell }{\lambda }_{\ell }}u_{\ell }\right)-\sum_{\ell }{{\rho }_{\ell }}y_{c\ell }q_{\ell }=0 \tag{1}
\end{equation}

























\noindent $\ell $  is the phase (brine/gas), $k\ $ is the rock absolute permeability, ${\lambda }_{\ell }$ is the phase mobility ratio, ${\mu }_{\ell }$ is the phase viscosity, $K_{r\ell }$ is the phase relative permeability, $S_{\ell }$ is the phase saturation,$u_{\ell }$ is the phase darcy velocity,\textit{ }$g$\textit{ }is the acceleration due to gravity, $\mathrm{z}$ is the depth,\textit{ }$y_{c,\ell }$ is the mass fraction of component $c$ in phase $\ell $, $t$ is time and $p$ is the pressure.

\noindent The system is closed by assuming,

\noindent $\sum_{\ell }{S_{\ell }}=1,\ \ Z_c=\ \frac{{\rho }_c}{{\rho }_T}\ \ ,{\rho }_T=\ \sum_c{{\rho }_c}$Eqn. 4(b)
\[\ \] 
$S_l=\ v_l\frac{{\rho }_T}{{\rho }_l}\ ,\ S_g=\ v_g\frac{{\rho }_T}{{\rho }_g}$Eqn. 4(c)

\noindent 

\noindent 
\subsection{1.2 Thermodynamic Equations }

\noindent The CO${}_{2}$-brine model includes two components (CO${}_{2}$ and H${}_{2}$O) that are transported by one or two fluid phases (the brine phase and the CO${}_{2}$ phase). We refer to the brine phase with the subscript $l\ $and to the CO${}_{2}$ phase with the subscript $g$ (although the CO${}_{2}$ phase can be in supercritical, liquid, or gas state). The water component is only present in the brine phase, while the CO${}_{2}$ component can be present in the CO${}_{2}$ phase as well as in the brine phase. Thus, considering the molar phase component fractions,$y_{c,p}$ (i.e., the fraction of the molar mass of phase $p$ represented by component $c$) 

\noindent The update of the fluid properties is done in two steps:

\begin{enumerate}
\item  The phase fractions ($v_p$) and phase component fractions ($y_{c,p}$) are computed as a function of pressure ($p$), temperature ($T$), component fractions ($z_c$), and a constant salinity.

\item  The phase densities (${\rho }_p$) and phase viscosities (${\mu }_p$) are computed as a function of pressure, temperature, the updated phase component fractions, and a constant salinity.
\end{enumerate}

\noindent Once the phase fractions, phase component fractions, phase densities, phase viscosities--and their derivatives with respect to pressure, temperature, and component fractions--have been computed, the

\noindent Note that the current implementation of the flow solver is isothermal and that the derivatives with respect to temperature are therefore discarded.

\noindent The models that are used in steps 1) and 2) are reviewed in more details below.

\noindent 

\noindent 
\paragraph{1.2.1 Computation of the phase fractions and phase component fractions (flash)}

\noindent We compute the values of CO${}_{2}$ solubility in brine as a function of pressure, temperature, and a constant salinity. we define the pressure ($p$) and temperature ($T$):

\noindent \textbf{Note that the pressures are in Pascal, temperatures are in Kelvin, and the salinity is a molality} (moles of NaCl per kg of brine). The temperature must be between 283.15 and 623.15 Kelvin. The table is populated using the model of Duan and Sun (2003). Specifically, we solve the following nonlinear CO${}_{2\ }$equation of state (Duan and Sun, 2003) for each pair to obtain the reduced volume,

\noindent $Z=\frac{p_rV_r}{T_r}=\ \ 1+\ \frac{a_1+{a_2}/{T^2_r}+\ {a_3}/{T^3_r}}{V_r}+\ \frac{a_4+{a_5}/{T^2_r}+\ {a_6}/{T^3_r}}{V^2_r}+\ \frac{a_7+{a_8}/{T^2_r}+\ {a_9}/{T^3_r}}{V^4_r}+\ \frac{a_{10}+{a_{11}}/{T^2_r}+\ {a_{12}}/{T^3_r}}{V^5_r}+\ \frac{a_{13}}{T^3_rV^2_r}\left(a_{14}+\ \frac{a_{15}}{V^2_r}\right)\mathrm{exp}\left(-\frac{a_{15}}{V^2_r}\right)$Eqn. 5

\noindent .

\noindent 

\noindent Where $p_r=\ {p}/{p_{crit}}$   is the reduced pressure and the reduced temperature $T_r=\ {T}/{T_{crit}}$

\noindent 

\begin{tabular}{|p{4.2in}|} \hline 
\textit{a1 = 8.99288497e-2, a2 = -4.94783127e-1, a3 = 4.77922245e-2, a4 = 1.03808883e-2, a5 = -2.82516861e-2, a6 = 9.49887563e-2, a7 = 5.20600880e-4, a8 = -2.93540971e-4, a9 = -1.77265112e-3, a10 = -2.51101973e-5, a11 = 8.93353441e-5, a12 = 7.88998563e-5, a13 = -1.66727022e-2, a14 = 1.39800000e0, a15 = 2.96000000e-2} \\ \hline 
\end{tabular}



\noindent Using the reduced volume, $V_r$, we compute the fugacity coefficient of CO${}_{2}$,

\noindent ${In}_{\phi }\left(T,P\right)=Z-1-InZ+\ \frac{a_1+{a_2}/{T^2_r}+\ {a_3}/{T^3_r}}{V_r}+\ \frac{a_4+{a_5}/{T^2_r}+\ {a_6}/{T^3_r}}{2V^2_r}+\ \frac{a_7+{a_8}/{T^2_r}+\ {a_9}/{T^3_r}}{{4V}^4_r}+\ \frac{a_{10}+{a_{11}}/{T^2_r}+\ {a_{12}}/{T^3_r}}{{5V}^5_r}\frac{a_{13}}{{2T}^3_rV^2_r}\left[a_{14}+1-\left(\ a_{14}+1+\frac{a_{15}}{V^2_r}\right)\right]\mathrm{exp}\left(-\frac{a_{15}}{V^2_r}\right)\mathrm{}$Eqn. 6

\noindent 

\noindent To conclude, we use the fugacity coefficient of CO${}_{2}$ to compute and store the solubility of CO${}_{2}$ in brine,$s_{{CO}_2}$

\noindent $In\frac{y_{{CO}_2}}{s_{{CO}_2}}P=\ \frac{{\mathrm{\Phi }}_{{CO}_2}}{RT}-\ {In}_{\phi }\left(T,P\right)+\ \sum_c{2{\lambda }_cm}+\ \sum_a{2{\lambda }_am}+\ \sum_{a,c}{{\varsigma }_{a,c}m^2}$Eqn.7

\noindent Where ${\mathrm{\Phi }}_{{CO}_2}$ is the chemical potential of the CO${}_{2}$ component, $R$ is the gas constant, and $m$ is the salinity. The mole fraction of CO${}_{2}$ in the vapor phase, $y_{{CO}_2}=\ {\left(p-\ p_{H_2O}\right)}/{p}$

\noindent Then, we compute the phase fractions as:

\noindent $v_l=\ \frac{1+\ s_{{CO}_2}}{1+\ {z_{{CO}_2}}/{\left(1-z_{{CO}_2}\right)}}$Eqn.8(a)

\noindent $v_g=1-\ v_l$Eqn. 8(b)

\noindent We conclude by computing the phase component fractions as:

\noindent $y_{{CO}_2,l}=\frac{s_{{CO}_2}}{1+\ s_{{CO}_2}}\ ,\ y_{H_2O,l}=1-\ y_{{CO}_2,l}$Eqn. 9(a)

\noindent 

\noindent $y_{{CO}_2,g}=1,\ y_{H_2O,g}=0$Eqn. 9(b)

\noindent 

\noindent 
\paragraph{1.2.2 Computation of the phase densities and phase viscosities}

\noindent 

\noindent \textbf{\underbar{CO${}_{2}$ phase density and viscosity}}

\noindent The nonlinear Helmholtz energy equation yields, 

\noindent $\frac{P}{RT{\rho }_g}=1+\ \psi {\phi }^r_{\psi }\left(\psi ,\tau \right)$Eqn. 10(a)

\noindent Were, 

\noindent $\psi =\ \frac{{\rho }_g}{{\rho }_{crit}}\ ,\ \tau =\ \frac{T_{crit}}{T}$Eqn. 10(b)

\noindent ${\mu }_g=\ {\mu }_o\left(T\right)+\ {\mu }_{excess}\left({\rho }_g,T\right)$Eqn. 10(c)

\noindent 

\noindent ${\mu }_{excess}\left({\rho }_g,T\right)=\ d_1{\rho }_g+d_2{{\rho }_g}^2+\ \frac{d_3{{\rho }_g}^6}{T^3}+\ d_4{{\rho }_g}^8+\frac{d_5{{\rho }_g}^8}{T}\ $Eqn. 10(d)

\begin{tabular}{|p{4.2in}|} \hline 
$d_1=0.4071119e-2,\ d_2=0.7198037e-4,d_3=0.2411697e-16,d_4=0.2971072e-22,d_5=-0.1627888e-22$\textit{} \\ \hline 
\end{tabular}



\noindent ${\mu }_o\left(T\right)=\ \frac{1.00697T^{0.5}}{B^*\left(T^*\right)},\ \ InB^*\left(T^*\right)=\sum^4_{i=0}{x_i{\left(InT^*\right)}^i}\ ,\ T^*=\omega T,\ \omega =\frac{1}{251.196}K$Eqn. 11

\noindent 

\begin{tabular}{|p{4.2in}|} \hline 
$x_0=0.235156,\ \ {\ x}_1=-0.491266,\ \ x_2=5.211155e-2,\ \ x_3=5.347906e-2,\ \ x_4=-1.537102e-2$\textit{} \\ \hline 
\end{tabular}



\noindent 

\noindent 

\noindent 

\noindent \textbf{\underbar{Brine phase density and viscosity}}

\noindent ${\rho }_{l,table}=\ A_1+\ A_2x+\ A_3x^2+\ A_4x^3$Eqn. 12(a)

\noindent $x=\ c_1{\mathrm{exp} \left(a_1m\right)\ }+c_2{\mathrm{exp} \left(a_2T\right)\ }+c_3\mathrm{exp}\mathrm{}\left(a_3P\right)$Eqn. 12(b)

\noindent ${\rho }_l=\ {\rho }_{l,table}+\ M_{{CO}_2}C_{{CO}_2}-C_{{CO}_2}{\rho }_{l,table}V_{\phi }$Eqn. 13(a)

\noindent $C_{{CO}_2}=\ \frac{y_{{CO}_2,l}{\rho }_{l,table}}{M_{H_2O}\left(1-y_{{CO}_2,l}\right)}$Eqn. 13(b)

\noindent $V_{\phi }=37.51-\left(T\times 9.585e-2\right)+\left(T^2\times 8.740e-4\right)-(T^3\times 5.044e-7)$Eqn. 13(c)

\noindent ${\mu }_l=a_zT+b_z$Eqn. 14(a)

\noindent $a_z=\ {\mu }_w(T)\times 0.000629\left(1-\mathrm{exp}\mathrm{}(-0.7m\right))\ $Eqn. 14(b)

\noindent $b_z=\ {\mu }_w(T)\left(1+0.0816m+0.0122m^2+0.000128m^3\right)$Eqn. 14(c)

\noindent 
\section{2. Physics Constrained Neural operator for the CO${}_{2}$-Brine case}

\noindent 

\noindent 
\subsection{2.1 Overall discretized equations loss}

\noindent 

\noindent The physics loss \textit{ansatz} is then,
\[{V\left(q_g,p\mathrm{;}{\lambda }_g\right)}_{pressure~equation,{CO}_2,g}\mathrm{=}\frac{\mathrm{1}}{n_s}\left(~{\left\|\mathrm{\nabla }.k\left({\rho }_gy_{{CO}_2,g}{\lambda }_g\mathrm{\nabla }\left(p-\ P_{{CO}_2,g}\right)\right)-{\rho }_gy_{{CO}_2,g}q_g\right\|}^{\mathrm{2}}_{\mathrm{2}}\right)\] 
Eqn. (15a)
\[{V\left(q_l,p\mathrm{;}{\lambda }_l\right)}_{pressure~equation,{CO}_2,l}\mathrm{=}\frac{\mathrm{1}}{n_s}\left(~{\left\|\mathrm{\nabla }.k\left({\rho }_ly_{{CO}_2,l}{\lambda }_l\mathrm{\nabla }\left(p-\ P_{{CO}_2,l}\right)\right)-{\rho }_ly_{{CO}_2,l}q_l\right\|}^{\mathrm{2}}_{\mathrm{2}}\right)\] 
Eqn. (15b)
\[{V\left(q_l,p\mathrm{;}{\lambda }_l\right)}_{pressure~equation,H_2O,l}\mathrm{=}\frac{\mathrm{1}}{n_s}\left(~{\left\|\mathrm{\nabla }.k\left({\rho }_ly_{H_2O,l}{\lambda }_l\mathrm{\nabla }\left(p-\ P_{H_2O,l}\right)\right)-{\rho }_ly_{H_2O,l}q_l\right\|}^{\mathrm{2}}_{\mathrm{2}}\right)\] 
Eqn. (15c)
\[{V\left(p,S_g\mathrm{;}t\right)}_{gas~equation,{CO}_2,g}\mathrm{=}\frac{\mathrm{1}}{n_s}{\left\|\varphi \frac{\partial }{\partial t}\left({\rho }_gy_{{CO}_2,g}S_g\right)-\ \mathrm{\nabla }.k\left({\rho }_gy_{{CO}_2,g}{\lambda }_g\mathrm{\nabla }\left(p-\ P_{{CO}_2,g}\right)\right)-{\rho }_gy_{{CO}_2,g}q_g\right\|}^{\mathrm{2}}_{\mathrm{2}}\] 
Eqn. (16a)
\[{V\left(p,S_g\mathrm{;}t\right)}_{gas~equation,{CO}_2,l}\mathrm{=}\frac{\mathrm{1}}{n_s}{\left\|\varphi \frac{\partial }{\partial t}\left({\rho }_ly_{{CO}_2,l}S_l\right)-\ \mathrm{\nabla }.k\left({\rho }_ly_{{CO}_2,l}{\lambda }_g\mathrm{\nabla }\left(p-\ P_{{CO}_2,l}\right)\right)-{\rho }_ly_{{CO}_2,l}q_l\right\|}^{\mathrm{2}}_{\mathrm{2}}\] 
Eqn. (16b)

\noindent 

\noindent 

\noindent ${V\left(p,S_l\mathrm{;}t\right)}_{brine~equation,H_2O,l}\mathrm{=}\frac{\mathrm{1}}{n_s}{\left\|\varphi \frac{\partial }{\partial t}\left({\rho }_ly_{H_2O,l}S_l\right)-\ \mathrm{\nabla }.k\left({\rho }_ly_{H_2O,l}{\lambda }_g\mathrm{\nabla }\left(p-\ P_{H_2O,l}\right)\right)-{\rho }_ly_{H_2O,l}q_l\right\|}^{\mathrm{2}}_{\mathrm{2}}$Eqn. (16c)

\noindent 
\[{\boldsymbol{\phi }}_{\boldsymbol{cfd}}\mathrm{=}{V\left(q_g,p\mathrm{;}{\lambda }_g\right)}_{pressure~equation,{CO}_2,g}\mathrm{+\ }{V\left(q_l,p\mathrm{;}{\lambda }_l\right)}_{pressure~equation,{CO}_2,l}\mathrm{+\ }{V\left(q_l,p\mathrm{;}{\lambda }_l\right)}_{pressure~equation,H_2O,l}\mathrm{+\ }{V\left(p,S_g\mathrm{;}t\right)}_{gas~equation,{CO}_2,g}\mathrm{+}{V\left(p,S_g\mathrm{;}t\right)}_{gas~equation,{CO}_2,l}\mathrm{+\ }{V\left(p,S_w\mathrm{;}t\right)}_{brine~equation,H_2O,l}\mathrm{\ }~\] 
Eqn. \eqref{GrindEQ__17_}
\[\boldsymbol{\phi }\boldsymbol{\mathrm{=}}~{\boldsymbol{\phi }}_{\boldsymbol{cfd}}\boldsymbol{+}{\boldsymbol{\phi }}_{\boldsymbol{data}}\] 
Eqn. \eqref{GrindEQ__18_}
\[\theta \mathrm{=}~{\left[{\theta }_p,{\theta }_s,{\theta }_g\right]}^T\] 
\[{\theta }^{j\mathrm{+1}}\mathrm{=}{\theta }^j-{\epsilon \nabla \phi }^j_{\theta }\] 

\subsection{2.2 Pseudocode}

\begin{tabular}{|p{3.9in}|} \hline 
\textbf{\textit{Algorithm 1}}: PINO CO${}_{2}$-Brine Reservoir simulator  \\ \hline 
\textbf{\textit{Input:}}  $\mathrm{\ \ }X_1\mathrm{=}\left\{K\mathrm{,\ }\varphi \right\}\mathrm{\in }{\mathbb{R}}^{B_0\mathrm{\times 1}\mathrm{\times }D\mathrm{\times }W\mathrm{\times }H}\mathrm{,\ }X_{N1}\mathrm{=}\left\{q_l,q_g,dt\ \right\}\mathrm{\in }{\mathbb{R}}^{B_0\mathrm{\times T}\mathrm{\times }D\mathrm{\times }W\mathrm{\times }H}$ \newline $Y_{pt,}$ --labelled pressure\newline $Y_{\mathrm{l}t},$ -- labelled water saturation\newline $Y_{\mathrm{g}t},$ -- labelled gas saturation\newline $f_{\mathrm{1}}\left(\mathrm{:,}{\theta }_p\right),$ \newline $f_{\mathrm{2}}\left(\mathrm{:,}{\theta }_l\right),$ \newline $f_3\left(\mathrm{:,}{\theta }_g\right),$ \newline $T\mathrm{=}$ -- Time \newline $epoch,$ $tol,$ $w_{\mathrm{1}},w_{2,}w_{3,}w_{4,}w_{5,\ }w_6,$ $\epsilon $ \newline \newline $\boldsymbol{j}\mathrm{\ =\ 0}$ \newline $\boldsymbol{while}\mathrm{\ \ }\left(\boldsymbol{j}\mathrm{\le }\boldsymbol{epoch}\right)\mathrm{\ }\boldsymbol{or}\mathrm{\ (}\boldsymbol{\phi }\mathrm{\le }\boldsymbol{tol}\mathrm{)\ }\boldsymbol{do}$ \newline  $Y_{0p}\mathrm{=}f_{\mathrm{1}}\left(X_1\mathrm{;}{\theta }_p\right)\mathrm{,\ }Y_{0s}\mathrm{=}f_{\mathrm{2}}\left(X_1\mathrm{;}{\theta }_l\right)\ \mathrm{,\ }Y_{\mathrm{0g}}\mathrm{=}f_3\left(X_1\mathrm{;}{\theta }_g\right)$  \newline  $\boldsymbol{Compute}\boldsymbol{:}\ v_l,v_g,\ y_{{CO}_2,l},y_{{CO}_2,g},y_{H_2O,l},y_{H_2O,g}\ $\textbf{using Eqn. (5-9)}\newline  $\boldsymbol{Compute}\boldsymbol{:\ }{\rho }_g,{\rho }_l,{\mu }_g,{\mu }_l\boldsymbol{\ }$\textbf{using Eqn. (10-14)}\newline  $\boldsymbol{Compute}\boldsymbol{:\ }Z_c=\ \frac{{\rho }_c}{{\rho }_T}$ \newline  \textbf{\textit{Compute: }}${\boldsymbol{\rho }}_{\boldsymbol{T}}\boldsymbol{=\ }\sum_{\boldsymbol{c}}{{\boldsymbol{\rho }}_{\boldsymbol{c}}}$\textbf{\textit{\newline  }}$\boldsymbol{Compute}\boldsymbol{:\ }{S_l}^*=\ v_l\frac{{\rho }_T}{{\rho }_l}\ ,\ {S_g}^*=\ v_g\frac{{\rho }_T}{{\rho }_g}$ \newline  $\boldsymbol{Compute}\boldsymbol{\ :\ }{{\phi }_l}^*\mathrm{=\ }{\left\|Y_{\mathrm{1}st,}\mathrm{-}{S_l}^*\right\|}^{\mathrm{2}}_{\mathrm{2}}\boldsymbol{\ }$\newline  $\boldsymbol{Compute}\boldsymbol{\ :\ }{{\phi }_g}^*\mathrm{=\ }{\left\|Y_{\mathrm{1g}t,}\mathrm{-}{S_g}^*\right\|}^{\mathrm{2}}_{\mathrm{2}}\boldsymbol{\ }$\newline \newline  $\boldsymbol{Compute}\boldsymbol{:}$\textbf{ }${V\left(q_g,p\mathrm{;}{\lambda }_g\right)}_{pressure~equation,{CO}_2,g}\mathrm{=}\frac{\mathrm{1}}{n_s}\left(~{\left\|\mathrm{\nabla }.k\left({\rho }_gy_{{CO}_2,g}{\lambda }_g\mathrm{\nabla }\left(p-\ P_{{CO}_2,g}\right)\right)-{\rho }_gy_{{CO}_2,g}q_g\right\|}^{\mathrm{2}}_{\mathrm{2}}\right)$\newline  \textbf{\textit{Compute :}} ${V\left(q_l,p\mathrm{;}{\lambda }_l\right)}_{pressure~equation,{CO}_2,l}\mathrm{=}\frac{\mathrm{1}}{n_s}\left(~{\left\|\mathrm{\nabla }.k\left({\rho }_ly_{{CO}_2,l}{\lambda }_l\mathrm{\nabla }\left(p-\ P_{{CO}_2,l}\right)\right)-{\rho }_ly_{{CO}_2,l}q_l\right\|}^{\mathrm{2}}_{\mathrm{2}}\right)$\newline  $\boldsymbol{Compute}\boldsymbol{:}$ ${V\left(q_l,p\mathrm{;}{\lambda }_l\right)}_{pressure~equation,H_2O,l}\mathrm{=}\frac{\mathrm{1}}{n_s}\left(~{\left\|\mathrm{\nabla }.k\left({\rho }_ly_{H_2O,l}{\lambda }_l\mathrm{\nabla }\left(p-\ P_{H_2O,l}\right)\right)-{\rho }_ly_{H_2O,l}q_l\right\|}^{\mathrm{2}}_{\mathrm{2}}\right)$\newline  $\boldsymbol{Compute}\boldsymbol{:}\ $${V\left(p,S_g\mathrm{;}t\right)}_{gas~equation,{CO}_2,g}\mathrm{=}\frac{\mathrm{1}}{n_s}{\left\|\varphi \frac{\partial }{\partial t}\left({\rho }_gy_{{CO}_2,g}S_g\right)-\ \mathrm{\nabla }.k\left({\rho }_gy_{{CO}_2,g}{\lambda }_g\mathrm{\nabla }\left(p-\ P_{{CO}_2,g}\right)\right)-{\rho }_gy_{{CO}_2,g}q_g\right\|}^{\mathrm{2}}_{\mathrm{2}}$\newline  $\boldsymbol{Compute}\boldsymbol{:}$ ${V\left(p,S_g\mathrm{;}t\right)}_{gas~equation,{CO}_2,l}\mathrm{=}\frac{\mathrm{1}}{n_s}{\left\|\varphi \frac{\partial }{\partial t}\left({\rho }_ly_{{CO}_2,l}S_l\right)-\ \mathrm{\nabla }.k\left({\rho }_ly_{{CO}_2,l}{\lambda }_g\mathrm{\nabla }\left(p-\ P_{{CO}_2,l}\right)\right)-{\rho }_ly_{{CO}_2,l}q_l\right\|}^{\mathrm{2}}_{\mathrm{2}}$\newline \newline  $\boldsymbol{Compute}\boldsymbol{:}$ ${V\left(p,S_l\mathrm{;}t\right)}_{brine~equation,H_2O,l}\mathrm{=}\frac{\mathrm{1}}{n_s}{\left\|\varphi \frac{\partial }{\partial t}\left({\rho }_ly_{H_2O,l}S_l\right)-\ \mathrm{\nabla }.k\left({\rho }_ly_{H_2O,l}{\lambda }_g\mathrm{\nabla }\left(p-\ P_{H_2O,l}\right)\right)-{\rho }_ly_{H_2O,l}q_l\right\|}^{\mathrm{2}}_{\mathrm{2}}$\newline \newline  ${\phi }_p\mathrm{=\ }{\left\|Y_{pt,}\mathrm{-}f_{\mathrm{1}}\left(X_{\mathrm{1}}\mathrm{;}{\theta }_p\right)\right\|}^{\mathrm{2}}_{\mathrm{2}}$  \newline  ${\phi }_s\mathrm{=\ }{\left\|Y_{lt,}\mathrm{-}f_{\mathrm{2}}\left(X_{\mathrm{1}}\mathrm{;}{\theta }_l\right)\right\|}^{\mathrm{2}}_{\mathrm{2}}$  \newline  ${\phi }_g\mathrm{=\ }{\left\|Y_{\mathrm{g}t,}\mathrm{-}f_3\left(X_{\mathrm{1}}\mathrm{;}{\theta }_g\right)\right\|}^{\mathrm{2}}_{\mathrm{2}}$          \newline  $\phi \mathrm{=\ }{w_{\mathrm{1}}\phi }_p\mathrm{+\ }{w_{\mathrm{2}}\phi }_s\mathrm{\ +\ }{w_3\phi }_g\mathrm{+\ }{w_4{\phi }_l}^*\mathrm{+\ }{w_5{\phi }_g}^*\mathrm{+\ }{w_6V\left(q_g,p\mathrm{;}{\lambda }_g\right)}_{pressure~equation,{CO}_2,g}\mathrm{+}w_7{V\left(q_l,p\mathrm{;}{\lambda }_l\right)}_{pressure~equation,{CO}_2,l}\mathrm{+\ }{w_8V\left(p,S_g\mathrm{;}t\right)}_{gas~equation,{CO}_2,g}\mathrm{+}w_9{V\left(q_l,p\mathrm{;}{\lambda }_l\right)}_{pressure~equation,H_2O,l}+\ w_{\mathrm{10}}{V\left(p,S_g\mathrm{;}t\right)}_{gas~equation,{CO}_2,l}+\ {V\left(p,S_l\mathrm{;}t\right)}_{brine~equation,H_2O,l}$\newline  $\boldsymbol{\mathrm{Update}}\mathrm{\ }\boldsymbol{\mathrm{models}}$:\newline $\theta \mathrm{=\ }{\left[{\theta }_p,{\theta }_l,{\theta }_g\right]}^T$\newline ${\theta }^{j\mathrm{+1}}\mathrm{=}{\theta }^j\mathrm{-}{\epsilon \nabla \phi }^j_{\theta }$\newline                $\boldsymbol{j}\boldsymbol{\ }\boldsymbol{\leftarrow }\mathrm{\ }\boldsymbol{j}\mathrm{+\ }\boldsymbol{\mathrm{1}}$ \newline \textbf{\textit{Output}}:$f_{\mathrm{1}}\left(\mathrm{:,}{\theta }_p\right),f_{\mathrm{2}}\left(\mathrm{:,}{\theta }_l\right)\mathrm{,\ }f_3\left(\mathrm{:,}{\theta }_g\right)$ \\ \hline 
\end{tabular}



\noindent 

\noindent 

\noindent 

\noindent $w_{\mathrm{1}}\mathrm{,\dots }w_{\mathrm{10}}$ are the weights associated to the loss functions associated to the 10 terms. $X_0\mathrm{=}\left\{k,\varphi \right\}\mathrm{\in }{\mathbb{R}}^{B_0\mathrm{\times 1}\mathrm{\times }W\mathrm{\times }H}$ are the dictionary inputs. $epoch,tol$ are the number or epochs and the tolerance level for the loss function $f_{\mathrm{1}}\left(\mathrm{:,}{\theta }_p\right),f_{\mathrm{2}}\left(\mathrm{:,}{\theta }_l\right)\mathrm{,\ }f_3\left(\mathrm{:,}{\theta }_g\right)$are the \textit{FNO} models for the pressure, brine saturation and gas saturations respectively.

\noindent 

\noindent 
\section{3. Results for Physics Constrained Black Oil Model (Norne Field).}

\noindent Below we show the application of a physics constrained neural operator to solve the black oil model of a real field.

\noindent \includegraphics*[width=6.67in, height=6.67in]{image1}

\noindent 

\noindent \textbf{\textit{Figure 1(a):}}\textit{ Forwarding of the Norne Field. }$N_x=46\ ,\ ~N_y=112\ ,\ N_z=22$\textit{. At Time = 8 days. Dynamic properties comparison between the pressure, water saturation, oil saturation and gas saturation field between Nvidia Modulus's PINO surrogate (left column), Flow reservoir simulator (middle column) and the difference between both approaches (last column). They are 22 oil/water/gas producers (green), 9 water injectors (blue) and 4 gas injectors) red. We can see good concordance between the surrogate's prediction and the numerical reservoir simulator (Flow)}

\noindent 

\noindent 

\noindent 

\noindent \includegraphics*[width=6.67in, height=6.67in]{image2}

\noindent \textbf{\textit{Figure 1(b):}}\textit{ Forwarding of the Norne Field. }$N_x=46\ ,\ ~N_y=112\ ,\ N_z=22$\textit{. At Time = 968 days. Dynamic properties comparison between the pressure, water saturation, oil saturation and gas saturation field between Nvidia Modulus's PINO surrogate (left column), Flow reservoir simulator (middle column) and the difference between both approaches (last column). They are 22 oil/water/gas producers (green), 9 water injectors (blue) and 4 gas injectors) red. We can see good concordance between the surrogate's prediction and the numerical reservoir simulator (Flow)}

\noindent 

\noindent 

\noindent 

\noindent 

\noindent 

\noindent \includegraphics*[width=6.52in, height=5.97in, trim=0.85in 0.09in 0.75in 0.19in]{image3}

\noindent 

\noindent \textbf{\textit{Figure 1(c):}}\textit{ Forwarding of the Norne Field. }$N_x=46\ ,\ ~N_y=112\ ,\ N_z=22$\textit{. At Time = 2104 days. Dynamic properties comparison between the pressure, water saturation, oil saturation and gas saturation field between Nvidia Modulus's PINO surrogate (left column), Flow reservoir simulator (middle column) and the difference between both approaches (last column). They are 22 oil/water/gas producers (green), 9 water injectors (blue) and 4 gas injectors) red. We can see good concordance between the surrogate's prediction and the numerical reservoir simulator (Flow)}

\noindent 

\noindent 

\noindent 

\noindent \includegraphics*[width=5.86in, height=6.40in, trim=0.74in 0.08in 0.69in 0.19in]{image4}

\noindent \textbf{\textit{Figure 1(d):}}\textit{ Forwarding of the Norne Field. }$N_x=46\ ,\ ~N_y=112\ ,\ N_z=22$\textit{. At Time = 3298 days. Dynamic properties comparison between the pressure, water saturation, oil saturation and gas saturation field between Nvidia Modulus's PINO surrogate (left column), Flow reservoir simulator (middle column) and the difference between both approaches (last column). They are 22 oil/water/gas producers (green), 9 water injectors (blue) and 4 gas injectors) red. We can see good concordance between the surrogate's prediction and the numerical reservoir simulator (Flow)}

\noindent 

\noindent 

\noindent 

\noindent 

\noindent 

\noindent 

\noindent 
\section{References}

\begin{enumerate}
\item \textbf{ }Z. Duan and R. Sun, An improved model calculating CO2 solubility in pure water and aqueous NaCl solutions from 273 to 533 K and from 0 to 2000 bar., Chemical Geology,vol. 193.3-4, pp. 257-271, 2003.

\item  R. Span and W. Wagner, A new equation of state for carbon dioxide covering the fluid region from the triple-point temperature to 1100 K at pressure up to 800 MPa, J. Phys.Chem. Ref. Data, vol. 25, pp. 1509-1596, 1996.

\item  Fenghour and W. A. Wakeham, The viscosity of carbon dioxide, J. Phys. Chem. Ref.Data, vol. 27, pp. 31-44, 1998.

\item  S. L. Phillips et al., A technical data book for geothermal energy utilization, Lawrence Berkeley Laboratory report, 1981.

\item  J. E. Garcia, Density of aqueous solutions of CO2. No. LBNL-49023. Lawrence Berkeley National Laboratory, Berkeley, CA, 2001.

\item  Zaytsev, I.D. and Aseyev, G.G. Properties of Aqueous Solutions of Electrolytes, BocaRaton, Florida, USA CRC Press, 1993.

\item  Engineering ToolBox, Water - Density, Specific Weight and Thermal Expansion Coefficients, 2003

\item  Engineering Tool Box, Water - Dynamic (Absolute) and Kinematic Viscosity, 2004

\item  Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar. Fourier Neural Operator for Parametric Partial Differential Equations. https://doi.org/10.48550/arXiv.2010.08895

\item  Zongyi Li, Hongkai Zheng, Nikola Kovachki, David Jin, Haoxuan Chen, Burigede Liu, Kamyar Azizzadenesheli, Anima Anandkumar.Physics-Informed Neural Operator for Learning Partial Differential Equations. https://arxiv.org/pdf/2111.03794.pdf
\end{enumerate}

\noindent 

\noindent  
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

