<!-- markdownlint-disable MD043 -->
# NVIDIA Modulus Sym Examples

## Introduction

This repository provides sample applications demonstrating use of specific Physics-ML
model architectures that are easy to train and deploy. These examples aim to show how
such models can help solve real world problems.

## Introductory

|Use case|Model|Level|Attributes|
| --- | --- |  --- | --- |
|Ldc| Fully Connected MLP PINN |Introductory|Steady state, Multi-GPU|
|Anti_derivative| Data and Physics informed DeepONet |Introductory|Steady state, Multi-GPU|
|Darcy Flow| FNO, AFNO, PINO |Introductory|Steady state, Multi-GPU|
|ODE_spring_mass| Fully Connected MLP PINN |Introductory|Steady state, Multi-GPU|
|Surface_pde| Fully Connected MLP PINN |Introductory|Steady state, Multi-GPU|

## Turbulence

|Use case|Model|Level|Attributes|
| --- | --- | --- | --- |
|Taylor_green| Fully Connected MLP PINN | Intermediate |Steady state, Multi-GPU|
|Turbulent_channel| Fourier Feature MLP PINN |Intermediate|Steady state, Multi-GPU|
|Super_resolution| Super Resolution Network, Pix2Pix |Intermediate|Steady state, Multi-GPU|

## Electromagnetics

|Use case|Model|Level|Attributes|
| --- | --- | --- | --- |
|Waveguide| Fourier Feature MLP PINN |Intermediate|Steady state, Multi-GPU|

## Solid Mechanics

|Use case|Model|Level|Attributes|
| --- | --- | --- | --- |
|Plane_displacement| Fully Connected MLP PINN, VPINN |Intermediate|Steady state, Multi-GPU|

## Design Optimization

|Use case|Model|Level|Attributes|
| --- | --- | --- | --- |
|Chip_2D| Fully Connected MLP PINN |Advanced|Steady state, Multi-GPU|
|Three_fin_3D| Fully Connected MLP PINN | Advanced |Steady state, Multi-Node|
|FPGA| Multiple Models (including Fourier Feature MLP PINN, SIRENS, etc.) |Advanced|Steady state, Multi-Node|
|Limerock| Fourier Feature MLP PINN |Advanced|Steady state, Multi-Node|

## Geophyscis

|Use case|Model|Level|Attributes|
| --- | --- | --- | --- |
|Reservoir simulation| FNO, PINO | Advanced | Steady state, Multi-Node|
|Seismic wave| Fully Connected MLP PINN |Intermediate|Steady state, Multi-Node|
|Wave_equation| Fully Connected MLP PINN |Intermediate|Steady state, Multi-Node|

## Healthcare

|Use case|Model|Level|Attributes|
| --- | --- | --- | --- |
|Aneurysm| Fully Connected MLP PINN |Intermediate|Steady state, Multi-Node|


## NVIDIA support

In each of the example READMEs, we indicate the level of support that will be provided.
Some examples are under active development/improvement and might involve rapid changes.
For stable examples, please refer the tagged versions.

## Feedback / Contributions

We're posting these examples on GitHub to better support the community, facilitate
feedback, as well as collect and implement contributions using
[GitHub issues](https://github.com/NVIDIA/modulus-launch/issues) and pull requests.
We welcome all contributions!
