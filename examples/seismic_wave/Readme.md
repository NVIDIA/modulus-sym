# PINNs for simulating 2D seismic wave propagation

This example uses PINNs for emulating 2D time-dependent seismic wave propagation using a simple domain geometry. 

## Problem overview
This sample illustrates how to solve the acoustiv wave equation
You can get more details on this sample from the [documentation](https://docs.nvidia.com/deeplearning/modulus/modulus-sym-v110/user_guide/foundational/2d_wave_equation.html)

## Dataset

This example does not require any dataset as it solves of the acoustic wave equation, the physical geometry and the boundary conditions.

## Model overview and architecture

We use a simple fully connected MLP to approximate the solution of the 2D time-dependent wave equation for the given boundary conditions. The neural network will have  inputs and  outputs .

## Getting Started

To get started, simply run

```bash
python wave_2d.py
```

## References

- [Modulus Documentation](https://docs.nvidia.com/deeplearning/modulus/modulus-sym-v110/user_guide/foundational/2d_wave_equation.html)
