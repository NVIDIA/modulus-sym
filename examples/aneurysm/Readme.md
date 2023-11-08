# PINNs for simulating flow in a complex structure of an Aneurysm

This example uses PINNs for emulating flow in an aneurysm taking the patient specific geometry as the input. 

## Problem overview
This sample illustrates the ability to use the geometry utilities to train a PINN model for a complex geometry. The trained surrogate PINN can be used to predict the flow in patient specific vessel geometries.
You can get more details on this sample from the [documentation](https://docs.nvidia.com/deeplearning/modulus/modulus-sym-v110/user_guide/intermediate/adding_stl_files.html)

## Dataset

This example does not require any Dataset as it solves of the Navier Stokes flow using the equations, the physical geometry and the boundary conditions.

## Model overview and architecture

We use a simple fully connected MLP to approximate the solution of the Navier-Stokes equations for the given boundary conditions. The neural network will have two inputs x,y and three outputs u, v, p.

## Getting Started

To get started, simply run

```bash
python aneurysm.py
```

## References

- [Modulus Documentation](https://docs.nvidia.com/deeplearning/modulus/modulus-sym-v110/user_guide/intermediate/adding_stl_files.html)
