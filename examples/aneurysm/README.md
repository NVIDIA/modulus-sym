# PINNs for simulating flow in a complex structure of an aneurysm

This example uses PINNs for emulating flow in an aneurysm taking the specific blood vessel geometry as the input. 

## Problem overview
This sample illustrates the features in Modulus Symbolic such as the geometry utilities to train a PINN model for a complex geometry. The trained surrogate PINN can be used to predict the pressure and velocity of blood flow inside the aneursym. You can get more details on this sample from the [documentation](https://docs.nvidia.com/deeplearning/modulus/modulus-sym-v110/user_guide/intermediate/adding_stl_files.html). This sample can be extended to incroporate parameterized geometry using CCSG instead of STL to leverage the surrogate model to provide patient specific geometry.  You can refer to the documentaion [here](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/advanced/parametrized_simulations.html#creating-nodes-and-architecture-for-parameterized-problems) for more details. 

## Dataset

This example does not require any dataset as it solves of the Navier Stokes flow using the equations, the physical geometry and the boundary conditions.

## Model overview and architecture

We use a simple fully connected MLP to approximate the solution of the Navier-Stokes equations for the given boundary conditions. The neural network will have two inputs x,y and three outputs u, v, p.

## Getting Started

To get started, simply run

```bash
python aneurysm.py
```

## References

- [Modulus Documentation](https://docs.nvidia.com/deeplearning/modulus/modulus-sym-v110/user_guide/intermediate/adding_stl_files.html)
