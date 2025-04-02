# Solving the Time-Fractional Burgers-Huxley Equation with PINNs  

## Overview  
This repository presents an implementation of Physics-Informed Neural Networks (PINNs) to solve the time-fractional Burgers-Huxley equation. The Burgers-Huxley equation describes reaction-diffusion dynamics and appears in various physical and biological systems. Introducing a fractional-order derivative enhances the model's ability to capture memory-dependent behaviors.  

## Problem Statement  
The time-fractional Burgers-Huxley equation is a nonlinear partial differential equation (PDE) with a fractional time derivative. Traditional numerical methods like finite difference methods (FDM) struggle with stability and computational efficiency when handling fractional derivatives. PINNs offer an alternative by incorporating the governing equation directly into the loss function, enabling efficient learning-based approximations.  

## Methodology  
### 1. **Fractional Derivative Approximation**  
   - Implemented the Caputo fractional derivative within the PINN framework.  
   - Used automatic differentiation in TensorFlow to compute required gradients.  

### 2. **PINN Formulation**  
   - Designed a neural network with input variables (time, space) and output \( u(x,t) \).  
   - Enforced physical constraints through a physics-informed loss function.  
   - Trained the model using Adam and L-BFGS optimizers for improved convergence.  

### 3. **Comparison with Finite Difference Method (FDM)**  
   - Implemented FDM to obtain numerical solutions as a benchmark.  
   - Compared PINN results against FDM to validate accuracy.  

## Implementation Details  
- **Framework:** Pytorch  
- **Loss Function:** Physics-informed loss combining equation residuals and boundary/initial conditions  
- **Optimization:** Adam  
- **Evaluation Metrics:** Mean squared error (MSE) and L2 error between numerical and PINN solutions  

## Results  
- PINNs successfully approximated the fractional Burgers-Huxley equation.  
- The method demonstrated strong agreement with FDM solutions.  
- Reduced the need for explicit meshing, making it adaptable to complex domains.  

## Future Work  
- Extend PINNs to higher-dimensional fractional PDEs.  
- Investigate different fractional derivative definitions (Caputo, Riemann-Liouville).  
- Apply Singular Value Decomposition (SVD) for dimensionality reduction before PINN training.  
