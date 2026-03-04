import sympy as sp
import numpy as np

# Define symbolic variables
beta1, beta2, beta3, beta4, beta5 = sp.symbols('beta1 beta2 beta3 beta4 beta5')
v, theta, T, P = sp.symbols('v theta T P')
beta = [beta1, beta2, beta3, beta4, beta5]

# Define the function E(beta, v, θ, T, P)
E = beta1 * v**2 + beta2 * sp.sin(theta) + beta3 * sp.exp(beta4 * T) + beta5 * sp.log(P)

# Compute the gradient (partial derivatives of E with respect to the betas)
gradient = [sp.diff(E, b) for b in beta]

# Compute the Hessian (second-order derivatives of E with respect to betas)
hessian = [[sp.diff(grad, b) for b in beta] for grad in gradient]

# Lambdify for numerical evaluation
E_func = sp.lambdify((beta1, beta2, beta3, beta4, beta5, v, theta, T, P), E, 'numpy')
grad_func = sp.lambdify((beta1, beta2, beta3, beta4, beta5, v, theta, T, P), gradient, 'numpy')
hessian_func = sp.lambdify((beta1, beta2, beta3, beta4, beta5, v, theta, T, P), hessian, 'numpy')

# Function to read data from file
def read_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                values = list(map(float, line.split()))
                data.append(values)
    return data

# Load data from data_test.txt
data = read_data('data_test.txt')

# Assuming data file format: v, theta, T, P
for i, row in enumerate(data):
    E, v_val, theta_val, T_val, P_val = row
    
    # Example beta values (you can change these as needed)
    beta_vals = [0.726249,	0.915339,	0.594327,	0.515358,	0.975149]

    # Calculate the gradient and Hessian for each data point
    grad_val = grad_func(*beta_vals, v_val, theta_val, T_val, P_val)
    hessian_val = hessian_func(*beta_vals, v_val, theta_val, T_val, P_val)

    print(f"\nData Point {i + 1}: v={v_val}, θ={theta_val}, T={T_val}, P={P_val}")
    print("Gradient:")
    for idx, val in enumerate(grad_val):
        print(f"Gradient at index {idx}: {val}")

    print("\nHessian:")
    for row in hessian_val:
        print(" ".join(f"{val:.6f}" for val in row))
