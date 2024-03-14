import numpy as np
from scipy.integrate import odeint

def lorentz_system(state, t, sigma, rho, beta):
    x, y, z, dxdt, dydt, dzdt = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt, 0, 0, 0]

def calculate_lyapunov_exponents(sigma, rho, beta, initial_state, num_steps, delta_t):
    # Initialize matrices to store Lyapunov exponents
    num_exponents = len(initial_state)
    exponents = np.zeros(num_exponents)
    tangent_space = np.eye(num_exponents)
    
    state = initial_state
    for i in range(num_steps):
        # Integrate the original and perturbed trajectories
        t = np.linspace(0, delta_t, 2)
        states = odeint(lorentz_system, state, t, args=(sigma, rho, beta))
        perturbed_states = odeint(lorentz_system, state + tangent_space * 1e-6, t, args=(sigma, rho, beta))
        
        # Compute the difference between the perturbed and original trajectories
        delta_states = perturbed_states[-1] - states[-1]
        
        # Perform QR decomposition to orthogonalize the tangent space
        tangent_space, _ = np.linalg.qr(np.dot(tangent_space, np.hstack((np.zeros((num_exponents, num_exponents)), delta_states.reshape(-1, num_exponents)))), mode='complete')
        
        # Compute the local Lyapunov exponents
        exponents += np.log(np.linalg.norm(tangent_space, axis=1))
        
        # Renormalize the tangent space
        tangent_space /= np.linalg.norm(tangent_space, axis=1)[:, None]
        
        # Update the state for the next iteration
        state = states[-1]

    # Average Lyapunov exponents over time
    exponents /= (num_steps * delta_t)
    
    return exponents

# Parameters for the Lorentz system
sigma = 16
rho = 45.92
beta = 8.0 / 2.0

# Initial state of the system [x, y, z, dx/dt, dy/dt, dz/dt]
initial_state = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

# Number of steps for Lyapunov exponent calculation
num_steps = 1000

# Time step for integration
delta_t = 0.01

# Calculate Lyapunov exponents
lyapunov_exponents = calculate_lyapunov_exponents(sigma, rho, beta, initial_state, num_steps, delta_t)

print("Lyapunov exponents:", lyapunov_exponents)
