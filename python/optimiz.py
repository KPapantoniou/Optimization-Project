import numpy as np
import matplotlib.pyplot as plt
import random
import sympy as sp
# ============================
# 1. Data Loading Function
# ============================

def load_data(file_path):
    data = np.loadtxt(file_path)
    v = data[:, 0]
    theta = data[:, 1]
    T = data[:, 2]
    P = data[:, 3]
    E = data[:, 4]
    return v, theta, T, P, E

# Load training and test data
v_train, theta_train, T_train, P_train, E_train = load_data('data_train.txt')
v_test, theta_test, T_test, P_test, E_test = load_data('data_test.txt')
beta1, beta2, beta3, beta4, beta5 = sp.symbols('beta1 beta2 beta3 beta4 beta5')
beta = [beta1, beta2, beta3, beta4, beta5]
v, theta, T, P = sp.symbols('v theta T P')
E = beta1 * v**2 + beta2 * sp.sin(theta) + beta3 * sp.exp(beta4 * T) + beta5 * sp.log(P)

# ============================
# 2. Model and MSE Definition
# ============================

def energy_model(beta, v, theta, T, P):
    return beta[0] * v**2 + beta[1] * np.sin(theta) + beta[2] * np.exp(beta[3] * T) + beta[4] * np.log(P)

def mse(beta, v, theta, T, P, E_true):
    E_pred = energy_model(beta, v, theta, T, P)
    return np.mean((E_pred - E_true)**2)

# ============================
# 3. Optimization Algorithms
# ============================

# --- Newton Trust Region (NewtonTR) ---
def newton_trust_region(beta, v, theta, T, P, E_true, max_iter=100, delta=1.0, tol=1e-6):
    def compute_gradient(beta):
        epsilon = 1e-6
        gradient = np.zeros(len(beta))
        for i in range(len(beta)):
            beta_plus = beta.copy()
            beta_minus = beta.copy()
            beta_plus[i] += epsilon
            beta_minus[i] -= epsilon
            gradient[i] = (mse(beta_plus, v, theta, T, P, E_true) - mse(beta_minus, v, theta, T, P, E_true)) / (2 * epsilon)
        gradient = [sp.diff(E, b) for b in beta]

        return gradient

    def compute_hessian(beta,gradient):
        epsilon = 1e-4
        n = len(beta)
        hessian = np.zeros((n, n))
        hessian = [[sp.diff(grad, b) for b in beta] for grad in gradient]

        for i in range(n):
            for j in range(n):
                beta_ijp = beta.copy()
                beta_ijp[i] += epsilon
                beta_ijp[j] += epsilon
                
                beta_ijm = beta.copy()
                beta_ijm[i] -= epsilon
                beta_ijm[j] -= epsilon
                
                beta_ip_jm = beta.copy()
                beta_ip_jm[i] += epsilon
                beta_ip_jm[j] -= epsilon
                
                beta_im_jp = beta.copy()
                beta_im_jp[i] -= epsilon
                beta_im_jp[j] += epsilon
                
                hessian[i, j] = (mse(beta_ijp, v, theta, T, P, E_true) - mse(beta_ip_jm, v, theta, T, P, E_true) - mse(beta_im_jp, v, theta, T, P, E_true) + mse(beta_ijm, v, theta, T, P, E_true)) / (4 * epsilon**2)
        return hessian

    for i in range(max_iter):
        grad = compute_gradient(beta)
        hessian = compute_hessian(beta,grad)
  
        print("Hessian:")
        for row in hessian:
            
            print(row)
        # print(hessian)
        if np.linalg.norm(grad) < tol:
            print("NewtonTR: Convergence achieved!")
            break

        try:
            p = np.linalg.solve(hessian, -grad)
        except np.linalg.LinAlgError:
            p = -grad

        if np.linalg.norm(p) > delta:
            p = (delta / np.linalg.norm(p)) * p

        beta_new = beta + p
        if mse(beta_new, v, theta, T, P, E_true) < mse(beta, v, theta, T, P, E_true):
            beta = beta_new
            delta *= 1.5
        else:
            delta *= 0.5

    return beta

# --- BFGS with Wolfe Line Search ---
def bfgs(beta, v, theta, T, P, E_true, max_iter=100, tol=1e-6):
    def compute_gradient(beta):
        epsilon = 1e-6
        grad = np.zeros(len(beta))
        for i in range(len(beta)):
            beta_plus = beta.copy()
            beta_minus = beta.copy()
            beta_plus[i] += epsilon
            beta_minus[i] -= epsilon
            grad[i] = (mse(beta_plus, v, theta, T, P, E_true) - mse(beta_minus, v, theta, T, P, E_true)) / (2 * epsilon)
        return grad

    n = len(beta)
    H = np.eye(n)
    grad = compute_gradient(beta)

    for i in range(max_iter):
        if np.linalg.norm(grad) < tol:
            print("BFGS: Convergence achieved!")
            break

        p = -H.dot(grad)

        alpha = 1.0
        c1, c2 = 1e-4, 0.9
        while mse(beta + alpha * p, v, theta, T, P, E_true) > mse(beta, v, theta, T, P, E_true) + c1 * alpha * grad.dot(p):
            alpha *= 0.5

        beta_new = beta + alpha * p
        grad_new = compute_gradient(beta_new)
        s = beta_new - beta
        y = grad_new - grad

        if s.dot(y) > 0:
            rho = 1.0 / y.dot(s)
            H = (np.eye(n) - rho * np.outer(s, y)).dot(H).dot(np.eye(n) - rho * np.outer(y, s)) + rho * np.outer(s, s)

        beta = beta_new
        grad = grad_new

    return beta

# --- Nelder-Mead ---
def nelder_mead(beta, v, theta, T, P, E_true, max_iter=100):
    simplex = [beta + np.random.uniform(-1, 1, size=5) for _ in range(6)]

    for i in range(max_iter):
        simplex = sorted(simplex, key=lambda b: mse(b, v, theta, T, P, E_true))
        best, worst = simplex[0], simplex[-1]
        centroid = np.mean(simplex[:-1], axis=0)
        reflection = centroid + (centroid - worst)

        if mse(reflection, v, theta, T, P, E_true) < mse(best, v, theta, T, P, E_true):
            expansion = centroid + 2 * (reflection - centroid)
            if mse(expansion, v, theta, T, P, E_true) < mse(reflection, v, theta, T, P, E_true):
                simplex[-1] = expansion
            else:
                simplex[-1] = reflection
        else:
            contraction = centroid + 0.5 * (worst - centroid)
            simplex[-1] = contraction

    return simplex[0]

# --- Genetic Algorithm (GA) ---
def genetic_algorithm(v, theta, T, P, E_true, pop_size=50, generations=100, mutation_rate=0.2):
    def random_beta():
        return np.random.uniform([10, -10, 50, 0.01, 0.1], [100, 10, 200, 0.1, 1])

    population = [random_beta() for _ in range(pop_size)]

    for gen in range(generations):
        population = sorted(population, key=lambda b: mse(b, v, theta, T, P, E_true))
        offspring = []
        for _ in range(pop_size // 2):
            parent1, parent2 = random.choices(population[:pop_size // 2], k=2)
            crossover = (parent1 + parent2) / 2
            if random.random() < mutation_rate:
                mutation = random_beta()
                crossover += mutation - crossover
            offspring.append(crossover)
        population = population[:pop_size // 2] + offspring

    return population[0]

# --- Particle Swarm Optimization (PSO) ---
# --- Particle Swarm Optimization (PSO) ---
def particle_swarm_optimization(v, theta, T, P, E_true, particles=30, max_iter=100):
    lb, ub = np.array([10, -10, 50, 0.01, 0.1]), np.array([100, 10, 200, 0.1, 1])
    swarm = [np.random.uniform(lb, ub) for _ in range(particles)]
    velocities = [np.random.uniform(-1, 1, size=5) for _ in range(particles)]
    personal_best = swarm.copy()
    personal_best_scores = [mse(b, v, theta, T, P, E_true) for b in swarm]
    global_best = min(personal_best, key=lambda b: mse(b, v, theta, T, P, E_true))
    global_best_score = mse(global_best, v, theta, T, P, E_true)

    w = 0.5  # inertia weight
    c1 = 1.5  # cognitive component
    c2 = 1.5  # social component

    for iteration in range(max_iter):
        for i in range(particles):
            # Update velocity
            r1, r2 = np.random.rand(2)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (personal_best[i] - swarm[i]) +
                             c2 * r2 * (global_best - swarm[i]))

            # Update position
            swarm[i] = swarm[i] + velocities[i]

            # Enforce bounds
            swarm[i] = np.clip(swarm[i], lb, ub)

            # Evaluate the new score
            score = mse(swarm[i], v, theta, T, P, E_true)

            # Update personal best
            if score < personal_best_scores[i]:
                personal_best[i] = swarm[i]
                personal_best_scores[i] = score

                # Update global best
                if score < global_best_score:
                    global_best = swarm[i]
                    global_best_score = score

        # Optionally print the progress
        # print(f"Iteration {iteration + 1}/{max_iter}, Best Score: {global_best_score}")

    return global_best


if __name__ == "__main__":
    # Load training data
    v_train, theta_train, T_train, P_train, E_train = load_data('data_train.txt')
    beta1 = random.uniform(10, 100)
    beta2 = random.uniform(-10, 10)
    beta3 = random.uniform(50, 200)
    beta4 = random.uniform(0.01, 0.1)
    beta5 = random.uniform(0.1, 1)
    
    # Initial guess for beta parameters
    initial_beta = np.array([beta1, beta2, beta3, beta4, beta5])    

    # Run each optimization algorithm and save results
    # Newton Trust Region
    beta_newton_tr = newton_trust_region(initial_beta, v_train, theta_train, T_train, P_train, E_train)
    mse_newton_tr = mse(beta_newton_tr, v_train, theta_train, T_train, P_train, E_train)
    message_newton = f"MSE: {mse_newton_tr}\tBETA: " + "\t".join(map(str, beta_newton_tr))
    # np.savetxt('newton_tr_results.txt', message_newton)
    with open('newton_tr_results.txt', 'w') as f:
        f.write(message_newton)
    # BFGS
    beta_bfgs = bfgs(initial_beta, v_train, theta_train, T_train, P_train, E_train)
    mse_bfgs = mse(beta_bfgs, v_train, theta_train, T_train, P_train, E_train)
    message_bfgs = f"MSE: {mse_bfgs}\tBETA: " + "\t".join(map(str, beta_bfgs))
    with open('bfgs_results.txt', 'w') as f:
        f.write(message_bfgs)
    # np.savetxt('bfgs_results.txt', beta_bfgs)

    # Nelder-Mead
    beta_nelder_mead = nelder_mead(initial_beta, v_train, theta_train, T_train, P_train, E_train)
    mse_nelder_mead = mse(beta_nelder_mead, v_train, theta_train, T_train, P_train, E_train)
    message_nelder = f"MSE: {mse_nelder_mead}\tBETA: " + "\t".join(map(str, beta_nelder_mead))
    with open('nelder_mead_results.txt', 'w') as f:
        f.write(message_nelder)
    # np.savetxt('nelder_mead_results.txt', beta_nelder_mead)
    
    # Genetic Algorithm
    beta_genetic = genetic_algorithm(v_train, theta_train, T_train, P_train, E_train)
    mse_genetic = mse(beta_genetic, v_train, theta_train, T_train, P_train, E_train)
    message_genetic = f"MSE: {mse_genetic}\tBETA: " + "\t".join(map(str, beta_genetic))
    with open('genetic_results.txt', 'w') as f:
        f.write(message_genetic)
    # np.savetxt('genetic_results.txt', beta_genetic)
    
    # Particle Swarm Optimization
    beta_pso = particle_swarm_optimization(v_train, theta_train, T_train, P_train, E_train)
    mse_pso = mse(beta_pso, v_train, theta_train, T_train, P_train, E_train)
    message_pso = f"MSE: {mse_pso}\tBETA: " + "\t".join(map(str, beta_pso))
    with open('pso_results.txt', 'w') as f:
        f.write(message_pso)
    # np.savetxt('pso_results.txt', beta_pso)

    print("Optimization results have been saved to respective files.")