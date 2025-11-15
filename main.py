import numpy as np

#### we will replace later with real data ####
A0 = 1.0 # baseline productivity before 2023
g = 0.05 # AI adoption rate
N = 1000000 # avg # of employees in tech (treat as constant for now)
P = 0.1 # job-finding rate (not in our current equations but part of model)
L = 1200000 # total # of tech job seekers
p = 0.2 # stickiness of wages (also affects employment)
R = 0.5 # relation of innovation and wage
phi = 0.000001 # relation of employee # and wage

def A(t):
    """
    A(t) = (A0 * e^(gt)) / N
    """
    return (A0 * np.exp(g * t)) / N


def derivatives(t, E, W):
    """
    dE/dt = -(E / A(t)) + p * (L - E)
    dW/dt = p * (R * A(t) - phi * E - W)
    """
    At = A(t)

    dE_dt = -(E / At) + p * (L - E)
    dW_dt = p * (R * At - phi * E - W)

    return dE_dt, dW_dt, At


def simulate(E0, W0, T=20.0, dt=0.01):
    """
    Basic Euler simulation from t=0 to t=T with step size dt.
    Returns arrays for t, E, W, A.
    """
    n_steps = int(T / dt)
    t_vals = np.linspace(0, T, n_steps + 1)

    E_vals = np.zeros(n_steps + 1)
    W_vals = np.zeros(n_steps + 1)
    A_vals = np.zeros(n_steps + 1)

    # initial conditions
    E_vals[0] = E0
    W_vals[0] = W0
    A_vals[0] = A(0.0)

    for i in range(n_steps):
        t = t_vals[i]
        E = E_vals[i]
        W = W_vals[i]

        dE_dt, dW_dt, At = derivatives(t, E, W)

        # Euler update
        E_vals[i + 1] = E + dt * dE_dt
        W_vals[i + 1] = W + dt * dW_dt
        A_vals[i + 1] = A(t_vals[i + 1])

    return t_vals, E_vals, W_vals, A_vals


def main():
    # rough initial conditions (we'll swap in real data later)
    E0 = 900_000.0 # starting tech employment
    W0 = 1.0 # normalized starting wage

    T = 20.0 # simulate 20 time units (e.g. years)
    dt = 0.01

    t, E, W, A_vals = simulate(E0, W0, T, dt)

    print("Simulation finished.")
    print("Final values:")
    print("  Employment:", E[-1])
    print("  Wage:", W[-1])
    print("  A(t):", A_vals[-1])


if __name__ == "__main__":
    main()