import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import matplotlib.pyplot as plt 

def terminal_condition(x, K):
    
    y = x - K
    y[(x - K) < 0] = 0
    
    return y

def bs_solver(L = 1, K = 1/2, sigma = 2, T = 10, n = 32, m = 2**(8)):
    
    x = np.linspace(0, L, n + 1)
    t = np.linspace(0, T, m + 1)
    h = L / n
    dt = T / m
    
    sig = sigma**2 / (4 * h**2) * dt
        
    # set up terminal condition
    v = terminal_condition(x, K)
    V = [v]
    
    b = np.zeros(n-1)
    
    # set up boundary condition
    b[-1] = sig * x[n-1]**2 * (L - K)
        
    # Assemble matrix
    diagonal = x**2
    diagonals = [2 * diagonal[1:-1], -diagonal[1:-2], -diagonal[2:-1]]
    A = sp.diags(diagonals, [0,1,-1]).tocsc()

    A = np.eye(n - 1, n - 1) + sig * A

    
    # solve Euler method applied on BS equation backward
    for j in range(m - 1):
        V = la.solve(A, v[1:n] + b)
    
    # return mesh, timesteps and solution
    return x, t, V

def plot():

    x, t, V = bs_solver()

    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle("Black Scholes Price")

    ax1.plot(V)
    ax1.set_title("Euler Method")

    plt.show()

plot()