import numpy as np
from numba import jit 
import matplotlib.pyplot as plt
from BlackScholes import black_scholes_price
from Derivatives import get_first_derivative, get_second_derivative

# Option Parameters
T = 0.5
S_0 = 1
K = 1
sigma = 0.3
r = 0.1
price = 0

# SOR Parameters
tol = 0.001
omega = 1.2

# Boundary Condition
S_max = 4 * S_0

# Grid Sizing
N = 250
M = 100
dt = T / N
ds = S_max / M

I = np.arange(0, M+1)
J = np.arange(0, N+1)
old_val = np.zeros(M-1)
new_val = np.zeros(M-1)
interval = np.linspace(0, 1.5, M)

# @jit(nopython = True, parallel = True)
def successive_over_relaxation():
    """
    Computes the Price of An American Option USing Successive OVer Relaxation
    """

    prices = np.zeros(M)

    for i in range(M):

        K = interval[i] 

        # Boundary and final conditions
        payoff = np.maximum(K - I[1:M] * ds, 0)
        old_layer = payoff
        bound_val = K * np.exp(-r * (N - J) * dt)

        # Calculating elements of M
        alpha = 0.25 * dt * (sigma**2 * (I**2) - r * I)
        alpha = alpha[1:]
        beta = -dt * 0.5 * (sigma**2 * (I**2) + r)
        beta = beta[1:]
        gamma = 0.25 * dt * (sigma**2 * (I**2) + r * I)
        gamma = gamma[1:]

        M2 = np.diag(1+beta[:M-1]) + np.diag(alpha[1:M-1], k=-1) + np.diag(gamma[:M-2], k=1)
        b = np.zeros(M-1)

        for j in range(N-1, -1, -1):

            b[0] = alpha[0] * (bound_val[j] + bound_val[j+1])
            rhs = M2 @ old_layer + b
            old_val = old_layer
            error = 1

            while error > tol:

                new_val[0] = np.maximum(payoff[0], old_val[0] + (omega/(1-beta[0]))*(rhs[0] - (1-beta[0])*old_val[0] + gamma[0] * old_val[1]))

                for k in range(1, M-2):

                    new_val[k] = np.maximum(payoff[k], old_val[k] + (omega / (1 - beta[k])) * (rhs[k] - (1 - beta[k]) * old_val[k] + alpha[k] * new_val[k-1] + gamma[k] * old_val[k+1]))
                
                new_val[M-2] = np.maximum(payoff[M-2], old_val[M-2] + (omega / (1 - beta[M-2])) * (rhs[M-2] - (1 - beta[M-2]) * old_val[M-2] + alpha[M-2] * new_val[M-3]))
                error = np.linalg.norm(new_val - old_val)
                old_val = new_val

            old_layer = new_val

        prices_t0 = np.concatenate(([bound_val[0]], old_layer, [0]))
        idown = int(np.floor(S_0 / ds))
        iup = int(np.ceil(S_0 / ds))

        if idown == iup:
            price = prices_t0[idown]
        else:
            price = prices_t0[idown] + ((iup - (S_0 / ds)) / (iup - idown)) * (prices_t0[iup] - prices_t0[idown])

        prices[i] = price     

    return prices[::-1], bound_val

if __name__ == "__main__":

    def main():

        N = 250
        M = 100
        dt = T / N

        # Compute Option Prices
        american, boundary = successive_over_relaxation()
        european = black_scholes_price(M)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("American vs European")

        ax1.plot(american, label = "American")
        ax1.plot(european, label = "European")
        ax1.legend(loc = 'best')
        ax1.set_title("Option Prices")

        ax2.plot(american - european[:-1])
        ax2.set_title("Difference")

        fig, (ax1) = plt.subplots(1, 1)

        ax1.plot(boundary)
        ax1.set_title("Exercise Boundary")

        # converge = np.zeros(25)

        # NN = np.linspace(10, 250, 25, dtype = np.int64)
        # MM = np.linspace(10, 100, 25, dtype = np.int64)

        # # Estimate Order of Convergence
        # for i in range(25):
        #     N = NN[i]
        #     M = MM[i]
        #     dt = T / N
            
        #     converge[i] = successive_over_relaxation()[int(M / 4)]

        # fig, (ax1, ax2) = plt.subplots(1, 2)

        # ax1.plot(converge)
        # ax1.set_title("Rate of Convergence")

        # Compute Derivatives
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Derivatives")

        s_naught = np.linspace(1e-8, 1.5, len(european))
        dt = s_naught[1] - s_naught[0]

        delta = get_first_derivative(european, dt)
        ax1.plot(delta)
        ax1.set_title("Option Delta")

        gamma = get_second_derivative(european, dt)
        ax2.plot(gamma)
        ax2.set_title("Option Gamma")

        plt.show()

        return 0

    main()