import numpy as np
from numba import jit
import matplotlib.pyplot as plt 
from BlackScholes import black_scholes_price
import BroadieGlasserman as BG

K = 1
Sigma = 0.3 

@jit(nopython = True, parallel = True)
def option_payoff(x):

    P = np.zeros(len(x))

    for i in range(len(x)):

        temp = K - x[i]

        if temp < 0:
            P[i] = 0
        else:
            P[i] = temp

    return P

@jit(nopython = True, parallel = True)
def get_coeff(index, dt):

    sigma_squared = Sigma**2

    alpha = np.zeros(len(index))
    for i in range(len(index)):
        alpha[i] = (dt / 4) * -(sigma_squared * index[i]**2)

    beta = np.zeros(len(index))
    for i in range(len(index)):
        beta[i] = (dt / 2) * (sigma_squared * index[i]**2)

    gamma = np.zeros(len(index))
    for i in range(len(index)):
        gamma[i] = (-dt / 4) * (sigma_squared * index[i]**2)

    return alpha, beta, gamma

@jit(nopython = True)
def get_matrix(alpha, beta, gamma):

    d = np.diag(1 + beta)
    ud = np.diag(gamma[:-1], 1)
    ld = np.diag(alpha[1:], -1)
    
    A = d + ud + ld

    return A

@jit(nopython = True)
def get_vector(prices, alpha, beta, gamma):

    d = np.diag(1 - beta)
    ud = np.diag(-gamma[:-1], 1)
    ld = np.diag(-alpha[1:], -1)
    
    A = d + ud + ld
    b = np.dot(A, prices[1:-1])
    
    return b

@jit(nopython = True)
def crank_nicolson(option_payoff, A, alpha, beta, gamma):

    for i in range(len(option_payoff)):

        V = get_vector(option_payoff, alpha, beta, gamma)
        option_payoff[1:-1] = np.linalg.solve(A, V)

    return option_payoff

if __name__ == '__main__':

    def plot():

        N = 100
        M = 100
        T = 0.5
        dt = T / N

        s_naught = np.linspace(1e-8, 4.0, M + 1)

        # Part Two: Crank Nicolson Method

        index = np.linspace(1, M, M - 1)
        alpha, beta, gamma = get_coeff(index, dt)
        A = get_matrix(alpha, beta, gamma)

        fig, ax1 = plt.subplots(1, 1)
        fig.suptitle("Black Scholes Price")

        payoff = option_payoff(s_naught)
        cn_prices = crank_nicolson(option_payoff(s_naught), A, alpha, beta, gamma)
        bs_prices = black_scholes_price(M)

        ax1.plot(payoff, label = "Put Option Payoff")
        ax1.plot(cn_prices, label = "Crank Nicolson Price")
        ax1.plot(bs_prices, label = "Black Scholes Price")
        ax1.set_title("Analytic vs Numerical")
        ax1.legend(loc = "best")

        # Part Three: Compute Delta

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle("Replicating Portfolio")

        dt = s_naught[1] - s_naught[0]
        delta = BG.get_delta(bs_prices, dt)
        ax1.plot(delta)
        ax1.set_title("Option Delta")

        black_scholes_simulation = BG.black_scholes_simulation(M).T
        print("Confidence Interval: ", BG.confidence_interval(black_scholes_simulation, 0.95))

        pathwise_derivative = BG.pathwise_derivative(M, black_scholes_simulation)
        likelihood_ratio = BG.likelihood_ratio(M, black_scholes_simulation)

        ax2.plot(pathwise_derivative, label = "Pathwise Derivative")
        ax2.plot(likelihood_ratio, label = "Likelihood Ratio")
        ax2.set_title("Broadie Glasserman")
        ax2.legend(loc = "best")

        ax3.plot(black_scholes_simulation, label = "Black Scholes Simulation")
        ax3.set_title("Monte Carlo Simulation")

        plt.show()

    plot()