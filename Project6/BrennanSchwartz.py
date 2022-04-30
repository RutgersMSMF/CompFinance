from click import option
import numpy as np
from numba import jit
from scipy.linalg import lu
import matplotlib.pyplot as plt
from BlackScholes import black_scholes_price
from Derivatives import get_first_derivative, get_second_derivative

T = 0.5
r = 0.1
K = 1.0
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
        alpha[i] = (dt / 4) * (r * index[i] - sigma_squared * index[i]**2)

    beta = np.zeros(len(index))
    for i in range(len(index)):
        beta[i] = (dt / 2) * (r + sigma_squared * index[i]**2)

    gamma = np.zeros(len(index))
    for i in range(len(index)):
        gamma[i] = (-dt / 4) * (r * index[i] + sigma_squared * index[i]**2)

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
def get_new_matrix(option_payoff):

    L = len(option_payoff)
    sigma_squared = Sigma**2
    mu = r - (sigma_squared / 2)
    h = 1 / L

    alpha = np.zeros(L)
    for i in range(L):
        alpha[i] = (-sigma_squared / 2 * h) + (r * h / 6) + (mu /2)

    beta = np.zeros(L)
    for i in range(L):
        beta[i] = (sigma_squared / h) + (4 * r * h / 6)

    gamma = np.zeros(L)
    for i in range(L):
        gamma[i] = (-sigma_squared / 2 * h) + (r * h / 6) - (mu /2)

    d = np.diag(alpha)
    ud = np.diag(beta[:-1], 1)
    udd = np.diag(gamma[1:-1], 2)
    
    A = d + ud + udd
    
    return A

@jit(nopython = True)
def get_B_matrix(dt, option_payoff):

    mu = r - (Sigma**2 / 2)

    a = np.zeros(len(option_payoff))
    b = np.zeros(len(option_payoff))
    c = np.zeros(len(option_payoff))

    for i in range(len(option_payoff)):

        h = i + 1

        # Construct B Matrix
        a[i] = ((Sigma**2 / 2 * h) + (h * r / 6) * dt) + (mu / 2 * dt) + (h / 6)
        b[i] = (-(Sigma**2 / h) + (4 * r * h / 6) * dt) + (4 * h / 6)
        c[i] = ((Sigma**2 / 2 * h) + (h * r / 6) * dt) - (mu / 2 * dt) + (h / 6)

    d = np.diag(b)
    ud = np.diag(c[:-1], 1)
    ld = np.diag(a[1:], -1)

    B = d + ud + ld

    return B

@jit(nopython = True)
def crank_nicolson(option_payoff, A, alpha, beta, gamma):

    for i in range(len(option_payoff)):

        V = get_vector(option_payoff, alpha, beta, gamma)
        option_payoff[1:-1] = np.linalg.solve(A, V)

    return option_payoff

def brennan_schwartz(option_payoff, A, alpha, beta, gamma):

    # Initial Condition
    psi = np.copy(option_payoff)

    # Construct Diagonals
    d = np.diag(np.ones(len(A)))
    ud = np.diag(4 * np.ones(len(A) - 1), 1)
    udd = np.diag(np.ones(len(A) - 2), 2)

    # Construct M Matrix
    dt = T / len(option_payoff)
    h = 1 / len(option_payoff)
    M = (h / 6) * (d + ud + udd)

    B = get_B_matrix(dt, option_payoff)

    mu = r - (Sigma**2 / 2)
    a = ((Sigma**2 / 2 * h) + (h * r / 6) * dt) + (mu / 2 * dt) + (h / 6)

    for i in range(len(option_payoff[1:-1]) - 1):

        # Compute D Matrix
        g = dt * a * psi[0]
        d = np.dot(M, option_payoff[1:-1]) - g

        P, lower_matrix, upper_matrix = lu(B)

        # print("Lower: ", lower_matrix[0])
        # print("Upper Matrix: ", upper_matrix[0])

        # Solve for Y Vector
        y = np.linalg.solve(upper_matrix, d)

        for j in range(len(y) - 1):

            # Brennan Schwartz Algorithm
            if j == 0:
                temp = y[j] / lower_matrix[j][j]
                option_payoff[i + 1] = np.maximum(temp, psi[j])

            else:
                temp = (y[j] - lower_matrix[j][j - 1] * option_payoff[j - 1]) / lower_matrix[j][j]
                option_payoff[i + 1] = np.maximum(temp, psi[j])

    return option_payoff

def plot():

    N = 100
    M = 100
    T = 0.5
    dt = T / N

    s_naught = np.linspace(1e-8, 4.0, M + 1)

    index = np.linspace(1, M, M - 1)
    alpha, beta, gamma = get_coeff(index, dt)
    A = get_matrix(alpha, beta, gamma)

    fig, ax1 = plt.subplots(1, 1)
    fig.suptitle("Black Scholes Price")

    payoff = option_payoff(s_naught)
    european = crank_nicolson(option_payoff(s_naught), A, alpha, beta, gamma)
    american = brennan_schwartz(option_payoff(s_naught), A, alpha, beta, gamma)
    bs_prices = black_scholes_price(M)

    ax1.plot(payoff, label = "Put Option Payoff")
    ax1.plot(american, label = "Brennan Schwartz Price")
    ax1.plot(bs_prices, label = "Black Scholes Price")
    ax1.set_title("Analytic vs Numerical")
    ax1.legend(loc = "best")

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Comparison")

    ax1.plot(american, label = "American Put")
    ax1.plot(european, label = "European Put")
    ax1.legend(loc = 'best')
    ax1.set_title("Option Prices")

    ax2.plot(american - european)
    ax2.set_title("Difference")

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Derivatives")

    dt = s_naught[1] - s_naught[0]

    delta = get_first_derivative(bs_prices, dt)
    ax1.plot(delta)
    ax1.set_title("Option Delta")

    gamma = get_second_derivative(bs_prices, dt)
    ax2.plot(gamma)
    ax2.set_title("Option Gamma")

    plt.show()

plot()

