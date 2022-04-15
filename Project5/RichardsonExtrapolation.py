import numpy as np
from numba import jit
import matplotlib.pyplot as plt 
import CrankNicolson as CN

S = np.array([0.5, 1.0, 1.5])
T = 0.5

@jit(nopython = True, parallel = True)
def convergence():

    Z = 25
    dt = np.zeros(Z)
    dx = np.zeros(Z)
    N = np.linspace(10, 250, Z)

    for i in range(len(dt)):

        dt[i] = T / N[i]
        dx[i] = dt[i] / 0.5

    return dt, dx

@jit(nopython = True, parallel = True)
def get_rate_of_convergence(prices):

    convergence = np.zeros(len(prices) - 3)
    count = 0

    for i in range(2, len(prices) - 1):

        top = np.log(np.abs((prices[i + 1] - prices[i]) / (prices[i] - prices[i - 1])))
        bottom = np.log(np.abs((prices[i] - prices[i - 1]) / (prices[i - 1] - prices[i - 2])))

        convergence[count] = top / bottom
        count+=1

    return convergence

@jit(nopython = True, parallel = True)
def richardson_extrapolation(prices):

    true_prices = np.zeros(len(prices)-1)

    for i in range(len(prices) - 1):

        true_prices[i] = (4**(i + 1) * prices[i + 1] - prices[i]) / (4**(i + 1) - 1)

    return true_prices

@jit(nopython = True, parallel = True)
def get_option_prices():

    Z = 25
    dt, dx = convergence()
    N = np.linspace(10, 250, Z)

    atm_prices = np.zeros(len(dt))

    for i in range(len(dt)):

        # Compute Payoff
        s_naught = np.linspace(0.5, 1.5, int(N[i] + 1))
        payoff = CN.option_payoff(s_naught)

        # Compute Coeffs
        index = np.linspace(1, N[i], int(N[i] - 1))
        alpha, beta, gamma = CN.get_coeff(index, dt[i])
        A = CN.get_matrix(alpha, beta, gamma)

        # Fetch Prices
        cn_prices = CN.crank_nicolson(payoff, A, alpha, beta, gamma)
        atm_prices[i] = cn_prices[int(N[i] / 2)]

    return atm_prices

if __name__ == '__main__':

    def plot():

        dt, dx = convergence()

        fig, (ax1, ax2) = plt.subplots(1,  2)
        fig.suptitle("Intervals")

        ax1.plot(dt)
        ax1.set_title("Time Interval")

        ax2.plot(dx)
        ax2.set_title("Price Interval")

        atm_prices = get_option_prices()
        extrapolation = richardson_extrapolation(atm_prices)

        rate_of_convergence = get_rate_of_convergence(atm_prices)
        print("Estimated Rate of Convergence: ", rate_of_convergence)

        fig, (ax1, ax2) = plt.subplots(1,  2)
        fig.suptitle("Richardson Extrapolation")

        # Slice After Extrapolation
        atm_prices = atm_prices[1:]

        ax1.plot(atm_prices, label = "Not Extrapolated")
        ax1.plot(extrapolation, label = "Extrapolated")
        ax1.set_title("ATM Prices")
        ax1.legend(loc = "best")

        ax2.plot(np.abs(atm_prices - extrapolation))
        ax2.set_title("Difference")

        plt.show()

    plot()