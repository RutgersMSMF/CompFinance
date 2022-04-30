import numpy as np
from numba import jit

@jit(nopython = True)
def get_order_of_convergence():

    return 0

@jit(nopython = True)
def get_first_derivative(option_prices, dt):

    delta = np.zeros(len(option_prices))

    for i in range(1, len(option_prices) - 1):

        # Central Difference Method to Compute Delta
        delta[i] = (option_prices[i + 1] - option_prices[i - 1]) / (2 * dt) * 100

    return delta[1:-1]

@jit(nopython = True)
def get_second_derivative(option_prices, dt):

    gamma = np.zeros(len(option_prices))

    for i in range(1, len(option_prices) - 1):

        # Central Difference Method to Compute Gamma
        gamma[i] = (option_prices[i + 1] - 2 * option_prices[i] + option_prices[i - 1]) / (4 * dt**2) * 100

    return gamma[1:-1]



    