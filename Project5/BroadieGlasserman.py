import numpy as np
from numba import jit 

@jit(nopython = True)
def confidence_interval(arr, clv):

    xbar = np.mean(arr)
    std = np.std(arr)

    return (xbar - clv * std / np.sqrt(len(arr)), xbar + clv * std / np.sqrt(len(arr)))

@jit(nopython = True, parallel = True)
def get_delta(option_prices, dt):

    delta = np.zeros(len(option_prices))

    for i in range(1, len(option_prices) - 1):

        # Central Difference Method to Compute Delta
        delta[i] = (option_prices[i + 1] - option_prices[i - 1]) / (2 * dt) * 100

    return delta[1:-1]

@jit(nopython = True, parallel = True)
def black_scholes_simulation(N):

    S = 1
    T = 0.5
    drift = 0
    sigma = 0.3
    dt = 1.0 / N
    arr = np.zeros((N, N))

    for i in range(N):

        # Generate Random Normals
        B = np.random.normal(0, 1, N)

        for j in range(N):

            # Initial Condition
            if j == 0:
                arr[i][j] = S 
            else:
                X = arr[i][j-1]
                arr[i][j] = X + (drift * X * dt) + (sigma * X * B[j-1] * np.sqrt(dt))

    return arr

@jit(nopython = True, parallel = True)
def pathwise_derivative(N, arr):

    K = 1
    T = 0.5
    dt = 1 / N
    sigma = 0.3
    delta = np.zeros(N)

    for i in range(len(arr)):

        B = np.random.normal(0, 1, N)
        sum = 0

        for j in range(len(arr[i])):

            sum += np.maximum(K - arr[i][j], 0) * np.exp(sigma * B[j] * np.sqrt(dt) - 0.5 * sigma**2 * T)

        delta[i] = (sum / N) * 100

    return delta

@jit(nopython = True, parallel = True)
def likelihood_ratio(N, arr):

    K = 1
    S = 1
    T = 0.5
    sigma = 0.3
    delta = np.zeros(N)

    for i in range(len(arr)):

        sum = 0

        for j in range(len(arr[i])):

            sum += np.maximum(K - arr[i][j], 0) * ((np.log(arr[i][j] / S) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))) * (1 / (S * sigma * np.sqrt(T)))

        delta[i] = (sum / N) * 100

    return delta

