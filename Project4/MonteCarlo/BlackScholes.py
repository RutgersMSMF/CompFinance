import numpy as np
from numba import jit
import matplotlib.pyplot as plt

# Define Parameters
S = 100                         # Stock Price
Drift = 0.0                     # Stochastic Drift 
Sigma = 0.10                    # Implied Volatility
N = 100                         # Simulation Count, Length

@jit(nopython = True)
def black_scholes(Tau):

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
                arr[i][j] = X + (Drift * X * dt) + (Sigma * X * B[j-1] * np.sqrt(dt))

    return arr

@jit(nopython = True)
def black_scholes_call():

    # Define Parameters
    S = 100                         # Stock Price
    K = [80, 90, 100, 110, 120]     # Strike Prices
    T = [0.1, 0.5, 1.0, 2.0 ,5.0]   # Time to Expiration

    arr = np.zeros((5, 5))

    for i in range(len(T)):

        # Fetch Black Scholes Simulation
        bs_simulation = black_scholes(T[i])

        for j in range(len(K)):

            win_rate = 0.0

            for k in range(len(bs_simulation[0])):

                if bs_simulation[k][-1] > K[j]:
                    win_rate+=1.0

            probability = win_rate / len(bs_simulation[0])
            price = S * probability - K[j] * probability

            if price > 0:
                arr[i][j] = price
            else:
                arr[i][j] = 0

    return arr

if __name__ == '__main__':

    def plot():

        stock = black_scholes(1)

        fig, (ax1) = plt.subplots(1, 1)
        fig.suptitle('Black Scholes Model')

        ax1.plot(stock.T)
        ax1.set_title("Stochastic Price Action")

        print(black_scholes_call())

        plt.show()

    plot()
