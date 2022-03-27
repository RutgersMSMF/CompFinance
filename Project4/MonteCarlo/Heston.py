import numpy as np
from numba import jit
import matplotlib.pyplot as plt

# Define Model Parameters 
S = 100                         # Stock Price
Theta = 0.01                    # Mean Reversion Level
Sigma = 0.01                    # Implied Volatility
Kappa = 2.0                     # Mean Reversion Speed
Rho = -0.5                      # Correlation 
Beta = 0.1                      # Vol of Vol
Tau = 1                         # Time to Expiry
N = 100                         # Simulation Count, Length

@jit(nopython = True)
def heston(Tau, kappa, beta, rho):

    dt = 1.0 / N
    volatility_process = np.zeros((N, N))
    stock_process = np.zeros((N, N))

    for i in range(N):

        # Generate Random Normals
        B = np.random.normal(0, 1, N)
        W = np.random.normal(0, 1, N)

        for j in range(N):

            # Set Initial Condtion
            if j == 0:
                volatility_process[i][j] = Sigma 
                stock_process[i][j] = S
            else:
                # Euler Method Simulation
                vol_drift =  kappa * (Theta - Sigma) * dt
                vol_diffusion = beta * np.sqrt(Sigma) * (rho * B[j - 1] * np.sqrt(dt) + np.sqrt(1 - rho**2) * W[j - 1] * np.sqrt(dt))
                volatility_process[i][j] = volatility_process[i][j - 1] + vol_drift + vol_diffusion
                stock_process[i][j] = stock_process[i][j - 1] + np.sqrt(volatility_process[i][j]) * stock_process[i][j - 1] * B[j - 1] * np.sqrt(dt)

    return volatility_process, stock_process

@jit(nopython = True)
def heston_model(kappa, beta, rho):

    # Define Parameters
    S = 100                         # Stock Price
    K = [80, 90, 100, 110, 120]     # Strike Prices
    T = [0.1, 0.5, 1.0, 2.0 ,5.0]   # Time to Expiration

    arr = np.zeros((5, 5))

    for i in range(len(T)):

        # Fetch Stein Stein Simulation
        heston_simulation = heston(T[i], kappa, beta, rho)[1]

        for j in range(len(K)):

            win_rate = 0

            for k in range(len(heston_simulation[0])):

                if heston_simulation[k][-1] > K[j]:
                    win_rate+=1

            probability = win_rate / len(heston_simulation[0])
            price = S * probability - K[j] * probability 

            if price > 0:
                arr[i][j] = price
            else:
                arr[i][j] = 0       

    return arr

if __name__ == '__main__':

    def plot():

        vol, stock = heston(1, Kappa, Beta, Rho)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Heston Model')

        ax1.plot(vol.T)
        ax1.set_title("Stochastic Volatility")
        ax2.plot(stock.T)
        ax2.set_title("Stochastic Price Action")

        prices = heston_model(Kappa, Beta, Rho)
        print(prices)

        fig, (ax1) = plt.subplots(1, 1)
        fig.suptitle('Heston Model')

        ax1.plot(prices.T)
        ax1.set_title("Option Prices")

        plt.show()

    plot()
