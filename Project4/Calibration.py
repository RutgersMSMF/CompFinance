import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from py_vollib.black_scholes.implied_volatility import implied_volatility
from MonteCarlo.SteinStein import stein_stein_model

# Define Parameters
S = 100                         # Stock Price
K = [80, 90, 100, 110, 120]     # Strike Prices
T = [0.1, 0.5, 1.0, 2.0 ,5.0]   # Time to Expiration
Kappa = 2.0                     # Mean Reversion Speed
Rho = -0.5                      # Correlation 
Beta = 0.05                     # Vol of Vol
N = 100

def get_implied_volatility(option_prices):
    """
        Jaeckel Method
        Fast Numerical Implementation of Black Scholes Implied Volatility
    """

    t = np.array([0.1, 0.5, 1, 2, 5])
    iVol = np.zeros((5, 5))
    S = 100
    K = 100

    for i in range(len(t)):

        for j in range(len(option_prices[i])):

            iVol[i][j] = implied_volatility(option_prices[i][j], S, K, t[i], 0.0, 'c')

    return iVol

def atm_vol_kappa():

    iv = np.zeros(N)
    kappa = np.linspace(0, 5, N)
    T = 1

    for i in range(N):

        call_prices = stein_stein_model(kappa[i], Beta, Rho)
        atm_call_prices = (call_prices[1][1] + call_prices[1][2]) / 2.0

        iVol = implied_volatility(atm_call_prices, 100, 100, T, 0.0, 'c')
        iv[i] = iVol

    return iv

def atm_vol_beta():

    iv = np.zeros(N)
    beta = np.linspace(0, 0.5, N)
    T = 1

    for i in range(N):

        call_prices = stein_stein_model(Kappa, beta[i], Rho)
        atm_call_prices = (call_prices[1][1] + call_prices[1][2]) / 2.0

        iVol = implied_volatility(atm_call_prices, 100, 100, T, 0.0, 'c')
        iv[i] = iVol

    return iv

def atm_vol_rho():

    iv = np.zeros(N)
    rho = np.linspace(-1, 1, N)
    T = 1

    for i in range(N):

        call_prices = stein_stein_model(Kappa, Beta, rho[i])
        atm_call_prices = (call_prices[1][1] + call_prices[1][2]) / 2.0

        iVol = implied_volatility(atm_call_prices, 100, 100, T, 0.0, 'c')
        iv[i] = iVol

    return iv

def plot():

    option_prices = np.array([
        [20.0000, 10.0015, 1.2635, 0.0004, 0.0000],
        [20.0111, 10.3040, 2.8359, 0.2275, 0.0042],
        [20.1121, 10.9035, 4.0220, 0.8196, 0.0847],
        [20.5040, 12.0279, 5.7096, 2.0772, 0.5656],
        [20.0016, 14.6873, 9.0667, 5.1660, 2.7221]
        ])

    iVols = get_implied_volatility(option_prices)

    # Part One
    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle('Volatility Surface')

    ax1.plot(iVols, label = ("0.1", "0.5", "1", "2", "5"))
    ax1.set_title("Black Scholes Implied Volatility")
    ax1.legend(loc = "best")

    vol_kappa = atm_vol_kappa()
    vol_beta = atm_vol_beta()
    vol_rho = atm_vol_rho()

    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle('Implied Volatility')

    ax1.plot(vol_kappa)
    ax1.set_title("Kappa")

    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle('Implied Volatility')

    ax1.plot(vol_beta)
    ax1.set_title("Beta")

    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle('Implied Volatility')

    ax1.plot(vol_rho)
    ax1.set_title("Rho")

    plt.show()

plot()