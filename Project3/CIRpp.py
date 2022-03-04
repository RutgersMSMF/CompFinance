import numpy as np
from numba import jit
import matplotlib.pyplot as plt

n = 1
N = 100000

@jit(nopython = True)
def cir_plus_plus():

    arr = np.zeros((n, N))
    euler_rates = np.zeros((n, N))
    euler_log_rates = np.zeros((n, N))
    milstein_rates = np.zeros((n, N))
    milstein_log_rates = np.zeros((n, N))

    # Initialize Variables
    r = 0.05
    x = 0.1
    theta = 0.1 
    kappa = 0.25 
    beta = 0.2 
    dt = 1 / N 

    for i in range(n):

        rand = np.random.normal(0, 1, N)

        for j in range(N):

            if j == 0:
                arr[i][j] = x
                euler_rates[i][j] = r + x
                euler_log_rates[i][j] = r + np.log(x)
                milstein_rates[i][j] = r + x
                milstein_log_rates[i][j] = r + np.log(x)

            else:
                X = arr[i][j - 1]

                if X > 0:
                    arr[i][j] = X + kappa * (theta - X) * dt + beta * np.sqrt(X) * np.sqrt(dt) * rand[j - 1]
                    euler_rates[i][j] = r + arr[i][j]
                    euler_log_rates[i][j] = np.log(arr[i][j])
                else:
                    X = np.abs(X)
                    arr[i][j] = X + kappa * (theta - X) * dt + beta * np.sqrt(X) * np.sqrt(dt) * rand[j - 1]
                    euler_rates[i][j] = r + arr[i][j]
                    euler_log_rates[i][j] = np.log(arr[i][j])

                X = arr[i][j - 1]

                if X > 0:
                    arr[i][j] = X + kappa * (theta - X) * dt + beta * np.sqrt(X) * np.sqrt(dt) * rand[j - 1] + (0.25) * beta**2 * dt * (rand[j - 1]**2 - 1) 
                    milstein_rates[i][j] = r + arr[i][j]
                    milstein_log_rates[i][j] = np.log(arr[i][j])
                else:
                    X = np.abs(X)
                    arr[i][j] = X + kappa * (theta - X) * dt + beta * np.sqrt(X) * np.sqrt(dt) * rand[j - 1] + (0.25) * beta**2 * dt * (rand[j - 1]**2 - 1) 
                    milstein_rates[i][j] = r + arr[i][j]
                    milstein_log_rates[i][j] = np.log(arr[i][j])

    return euler_rates.T, euler_log_rates.T, milstein_rates.T, milstein_log_rates.T

# Return Zero Coupon Bond Prices
def ZCB(arr):

    zcb = np.zeros(len(arr))

    for i in range(len(arr)):
        zcb[i] = arr[i] / arr[-1]

    return zcb

def plot():

    euler, euler_log, milstein, milstein_log = cir_plus_plus()
    euler_zcb = ZCB(euler)
    milstein_zcb = ZCB(milstein)
    euler_log_zcb = ZCB(np.exp(euler_log))
    milstein_log_zcb = ZCB(np.exp(milstein_log))

    # Part One
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig1.suptitle('CIR Plus Plus')

    ax1.plot(euler, label = 'Euler')
    ax1.plot(milstein, label = 'Milstein')
    ax1.legend(loc = 'best')
    ax1.set_title("CIR Process")

    ax2.plot(euler_log, label = 'Euler Log')
    ax2.plot(milstein_log, label = 'Milstein Log')
    ax2.legend(loc = 'best')
    ax2.set_title("CIR Process")

    ax3.plot(euler_zcb, label = 'Euler')
    ax3.plot(euler_log_zcb, label = 'Euler Log')
    ax3.plot(milstein_zcb, label = 'Milstein')
    ax3.plot(milstein_log_zcb, label = 'Milstein Log')
    ax3.legend(loc = 'best')
    ax3.set_title("Zero Coupon Bond Prices")

    plt.show()

plot()