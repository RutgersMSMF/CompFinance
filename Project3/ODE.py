import numpy as np
import matplotlib.pyplot as plt

# Global Variables 
N = 100

def a(beta, kappa, theta, s):

    gamma = np.sqrt(kappa**2 + 2 * beta**2)

    return (-2 * kappa * theta / beta**2) * (np.log(2 * gamma) + (0.5 * (kappa + gamma) * s) - np.log((gamma + kappa) * (np.exp(gamma * s) - 1) + (2 * gamma)))

def b(beta, kappa, s):

    gamma = np.sqrt(kappa**2 + 2 * beta**2)

    return (2 * (np.exp(gamma * s) - 1)) / ((gamma + kappa) * ((np.exp(gamma * s) - 1)) + (2 * gamma))

# Returns Interest Rate Derived from ODE
def get_interest_rate(t, X):

    # Bond Parameters
    r = 0.05 
    theta = 0.1
    kappa = 0.25
    beta = 0.2
    T = 100

    A = a(beta, kappa, theta, (T - t))
    B = b(beta, kappa, (T - t))

    ode = np.exp(-A - (B * X))

    return r + ode

def interest_rates():

    arr = np.zeros(N)

    X = 0.1
    for i in range(len(arr)):
        arr[i] = get_interest_rate(i, X)
        X = arr[i]

    return arr

# Track Money Market Account
def money_market(index, rates):

    s = 1
    i = 0
    while i < index:

        s = s * (1 + rates[i])
        i+=1

    return s

# Return Zero Coupon Bond Prices
def ZCB(index, rates):

    return money_market(index, rates) / money_market(len(rates), rates)

def ode():

    # Initial Condition Test
    beta = 1
    kappa = 2 
    theta = 2

    print("Fellers Condition: ", beta**2 <= 2 * kappa * theta)
    print("A Naught: ", a(beta, kappa, theta, 0))
    print("B Naught: ", b(beta, kappa, 0))

    # Plot
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.suptitle('Zero Coupon Bonds')

    s = np.linspace(1e-8, 1, N)
    
    ax1.plot(a(beta, kappa, theta, s))
    ax1.set_title("Solution A")

    ax2.plot(b(beta, kappa, s))
    ax2.set_title("Solution B")

    time_one = get_interest_rate(1, 0.1)
    time_ten = get_interest_rate(10, 0.1)
    time_hunnit = get_interest_rate(100, 0.1)

    rates = interest_rates()

    print("####################")

    print("Interest Rate (t = 1): ", time_one)
    print("Zero Coupon Bond Price: ", ZCB(1, rates))

    print("####################")

    print("Interest Rate (t = 10): ", time_ten)
    print("Zero Coupon Bond Price: ", ZCB(10, rates))

    print("####################")

    print("Interest Rate (t = 100): ", time_hunnit)
    print("Zero Coupon Bond Price: ", ZCB(100, rates))

    zcb = np.zeros(N + 1)
    for i in range(N + 1):
        zcb[i] = ZCB(i, rates)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Zero Coupon Bonds')

    ax1.plot(rates)
    ax1.set_title("Interest Rates")

    ax2.plot(zcb)
    ax2.set_title("Zero Coupon Bonds")

    plt.show()

ode()