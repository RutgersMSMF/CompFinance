import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define Parameters
S = 100                     # Stock Price
Theta = 0.01                # Mean Reversion Level
Kappa = 2                   # Mean Reversion Speed
Rho = -0.5                  # Correlation
Sigma = 0.1                 # Vol of Vol
Veta = 0.01                 # Implied Volatility
Tau = 1                     # Time
N = 100                     # Simulation Count, Length

@jit(nopython = True)
def heston(y, t, K, T, S, R, U):

    I = 0 + 1j

    arr1 = y[1] * K * T
    arr2 = -0.5 * U * (I.imag + U) + y[1] * (I.imag * U * R * S - K) + 0.5 * y[1]**2 * S

    return arr1, arr2

def solve_heston(t, B, R, U):

    I = 0 + 1j
    t = np.array([0, t])
    y = np.array([0, 0])
    sol = odeint(heston, y, t, args = (Kappa, Theta, Sigma, R, U))

    ans = np.zeros(N)
    ans = np.exp(sol[1][0] + sol[1][1] * B)

    return ans

@jit(nopython = True)
def heston_tilde(y, t, K, T, S, R, U):

    I = 0 + 1j

    arr1 = y[1] * K * T
    arr2 = 0.5 * U * (I.imag - U) + y[1] * (I.imag * U * R * S - K) + 0.5 * y[1]**2 * S

    return arr1, arr2

def solve_heston_tilde(t, B, R, U):

    t = np.array([0, t])
    y = np.array([0, 0])
    sol = odeint(heston_tilde, y, t, args = (Kappa, Theta, Sigma, R, U))

    ans = np.zeros(N)
    ans = np.exp(sol[1][0] + sol[1][1] * B)

    return ans

def atm_option_prices(V):

    K = 100
    T = 1
    normalization = 1 / (2 * np.pi)

    I = 0 + 1j
    y = np.array([0, 0])
    U = np.linspace(-0.5, 0.5, N)

    Q = 0
    Q_tilde = 0
    X = np.log(K / S)
    UU = np.linspace(-50, X, N)

    for i in range(N):

        sol = odeint(heston, y, np.array([0, T]), args = (Kappa, Theta, Sigma, Rho, U[i]))
        Q += normalization * np.exp(-I.imag * U[i] * UU[i]) * np.exp(sol[1][0] + sol[1][1] * V)

        sol = odeint(heston_tilde, y, np.array([0, T]), args = (Kappa, Theta, Sigma, Rho, U[i]))
        Q_tilde += normalization * np.exp(-I.imag * U[i] * UU[i]) * np.exp(sol[1][0] + sol[1][1] * V)

    op = (S * (Q_tilde / N)) - (K * (Q / N))

    return op

def get_option_prices(V):

    K = np.array([80, 90, 100, 110, 120])
    T = np.array([0.1, 0.5, 1, 2, 5])
    op = np.zeros((5, 5))
    normalization = 1 / (2 * np.pi)

    I = 0 + 1j
    Y = np.linspace(-0.5, 0.5, N)

    for i in range(len(T)):

        for j in range(len(K)):

            Q = 0
            Q_tilde = 0
            X = np.log(K[j] / S)
            U = np.linspace(-50, X, N)

            for k in range(N-1):

                step_size = Y[k+1] - Y[k]

                x0 = np.exp(-I.imag * Y[k] * U[k]) * solve_heston(T[i], V, Rho, Y[k])
                x1 = np.exp(-I.imag * Y[k+1] * U[k+1]) * solve_heston(T[i], V, Rho, Y[k+1])

                # Trapezoid Rule
                Q += step_size * 0.5 * (x1 + x0) 

                # Change of Measure

                x0 = np.exp(-I.imag * Y[k] * U[k]) * solve_heston_tilde(T[i], V, Rho, Y[k])
                x1 = np.exp(-I.imag * Y[k+1] * U[k+1]) * solve_heston_tilde(T[i], V, Rho, Y[k+1])

                # Trapezoid Rule
                Q_tilde += step_size * 0.5 * (x1 + x0) 

            price = (S * (Q_tilde)) - (K[j] * (Q))

            if price > 0:
                op[i][j] = normalization * price 
            else:
                op[i][j] = 0

    return op

def bisection(a, b):

    c = (a + b) / 2

    return c

def find_root(a, b, call_price):

    iVol = (a + b) / 2

    count = 0
    max_itr = 100
    converge = False
    while (converge == False) and (count < max_itr):

        atm_price = atm_option_prices(iVol)

        if (np.abs(call_price - atm_price) < 0.01):
            converge = True

        else:
            if atm_price > call_price:
                iVol = bisection(a, b)
                b = bisection(iVol, b)

            if atm_price < call_price:
                iVol = bisection(a, b)
                a = bisection(a, iVol)

        count+=1

        if count >= max_itr:
            print("Convergence Not Reached in 100 Iteration")
            return 0
        
    return iVol

def optimization():

    a = 0.01
    b = 1.00
    call_price = 4.0220

    # Step One: Find Veta Such That Heston Produces ATM Call Price
    iVol = find_root(a, b, call_price)
    
    # Step Two: Compute All Option Prices 
    heston_prices = get_option_prices(iVol)

    # Intermediate Step: Option Prices
    option_prices = np.array([
        [20.0000, 10.0015, 1.2635, 0.0004, 0.0000],
        [20.0111, 10.3040, 2.8359, 0.2275, 0.0042],
        [20.1121, 10.9035, 4.0220, 0.8196, 0.0847],
        [20.5040, 12.0279, 5.7096, 2.0772, 0.5656],
        [20.0016, 14.6873, 9.0667, 5.1660, 2.7221]
        ])

    # Step Three: Compute The Squared Error
    error = mse(option_prices, heston_prices)

    # Step Four: Minimize Squared Error
    errors = [sum_error(error)]
    minimized_prices = minimize_mse(iVol, errors)

    return heston_prices, minimized_prices

@jit(nopython = True)
def sum_error(arr):

    s = 0

    for i in range(len(arr)):

        for j in range(len(arr[i])):

            s+= arr[i][j]

    return s

@jit(nopython = True)
def mse(option_prices, heston_prices):

    mse = np.zeros((5, 5))

    for i in range(5):

        for j in range(5):

            mse[i][j] = (option_prices[i][j] - heston_prices[i][j])**2

    return mse

def minimize_mse(iVol, errors):

    option_prices = np.array([
        [20.0000, 10.0015, 1.2635, 0.0004, 0.0000],
        [20.0111, 10.3040, 2.8359, 0.2275, 0.0042],
        [20.1121, 10.9035, 4.0220, 0.8196, 0.0847],
        [20.5040, 12.0279, 5.7096, 2.0772, 0.5656],
        [20.0016, 14.6873, 9.0667, 5.1660, 2.7221]
        ])

    a = 0.01
    b = 1.00
    
    lower = bisection(a, iVol)
    heston_prices = get_option_prices(lower)
    lower_error = sum_error(mse(option_prices, heston_prices))

    upper = bisection(b, iVol)
    heston_prices = get_option_prices(upper)
    upper_error = sum_error(mse(option_prices, heston_prices))

    iVol = bisection(lower, upper)

    if lower_error < upper_error:
        errors.append(lower_error)
        b = iVol
    else:
        errors.append(upper_error)
        a = iVol

    while np.abs(errors[-1] - errors[-2]) > 1e-3:

        lower = bisection(a, iVol)
        heston_prices = get_option_prices(lower)
        lower_error = sum_error(mse(option_prices, heston_prices))

        upper = bisection(b, iVol)
        heston_prices = get_option_prices(upper)
        upper_error = sum_error(mse(option_prices, heston_prices))

        iVol = bisection(lower, upper)

        if lower_error < upper_error:
            errors.append(lower_error)
            b = iVol
        else:
            errors.append(upper_error)
            a = iVol

    print("Optimal Sigma: ", iVol)
    print("MSE: ", errors[-1])

    return heston_prices

def plot():

    option_prices = np.array([
        [20.0000, 10.0015, 1.2635, 0.0004, 0.0000],
        [20.0111, 10.3040, 2.8359, 0.2275, 0.0042],
        [20.1121, 10.9035, 4.0220, 0.8196, 0.0847],
        [20.5040, 12.0279, 5.7096, 2.0772, 0.5656],
        [20.0016, 14.6873, 9.0667, 5.1660, 2.7221]
        ])

    # Part One
    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle('Inversion Formula Density')

    U = np.linspace(-50, 50, N)
    psi_pdf = np.zeros(N)
    tilde_pdf = np.zeros(N)

    for i in range(N):
        psi_pdf[i] = solve_heston(Tau, Veta, Rho, U[i])
        tilde_pdf[i] = solve_heston_tilde(Tau, Veta, Rho, U[i])

    ax1.plot(psi_pdf, label = "Psi")
    ax1.plot(tilde_pdf, label = "Psi Tilde")
    ax1.set_title("Psi")
    ax1.legend(loc = "best")

    vol_surface = optimization()
    normalization = 1.0 / (2.0 * np.pi)
    
    # Part Two
    fig2, (ax1) = plt.subplots(1, 1)
    fig2.suptitle('Heston Option Prices')

    ax1.plot(normalization * vol_surface[0].T, label = "Heston")
    ax1.plot(option_prices.T, label = "Stein Stein")
    ax1.set_title("Heston Option Prices")
    ax1.legend(loc = "best")

    # Part Two
    fig2, (ax1) = plt.subplots(1, 1)
    fig2.suptitle('Heston Option Prices')

    ax1.plot(normalization * vol_surface[1].T, label = "Heston")
    ax1.plot(option_prices.T, label = "Stein Stein")
    ax1.set_title("Heston Option Prices Minimized")
    ax1.legend(loc = "best")

    plt.show()

plot()
