import numpy as np 
from numba import jit
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define Model Parameters 
S = 100                         # Stock Price
Theta = 0.1                     # Mean Reversion Level
Sigma = 0.1                     # Implied Volatility
Kappa = 2.0                     # Mean Reversion Speed
Rho = -0.5                      # Correlation 
Beta = 0.05                     # Vol of Vol
Tau = 1                         # Time to Expiry
N = 100                         # Simulation Count, Length

@jit(nopython = True)
def psi(y, t, K, T, B, R, U):

    I = 0 + 1j

    arr1 = y[1] * K * T + 0.5 * (y[1]**2 + y[2]) * B**2
    arr2 = y[2] * K * T + y[1] * (y[2] * B**2 + I.imag * U * R * B - K)
    arr3 = -U * (I.imag  + U) + 2 * y[2] * (I.imag * U * R * B - K) + y[2]**2 * B**2

    return arr1, arr2, arr3

@jit(nopython = True)
def psi_tilde(y, t, K, T, B, R, U):

    I = 0 + 1j

    arr1 = (y[1] * K * T) + (0.5 * (y[1]**2 + y[2]) * B**2)
    arr2 = (y[2] * K * T) + (y[1] * (y[2] * B**2 + I.imag * U * R * B + B * R - K))
    arr3 = (U * (I.imag - U)) + (2 * y[2] * (I.imag * U * R * B + B * R - K)) + (y[2]**2 * B**2)

    return arr1, arr2, arr3

def get_option_price(Strike, T, iVol, K, B, R):

    normalization = 1 / (2 * np.pi)

    I = 0 + 1j
    y = np.array([0, 0, 0])
    U = np.linspace(-0.5, 0.5, N)

    Q = 0
    Q_tilde = 0
    X = np.log(Strike / S)
    UU = np.linspace(-50, X, N)

    for i in range(N):

        sol = odeint(psi, y, np.array([0, T]), args = (K, Theta, B, R, U[i]))
        Q += normalization * np.exp(-I.imag * U[i] * UU[i]) * np.exp(sol[1][0] + sol[1][1] * iVol + 0.5 * sol[1][2] * iVol**2)

        sol = odeint(psi_tilde, y, np.array([0, T]), args = (K, Theta, B, R, U[i]))
        Q_tilde += normalization * np.exp(-I.imag * U[i] * UU[i]) * np.exp(sol[1][0] + sol[1][1] * iVol + 0.5 * sol[1][2] * iVol**2)

    op = (S * (Q_tilde / N)) - (Strike * (Q / N))

    return op

if __name__ == '__main__':

    def solve_psi(t, B, R, U):

        t = np.array([0, t])
        y = np.array([0, 0, 0])
        sol = odeint(psi, y, t, args = (Kappa, Theta, B, R, U))

        ans = np.zeros(N)
        ans = np.exp(sol[1][0] + (sol[1][1] * Sigma) + (0.5 * sol[1][2] * Sigma**2))

        return ans

    def solve_psi_tilde(t, B, R, U):

        t = np.array([0, t])
        y = np.array([0, 0, 0])
        sol = odeint(psi_tilde, y, t, args = (Kappa, Theta, B, R, U))

        ans = np.zeros(N)
        ans = np.exp(sol[1][0] + (sol[1][1] * Sigma) + (0.5 * sol[1][2] * Sigma**2))

        return ans

    def density(): 

        arr1 = np.zeros((3, N))
        arr2 = np.zeros((3, N))

        I = 0 + 1j
        Y = np.linspace(-0.5, 0.5, N)
        U = np.linspace(-50, 50, N)
        
        rho = np.array([-0.5, 0, 0.5])
        beta = np.array([0.01, 0.05, 0.09])

        for i in range(len(rho)):

            for j in range(N):

                arr1[i][j] = solve_psi(Tau, Beta, rho[i], U[j])
                arr2[i][j] = solve_psi(Tau, beta[i], Rho, U[j])

        return arr1, arr2

    def option_prices():

        K = np.array([80, 90, 100, 110, 120])
        T = np.array([0.1, 0.5, 1, 2, 5])
        op = np.zeros((5, 5))

        I = 0 + 1j
        Y = np.linspace(-0.5, 0.5, N)

        for i in range(len(T)):

            for j in range(len(K)):

                Q = 0
                Q_tilde = 0
                X = -np.log(K[j] / S)
                U = np.linspace(-50, X, N)

                for k in range(N - 1):

                    step_size = Y[k+1] - Y[k]

                    x0 = np.exp(-I.imag * Y[k] * U[k]) * solve_psi(T[i], Beta, Rho, Y[k])
                    x1 = np.exp(-I.imag * Y[k+1] * U[k+1]) * solve_psi(T[i], Beta, Rho, Y[k + 1])

                    # Trapezoid Rule
                    Q += step_size * 0.5 * (x1 + x0) 

                    # Change of Measure

                    x0 = np.exp(-I.imag * Y[k] * U[k]) * solve_psi_tilde(T[i], Beta, Rho, Y[k])
                    x1 = np.exp(-I.imag * Y[k+1] * U[k+1]) * solve_psi_tilde(T[i], Beta, Rho, Y[k + 1])

                    # Trapezoid Rule
                    Q_tilde += step_size * 0.5 * (x1 + x0) 

                price = (S * (Q_tilde)) - (K[j] * (Q))

                if price > 0:
                    op[i][j] = price
                else:
                    op[i][j] = 0

        return op

    def plot():

        # Part One
        fig, (ax1) = plt.subplots(1, 1)
        fig.suptitle('Stein Stein Density')

        I = 0 + 1j
        Y = np.linspace(-0.5, 0.5, N)
        U = np.linspace(-50, 50, N)
        psi_pdf = np.zeros(N)
        tilde_pdf = np.zeros(N)

        for i in range(N):
            psi_pdf[i] = solve_psi(Tau, Beta, Rho, U[i])
            tilde_pdf[i] = solve_psi_tilde(Tau, Beta, Rho, U[i])

        ax1.plot(psi_pdf, label = "Psi")
        ax1.plot(tilde_pdf, label = "Psi Tilde")
        ax1.set_title("Psi")
        ax1.legend(loc = "best")

        # Part Two
        fig1, (ax1, ax2) = plt.subplots(1, 2)
        fig1.suptitle('Inversion Formula Density')

        v_pdf = density()
        normalization = 1.0 / (2.0 * np.pi)

        ax1.plot(normalization * v_pdf[0].T, label = ("-0.5", "0", "0.5"))
        ax1.set_title("Rho")
        ax1.legend(loc = "best")

        ax2.plot(normalization * v_pdf[1].T, label = ("0.01", "0.05", "0.09"))
        ax2.set_title("Beta")
        ax2.legend(loc = "best")

        # Part Three
        fig1, (ax1) = plt.subplots(1, 1)
        fig1.suptitle('Risk Neutral Price')

        prices = option_prices()

        ax1.plot(prices.T * normalization, label = ("0.1", "0.5", "1", "2", "5"))
        ax1.set_title("Option Prices")
        ax1.legend(loc = "best")

        plt.show()

    plot()
