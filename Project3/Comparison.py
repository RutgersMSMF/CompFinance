import numpy as np
import matplotlib.pyplot as plt

N = 100

# Geometric Brownian Motion
def geometric_brownian_motion():

    arr = np.zeros((3, N))

    mu = 0 
    sigma = 1
    dt = 1 / 100

    # Generate Random Normals
    rand = np.random.default_rng().normal(mu, sigma, N)

    for i in range(3):

        for j in range(N):

            if j == 0:
                # Initial Value
                arr[i][j] = 1

            else:

                if i == 0:
                    # Analytic Solution
                    X = arr[i][j - 1]
                    arr[i][j] = X * np.exp((mu - sigma**2 / 2) * dt + (sigma * np.sqrt(dt) * rand[j - 1]))

                if i == 1:
                    # Euler Method
                    X = arr[i][j - 1]
                    arr[i][j] = X + (mu * X * dt) + (sigma * X * np.sqrt(dt) * rand[j - 1])

                if i == 2:
                    # Milstein Method
                    X = arr[i][j - 1]
                    arr[i][j] = X + (mu * X * dt) + (sigma * X * np.sqrt(dt) * rand[j - 1]) + (0.5 * sigma**2 * X * (np.sqrt(dt)**2 * rand[j - 1]**2 - dt))

    return arr.T

def plot():

    gbm = geometric_brownian_motion()

    # Part One
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    fig1.suptitle('Brownian Processes !!')

    ax1.plot(gbm, label = ["Analytic", "Euler", "Milstein"])
    ax1.set_title("Geometric Brownian Motion")
    ax1.legend(loc = 'best')

    euler = []
    milstein = []
    for i in range(N):
        euler.append(np.abs(gbm[i][0] - gbm[i][1]))
        milstein.append(np.abs(gbm[i][0] - gbm[i][2]))

    ax2.plot(euler, label = "Analytic - Euler")
    ax2.plot(milstein, label = "Analytic - Milstein")
    ax2.set_title("Difference")
    ax2.legend(loc = 'best')

    plt.show()

plot()