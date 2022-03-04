import numpy as np
import matplotlib.pyplot as plt

n = 1
N = 100

# Brownian Motion
def brownian_motion():

    mu = 0
    sigma = 1
    dt = 1 / N

    arr = np.zeros((n, N))

    for i in range(n):

        # Generate Random Normals
        rand = np.random.default_rng().normal(mu, sigma, N)

        for j in range(N):

            if j == 0:
                # Initial Value
                arr[i][j] = 0

            else:
                arr[i][j] = sigma * np.sqrt(dt) * rand[j - 1]

    return arr

# Geometric Brownian Motion
def euler_geometric(bm):

    arr = np.zeros((n, N))

    mu = 0 
    sigma = 1
    dt = 1 / N

    for i in range(n):

        for j in range(N):

            if j == 0:
                # Initial Value
                arr[i][j] = 1

            else:
                # Analytic Solution
                X = arr[i][j - 1]
                arr[i][j] = X + (mu * X * dt) + (sigma * X * bm[i][j - 1])

    return arr.T

# Geometric Brownian Motion
def milstein_geometric(bm):

    arr = np.zeros((n, N))

    mu = 0 
    sigma = 1
    dt = 1 / N

    for i in range(n):

        for j in range(N):

            if j == 0:
                # Initial Value
                arr[i][j] = 1

            else:
                X = arr[i][j - 1]
                arr[i][j] = X + (mu * X * dt) + (sigma * X * bm[i][j - 1]) + (0.25 * sigma**2 * X * dt * (bm[i][j - 1]**2 - 1))

    return arr.T

# Brownian Bridge Process
def explicit_bridge(bm):

    arr = np.zeros((n, N))
    dt = 1 / N
    # Final Value
    b = 0

    for i in range(n):

        for j in range(N):

            if j == 0:
                # Initial Value
                arr[i][j] = 0

            else:
                # Euler Method
                X = arr[i][j - 1]
                arr[i][j] = X + ((b - X) / (1 - j * dt)) * dt + bm[i][j - 1]

    return arr.T

def brownian_bridge_iteration(X, b, j, dt, rv):

    Xdt = X

    i = 0
    max_itr = 1000
    err_tol = 10**-6
    while i < max_itr:

        last = Xdt
        Xdt = X + ((b - Xdt) / (1 - j * dt)) * dt + rv
        i+=1

        if np.abs(Xdt - last) < err_tol:
            return Xdt

    return 0

# Brownian Bridge Process
def implicit_bridge(bm):

    arr = np.zeros((n, N))
    dt = 1 / N
    # Final Value
    b = 0

    for i in range(n):

        for j in range(N):

            if j == 0:
                # Initial Value
                arr[i][j] = 0

            else:
                # Euler Method
                X = arr[i][j - 1]
                # Solve For Roots of Implicit Method
                arr[i][j] = brownian_bridge_iteration(X, b, j, dt, bm[i][j - 1])

    return arr.T

# Brownian Excursion
def explicit_excursion(bm):

    arr = np.zeros((n, N))
    dt = 1 / N
    sigma = 1

    for i in range(n):

        for j in range(N):

            if j == 0:
                # Initial Value
                arr[i][j] = 1

            else:
                # Euler Method
                X = arr[i][j - 1]
                arr[i][j] = X + (X * sigma * dt * bm[i][j - 1])

    return arr.T

def brownian_excursion_iteration(X, sigma, dt, rv):

    Xdt = X

    i = 0
    max_itr = 1000
    err_tol = 10**-6
    while i < max_itr:

        last = Xdt
        Xdt = X + (Xdt * sigma * dt * rv)
        i+=1

        if np.abs(Xdt - last) < err_tol:
            return Xdt

    return 0

# Brownian Excursion
def implicit_excursion(bm):

    arr = np.zeros((n, N))
    dt = 1 / N
    sigma = 1

    for i in range(n):

        for j in range(N):

            if j == 0:
                # Initial Value
                arr[i][j] = 1

            else:
                # Euler Method
                X = arr[i][j - 1]
                arr[i][j] = brownian_excursion_iteration(X, sigma, dt, bm[i][j - 1])

    return arr.T

def plot():

    brownian = brownian_motion()

    # Part One
    fig1, (ax2, ax3) = plt.subplots(1, 2)
    fig1.suptitle('Geometric Brownian Motion')

    ax2.plot(euler_geometric(brownian), label = "Euler")
    ax2.plot(milstein_geometric(brownian), label = "Milstein")
    ax2.legend(loc = 'best')
    ax3.plot(np.abs(euler_geometric(brownian) - milstein_geometric(brownian)))
    ax3.set_title("Difference")

    # Part Two
    fig2, (ax2, ax3) = plt.subplots(1, 2)
    fig2.suptitle('Brownian Bridge Process')

    ax2.plot(explicit_bridge(brownian), label = "Explicit")
    ax2.plot(implicit_bridge(brownian), label = "Implicit")
    ax2.axhline(y = 0.0, linestyle = '--')
    ax2.legend(loc = 'best')   
    ax3.plot(np.abs(explicit_bridge(brownian) - implicit_bridge(brownian)))
    ax3.set_title("Difference")

    # Part Three
    fig3, (ax2, ax3) = plt.subplots(1, 2)
    fig3.suptitle('Brownian Excursion Process')

    ax2.plot(explicit_excursion(brownian), label = "Explicit")
    ax2.plot(implicit_excursion(brownian), label = "Implicit")
    ax2.legend(loc = 'best')   
    ax3.plot(np.abs(explicit_excursion(brownian) - implicit_excursion(brownian)))
    ax3.set_title("Difference")

    plt.show()

plot()