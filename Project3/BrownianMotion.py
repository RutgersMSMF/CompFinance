import numpy as np
import matplotlib.pyplot as plt

n = 1
N = 100

def weiner_process():

    n = 100
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
                X = arr[i][j - 1]
                arr[i][j] = X + sigma * np.sqrt(dt) * rand[j - 1]

    return arr.T


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

# Forward Euler Method
def explicit_ornstein(bm):

    mu = 0
    dt = 1 / N
    theta = 1

    arr = np.zeros((n, N))

    for i in range(n):

        for j in range(N):

            if j == 0:
                arr[i][j] = 0

            else:
                X = arr[i][j -1]
                arr[i][j] = X + theta * (mu - X) * dt + bm[i][j - 1]

    return arr.T

def fixed_point_iteration(X, mu, theta, dt, rv):

    Xdt = X

    i = 0
    max_itr = 1000
    err_tol = 10**-6
    while i < max_itr:

        last = Xdt
        Xdt = X + theta * (mu - Xdt) * dt + rv
        i+=1

        if np.abs(Xdt - last) < err_tol:
            return Xdt

    return 0

# Backward Euler Method
def implicit_ornstein(bm):

    mu = 0
    dt = 1 / N
    theta = 1

    arr = np.zeros((n, N))

    for i in range(n):

        for j in range(N):

            if j == 0:
                arr[i][j] = 0

            else:
                X = arr[i][j -1]
                # Solve For Roots of Implicit Method
                arr[i][j] =  fixed_point_iteration(X, mu, theta, dt, bm[i][j - 1])

    return arr.T

def euler_feller(bm):

    dt = 1 / N

    arr = np.zeros((n, N))
    arr_log = np.zeros((n, N))

    for i in range(n):

        for j in range(N):

            if j == 0:
                arr[i][j] = 1
                arr_log[i][j] = np.log(arr[i][j])

            else:
                X = arr[i][j -1]
                arr[i][j] = X + ((1 - X) * dt) + (np.sqrt(X) * bm[i][j - 1])
                arr_log[i][j] = np.log(arr[i][j])


    return arr_log.T

def milstein_feller(bm):

    dt = 1 / N
    sigma = 1
    arr = np.zeros((n, N))
    arr_log = np.zeros((n, N))

    for i in range(n):

        for j in range(N):

            if j == 0:
                arr[i][j] = 1
                arr_log[i][j] = np.log(arr[i][j])

            else:
                X = arr[i][j - 1]
                arr[i][j] = X + ((1 - X) * dt) + (np.sqrt(X) * bm[i][j - 1]) + (0.25) * sigma**2 * X * dt* (bm[i][j - 1]**2 - 1)
                arr_log[i][j] = np.log(arr[i][j])


    return arr_log.T

def plot():

    brownian = brownian_motion()

    # Initial Brownian Motion
    fig, ax = plt.subplots(1, 1)
    fig.suptitle("Brownian Motion")
    ax.plot(weiner_process())

    # Part One
    fig2, (ax2, ax3) = plt.subplots(1, 2)
    fig2.suptitle('Ornstein Uhlenbeck Process !!')

    ax2.plot(explicit_ornstein(brownian), label = 'Explicit')
    ax2.plot(implicit_ornstein(brownian), label = 'Implicit')
    ax2.legend(loc = 'best')
    ax3.plot(np.abs(explicit_ornstein(brownian) - implicit_ornstein(brownian)))
    ax3.set_title("Difference")

    # Part Two
    fig3, (ax2, ax3) = plt.subplots(1, 2)
    fig3.suptitle('Feller Process')

    ax2.plot(np.exp(euler_feller(brownian)), label = 'Euler')
    ax2.plot(np.exp(milstein_feller(brownian)), label = 'Milstein')
    ax2.legend(loc = 'best')
    ax3.plot(np.abs(np.exp(euler_feller(brownian)) - np.exp(milstein_feller(brownian))))
    ax3.set_title("Difference")

    plt.show()

plot()