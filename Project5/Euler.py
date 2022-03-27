import numpy as np
from numba import jit
import matplotlib.pyplot as plt 

N = 1000
B = np.linspace(1, 0, N)
F = np.linspace(0, 1, N)
dt = B[1] - B[0]

@jit(nopython = True)
def f(t):

    return -0.5 * (1 - t)

@jit(nopython = True)
def f_prime(t):

    return (f(t) / (1 - t)) + 1

@jit(nopython = True)
def explicit_euler(initial_value, I):

    arr = np.zeros(N)

    for i in range(N):

        if i == 0:
            arr[i] = initial_value

        else:
            arr[i] = I[i - 1] + dt * f(I[i - 1])

    return arr

@jit(nopython = True)
def fixed_point_iteration(X, Y):

    i = 0
    max_itr = 1000
    err_tol = 10**-8
    while i < max_itr:

        X = Y
        Y = X + dt * f(Y)
        i+=1

        if np.abs(X - Y) < err_tol:
            return Y

    print("Convergence not reached in 1000 Iterations")
    return 0

@jit(nopython = True)
def implicit_euler(initial_value, I):

    arr = np.zeros(N)

    for i in range(N):

        if i == 0:
            arr[i] = initial_value

        else:
            temp = I[i - 1] + dt * f(I[i - 1])
            arr[i] = fixed_point_iteration(arr[i-1], temp)

    return arr

def plot():

    # Backwards Iteration
    initial_value = 0
    explicit_backwards = explicit_euler(initial_value, B)

    # Forward Iteration
    initial_value = -1/2
    explicit_forward = explicit_euler(initial_value, F)

    # Explicit Euler Method
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Explicit Euler Method')

    ax1.plot(explicit_backwards)
    ax1.set_title("Backward Iteration")

    ax2.plot(explicit_forward)
    ax2.set_title("Forward Iteration")

    # Backwards Iteration
    initial_value = 0
    implicit_backwards = implicit_euler(initial_value, B)

    # Forward Iteration
    initial_value = -1/2
    implicit_forward = implicit_euler(initial_value, F)

    # Explicit Euler Method
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Implicit Euler Method')

    ax1.plot(implicit_backwards)
    ax1.set_title("Backward Iteration")

    ax2.plot(implicit_forward)
    ax2.set_title("Forward Iteration")

    plt.show()

plot()