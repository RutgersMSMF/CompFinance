import numpy as np
from numba import jit
import matplotlib.pyplot as plt 

N = 100

@jit(nopython = True)
def f_prime(lv, t):

    return (lv / (1 - t)) + 1

@jit(nopython = True, parallel = True)
def explicit_euler(t, initial_value):

    arr = np.zeros(N)
    dt = np.abs(t[1] - t[0])

    for i in range(N):

        if i == 0:

            arr[i] = initial_value

        else:

            arr[i] = arr[i - 1] + dt * f_prime(arr[i - 1], t[i])

    return arr.T

@jit(nopython = True)
def fixed_point_iteration(X, t, dt):

    i = 0
    Y = X
    max_itr = 100
    err_tol = 10**-8

    while i < max_itr:

        last = Y
        Y = X + dt * f_prime(X, t)
        i+=1

        if np.abs(Y - last) < err_tol:
            return Y

    print("Convergence not reached in 100 Iterations")
    return 0

@jit(nopython = True, parallel = True)
def implicit_euler(t, initial_value):

    arr = np.zeros(N)
    dt = np.abs(t[1] - t[0])

    for i in range(N):

        if i == 0:

            arr[i] = initial_value

        else:

            arr[i]= fixed_point_iteration(arr[i - 1], t[i], dt)

    return arr.T

def plot():

    # Initial Values
    initial_value = np.linspace(-1, 1, 25)

    # Forward Iteration (0 => 1)
    a = 1e-8
    b = 1 - 1e-8
    forward = np.linspace(a, b, N)

    # Backwards Iteration (1 => 0)
    a = 1 - 1e-8
    b = 1e-8
    backward = np.linspace(a, b, N)

    # Create Figure Object
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Euler Method')

    # Call Euler Forward
    for i in range(len(initial_value)):

        explicit_forward = explicit_euler(forward, initial_value[i])
        implicit_forward = implicit_euler(forward, initial_value[i])

        ax1.plot(explicit_forward)
        ax1.plot(implicit_forward)

    ax1.set_title("Forward Iteration: t = 0")
    ax1.set_ylim(-5, 5)

    # Call Euler Backward
    for i in range(len(initial_value)):

        explicit_backward = explicit_euler(backward, initial_value[i])
        implicit_backward = implicit_euler(backward, initial_value[i])

        ax2.plot(explicit_backward)
        ax2.plot(implicit_backward)
    
    ax2.set_title("Backward Iteration: t = 1")
    ax2.set_ylim(-5, 5)

    plt.show()

plot()