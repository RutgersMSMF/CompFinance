import numpy as np 
import matplotlib.pyplot as plt

def cfx(t):
    # Characteristic Function of Gaussian

    return np.exp(-t**2 / 2)

def cfxx(t):
    # Characteristic Function of Gaussian Squared

    return np.sqrt(2)  / np.sqrt(2 - 4*t)

def laplace_density(t):
    # Characteristic Function of Laplace Density

    return 1 / (1 + t**2)

def cauchy_density(t):

    return np.exp(-np.abs(t))

def cauchy_expectation_variable():
    # Mean of Cauchy Sampling

    N = 1000
    arr = np.zeros(N)
    for i in range(N):
        rc = np.random.default_rng().standard_cauchy(N)
        arr[i] = sum(rc) / N

    return arr

def plot():

    x = np.linspace(-6, 6, 1000)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    fig.suptitle('Characteristic Functions !!')

    ax1.plot(cfx(x))
    ax1.set_title("Gaussian")

    z = np.linspace(-1/2, 6, 100)

    ax2.plot(cfxx(z))
    ax2.set_title("Gaussian Squared")

    ax3.plot(laplace_density(x))
    ax3.set_title("LaPlace Density")

    ax4.plot(cauchy_density(x))
    ax4.set_title("Cauchy Density")

    ax5.plot(cauchy_expectation_variable())
    ax5.set_title("Cauchy Sample Means")

    ax6.plot(cauchy_density(x) + cauchy_density(x))
    ax6.set_title("Cauchy Sum Density")

    plt.show()


plot()