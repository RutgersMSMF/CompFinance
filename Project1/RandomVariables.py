import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

N = 1000

# Box Muller Method
def boxMuller():

    arr = np.zeros(N)

    # Generate Uniform Random Variables
    rv = np.random.default_rng().uniform(0, 1, N)

    for i in range(N - 1):

        if i % 2 == 0:
            arr[i] = np.sqrt(-2 * np.log(rv[i])) * np.cos(2 * np.pi * rv[i + 1])

        else:
            arr[i] = np.sqrt(-2 * np.log(rv[i])) * np.sin(2 * np.pi * rv[i + 1])

    return arr

# Method For Standard Normal Distribution
def Gaussian(x):
    mu = 0
    sigma = 1

    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((x - mu) / sigma)**2)

# Method For Exponential Distribution
def Exponential(x, Lambda):

    if x > 0:
        return Lambda * np.exp(-Lambda * x)

    else: 
        return 0

# Acceptance Rejection Algorithm
def acceptanceRejection(Lambda):

    arr = []
    M = 1.01
    beta = 1 / Lambda
    for i in range(N):

        # Step One: Sample Random Exponential
        rv1 = np.random.default_rng().exponential(beta, 1)

        # Step Two: Sample Uniform Random Variables
        rv2 = np.random.default_rng().uniform(0, 1, 1)

        if rv2[0] < (Gaussian(rv1[0]) / (M * Exponential(rv1[0], Lambda))):
            arr.append(rv1[0])

        else:
            i-=1

    return np.asarray(arr)

# NumPy Method 
def numpyGenerator():

    mu = 0
    sigma = 1
    rv = np.random.default_rng().normal(mu, sigma, N)

    return rv

# QQ Plot
def plot():

    # Call Box Muller Method
    sm.qqplot(boxMuller(), line = "45")
    plt.title("Box Muller")

    # Call Acceptance Rejection Method
    Lambdas = [1/2, 1, 3/2]
    for i in range(3):
        sm.qqplot(acceptanceRejection(Lambdas[i]), line = "45")
        plt.title("Acceptance Rejection: Lambda = " + str(Lambdas[i]))

    # Call Numpy Generator
    sm.qqplot(numpyGenerator(), line = "45")
    plt.title("Numpy Gaussian Generator")

    plt.show()
    
plot()
