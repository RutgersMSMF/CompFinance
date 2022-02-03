import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

N = 1000

# Chi Squared
def chi():

    mu = 0
    sigma = 1
    arr = np.zeros(N)

    # Step One: Generate Samples for X
    rvx = np.random.default_rng().normal(mu, sigma, N)

    # Step Two: Generate Samples for Y
    rvy = np.random.default_rng().normal(mu, sigma, N)

    for i in range(N):
        arr[i] = rvx[i]**2 + rvy[i]**2

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

        if rv2[0] < (rv1[0]**2 / (M * Exponential(rv1[0], Lambda))):
            arr.append(rv1[0])

        else:
            i-=1

    return np.asarray(arr)

# QQ Plot
def plot():

    # Call Chi Squared
    sm.qqplot(chi(), line = "45")
    plt.title("Chi Squared")

    # Call Acceptance Rejection Method
    Lambdas = [1/2, 1, 3/2]
    for i in range(3):
        sm.qqplot(acceptanceRejection(Lambdas[i]), line = "45")
        plt.title("Acceptance Rejection: Lambda = " + str(Lambdas[i]))

    plt.show()
    
plot()

