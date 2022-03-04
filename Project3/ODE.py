import numpy as np

def a(beta, kappa, theta, s):

    gamma = np.sqrt(kappa**2 + 2 * beta**2)

    return (-2 * kappa * theta / beta**2) * (np.log(2 * gamma) + (0.5 * (kappa + gamma) * s) - np.log((gamma + kappa) * (np.exp(gamma * s) - 1) + (2 * gamma)))

def b(beta, kappa, s):

    gamma = np.sqrt(kappa**2 + 2 * beta**2)

    return (2 * (np.exp(gamma * s) - 1)) / ((gamma + kappa) * ((np.exp(gamma * s) - 1)) + (2 * gamma))

def ode():

    beta = 1
    kappa = 2 
    theta = 2

    print("Fellers Condition: ", beta**2 <= 2 * kappa * theta)
    print("A Naught: ", a(beta, kappa, theta, 0))
    print("B Naught: ", b(beta, kappa, 0))

    return 0

ode()