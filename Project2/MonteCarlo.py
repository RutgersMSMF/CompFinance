import numpy as np
import matplotlib.pyplot as plt

K = 1
mu = 1
sigma = 1
N = 100000

def confidence_interval(arr, clv):

    xbar = np.mean(arr)
    std = np.std(arr)

    return (xbar - clv * std / np.sqrt(len(arr)), xbar + clv * std / np.sqrt(len(arr)))

def bachelier():

    X = np.random.default_rng().normal(mu, sigma, N)
    arr = np.zeros(N)

    for i in range(N):

        if K - X[i] > 0: 
            arr[i] = K - X[i]
        else:
            arr[i] = 0

    return arr

def antithetic_bachelier():

    X = np.random.default_rng().normal(mu, sigma, N)
    arr = np.zeros(N)

    for i in range(N):

        if (K - X[i] > 0) and (K + X[i] - 2 > 0): 
            arr[i] = (K - X[i]) + (K + X[i] - 2) / 2.0

        elif (K - X[i] > 0): 
            arr[i] = (K - X[i]) / 2.0
            
        elif (K + X[i] - 2 > 0):
            arr[i] = (K + X[i] - 2) / 2.0

        else:
            arr[i] = 0

    return arr

def black_scholes(): 

    X = np.random.default_rng().normal(mu, sigma, N)
    arr = np.zeros(N)

    for i in range(N):

        if K - np.exp(X[i]) > 0: 
            arr[i] = K - np.exp(X[i])
        else:
            arr[i] = 0

    return arr

def antithetic_black_scholes():

    X = np.random.default_rng().normal(mu, sigma, N)
    arr = np.zeros(N)

    for i in range(N):

        if (K - np.exp(X[i]) > 0) and (K + np.exp(X[i]) - 2 > 0): 
            arr[i] = (K - np.exp(X[i])) + (K + np.exp(X[i]) - 2) / 2.0

        elif (K - np.exp(X[i]) > 0): 
            arr[i] = (K - np.exp(X[i])) / 2.0
            
        elif (K + np.exp(X[i]) - 2 > 0):
            arr[i] = (K + np.exp(X[i]) - 2) / 2.0

        else:
            arr[i] = 0

    return arr

def control_variate(c):

    arr = np.zeros((len(c), N))
    bach = 1 / np.sqrt(2 * np.pi)

    for i in range(len(c)):

        X = np.random.default_rng().normal(mu, sigma, N)

        for j in range(N):

            if (K - np.exp(X[j]) > 0) and (K - X[i] > 0):
                arr[i][j] = (K - np.exp(X[j])) + c[i] * ((K - X[i]) - bach)

            elif (K - np.exp(X[j]) > 0):
                arr[i][j] = (K - np.exp(X[j])) + c[i] * -bach

            elif K - X[i] > 0:
                arr[i][j] = c[i] * ((K - X[i]) - bach)
            
            else:
                arr[i][j] = 0

    return arr

c = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
con = control_variate(c)

s = 0
mo = []
for i in range(len(con)):
    for j in range(len(con[i])):
        s+= con[i][j]
    
    mo.append(s)

    print("Control Variate: ", c[i])
    print("Expectation: ", s / len(con[i]))
    print("Confidence Interval: ", confidence_interval(con[i], 0.95))
    print("!!!!!!!!!!")

print("Most Optimal C (Array Index): ", mo.index(max(mo)))

def expectation():

    s = 0
    b = bachelier()
    for i in range(len(b)):
        s += b[i]

    print("Bachelier Expectation: ", s / len(b))
    print("Bachelier Confidence Interval: ", confidence_interval(b, 0.95))

    s = 0
    ab = antithetic_bachelier()
    for i in range(len(ab)):
        s += ab[i]

    print("Antithetic Bachelier Expectation: ", s / len(ab))
    print("Antithetic Bachelier Confidence Interval: ", confidence_interval(ab, 0.95))

    s = 0
    bs = black_scholes()
    for i in range(len(bs)):
        s += bs[i]

    print("Black Scholes Expectation: ", s / len(bs))
    print("Black Scholes Confidence Interval: ", confidence_interval(bs, 0.95))

    s = 0
    abs = antithetic_black_scholes()
    for i in range(len(abs)):
        s += abs[i]

    print("Antithetic Black Scholes Expectation: ", s / len(abs))
    print("Antithetic Black Scholes Confidence Interval: ", confidence_interval(abs, 0.95))

expectation()

# def plot():

#     fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
#     fig.suptitle('Monte Carlo !!')

#     ax1.plot(bachelier())
#     ax1.set_title("Bachelier")

#     ax2.plot(antithetic_bachelier())
#     ax2.set_title("Antithetic Bachelier")

#     ax3.plot(black_scholes())
#     ax3.set_title("Black Scholes")

#     ax4.plot(antithetic_black_scholes())
#     ax4.set_title("Antithetic Black Scholes")

#     plt.show()

# plot()