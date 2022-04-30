import numpy as np
import matplotlib.pyplot as plt 
from py_vollib.black_scholes import black_scholes

def black_scholes_price(length):

    S = np.linspace(1e-8, 4.0, length + 1)
    K = 1
    Sigma = 0.3 
    T = 0.5 
    r = 0

    put_price = np.zeros(len(S))

    for i in range(len(S)):

        put_price[i] = black_scholes('p', S[i], K, Sigma, r, T) 
        
    return put_price

if __name__ == '__main__':

    def plot():

        N = 100
        prices = black_scholes_price(N)

        fig, ax1 = plt.subplots(1,  1)
        fig.suptitle("Black Scholes Price")

        ax1.plot(prices)
        ax1.set_title("Risk Neutral Price")

        plt.show()

    plot()