import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

SwapRates = [0.0325, 0.0375, 0.04, 0.0425, 0.04375, 0.045, 0.04625, 0.0475, 0.04875, 0.05, 0.05125, 0.0525]

def moneyMarket(index, rate):

    s = 1
    i = 0
    while i < index:

        s = s * (1 + rate)
        i+=1

    return s


def ZCB():

    zcb = []

    for i in range(len(SwapRates)):
        zcb.append(1 / moneyMarket(i, SwapRates[i]))

    return zcb

def Yields(bonds):

    yields = []

    for i in range(len(SwapRates)):
        sum = 0

        for j in range(i):
            sum += bonds[j]
        
        yields.append(SwapRates[i] * sum)

    return yields

def plot():

    bonds = ZCB()
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Bootstrapping !!')

    cs = CubicSpline(x, bonds)
    ax1.plot(bonds, 'x', label = "Data")
    ax1.plot(cs(x), label = "Cubic Spline")
    ax1.set_title("Bond Prices")
    ax1.legend(loc = "best")

    cs = CubicSpline(x, Yields(bonds))
    ax2.plot(Yields(bonds), 'x', label = "Data")
    ax2.plot(cs(x), label = "Cubic Spline")
    ax2.set_title("Bond Yields")
    ax1.legend(loc = "best")

    plt.show()

plot()