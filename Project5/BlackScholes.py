import numpy as np
from numba import jit
import matplotlib.pyplot as plt 
from py_vollib.black_scholes.black_scholes import black_scholes

def black_scholes_price():

    put_price = black_scholes('p',100,90,.5,.01,.2) 

    return put_price