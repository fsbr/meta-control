# functions for calculating the myopic value of computation
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve fit


# put the cost into this function
def U(q, t, a=1 , b=1):
    return a*q - np.exp(b*t)

if __name__ == "__main__":
    t = np.linspace(0, 1, 1000)
    u = U(1, t)    
    plt.plot(t,u)
    plt.show()
    
