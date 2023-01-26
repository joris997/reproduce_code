import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def dx(t,x):
    return -x**(1./3.) - x**3

if __name__ == "__main__":
    x0 = np.array([50.0])
    Tmax = 2/3*((abs(x0)**2)**(1./3.))

    trange = [0, Tmax[0]+2.]
    sol = solve_ivp(dx,trange,x0)

    print(Tmax)
    print(trange)

    fig, axs = plt.subplots(1,2,figsize=(10,5))
    axs = axs.ravel()
    axs[0].plot(sol.t,sol.y[0,:])
    axs[0].scatter(Tmax,0.0)
    plt.savefig("fluff/FxT.png")

