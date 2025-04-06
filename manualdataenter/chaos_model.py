import numpy as np
from scipy.integrate import odeint

def chaos_system(state, t, sigma, rho, beta, d0):
    x, y, z, y1, z1 = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - x * y1 - beta * z
    dy1dt = x * z - 2 * x * z1 - d0 * y1
    dz1dt = 2 * x * y1 - 4 * beta * z1
    return [dxdt, dydt, dzdt, dy1dt, dz1dt]

def solve_chaos(state0, t, sigma, rho, beta, d0):
    return odeint(chaos_system, state0, t, args=(sigma, rho, beta, d0))