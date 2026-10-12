# First python workshop
# First input libraries
import numpy as np

# Defining variables
t_max = 150e-3 #second
dt = 1e-3 #second
tau = 20e-3 #second
el = -60e-3 #mV
vr = -70e-3 #mV
vth = -50e-3 #mV
r = 100e6 #ohm
i_mean = 25e-11 #ampere

print(t_max,dt,tau,el,vr,vth,r,i_mean)

# Creating a sinusoidal signal
# Loop in 10 steps (from 0 to 9)
for step in range(10):
    # compute value of t
    t = step * dt
    # compute value of i
    i = i_mean * (1 + np.sin((2 * t *np.pi)/0.01))

    print(f'{t:.3f} {i:.4e}')