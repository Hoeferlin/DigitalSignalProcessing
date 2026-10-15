# LIF_Neuron_Part1 Exercise 4
# 10.14.2026
# The goal of the exercise is to simulate the evolution of the membrane equation in discrete time steps

# Equation: tm = (V(t+dt) - V(t))/dt = (el - V(t) + R*I(t))/tau

import numpy as np
import matplotlib.pyplot as plt

# Defining variables
t_max = 150e-3 #second
dt = 1e-3 #second
tau = 20e-3 #second
el = -60e-3 #mV
vr = -70e-3 #mV
vth = -50e-3 #mV
r = 100e6 #ohm
i_mean = 25e-11 #ampere

# initialize step_end and v0
step_end = 10
v = el

for step in range(step_end):
    # compute the value of t
    t = step * dt

    # compute the value of i at this time step
    i = i_mean * (1 + np.sin((2 * t *np.pi)/0.01))

    # compute v
    v = v + (dt / tau) * (el - v + r * i)

    # print answer for t and v
    print(f"{t:.3f} {v:.4e}")


# Exercise 5: Plotting I(t)

# initialize end_step
step_end = 25

# Initialize plot
plt.figure()
plt.title("Exercise 5: Plotting I(t)")
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")

# Loop for step_end steps
for step in range(step_end):
    #compute value of t
    t = step * dt

    i = i_mean * (1 + np.sin((2 * t *np.pi)/0.01))

    plt.plot(t, i, 'ro')

plt.show()

# Exercise 6: Plotting membrane potential

# Initialize step_end
step_end = int(t_max/dt)

# Initialize plot
plt.figure()
plt.title("Exercise 6: Plotting Membrane Potential")
plt.xlabel("Time (s)")
plt.ylabel("Membrane Potential (V)")

# Loop for step_end steps
for step in range(step_end):
    # compute value of t
    t = step * dt

    # compute value of i at this time step
    i = i_mean * (1 + np.sin((2 * t *np.pi)/0.01))

    # compute v
    v = v + (dt / tau) * (el - v + r * i)

    plt.plot(t, v, 'k.')

plt.show()