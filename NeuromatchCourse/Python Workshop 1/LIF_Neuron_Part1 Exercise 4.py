# LIF_Neuron_Part1 Exercise 4
# 10.14.2026
# The goal of the exercise is to simulate the evolution of the membrane equation in discrete time steps

# Equation: tm = (V(t+dt) - V(t))/dt = (el - V(t) + R*I(t))/tau

import numpy as np
import matplotlib.pyplot as plt
import random

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

# Section 4: Random synaptic input
# Exercise 7: Adding randomness

# Set random number generator seed
np.random.seed(2020)

# Initialize step_end
step_end = int(t_max/dt)
v = el

# Initialize plot
plt.figure()
plt.title("Exercise 7: Random Synaptic Input")
plt.xlabel("Time (s)")
plt.ylabel("Membrane Potential (V)")

# Loop for step_end steps
for step in range(step_end):
    # compute value of t
    t = step * dt

    # random numbers
    random_number = 2 * np.random.random() - 1
    # compute value of i at this time step
    i = i_mean * (1 + 0.1*(t_max/dt)**(0.5) * random_number)

    # compute v
    v = v + (dt / tau) * (el - v + r * i)

    plt.plot(t, v, 'k.')

plt.show()

# Exercise 8: Storing stimulations in a list

# Set random number generator seed
np.random.seed(2020)

# Initialize step_end
step_end = int(t_max/dt)
n = 50
v_n = el * np.ones(n)

# Initialize plot
plt.figure()
plt.title("Exercise 8: Storing Stimulations in a list")
plt.xlabel("Time (s)")
plt.ylabel("Membrane Potential (V)")

# loop for step_end steps
for step in range(step_end):
    # compute value of t
    t = step * dt

    # Loop 50 times to get 50 cv
    for j in range(0,n):
         # random numbers
         random_number = 2 * np.random.random() - 1
         
         # compute value of i at this time step
         i = i_mean * (1 + 0.1*(t_max/dt)**(0.5) * random_number)
         
         # compute v
         v_n[j] = v_n[j] + (dt / tau) * (el - v_n[j] + r * i)

    plt.plot([t]*n, v_n, 'k.', alpha=0.1)

plt.show()


