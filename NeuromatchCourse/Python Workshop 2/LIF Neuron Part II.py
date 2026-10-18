# Python Workshop 2 - LIF Neuron Part II
# Week 0 Refresher
# 10.18.2026

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

# Section 1: Histograms
# Exercise 1: Plotting a histogram

# Random seed generator
rng = np.random.default_rng(2020)

# Initialize time range, bins, realizations, and equations
t_range =np.arange(0, t_max, dt)
step_end = len(t_range)
n = 10000
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max/dt)**(0.5) * (2 * rng.random([n, step_end]) - 1))
nbins = 50

# Loop over time steps
for step, t in enumerate(t_range):

    # skip first iteration
    if step==0:
        continue

    # Compute v_n
    v_n[:, step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step-1] + r * i[:, step])

# Initialize the figure
plt.figure()
plt.ylabel("Frequency")
plt.xlabel("Membrane Voltage (V)")

# Plot histogram at t_max/10
plt.hist(v_n[:,int(step_end/10)],nbins, label = f't = {t_max / 10} s', histtype = 'stepfilled', linewidth = 0)

# Plot histogram at t_max
plt.hist(v_n[:,-1],nbins, label = f't = {t_max} s', histtype = 'stepfilled', linewidth = 0)

plt.legend()
plt.show()

# Section 2: Dictionaries & introducing spikes
# Exercise 2: adding spiking to the LIF Neuron

# Insert the reset condition, collect spike times of each realization in a dictionary variable, 'spikes' with N = 500

# Random seed generator
rng = np.random.default_rng(2020)

# Initialize time range, bins, realizations, and equations
t_range =np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max/dt)**(0.5) * (2 * rng.random([n, step_end]) - 1))

# Initialize spikes and spikes_n
spikes = {j: [] for j in range(n)}
spikes_n = np.zeros([step_end])

# Loop over time steps
for step, t in enumerate(t_range):

    # Skip first
    if step==0:
        continue

    # Compute v_n
    v_n[:,step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r * i[:, step])

    # Loop over simulations
    for j in range(n):

        # Check if voltage above threshold
        if v_n[j, step] >= vth:

            # Reset to reset voltage
            v_n[j,step] = vr

            # Add spike time
            spikes[j] += [t]

            # Add spike count to this step
            spikes_n[step] += 1

# Collect mean Vm and mean spiking rate
v_mean = np.mean(v_n, axis=0)
spikes_mean = spikes_n / n

# Initialize figures
plt.figure()

 # plot simulations and sample mean
ax1 = plt.subplot(3,1,1)
for j in range(n):
    plt.scatter(t_range, v_n[j], color="k", marker=".", alpha=0.01)
plt.plot(t_range, v_mean, 'C1', alpha=0.8,linewidth = 3)
plt.ylabel('Membrane Voltage (V)')

# Plot spikes
plt.subplot(3,1,2, sharex=ax1)
# for each neuron j: collect spike times and plot them at height j
for j in range(n):
    times = np.array(spikes[j])
    plt.scatter(times, j * np.ones_like(times), color = 'C0', marker=".", alpha=0.2)

plt.ylabel('neuron')

# plot firing rate
plt.subplot(3,1,3, sharex=ax1)
plt.plot(t_range, spikes_mean)
plt.xlabel('time (s)')
plt.ylabel('rate (Hz)')

plt.tight_layout()
plt.show()

# Section 3: Boolean indexes
