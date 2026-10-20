# Python Workshop 2 - LIF Neuron Part II
# Week 0 Refresher
# 10.18.2026
#%% Import and variables
import numpy as np
import matplotlib.pyplot as plt
import random

# Defining variables
t_max = 150e-3   # second
dt = 1e-3        # second
tau = 20e-3      # second
el = -60e-3      # milivolt
vr = -70e-3      # milivolt
vth = -50e-3     # milivolt
r = 100e6        # ohm
i_mean = 25e-11  # ampere

#%%
def plot_all(t_range, v, raster=None, spikes=None, spikes_mean=None):
  """
  Plots Time evolution for
  (1) multiple realizations of membrane potential
  (2) spikes
  (3) mean spike rate (optional)

  Args:
    t_range (numpy array of floats)
        range of time steps for the plots of shape (time steps)

    v (numpy array of floats)
        membrane potential values of shape (neurons, time steps)

    raster (numpy array of floats)
        spike raster of shape (neurons, time steps)

    spikes (dictionary of lists)
        list with spike times indexed by neuron number

    spikes_mean (numpy array of floats)
        Mean spike rate for spikes as dictionary

  Returns:
    Nothing.
  """

  v_mean = np.mean(v, axis=0)
  fig_w, fig_h = plt.rcParams['figure.figsize']
  plt.figure(figsize=(fig_w, 1.5 * fig_h))

  ax1 = plt.subplot(3, 1, 1)
  for j in range(n):
    plt.scatter(t_range, v[j], color="k", marker=".", alpha=0.01)
  plt.plot(t_range, v_mean, 'C1', alpha=0.8, linewidth=3)
  plt.xticks([])
  plt.ylabel(r'$V_m$ (V)')

  if raster is not None:
    plt.subplot(3, 1, 2)
    spikes_mean = np.mean(raster, axis=0)
    plt.imshow(raster, cmap='Greys', origin='lower', aspect='auto')

  else:
    plt.subplot(3, 1, 2, sharex=ax1)
    for j in range(n):
      times = np.array(spikes[j])
      plt.scatter(times, j * np.ones_like(times), color="C0", marker=".", alpha=0.2)

  plt.xticks([])
  plt.ylabel('neuron')

  if spikes_mean is not None:
    plt.subplot(3, 1, 3, sharex=ax1)
    plt.plot(t_range, spikes_mean)
    plt.xlabel('time (s)')
    plt.ylabel('rate (Hz)')

  plt.tight_layout()
  plt.show()

 

#%%
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

#%%
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

#%%
# Section 3: Boolean indexes
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

    # Initialize boolean numpy array 'spiked' with v_n > v_thr
    spiked = (v_n[:, step] >= vth)

    # set relevant values of v_n to resting potential using spiked
    v_n[spiked,step] = vr

    for j in np.where(spiked)[0]:
        spikes[j] += [t]
        spikes_n[step] += 1

# collect mean spiking rate 

spikes_mean = spikes_n / n
# plot multipler ealizations of Vm, spikes, and mean spike rate
plot_all(t_range, v_n, spikes = spikes, spikes_mean = spikes_mean)

# Exercise 4: making a binary raster plot
# Random seed generator
rng = np.random.default_rng(2020)

# Initialize time range, bins, realizations, and equations
t_range =np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max/dt)**(0.5) * (2 * rng.random([n, step_end]) - 1))

# initialize binary numpy array for raster plot
raster = np.zeros([n, step_end])

# Loop over time steps
for step, t in enumerate(t_range):

   if step==0:
      continue

   # compute v_n
   v_n[:,step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r * i[:, step])

   # Initialize boolean numpy array 'spiked' with v_n > v_thr
   spiked = (v_n[:, step] >= vth)

   # set relevant values of v_n to resting potential using spiked
   v_n[spiked,step] = vr

   # Set relevant elements in raster to 1 using spiked
   raster[spiked,step] = 1

# Plot multiple realizations of Vm, spikes, and mean spike rate
plot_all(t_range,v_n, raster)

#%% 
# Section 4: Refractory Period
# Exercise 5: investigating refractory periods

# Setting random seed
rng = np.random.default_rng(2020)

# Initialize time range, bins, realizations, and equations
t_range =np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max/dt)**(0.5) * (2 * rng.random([n, step_end]) - 1))

# Initialize binary numpy array for raster plot
raster = np.zeros([n,step_end])

# Initialize t_ref and last_spike
t_ref = 0.01
last_spike = -t_ref * np.ones([n])
# Loop over time steps
for step, t in enumerate(t_range):

   if step==0:
      continue

   # compute v_n
   v_n[:,step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r * i[:, step])

   # Initialize boolean numpy array 'spiked' with v_n > v_thr
   spiked = (v_n[:, step] >= vth)

   # set relevant values of v_n to resting potential using spiked
   v_n[spiked,step] = vr

   # Set relevant elements in raster to 1 using spiked
   raster[spiked,step] = 1

   # Initialize boolean numpy array clamped using last_spike, t, and t_ref
   clamped = (last_spike + t_ref > t)

   # reset clamped neurons to vr
   v_n[clamped,step] = vr

   # Update numpy array last_spike with time t for spiking neurons
   last_spike[spiked] = t

# Plot multiple realizations of Vm, spikes, and mean spike rate
plot_all(t_range,v_n, raster)
#%%
# Interactive demo 1: random refractory period

# Setting random seed
rng = np.random.default_rng(2020)

# Initialize time range, bins, realizations, and equations
t_range =np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max/dt)**(0.5) * (2 * rng.random([n, step_end]) - 1))

# Initialize binary numpy array for raster plot
raster = np.zeros([n,step_end])

# Initialize t_ref and last_spike
t_ref = 0.01 + 0.007 * rng.random(n)
t_ref[t_ref<0] = 0
last_spike = -t_ref * np.ones([n])

# Loop over time steps
for step, t in enumerate(t_range):

   if step==0:
      continue

   # compute v_n
   v_n[:,step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r * i[:, step])

   # Initialize boolean numpy array 'spiked' with v_n > v_thr
   spiked = (v_n[:, step] >= vth)
   v_n[spiked,step] = vr
   raster[spiked,step] = 1

   # Initialize boolean numpy array clamped using last_spike, t, and t_ref
   clamped = (last_spike + t_ref > t)
   v_n[clamped,step] = vr
   last_spike[spiked] = t

# Plot multiple realizations of Vm, spikes, and mean spike rate
plot_all(t_range,v_n, raster)

# plot histograms of t_ref
plt.figure(figsize=(8,4))
plt.hist(t_ref, bins=32, histtype = 'stepfilled', linewidth=0, color='C1')
plt.xlabel(r'$t_{ref}$ (s)')
plt.ylabel('count')
plt.tight_layout()
plt.show()
# %%
# Section 5: Using Functions
# Exercise 6: rewriting code with functions

def ode_step(v, i, dt):
  """
  Evolves membrane potential by one step of discrete time integration

  Args:
    v (numpy array of floats)
      membrane potential at previous time step of shape (neurons)

    i (numpy array of floats)
      synaptic input at current time step of shape (neurons)

    dt (float)
      time step increment

  Returns:
    v (numpy array of floats)
      membrane potential at current time step of shape (neurons)
  """
  v = v + dt/tau * (el - v + r*i)

  return v


def spike_clamp(v, delta_spike):
  """
  Resets membrane potential of neurons if v>= vth
  and clamps to vr if interval of time since last spike < t_ref

  Args:
    v (numpy array of floats)
      membrane potential of shape (neurons)

    delta_spike (numpy array of floats)
      interval of time since last spike of shape (neurons)

  Returns:
    v (numpy array of floats)
      membrane potential of shape (neurons)
    spiked (numpy array of floats)
      boolean array of neurons that spiked  of shape (neurons)
  """
  # Boolean array spiked indexes neurons with v>=vth
  spiked = (v >= vth)
  v[spiked] = vr

  # Boolean array clamped indexes refractory neurons
  clamped = (t_ref > delta_spike)
  v[clamped] = vr

  return v, spiked

# Setting random seed
rng = np.random.default_rng(2020)

# Initialize time range, bins, realizations, and equations
t_range =np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max/dt)**(0.5) * (2 * rng.random([n, step_end]) - 1))

# Initialize binary numpy array for raster plot
raster = np.zeros([n,step_end])

# Initialize t_ref and last_spike
mu = 0.01
sigma = 0.007
t_ref = mu + sigma * rng.random(n)
t_ref[t_ref<0] = 0
last_spike = -t_ref * np.ones([n]) 

# Loop over time steps
for step, t in enumerate(t_range):

   if step==0:
      continue

   # compute v_n
   v_n[:,step] = ode_step(v_n[:, step - 1], i[:,step], dt)

   # Rest membrane potential and clamp
   v_n[:,step], spiked = spike_clamp(v_n[:,step],t - last_spike)

   # Update raster and last_spike
   raster[spiked,step] = 1
   last_spike[spiked] = t

# Plot multiple realizations of Vm, spikes, and mean spike rate
plot_all(t_range,v_n, raster)

# plot histograms of t_ref
plt.figure(figsize=(8,4))
plt.hist(t_ref, bins=32, histtype = 'stepfilled', linewidth=0, color='C1')
plt.xlabel(r'$t_{ref}$ (s)')
plt.ylabel('count')
plt.tight_layout()
plt.show()
# %%
# Section 6: Using classes
