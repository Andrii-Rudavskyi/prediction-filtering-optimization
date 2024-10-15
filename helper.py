# alpha = 1 - exp(-dt / Tc) # results in exponential smoothing with Tc time constant
# dt is the time step of the signal Ts(t)-Ts(t-1)

# propose 1/Tc = 1/Tmax + 1/ΔT * (v/sig_v)^2
# Tmax is the maximum time constant (smoothing window when v=0)
# ΔT is the nominal camera time step = 1000/fps

import numpy as np
from matplotlib import pyplot as plt

# Function to compute adaptive alpha based on the velocity and noise
def alpha(Tmax, v, sig_v, fps, dt):
    delT = 1000 / fps  # Nominal camera time step
    invTc = 1 / Tmax + 1 / delT * (v / sig_v) ** 2
    return 1 - np.exp(-dt * invTc)  # Exponential smoothing with time constant Tc

# EWMA with adaptive smoothing and velocity noise estimation
def ewma_smoothing(x, Ts, Tmax, fps, initial_ex=0.0, initial_ev=0.0, initial_sig_v=1.0):
    ex = initial_ex  # Smoothed variable
    ev = initial_ev  # Smoothed velocity
    sig_v = initial_sig_v  # Initial velocity noise
    
    smoothed_values = [ex]  # Store the smoothed values
    velocities = []  # Store the instant velocities
    smoothed_velocities = [ev]  # Store the smoothed velocities
    sig_v_list = [sig_v]  # Store the velocity noise over time
    
    for i in range(1, len(x)):
        dt = Ts[i] - Ts[i - 1]  # Time difference between consecutive measurements
        if dt <= 0:
            continue  # Skip if there's an issue with time steps (e.g., zero or negative dt)

        # Compute the current instant velocity (raw)
        v_instant = (x[i] - x[i - 1]) / dt

        # Calculate alpha for this time step
        current_alpha = alpha(Tmax, v_instant, sig_v, fps, dt)

        # Update the smoothed value using EWMA
        ex = current_alpha * x[i] + (1 - current_alpha) * ex
        
        # Calculate the smoothed velocity
        ev = (ex - smoothed_values[-1]) / dt

        # Store the current instant velocity and smoothed velocity
        velocities.append(v_instant)
        smoothed_velocities.append(ev)

        # Estimate the velocity noise as std(instant velocities - smoothed velocities)
        if i >= 2:
            # Compute the velocity noise as the difference between raw and smoothed velocities
            velocity_differences = np.array(velocities[-min(i, 10):]) - np.array(smoothed_velocities[-min(i, 10):])
            sig_v = np.std(velocity_differences)  # Noise as standard deviation of differences

        # Store the updated values
        smoothed_values.append(ex)
        sig_v_list.append(sig_v)

    return np.array(smoothed_values), np.array(velocities), np.array(smoothed_velocities), np.array(sig_v_list)

# Example Usage
# Sample data and time axis (non-uniform intervals)
Ts = 1000*np.array([0, 0.1, 0.25, 0.35, 0.55, 0.65, 0.8])  # Time in milliseconds (non-uniform time intervals)
x = np.sin(2 * np.pi * Ts)  # Example signal, could be real measurements
fps = 30  # Frames per second of the camera
Tmax = 1000  # Maximum time constant for smoothing

# Run the EWMA smoothing
smoothed_x, velocities, smoothed_velocities, sig_v_estimates = ewma_smoothing(x, Ts, Tmax, fps)

# Output results
print("Smoothed Values:", smoothed_x)
print("Velocities:", velocities)
print("Velocity Noise Estimates:", sig_v_estimates)

# Plot raw vs. smoothed data
plt.figure(figsize=(10, 5))

# Subplot 1: Raw vs. Smoothed Data
plt.subplot(2, 1, 1)
plt.plot(Ts, x, label="Raw Data", marker='o', linestyle='--')
plt.plot(Ts, smoothed_x, label="Smoothed Data", marker='x')
plt.title("Raw vs. Smoothed Data")
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.legend()

# Subplot 2: Raw vs. Smoothed Velocities
plt.subplot(2, 1, 2)
plt.plot(Ts[1:], velocities, label="Raw Velocities", marker='o', linestyle='--')
plt.plot(Ts, smoothed_velocities, label="Smoothed Velocities", marker='x')
plt.title("Raw vs. Smoothed Velocities")
plt.xlabel("Time (s)")
plt.ylabel("Velocity")
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()