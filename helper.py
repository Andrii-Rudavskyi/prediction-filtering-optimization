# alpha = 1 - exp(-dt / Tc) # results in exponential smoothing with Tc time constant
# dt is the time step of the signal Ts(t)-Ts(t-1)

# propose 1/Tc = 1/Tmax + 1/ΔT * (v/sig_v)^2
# Tmax is the maximum time constant (smoothing window when v=0)
# ΔT is the nominal camera time step = 1000/fps

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# Function to compute adaptive alpha based on the velocity and noise
def alpha(Tmax, v, sig_v, fps, dt):
    delT = 1000 / fps  # Nominal camera time step
    invTc = 1 / Tmax + 1 / delT * (v / sig_v) ** 2
    return 1 - np.exp(-dt * invTc)  # Exponential smoothing with time constant Tc

# EWMA with adaptive smoothing and velocity noise estimation
def ewma_smoothing(x, Ts, Tmax, fps, initial_sig_v=0.07):
    ex = x[0]  # Smoothed variable
    ev = 0  # Smoothed velocity
    sig_v = initial_sig_v  # Initial velocity noise
    
    smoothed_values = [ex]  # Store the smoothed values
    velocities = []  # Store the instant velocities
    smoothed_velocities = [ev]  # Store the smoothed velocities
    sig_v_list = [sig_v]  # Store the velocity noise over time
    alphas = []  # Store the alpha values over time
    velocity_differences = []  # Store the differences between raw and smoothed velocities
    
    for i in range(1, len(x)):
        dt = Ts[i] - Ts[i - 1]  # Time difference between consecutive measurements
        if dt <= 0:
            continue  # Skip if there's an issue with time steps (e.g., zero or negative dt)

        # Compute the current instant velocity (raw)
        v_instant = (x[i] - x[i - 1]) / dt

        # Calculate alpha for this time step
        current_alpha = alpha(Tmax, v_instant, sig_v, fps, dt)
        alphas.append(current_alpha)

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
            # sig_v = np.std(velocity_differences)  # Noise as standard deviation of differences
            sig_v = initial_sig_v

        # Store the updated values
        smoothed_values.append(ex)
        sig_v_list.append(np.std(velocity_differences))

    return np.array(smoothed_values), np.array(velocities), np.array(smoothed_velocities), np.array(sig_v_list), np.array(alphas)

# Example Usage
# Load the CSV file using pandas
data = pd.read_csv("data/14-10-2024-10-38-04_cnsdk_blink_raw_data_trace.csv")
# Trim spaces from the column names
data.columns = data.columns.str.strip()

# Extract the "leftEyeX2D" and "cameraTimestamp" columns as numpy arrays
x = data["leftEye3DZ"].to_numpy()
Ts = data["cameraTimestamp"].to_numpy()
Ts = Ts - Ts[0]  # Start the time from zero


fps = 90  # Frames per second of the camera
Tmax = 1000
smoothed_x, velocities, smoothed_velocities, sig_v_estimates, alphas = ewma_smoothing(x, Ts, Tmax, fps)

# Output results
# print("Smoothed Values:", smoothed_x)
# print("Velocities:", velocities)
# print("Velocity Noise Estimates:", sig_v_estimates)

# Plot raw vs. smoothed data
plt.figure(figsize=(10, 5))

# Subplot 1: Raw vs. Smoothed Data
plt.subplot(4, 1, 1)
plt.plot(Ts, x, label="Raw Data", linestyle='-')
plt.plot(Ts, smoothed_x, label="Smoothed Data", linestyle='-')
plt.title("Raw vs. Smoothed Data")
plt.xlabel("Time (s)")
plt.ylabel("Position")
# plt.legend()

# Subplot 2: Raw vs. Smoothed Velocities
plt.subplot(4, 1, 2)
plt.plot(Ts[1:], velocities, label="Raw Velocities", linestyle='-')
plt.plot(Ts, smoothed_velocities, label="Smoothed Velocities")
plt.title("Raw vs. Smoothed Velocities")
plt.xlabel("Time (s)")
plt.ylabel("Velocity")
# plt.legend()

plt.subplot(4, 1, 3)
plt.plot(Ts[1:], alphas, label="Alphas", linestyle='-')
plt.title("Alphas")
plt.xlabel("Time (s)")
plt.ylabel("Alphas")

plt.subplot(4, 1, 4)
plt.plot(Ts, sig_v_estimates, label="sig_v", linestyle='-')
plt.title("sig_v")
plt.xlabel("Time (s)")
plt.ylabel("Alphas")

# Show the plots
plt.tight_layout()
plt.show()