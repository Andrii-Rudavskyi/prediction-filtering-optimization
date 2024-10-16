import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sn

resolutionWidth = 3840
resolutionHeight = 2160
pixelPitch = 0.08964
VD = 600
slant = 0.33
IPD = 63

screenWidth = resolutionWidth * pixelPitch
screenHeight = resolutionHeight * pixelPitch

ph0 = 0

xc = screenWidth / 2
yc = 0

def calculate_phase_error(deltaZ, deltaX2d, deltaY2d, Z, x0, y0):
    donopx = VD / (2 * IPD)
    phase_error = -donopx * (((xc - x0) + slant * (yc - y0)) * deltaZ / np.power(Z, 2) + deltaX2d + slant * deltaY2d)
    return phase_error

def extract_noise_2(signal, time, windowSize = 5, polynomialOrder = 2):
    #Smooths signal by appling moving window. Points withing the window are fitted with the polynomial and the middle
    # of the window is estimated by calculating polynom value at that point.
    filtered_signal = []
    filtered_time = []
    for i in range(0, len(signal) - windowSize + 1):
        t = time[i:i + windowSize]
        x = signal[i:i + windowSize]
        p = np.polyfit(t, x, polynomialOrder)

        t_est = t[i + (windowSize - 1)/2]
        x_est = 0
        for j in range(0, polynomialOrder + 1):
            x_est = x_est + p[j] * np.power(t_est, polynomialOrder - j)

        filtered_time = np.append(filtered_time, t_est)
        filtered_signal = np.append(filtered_signal, x_est)

    filtered_noise = signal[int((windowSize - 1) / 2):int(len(signal) - (windowSize - 1) / 2)] - filtered_signal
    return filtered_time, filtered_signal, filtered_noise

def calculate_delay_error(t, x, predictionTime = 0.1):
    # just shift in time by expected latency. Once prediction filters applied another function should be implemented that calculates residual error and probably another that calculates noise.
    predicted_t = t - predictionTime
    error = x - np.interp(predicted_t, t, x)
    return error

dataFolder = './data/'
imagesFolder = './images/'

# Focal distance to convert to angular space. TODO: Load it properly from intrinsics.
fx = 330
fy = 330
latency = 0.05

for filename in os.listdir(dataFolder):
    if filename.endswith('cnsdk_blink_raw_data_trace.csv'):
        eyetrackerdata = pd.read_csv(dataFolder + filename)
        imudata = pd.read_csv(dataFolder + filename)

        eyetrackerdata_timestapms = eyetrackerdata[' cameraTimestamp']
        #Start trace from zero
        eyetrackerdata_timestapms = 0.001 * (eyetrackerdata_timestapms - eyetrackerdata_timestapms[0])

        #Convert to angular space
        leftEyeX = eyetrackerdata[' leftEyeX'] / fx
        leftEyeY = eyetrackerdata[' leftEyeY'] / fy
        leftEye3DZ = eyetrackerdata[' leftEye3DZ']

        signal = [leftEyeX, leftEyeY, leftEye3DZ]
        signal_label = ['leftEyeX', 'leftEyeY', 'leftEye3DZ']

        filtered_time = []
        filtered_signal = []
        noise = []
        delay_error = []

        #Calculate noise, filtered signal and delay error for x2d,y2d and z
        for i in range(0, 3):
            filtered_time_tmp, filtered_signal_tmp, noise_tmp = extract_noise_2(signal[i], eyetrackerdata_timestapms)
            delay_error_tmp = calculate_delay_error(filtered_time_tmp, filtered_signal_tmp, predictionTime=latency)
            filtered_time.append(filtered_time_tmp)
            filtered_signal.append(filtered_signal_tmp)
            noise.append(noise_tmp)
            delay_error.append(delay_error_tmp)

        # filtered_time_x, filtered_signal_x, noise_x = extract_noise_2(signal[0], eyetrackerdata_timestapms)
        # delay_error_x = calculate_delay_error(filtered_time_x, filtered_signal_x, predictionTime=latency)
        #
        # filtered_time_y, filtered_signal_y, noise_y = extract_noise_2(signal[1], eyetrackerdata_timestapms)
        # delay_error_y = calculate_delay_error(filtered_time_y, filtered_signal_y, predictionTime=latency)
        #
        # filtered_time_z, filtered_signal_z, noise_z = extract_noise_2(signal[2], eyetrackerdata_timestapms)
        # delay_error_z = calculate_delay_error(filtered_time_z, filtered_signal_z, predictionTime=latency)
        #
        # filtered_time = [filtered_time_x, filtered_time_y, filtered_time_z]
        # filtered_signal = [filtered_signal_x, filtered_signal_y, filtered_signal_z]
        # noise = [noise_x, noise_y, noise_z]
        # delay_error = [delay_error_x, delay_error_y, delay_error_z]

        n = 100

        size = int(resolutionWidth/n), int(resolutionHeight/n)
        error_heatmap = np.zeros(size)

        for k in range(0, len(delay_error[0])):
            for i in range(0, int(resolutionWidth / n)):
                for j in range(0, int(resolutionHeight / n)):
                    phase_error = calculate_phase_error(delay_error[2][k], delay_error[0][k], delay_error[1][k], filtered_signal[2][k], i * n * pixelPitch, j * n * pixelPitch)
                    error_heatmap[i, j] = phase_error

            frameIDstr = 0
            if k < 10:
                frameIDstr = '000' + str(k)
            if (k >= 10 and k < 100):
                frameIDstr = '00' + str(k)
            if (k >= 100 and k < 1000):
                frameIDstr = '0' + str(k)
            if (k >= 1000):
                frameIDstr = str(k)

            plt.figure(figsize=(10, 8))
            for l in range(0, 3):
                plt.subplot(4, 1, l + 1)
                plt.plot(filtered_time[l], filtered_signal[l], '-o')
                plt.title(str(signal_label[l]))
                plt.plot(filtered_time[l], delay_error[l], '-o')
                plt.legend(['Signal', 'Delay Error'])
                plt.axvline(x=filtered_time[l][k])
                print("filtered_time[l][k] ", filtered_time[l][k])


            plt.subplot(4, 1, 4)
            sn.heatmap(error_heatmap.transpose(), vmin=-0.5, vmax=0.5)
            plt.savefig(imagesFolder + 'frame' + frameIDstr + '.png')
            #plt.show()

# fps = 60
# output_dir = './images'
# # Generate the MP4 video using ffmpeg
# ffmpeg_command = f"ffmpeg -y -r {fps} -i {output_dir}/frame%04d.png -vcodec libx264 -pix_fmt yuv420p {'./' + 'video.mp4'}"
# exit_status = os.system(ffmpeg_command)



