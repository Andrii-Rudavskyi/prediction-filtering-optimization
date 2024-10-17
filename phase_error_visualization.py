import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sn
from helper2 import calculate_delay_error, calculate_predicted_error, predict, extract_noise_2, calculate_phase_error

dataFolder = './data/'
imagesFolder = './images/'

resolutionWidth = 3840
resolutionHeight = 2160
pixelPitch = 0.08964

screenWidth = resolutionWidth * pixelPitch
screenHeight = resolutionHeight * pixelPitch

#Camera position with respect to the screen TODO: load it from the product config data
xc = screenWidth / 2
yc = 0

latency = 0.1

# Focal distance to convert to angular space. TODO: Load it properly from intrinsics (product/device specific).
fx = 330
fy = 330

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

        n = 100

        size = int(resolutionWidth/n), int(resolutionHeight/n)
        error_heatmap = np.zeros(size)

        for k in range(0, len(delay_error[0])):
            for i in range(0, int(resolutionWidth / n)):
                for j in range(0, int(resolutionHeight / n)):
                    phase_error = calculate_phase_error(xc, yc, delay_error[2][k], delay_error[0][k], delay_error[1][k], filtered_signal[2][k], i * n * pixelPitch, j * n * pixelPitch)
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



