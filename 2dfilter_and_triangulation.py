import numpy as np
import pandas as pd

from StereoEyePositionFilter import StereoEyePositionFilter, Point2D, StereoFilterParameters
from Triangulation import Triangulation

import matplotlib.pyplot as plt

dataID = '2024-12-02___14_24_31'
dataPath = './data/windows_traces/'
dataPath = dataPath + dataID + '/'

#instantite Triangulation class
triangulation = Triangulation(dataPath + 'resources')
landmasks_data = pd.read_csv(dataPath + dataID + '_eyeCoordinates.csv')

frameNumber = landmasks_data[' frameNumber']
captureTime = landmasks_data['CaptureTime']

leftEyes_0_x = landmasks_data[' Camera1Left2D.x']
leftEyes_0_y = landmasks_data[' Camera1Left2D.y']
leftEyes_1_x = landmasks_data[' Camera2Left2D.x']
leftEyes_1_y = landmasks_data[' Camera2Left2D.y']

rightEyes_0_x = landmasks_data[' Camera1Right2D.x']
rightEyes_0_y = landmasks_data[' Camera1Right2D.y']
rightEyes_1_x = landmasks_data[' Camera2Right2D.x']
rightEyes_1_y = landmasks_data[' Camera1Right2D.y']

left_x = landmasks_data[' Left3D.x']
left_y = landmasks_data[' Left3D.y']
left_z = landmasks_data[' Left3D.z']

right_x = landmasks_data[' Right3D.x']
right_y = landmasks_data[' Right3D.y']
right_z = landmasks_data[' Right3D.z']

leftEyes_0_x_2Dfiltered_simulated = []
leftEyes_0_y_2Dfiltered_simulated = []

left_x3D_2D_filtered = []
left_y3D_2D_filtered = []

#initialize 2D filter

stereoFilterParameters = StereoFilterParameters('./data/windows_traces/2024-12-02___14_24_31', filter2D=True)
stereoFilter = StereoEyePositionFilter(stereoFilterParameters=stereoFilterParameters)

for i in range(0, len(leftEyes_0_x)):
    leftEyes = [Point2D(x=leftEyes_0_x[i], y=leftEyes_0_y[i]), Point2D(x=leftEyes_1_x[i], y=leftEyes_1_y[i])]
    rightEyes = [Point2D(x=rightEyes_0_x[i], y=rightEyes_0_y[i]), Point2D(x=rightEyes_1_x[i], y=rightEyes_1_y[i])]

    #print('Before: ', leftEyes[0].x, leftEyes[0].y, rightEyes[1].x, rightEyes[1].y)

    leftEyes_2dfiltered, rightEyes_2dfiltered = stereoFilter.filterEyes(frameNumber=frameNumber[i], captureTime=captureTime[i], leftEyes=leftEyes, rightEyes=rightEyes)

    #prepare left and right eyes for triangulation
    points1 = np.array([[leftEyes_2dfiltered[0].x, leftEyes_2dfiltered[0].y],
                        [rightEyes_2dfiltered[0].x, rightEyes_2dfiltered[0].y]], dtype=np.float64)
    points2 = np.array([[leftEyes_2dfiltered[1].x, leftEyes_2dfiltered[1].y],
                        [rightEyes_2dfiltered[1].x, rightEyes_2dfiltered[1].y]], dtype=np.float64)

    #triangulate left and right eyes; rectification of the points is done inside of triangulation fucntion
    xyz = triangulation.triangulate(points1, points2)
    left_x3D_2D_filtered.append(xyz[0][0])
    left_y3D_2D_filtered.append(xyz[0][1])

    #print('After: ', leftEyes_2dfiltered[0].x, leftEyes_2dfiltered[0].y, rightEyes_2dfiltered[1].x, rightEyes_2dfiltered[1].y)

    leftEyes_0_x_2Dfiltered_simulated.append(leftEyes_2dfiltered[0].x)
    leftEyes_0_y_2Dfiltered_simulated.append(leftEyes_2dfiltered[0].y)

fig, axs = plt.subplots(2,2)
axs[0, 0].plot(frameNumber, leftEyes_0_x)
axs[0, 0].plot(frameNumber, leftEyes_0_x_2Dfiltered_simulated)

axs[0, 1].plot(frameNumber, left_x)
axs[0, 1].plot(frameNumber, left_x3D_2D_filtered)

axs[1, 0].plot(frameNumber, leftEyes_0_y)
axs[1, 0].plot(frameNumber, leftEyes_0_y_2Dfiltered_simulated)

axs[1, 1].plot(frameNumber, left_y)
axs[1, 1].plot(frameNumber, left_y3D_2D_filtered)

plt.show()