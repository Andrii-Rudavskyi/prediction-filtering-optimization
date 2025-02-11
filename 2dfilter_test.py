import pandas as pd
from StereoEyePositionFilter import StereoEyePositionFilter, Point2D

import matplotlib.pyplot as plt

data = pd.read_csv('./data/2Dfilter_data/2d.csv')

frameNumber = data['frameNumber']
captureTime = data['captureTime']

leftEyes_0_x = data['leftEyes[0].x']
leftEyes_0_y = data['leftEyes[0].y']
leftEyes_1_x = data['leftEyes[1].x']
leftEyes_1_y = data['leftEyes[1].y']

rightEyes_0_x = data['rightEyes[0].x']
rightEyes_0_y = data['rightEyes[0].y']
rightEyes_1_x = data['rightEyes[1].x']
rightEyes_1_y = data['rightEyes[1].y']

leftEyes_0_x_2Dfiltered = data['leftEyes[0].x.2d']
leftEyes_0_y_2Dfiltered = data['leftEyes[0].y.2d']
leftEyes_1_x_2Dfiltered = data['leftEyes[1].x.2d']
leftEyes_1_y_2Dfiltered = data['leftEyes[1].y.2d']

rightEyes_0_x_2Dfiltered = data['rightEyes[0].x.2d']
rightEyes_0_y_2Dfiltered = data['rightEyes[0].y.2d']
rightEyes_1_x_2Dfiltered = data['rightEyes[1].x.2d']
rightEyes_1_y_2Dfiltered = data['rightEyes[1].y.2d']

leftEyes_0_x_2Dfiltered_simulated = []

stereoFilter = StereoEyePositionFilter(filter2D=True)

for i in range(0, len(leftEyes_0_x)):
    leftEyes = [Point2D(x=leftEyes_0_x[i], y=leftEyes_0_y[i]), Point2D(x=leftEyes_1_x[i], y=leftEyes_1_y[i])]
    rightEyes = [Point2D(x=rightEyes_0_x[i], y=rightEyes_0_y[i]), Point2D(x=rightEyes_1_x[i], y=rightEyes_1_y[i])]
    leftEyes_2dfiltered, rightEyes_2dfiltered = stereoFilter.filterEyes(frameNumber=frameNumber[i], captureTime=captureTime[i], leftEyes=leftEyes, rightEyes=rightEyes)

    leftEyes_0_x_2Dfiltered_simulated.append(leftEyes_2dfiltered[0].x)

plt.plot(frameNumber, leftEyes_0_x)
plt.plot(frameNumber, leftEyes_0_x_2Dfiltered)
plt.plot(frameNumber, leftEyes_0_x_2Dfiltered_simulated)
plt.legend(['raw', 'filtered', 'filtered_simulated'])
plt.show()