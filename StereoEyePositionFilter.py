import os
import configparser
import numpy as np

class Point2D:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class StereoFilterParameters:
    def __init__(self, resourcesPath=None, filter2D=None, fitOrder=None, bufferSize=None, filterMethod=None, loggingEnabled=None, diff_threshold=None, selection_hysteresis=None):
        self.resourcesPath = resourcesPath
        self.filter2D = filter2D if filter2D is not None else True
        self.fitOrder = fitOrder if fitOrder is not None else 1
        self.bufferSize = bufferSize if bufferSize is not None else 12
        self.filterMethod = filterMethod if filterMethod is not None else 3
        self.loggingEnabled = loggingEnabled if loggingEnabled is not None else False
        self.diff_threshold = diff_threshold if diff_threshold is not None else 5
        self.selection_hysteresis = selection_hysteresis if selection_hysteresis is not None else 10

        if resourcesPath is not None and os.path.exists(resourcesPath + '/ft_user.ini'):
            print('Parsing parameters from:' + resourcesPath + '/ft_user.ini')

            config = configparser.ConfigParser(inline_comment_prefixes=';')
            config.sections()

            config.read(resourcesPath + '/ft_user.ini')
            config.sections()
            if filter2D is None:
                self.filter2D = config.getboolean('EyeStabilizationParams', 'use2Dfiltering')
            if fitOrder is None:
                self.fitOrder = config.getint('EyeStabilizationParams', 'fitOrder')
            if bufferSize in None:
                self.bufferSize = config.getint('EyeStabilizationParams', 'bufferSize')
            if filterMethod is None:
                self.filterMethod = config.getint('EyeStabilizationParams', 'filterMethod')
            if loggingEnabled is None:
                self.loggingEnabled = config.getboolean('EyeStabilizationParams', 'enableLogging')
            if selection_hysteresis is None:
                self.selection_hysteresis = config.getint('EyeStabilizationParams', 'selection_hysteresis')
            if diff_threshold is None:
                self.diff_threshold = config.getint('EyeStabilizationParams', 'diff_threshold')

    def print_parameters(self):
        print('2D filter parameters:')
        print('filter2D:  ', self.filter2D)
        print('fitOrder: ', self.fitOrder)
        print('bufferSize: ', self.bufferSize)
        print('filterMethod: ', self.filterMethod)
        print('loggingEnabled: ', self.loggingEnabled)
        print('diff_threshold: ', self.diff_threshold)
        print('selection_hysteresis: ', self.selection_hysteresis)

class StereoEyePositionFilter:
    def __init__(self, stereoFilterParameters = StereoFilterParameters()):
        self.stereoFilterParameters = stereoFilterParameters
        self.stereoFilterParameters.print_parameters()

        self.frameNumbers = []
        self.leftCam = []
        self.rightCam = []
        self.select = []

    def filterMethod2(self, left_eyes, right_eyes, order):
        if len(self.frameNumbers) <= 4 + order + 2:
            return left_eyes, right_eyes

        frame_number = self.frameNumbers[-1]
        num_points = len(self.leftCam)

        A = np.zeros((num_points * 4, 4 + order + (1 if order > 0 else 0)))
        Y_x = np.zeros((num_points * 4, 1))
        Y_y = np.zeros((num_points * 4, 1))

        counter = 0

        for mm in range(num_points):
            fn_it = self.frameNumbers[counter]
            left_it = self.leftCam[counter]
            right_it = self.rightCam[counter]

            counter = counter + 1

            f = (float)(fn_it - frame_number)

            A[mm * 4 + 0, 0] = 1
            A[mm * 4 + 0, 1] = 0
            A[mm * 4 + 0, 2] = 0
            A[mm * 4 + 0, 3] = 0
            if order > 0: A[mm * 4 + 0, 4] = f
            if order > 0: A[mm * 4 + 0, 5] = 0  # z
            if order > 1: A[mm * 4 + 0, 6] = f * f
            if order > 2: A[mm * 4 + 0, 7] = f * f * f

            A[mm * 4 + 1, 0] = 1
            A[mm * 4 + 1, 1] = 1
            A[mm * 4 + 1, 2] = 0
            A[mm * 4 + 1, 3] = 0
            if order > 0: A[mm * 4 + 1, 4] = f
            if order > 0: A[mm * 4 + 1, 5] = 0  # z
            if order > 1: A[mm * 4 + 1, 6] = f * f
            if order > 2: A[mm * 4 + 1, 7] = f * f * f

            A[mm * 4 + 2, 0] = 0
            A[mm * 4 + 2, 1] = 0
            A[mm * 4 + 2, 2] = 1
            A[mm * 4 + 2, 3] = 0
            if order > 0: A[mm * 4 + 2, 4] = f
            if order > 0: A[mm * 4 + 2, 5] = f  # z
            if order > 1: A[mm * 4 + 2, 6] = f * f
            if order > 2: A[mm * 4 + 2, 7] = f * f * f

            A[mm * 4 + 3, 0] = 0
            A[mm * 4 + 3, 1] = 0
            A[mm * 4 + 3, 2] = 1
            A[mm * 4 + 3, 3] = 1
            if order > 0: A[mm * 4 + 3, 4] = f
            if order > 0: A[mm * 4 + 3, 5] = f  # z
            if order > 1: A[mm * 4 + 3, 6] = f * f
            if order > 2: A[mm * 4 + 3, 7] = f * f * f

            Y_x[mm * 4 + 0, 0] = left_it[0].x
            Y_x[mm * 4 + 1, 0] = left_it[1].x
            Y_x[mm * 4 + 2, 0] = right_it[0].x
            Y_x[mm * 4 + 3, 0] = right_it[1].x

            Y_y[mm * 4 + 0, 0] = left_it[0].y
            Y_y[mm * 4 + 1, 0] = left_it[1].y
            Y_y[mm * 4 + 2, 0] = right_it[0].y
            Y_y[mm * 4 + 3, 0] = right_it[1].y

        p_x = np.linalg.inv(A.T @ A) @ A.T @ Y_x
        p_y = np.linalg.inv(A.T @ A) @ A.T @ Y_y

        left_eyes[0].x = p_x[0, 0]
        right_eyes[0].x = p_x[0, 0] + p_x[1, 0]
        left_eyes[1].x = p_x[2, 0]
        right_eyes[1].x = p_x[2, 0] + p_x[3, 0]

        left_eyes[0].y = p_y[0, 0]
        right_eyes[0].y = p_y[0, 0] + p_y[1, 0]
        left_eyes[1].y = p_y[2, 0]
        right_eyes[1].y = p_y[2, 0] + p_y[3, 0]

        return left_eyes, right_eyes

    def filterMethod3(self, leftEyes = [Point2D(), Point2D()], rightEyes = [Point2D(), Point2D()]):
        leftEyes, rightEyes = self.filterMethod2(left_eyes=leftEyes, right_eyes=rightEyes, order=0)
        leftEyes_0 = [Point2D(leftEyes[0].x, leftEyes[0].y), Point2D(leftEyes[1].x, leftEyes[1].y)]
        rightEyes_0 = [Point2D(rightEyes[0].x, rightEyes[0].y), Point2D(rightEyes[1].x, rightEyes[1].y)]

        leftEyes, rightEyes = self.filterMethod2(left_eyes=leftEyes, right_eyes=rightEyes, order=1)
        leftEyes_1 = [Point2D(leftEyes[0].x, leftEyes[0].y), Point2D(leftEyes[1].x, leftEyes[1].y)]
        rightEyes_1 = [Point2D(rightEyes[0].x, rightEyes[0].y), Point2D(rightEyes[1].x, rightEyes[1].y)]

        diff1 = Point2D(x=leftEyes_1[0].x - leftEyes_0[0].x, y=leftEyes_1[0].y - leftEyes_0[0].y)
        diff2 = Point2D(x=leftEyes_1[1].x - leftEyes_0[1].x, y=leftEyes_1[1].y - leftEyes_0[1].y)
        diff3 = Point2D(x=rightEyes_1[0].x - rightEyes_0[0].x, y=rightEyes_1[0].y - rightEyes_0[0].y)
        diff4 = Point2D(x=rightEyes_1[1].x - rightEyes_0[1].x, y=rightEyes_1[1].y - rightEyes_0[1].y)

        avg_abserr_x = (abs(diff1.x) + abs(diff2.x) + abs(diff3.x) + abs(diff4.x))/4
        avg_abserr_y = (abs(diff1.y) + abs(diff2.y) + abs(diff3.y) + abs(diff4.y))/4

        one = Point2D(x=1, y=1)

        new_select = Point2D()
        new_select.x = min(1, max(0, avg_abserr_x / self.stereoFilterParameters.diff_threshold))
        new_select.y = min(1, max(0, avg_abserr_y / self.stereoFilterParameters.diff_threshold))

        self.select.append(new_select)
        if (len(self.select) > self.stereoFilterParameters.selection_hysteresis):
            self.select.pop(0)

        f = Point2D(x=0, y=0)

        for ss in self.select:
            f.x = max(f.x, ss.x)
            f.y = max(f.y, ss.y)

        leftEyes[0].x = f.x * leftEyes_1[0].x + (one.x - f.x) * leftEyes_0[0].x
        leftEyes[0].y = f.y * leftEyes_1[0].y + (one.y - f.y) * leftEyes_0[0].y

        leftEyes[1].x = f.x * leftEyes_1[1].x + (one.x - f.x) * leftEyes_0[1].x
        leftEyes[1].y = f.y * leftEyes_1[1].y + (one.y - f.y) * leftEyes_0[1].y

        rightEyes[0].x = f.x * rightEyes_1[0].x + (one.x - f.x) * rightEyes_0[0].x
        rightEyes[0].y = f.y * rightEyes_1[0].y + (one.y - f.y) * rightEyes_0[0].y

        rightEyes[1].x = f.x * rightEyes_1[1].x + (one.x - f.x) * rightEyes_0[1].x
        rightEyes[1].y = f.y * rightEyes_1[1].y + (one.y - f.y) * rightEyes_0[1].y

        return leftEyes, rightEyes

    def filterEyes(self, frameNumber, captureTime, leftEyes = [Point2D(), Point2D()], rightEyes = [Point2D(), Point2D()]):

        if self.stereoFilterParameters.filter2D is False:
            return leftEyes, rightEyes

        currentFrame = 0

        if len(self.frameNumbers) and self.frameNumbers[-1] == frameNumber:
            return leftEyes, rightEyes

        left_cam = [Point2D(leftEyes[0].x, leftEyes[0].y), Point2D(rightEyes[0].x, rightEyes[0].y)]
        right_cam = [Point2D(leftEyes[1].x, leftEyes[1].y), Point2D(rightEyes[1].x, rightEyes[1].y)]

        self.frameNumbers.append(frameNumber)
        self.leftCam.append(left_cam)
        self.rightCam.append(right_cam)

        if len(self.leftCam) > self.stereoFilterParameters.bufferSize:
            self.leftCam.pop(0)

        if len(self.rightCam) > self.stereoFilterParameters.bufferSize:
            self.rightCam.pop(0)
        if len(self.frameNumbers) > self.stereoFilterParameters.bufferSize:
            self.frameNumbers.pop(0)

        if self.stereoFilterParameters.filterMethod == 2:
            leftEyes, rightEyes = self.filterMethod2(left_eyes=leftEyes, right_eyes=rightEyes, order=self.stereoFilterParameters.fitOrder)
        elif self.stereoFilterParameters.filterMethod == 3:
            leftEyes, rightEyes = self.filterMethod3(leftEyes=leftEyes, rightEyes=rightEyes)

        return leftEyes, rightEyes
