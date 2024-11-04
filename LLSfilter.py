import numpy as np

class LLSFilterParameters:
    def __init__(self, maxSizeHistory = np.array([12, 12, 12]), smoothVscale = np.array([2.5, 2.5, 2.5]),
                 numOutliers = np.array([2, 2, 2]), useFizedZ = False, fixedZ = 50, predictionTime = 0.020):
        self.maxSizeHistory = maxSizeHistory
        self.smoothVscale = smoothVscale
        self.numOutliers = numOutliers
        self.useFizedZ = useFizedZ
        self.fixedZ = fixedZ
        self.predictionTime = predictionTime
        print("maxSizeHistory", self.maxSizeHistory)
        print("smoothVscale", self.smoothVscale)
        print("numOutliers", self.numOutliers)
        print("useFizedZ", self.useFizedZ)
        if (self.useFizedZ):
            print("fixedZ", self.fixedZ)



class LLSfilter:
    def __init__(self, pilterParameters):
        self.pilterParameters = pilterParameters

    def LLS2(self, xvals, yvals, xpredict):
        XSx = 0
        XSy = 0
        XSxy = 0
        XSxx = 0
        XSxxx = 0
        XSxxy = 0
        XSxxxx = 0

        n = yvals.size()
        n_inverse = 1.0 / n

        for ii in range(0, n):
            XSx += xvals[ii]
            XSy += yvals[ii]
            XSxy += xvals[ii] * yvals[ii]
            XSxx += xvals[ii] * xvals[ii]

            XSxxx += xvals[ii] * xvals[ii] * xvals[ii]
            XSxxy += xvals[ii] * xvals[ii] * yvals[ii]
            XSxxxx += xvals[ii] * xvals[ii] * xvals[ii] * xvals[ii]

        Sxx = XSxx - XSx * XSx * n_inverse
        Sxy = XSxy - XSx * XSy * n_inverse
        Sxx2 = XSxxx - XSx * XSxx * n_inverse
        Sx2y = XSxxy - XSxx * XSy * n_inverse
        Sx2x2 = XSxxxx - XSxx * XSxx * n_inverse

        d = Sxx * Sx2x2 - Sxx2 * Sxx2
        d_inverse = 1 / d

        a = (Sx2y * Sxx - Sxy * Sxx2) * d_inverse
        b = (Sxy * Sx2x2 - Sx2y * Sxx2) * d_inverse
        c = (XSy - b * XSx - a * XSxx) * n_inverse

        v = 2 * a * xpredict + b

        predicted = a * xpredict * xpredict + b * xpredict + c

        return v, predicted

    def genericFilterWithHistory(self, history, currentTime, parameters):
        #history is a pair of timestamp and coordinate

        #Values to use for curve fitting
        xs = []
        ys = []
        zs = []

        # Times to use for curve fitting
        timesX = []
        timesY = []
        timesZ = []

        # Each dimension mightb include a different range of the history
        istartX = max(0, len(history) - parameters.maxSizeHistory[0])
        istartY = max(0, len(history) - parameters.maxSizeHistory[1])
        istartZ = max(0, len(history) - parameters.maxSizeHistory[2])

        #Pivot
        imid = (istartX + len(history)) / 2;
        pivotTime = history[imid][1] # pivot time in the middle of history

        for ii in range(0, len(history)):
            if (ii >= istartX):
                timesX.append(history[ii][1] - pivotTime)
            if (ii >= istartY):
                timesY.append(history[ii][1] - pivotTime)
            if (ii >= istartZ):
                timesZ.append(history[ii][1] - pivotTime)

            x = history[ii][0].x
            y = history[ii][0].y
            z = history[ii][0].z

            if (parameters.useFixedZ):
                x = x / z * parameters.fixedZ
                y = y / z * parameters.fixedZ

            if (ii >= istartX):
                xs.append(x)

            if (ii >= istartY):
                ys.append(y)

            if (ii >= istartZ):
                zs.append(z)

        #Calculate velocities in the second half of the history
        vx, predictedX = self.LLS2(timesX, xs, (history[-1][1] - pivotTime) / 2)
        vy, predictedY = self.LLS2(timesY, ys, (history[-1][1] - pivotTime) / 2)
        vz, predictedZ = self.LLS2(timesZ, zs, (history[-1][1] - pivotTime) / 2)

        # Calculate velocities in the first half of the history
        vx2, predictedX = self.LLS2(timesX, xs, (history[0][1] - pivotTime) / 2)
        vy2, predictedY = self.LLS2(timesY, ys, (history[0][1] - pivotTime) / 2)
        vz2, predictedZ = self.LLS2(timesZ, zs, (history[0][1] - pivotTime) / 2)

        if (parameters.useFixedZ):
            vx = vx * vz / parameters.useFixedZ
            vy = vy * vz / parameters.useFixedZ
            vx2 = vx2 * vz2 / parameters.useFixedZ
            vy2 = vy2 * vz2 / parameters.useFixedZ

        vx = max(abs(vx), abs(vx2))
        vy = max(abs(vy), abs(vy2))
        vz = max(abs(vz), abs(vz2))

        sx = 1
        sy = 1
        sz = 1

        if (parameters.smoothVscale[0] != 0):
            unbound_sx = sqrt(abs(vx)) / parameters.smoothVscale[0]
            sx = max(0, min(1, unbound_sx))
        if (parameters.smoothVscale[1] != 0):
            unbound_sy = sqrt(abs(vy)) / parameters.smoothVscale[1]
            sy = max(0, min(1, unbound_sy))
        if (parameters.smoothVscale[2] != 0):
            unbound_sz = sqrt(abs(vz)) / parameters.smoothVscale[2]
            sz = max(0, min(1, unbound_sz))

        # Time with respect to the center of the maintained history
        # This is negative and will result in a smoothing effect

        minPredictionTime = pivotTime - currentTime
        # Time with respect to the last time in history + 0.2s
        # This limits the amount of time that we're trying to predict in the future

        maxPredictionTime = history[-1][1] + 0.2 - currentTime;

        # Times to predict for (between minPredictionTime and predictionTime)
        ptimex = minPredictionTime + (parameters.predictionTime - minPredictionTime) * sx
        ptimey = minPredictionTime + (parameters.predictionTime - minPredictionTime) * sy
        ptimez = minPredictionTime + (parameters.predictionTime - minPredictionTime) * sz



