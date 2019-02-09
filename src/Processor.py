#!/usr/bin/env python3
import numpy as np
import cv2

class VideoProcessor:
    def __init__(
        delay=-1,
        rate=0,
        fnumber=0,
        length=0,
        stop=True,
        modify=False,
        curPos=0,
        curIndex=0,
        curLevel=0,
        digits=0,
        extension=".avi",
        levels=4,
        alpha=10,
        lambda_c=80,
        fl=0.05,
        fh=0.4,
        chromAttenuation=0.1,
        delta=0,
        exaggeration_factor=2.0,
        lambda_=0):
        self.delay = delay
        self.rate = rate
        self.fnumber = fnumber
        self.length = length
        self.stop = stop
        self.modify = modify
        self.curPos = curPos
        self.curIndex = curIndex
        self.curLevel = curLevel
        self.digits = digits
        self.extension = extension
        self.levels = levels
        self.alpha = alpha
        self.lambda_c = lambda_c
        self.fl = fl
        self.fh = fh
        self.chromAttenuation = chromAttenuation
        self.delta = delta
        self.exaggeration_factor = exaggeration_factor
        self.lambda_ = lambda_
        self.spatialType = ""
        self.temporalType = ""

        def setDelay(self, d):
            self.delay = d

        def getNumberOfProcessedFrames():
            return fnumber

        def getNumberOfPlayedFrames(self):
            return self.curPos

        def getFrameSize(self, cap):
            if cap.isOpened():
                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                return (w, h)

        def getFrameNumber(self, cap):
            f = cap.get(cv2.CAP_PROP_POS_FRAMES)
            return f

        def getPositionMS(self, cap):
            t = cap.get(cv2.CAP_PROP_POS_MSEC)
            return t

        def getFrameRate(self, cap):
            r = cap.get(cv2.CAP_PROP_FPS)
            return r

        def getLength(self):
            return self.length

        def getLengthMS(self):
            l = 1000.0 * self.length / self.rate
            return l

        def calculateLength(self, filename):
            l = 0
            cap = cv2.VideoCapture(filename)
            while cap.isOpened():
                l += 1
            self.length = l
            cap.release()

        def spatialFilter(self, src, pyramid, spatialType):
            if spatialType == 'LAPLACIAN':
                return buildLaplacianPyramid(src, self.levels)
            elif spatialType == 'GAUSSIAN':
                return buildGaussianPyramid(src, self.levels)

        def temporalFilter(self, src):
            if self.temporalType == 'IIR':
                dst = temporalIIRFilter(src, self.levels)
                return dst
            elif self.temporalType == 'IDEAL':
                dst = temporalIdealFilter(src)
                return dst

        def amplify(self, src):
            if self.spatialType is 'LAPLACIAN':	# Motion Magnification
                curAlpha = self.lambda_ / self.delta / 8 - 1
                curAlpha *= self.exaggeration_factor
		        if self.curLevel == self.levels or self.curLevel == 0:
			        return src * 0
		        else:
			        return src*cv2.min(self.alpha, curAlpha)
	        elif self.spatialType is 'GAUSSIAN':	 # Color Magnification
		        return src * self.alpha

        def attenuate(self, src):
            planes = [src[:,:,0], src[:,:,1], src[:,:,2]]
            planes[1] = planes[1] * self.chromAttenuation
            planes[2] = planes[2] * self.chromAttenuation
            return planes
        """
        frames is a numpy array
        """
        def concat(self, frames):
            frameSize = frames[0].shape
            temp = np.zeros((3, framesSize[0]*framesSize[1],self.length-1))
            for i in range(self.length-1):
                reshaped = frames[i].reshape(3,frames[i].shape[1]*frames[i].shape[0])
                temp[:,:,i] = reshaped
            return temp

        def deConcat(self, src, frameSize):
            frames = []
            for i in range(self.length-1):
                line = src[:,:,i]
                reshaped = line.reshape(3, frameSize[0], frameSize[1])
                frames.append(reshaped)
            return np.array(frames)

        #TODO: GET CODEC
        # def getCodec(self, cap):
        #     value = int(cap.get(cv2.CAP_PROP_FOURCC))
        #     codec = []

        def getTempFile(self, tempFileList):
            if len(tempFileList) == 0:
                str_ = tempFileList[-1]
                tempFileList = tempFileList[:-1]
            else:
                str_ = ""
            return str_

        def setSpatialFilter(self, type_):
            self.spatialType = type_

        def setTemporalFilter(self, type_):
            self.temporalType = type_

        def motionMagnify(self):
            self.setSpatialFilter('LAPLACIAN')
            self.setTemporalFilter('IIR')










