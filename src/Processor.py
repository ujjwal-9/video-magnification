#!/usr/bin/env python3
import numpy as np
import cv2
from spatialfilter import buildLaplacianPyramid, buildGaussianPyramid

class VideoProcessor:
    def __init__(
        self,
        writer=None,
        tempWriter = None,
        input_ = None,
        outputFile = None,
        cap=None,
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
        lambda_=0,
        lowpass1=None,
        lowpass2=None):
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
        self.cap = cap
        self.writer = writer #cv::VideoWriter
        self.tempWriter = tempWriter
        self.input_ = input_
        self.outputFile = outputFile

        def setDelay(self, d):
            self.delay = d

        def getNumberOfProcessedFrames():
            return fnumber

        def getNumberOfPlayedFrames(self):
            return self.curPos

        def getFrameSize(self):
            if self.cap.isOpened():
                w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                return (w, h)

        def getFrameNumber(self):
            f = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            return f

        def getPositionMS(self):
            t = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            return t

        def getFrameRate(self):
            r = self.cap.get(cv2.CAP_PROP_FPS)
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
                dst = self.temporalIIRFilter(src, self.levels)
                return dst
            elif self.temporalType == 'IDEAL':
                dst = self.temporalIdealFilter(src)
                return dst

        #temporal IIR filtering an image
        def temporalIIRFilter(src):
            temp1 = (1-self.fh)*lowpass1[self.curLevel] + self.fh*src
            temp2 = (1-self.fl)*lowpass2[self.curLevel] + self.fl*src
            lowpass1[self.curLevel] = temp1
            lowpass2[self.curLevel] = temp2
            return lowpass1[self.curLevel] - lowpass2[self.curLevel]

        #temporalIdalFilter - temporal IIR filtering an image pyramid of concatenated frames
        def temporalIdealFilter(src):
            channels = [src[:,:,0], src[:,:,1], src[:,:,2]]
            for i in range(3):
                current = src[i]
                width = cv2.getOptimalDFTSize(current.shape[1])
                height = cv2.getOptimalDFTSize(current.shape[0])
                tempImg = cv2.copyMakeBorder(current,0,height-current.shape[0],0,width-current.shape[1],cv2.BORDER_CONSTANT, 0)
                tempImg = cv2.dft(tempImg, flags=cv2.DFT_ROWS | cv2.DFT_SCALE)
                filter_ = tempImg
                filter_ = createIdealBandpassFilter(filter_, self.fl, self.fh, self.rate)
                tempImg = cv2.mulSpectrums(tempImg, filter_, cv2.DFT_ROWS)
                tempImg = cv2.idft(tempImg, flags=cv2.DFT_ROWS | cv2.DFT_SCALE)
                channels[i] = tempImg[:current.shape[0], :current.shape[1]]
            channels = cv2.normalize(channels, 0, 1, cv2.NORM_MINMAX)
            return channels

        #create a 1D ideal band-pass filter
        def createIdealBandpassFilter(filter_, fl, fh, rate):
            width = filter_.shape[1]
            height = filter_.shape[0]

            fl = 2 * fl * width / rate
            fh = 2 * fh * width / rate

            for i in range(height):
                for j in range(width):
                    if j>=fl and j<=fh:
                        response = 1.0
                    else:
                        response = 0.0
                    filter_[i,j] = response
            return filter_

        def amplify(self, src):
            if self.spatialType is 'LAPLACIAN':	# Motion Magnification
                curAlpha = self.lambda_ / self.delta / 8 - 1
                curAlpha *= self.exaggeration_factor
                if self.curLevel == self.levels or self.curLevel == 0:
			        return src * 0
                else:
                    return src * cv2.min(self.alpha, curAlpha)
            elif self.spatialType is 'GAUSSIAN':	 # ColorMagnification
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
            temp = np.zeros((3, frameSize[0]*frameSize[1],self.length-1))
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
            if len(tempFileList) != 0:
                str_ = tempFileList[-1]
                tempFileList = tempFileList[:-1]
            else:
                str_ = ""
            return str_

        def getCurTempFile(self):
            str_ = self.tempFile
            return str_

        def setInput(self, fileName):
            self.fnumber = 0
            self.tempFile = fileName
            if self.isOpened():
                self.cap.release()
            if self.cap.open(fileName):
                self.length = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                self.rate = self.getFrameRate()
                self.input_ = self.getNextFrame()

        # def setOutput(self, fileName, codec, framerate, isColor):
        #     self.outputFile = fileName

        def setOutput(self, fileName, ext, numberOfDigits, startIndex):
            if numberOfDigits < 0:
                return False
            self.outputFile = fileName
            self.extension = ext
            self.digits = numberOfDigits
            self.curIndex = startIndex
            return True

        # def createTemp(self, framerate, isColor):


        def setSpatialFilter(self, type_):
            self.spatialType = type_

        def setTemporalFilter(self, type_):
            self.temporalType = type_

        def motionMagnify(self):
            self.setSpatialFilter('LAPLACIAN')
            self.setTemporalFilter('IIR')










