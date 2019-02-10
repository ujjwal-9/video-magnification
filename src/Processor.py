#!/usr/bin/env python3
import numpy as np
import cv2
from spatialfilter import buildLaplacianPyramid, buildGaussianPyramid, reconImgFromLaplacianPyramid, upsamplingFromGaussianPyramid
import datetime

class Processor(object):
    def __init__(
        self,
        f = "",
        cap = cv2.VideoCapture(),
        writer = cv2.VideoWriter(),
        tempWriter = cv2.VideoWriter(),
        delay = -1,
        rate = 0,
        fnumber = 0,
        length = 0,
        stop = True,
        modify = False,
        curPos = 0,
        curIndex = 0,
        curLevel = 0,
        digits = 0,
        extension = ".avi",
        levels = 4,
        alpha = 10,
        lambda_c = 80,
        fl = 0.05,
        fh = 0.4,
        chromAttenuation = 0.1,
        delta = 0,
        exaggeration_factor = 2.0,
        lambda_ = 0,
        lowpass1 = np.array([]),
        lowpass2 = np.array([]),
        tempFile = None,
        outputFile = None):
        self.file = f
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
        self.writer = writer #cv::VideoWriter
        self.tempWriter = tempWriter
        self.outputFile = outputFile
        self.tempFile = tempFile
        self.tempFileList = []
        self.lowpass1 = lowpass1
        self.lowpass2 = lowpass2
        if len(self.file) != 0:
            self.cap = cv2.VideoCapture(self.file)
        else:
            self.cap = cap

    def setDelay(self, d):
        self.delay = d

    def getNumberOfProcessedFrames(self):
        return self.fnumber

    def getNumberOfPlayedFrames(self):
        return self.curPos

    def getFrameSize(self):
        if self.cap.isOpened():
            w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            return (int(w), int(h))

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

    def calculateLength(self):
        # l = 0
        tempCap = cv2.VideoCapture(self.file)
        self.length = int(tempCap.get(cv2.CAP_PROP_FRAME_COUNT))
        # while tempCap.isOpened():
        #     print(l)
        #     l += 1
        # self.length = l
        tempCap.release()

    def spatialFilter(self, src, pyramid):
        if self.spatialType == 'LAPLACIAN':
            return buildLaplacianPyramid(src, self.levels)
        elif self.spatialType == 'GAUSSIAN':
            return buildGaussianPyramid(src, self.levels)

    def temporalFilter(self, src):
        if self.temporalType == 'IIR':
            dst = self.temporalIIRFilter(src, self.levels)
            return dst
        elif self.temporalType == 'IDEAL':
            dst = self.temporalIdealFilter(src)
            return dst

    #temporal IIR filtering an image
    def temporalIIRFilter(self, src):
        temp1 = (1-self.fh)*self.lowpass1[self.curLevel] + self.fh*src
        temp2 = (1-self.fl)*self.lowpass2[self.curLevel] + self.fl*src
        self.lowpass1[self.curLevel] = temp1
        self.lowpass2[self.curLevel] = temp2
        return self.lowpass1[self.curLevel] - self.lowpass2[self.curLevel]

    #temporalIdalFilter - temporal IIR filtering an image pyramid of concatenated frames
    def temporalIdealFilter(self, src):
        channels = [src[:,:,0], src[:,:,1], src[:,:,2]]
        for i in range(3):
            current = src[i]
            width = cv2.getOptimalDFTSize(current.shape[1])
            height = cv2.getOptimalDFTSize(current.shape[0])
            tempImg = cv2.copyMakeBorder(current,0,height-current.shape[0],0,width-current.shape[1],cv2.BORDER_CONSTANT, 0)
            tempImg = cv2.dft(tempImg, flags=cv2.DFT_ROWS | cv2.DFT_SCALE)
            filter_ = tempImg
            filter_ = self.createIdealBandpassFilter(filter_, self.fl, self.fh, self.rate)
            tempImg = cv2.mulSpectrums(tempImg, filter_, cv2.DFT_ROWS)
            tempImg = cv2.idft(tempImg, flags=cv2.DFT_ROWS | cv2.DFT_SCALE)
            channels[i] = tempImg[:current.shape[0], :current.shape[1]]
        channels = cv2.normalize(channels, 0, 1, cv2.NORM_MINMAX)
        return channels

    #create a 1D ideal band-pass filter
    def createIdealBandpassFilter(self, filter_, fl, fh, rate):
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
            input_ = self.getNextFrame()
            return True
        else:
            return False

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

    def createTemp(self, name="evm", framerate=0.0, isColor=True):
        if self.file is None:
            self.tempFile = "../Results/" + name + "_" + datetime.datetime.now().strftime("%y%m%d%H%M")+".avi"
        else:
            self.tempFile = "../Results/" + (str(self.file.split('/')[-1])).split('.')[0] + ".avi"
        self.tempFileList.append(self.tempFile)
        # print(type(self.getFrameSize(),))
        if framerate == 0.0:
            framerate = self.getFrameRate()
        self.tempWriter = cv2.VideoWriter(
            self.tempFile,
            cv2.VideoWriter_fourcc('M','J','P','G'),
            int(framerate),
            self.getFrameSize(),
            isColor)

    def setSpatialFilter(self, type_):
        self.spatialType = type_

    def setTemporalFilter(self, type_):
        self.temporalType = type_

    def stopIt(self):
        self.stop = True

    def prevFrame(self):
        if self.isStop():
            self.pauseIt()
        if self.curPos >= 0:
            self.curPos = -1
            self.jumpTo(self.curPos)

    def nextFrame(self):
        if self.isStop():
            self.pauseIt()
        self.curPos += 1
        if self.curPos <= self.length:
            self.curPos += 1
            self.jumpTo(self.curPos)

    def jumpTo(self, index):
        if index >= self.length:
            return 1
        re = self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        if re and not self.isStop():
            frame = self.cap.read()
        return re

    def jumpToMS(self, pos):
        return self.cap.set(cv2.CAP_PROP_POS_MSEC, pos)

    def close(self):
        self.rate = 0
        self.length = 0
        self.modify = 0
        self.cap.release()
        self.writer.release()
        self.tempWriter.release()

    def isStop(self):
        return self.stop

    def isModified(self):
        return self.modify

    def isOpened(self):
        return self.cap.isOpened()

    def getNextFrame(self):
        return self.cap.read()

    def writeNextFrame(self, frame):
        if len(self.extension):
            curindex = str(self.curIndex + 1)
            length = len(self.outputFile + curindex + self.extension)
            ss = self.outputFile + '0' * (self.digits - length) + curindex + self.extension
            return cv2.imwrite(ss,frame)
        else:
            return self.writer.write(frame)

    def playIt(self):
        if self.isOpened():
            return
        self.stop = False
        while not self.isStop():
            input_ = self.getNextFrame()
            if not input_:
                break
            self.curPos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

    def pauseIt(self):
        self.stop = True

    def isOpened(self):
        return self.cap.isOpened()

    def writeOutput(self):
        if not self.isOpened() or self.writer.isOpened():
            return
        pos = self.curPos
        self.jumpTo(0)
        input_ = self.getNextFrame()
        while input_:
            if len(self.outputFile) != 0:
                self.writeNextFrame(input_)
        self.modify = False
        self.writer.release()
        self.jumpTo(pos)

    def revertVideo(self):
        self.jumpTo(0)
        self.curPos = 0
        self.pauseIt()

    def motionMagnify(self):
        self.setSpatialFilter('LAPLACIAN')
        self.setTemporalFilter('IIR')
        self.createTemp(name="motion")
        if not self.isOpened():
            return
        self.modify = True
        self.stop = False
        pos = self.curPos
        self.jumpTo(0)
        while not self.isStop():
            input_ = self.getNextFrame()
            if not input_:
                break
            input_ = cv2.convertScaleAbs(input_, alpha=1.0/255.0)
            input_ = cv2.cvtColor(input_, cv2.COLOR_BGR2Lab)
            pyramid = self.spatialFilter(input_)
            if self.fnumber == 0:
                self.lowpass1 = pyramid
                self.lowpass2 = pyramid
                filtered = pyramid
            else:
                for i in range(self.levels):
                    self.curLevel = i
                    filtered[i] = self.temporalFilter(pyramid[i])
                filterSize = filtered[0].shape
                w = filterSize[1]
                h = filterSize[0]

                self.delta = self.lambda_c / 8.0 / (1.0 + self.alpha)
                self.exaggeration_factor = 2.0
                import math
                self.lambda_ = math.sqrt(w * w + h * h) / 3
                for i in reversed(range(self.levels+1)):
                    self.curLevel = i
                    filtered[i] = self.amplify(filtered[i])
                    self.lambda_ /= 2.0

            motion = reconImgFromLaplacianPyramid(filtered, self.levels)
            motion = self.attenuate(motion)
            if self.fnumber > 0:
                input_ += motion
            output = input_
            output = cv2.cvtColor(output, cv2.COLOR_Lab2BGR)
            output = cv2.convertScaleAbs(output, alpha=255.0, beta=1.0/255.0)
            output = output.astype(np.uint8)
            self.tempWriter.write(output)
        self.tempWriter.release()
        self.setInput(self.tempFile)
        self.jumpTo(pos)

    def colorMagnify(self):
        self.setSpatialFilter('GAUSSIAN')
        self.setTemporalFilter('IDEAL')

        self.createTemp(name="color")

        frames = []
        downSampledFrames = []

        if not self.isOpened():
            return

        self.modify = True
        self.stop = False
        pos = self.curPos

        self.jumpTo(0)

        input_ = self.getNextFrame()

        while input_ and not self.isStop():
            temp = input_.astype(np.int32)
            frames.append(temp)
            pyramid = self.spatialFilter(temp)
            downSampledFrames.append(pyramid[self.levels-1])

        frames = np.array(frames)
        downSampledFrames = np.array(downSampledFrames)

        if self.isStop():
            self.fnumber = 0
            return

        videoMat = self.concat(downSampledFrames)
        filtered = self.temporalFilter(videoMat)
        filtered = self.amplify(filtered)
        filteredFrames = self.deConcat(filtered,downSampledFrames[0].shape)

        self.fnumber = 0
        for i in range(self.length-1):
            if self.isStop():
                break
            motion = upsamplingFromGaussianPyramid(filteredFrames[i], self.levels)
            motion = np.resize(motion, (frames[0]).shape)
            temp = frames[i] + motion
            output = temp
            minVal, maxVal, _, _ = cv2.minMaxLoc(output)
            output = cv2.convertScaleAbs(output, alpha=255.0/(maxVal-minVal), beta=(-minVal)*255.0/(maxVal-minVal))
            output = output.astype(np.uint8)
            self.tempWriter.write(output)

        self.tempWriter.release()
        self.setInput(self.tempFile)
        self.jumpTo(pos)

if __name__ == '__main__':
    import os
    DATA_DIR = '../Videos/'
    for file_ in os.listdir(DATA_DIR):
        if file_.endswith(".mp4"):
            print("Processing : " + file_)
            m = Processor(f=DATA_DIR+file_)
            m.motionMagnify()