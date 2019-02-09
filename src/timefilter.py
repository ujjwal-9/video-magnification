#!/usr/bin/env python3

import cv2

#temporal IIR filtering an image
def temporalIIRFilter(src, level):
    temp1 = (1.0 - cuttofFreqHigh) * lowpass1[level] + cuttofFreqHigh * src
    temp2 = (1.0 - cuttofFreqLow) * lowpass2[level] + cuttofFreqLow * src
    lowpass1[level] = temp1
    lowpass2[level] = temp2
    return lowpass1[level] - lowpass2[level]

#temporalIdalFilter - temporal IIR filtering an image pyramid of concatenated frames
def temporalIdealFilter(src):
	channels = [src[:,:,0], src[:,:,1], src[:,:,2]]
	for i in range(3):
		current = src[i]
		width = cv2.getOptimalDFTSize(current.shape[1])
		height = cv2.getOptimalDFTSize(current.shape[0])
		tempimg = cv2.copyMakeBorder(current,0,height-current.shape[0],0,width-current.shape[1],cv2.BORDER_CONSTANT, 0)
		tempimg = cv2.dft(tempimg, flags=cv2.DFT_ROWS | cv2.DFT_SCALE, tempimg.shape[0])
		filter_ = tempimg
		filter_ = createIdealBandpassFilter(filter_, cuttofFreqLow, cuttofFreqHigh, rate)
		tempimg = cv2.mulSpectrums(tempimg, filter_, cv2.DFT_ROWS)
		tempimg = cv2.idft(tempimg, flags=cv2.DFT_ROWS | cv2.DFT_SCALE, tempimg.shape[0])
		channels[i] = tempimg[:current.shape[0], :current.shape[1]]
	channels = cv2.normalize(channels, 0, 1, cv2.NORM_MINMAX)
	return channels

#create a 1D ideal band-pass filter
def createIdealBandpassFilter(filter_, fl, fh, rate):
	width = filter_.shape[1]
	height = filter_.shape[0]

	Fl = 2 * fl * width / rate;
    Fh = 2 * fh * width / rate;

    for i in range(height):
    	for i in range(width):
    		if j>=fl and j<=fh:
    			response = 1.0
    		else:
    			response = 0.0
    		filter_[i,j] = response
    return filter_