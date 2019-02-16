#!/usr/bin/env python3
import cv2
import numpy as np

def buildGaussianPyramid(img, levels):
	if levels < 1:
		print("Level should be greater than 1")
		return False
	currImg = img
	pyramid = []
	for l in range(levels):
		down = cv2.pyrDown(currImg)
		pyramid.append(down)
		currImg = down
	return np.array(pyramid)

def buildLaplacianPyramid(img, levels):
	print(img.shape)
	if levels < 1:
		print("Levels should be greater than 1")
		return False
	currImg = img
	pyramid = []
	for l in range(levels):
		down = cv2.pyrDown(currImg)
		h,w,c = currImg.shape
		# print("currImg:", currImg.shape)
		up = cv2.pyrUp(down, dstsize=(w,h))
		# print("up:", up.shape)
		lap = currImg - up
		pyramid.append(lap)
		currImg = down
	pyramid.append(currImg)
	return np.array(pyramid)

def reconImgFromLaplacianPyramid(pyramid, levels):
	currentImg = pyramid[levels]
	for l in reversed(range(levels)):
		h,w,c = pyramid[l].shape
		up = cv2.pyrUp(currentImg, dstsize=(w,h))
		currentImg = up + pyramid[l]
	return currentImg

def upsamplingFromGaussianPyramid(src, levels):
	currentLevel = src
	for i in range(levels):
		up = cv2.pyrUp(currentLevel)
		currentLevel = up
	return currentLevel

if __name__ == '__main__':
	img = cv2.imread('../Videos/test.jpg')
	print("image: ", img.shape)
	gpy = buildGaussianPyramid(img, 4)
	print("Gaussian: ", gpy.shape)
	lpy = buildLaplacianPyramid(img, 4)
	print("Laplacian: ", lpy.shape)