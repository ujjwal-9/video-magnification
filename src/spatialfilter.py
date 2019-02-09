#!/usr/bin/env python3
import cv2
import numpy as np

def buildGaussianPyramid(src, levels):
	G = src.copy()
	gpA = [G]
	for i in range(levels):
	    G = cv2.pyrDown(G)
	    gpA.append(G)
	gpA = np.array(gpA)
	return gpA

def buildLaplacianPyramid(src, levels):
	lpA = [src[levels]]
	for i in range(5,0,-1):
	    GE = cv2.pyrUp(src[i])
	    L = cv2.subtract(src[i-1],GE)
	    lpA.append(L)
	lpA = np.array(lpA)
	return lpA

def reconImgFromLaplacianPyramid(pyramid, levels):
	currentImg = pyramid[levels]
	for l in reversed(range(levels)):
		up = cv2.pyrUp(currentImg, pyramid[l].shape)
		currentImg = up + pyramid[l]
	return currentImg

def upsamplingFromGaussianPyramid(src, levels):
	currentLevel = src
	for i in range(levels):
		up = cv2.pyrUp(currentLevel)
		currentLevel = up
	return currentLevel
