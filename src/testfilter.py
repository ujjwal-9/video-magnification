#!/usr/bin/env python3

import numpy as np
import cv2
import matplotlib.pyplot as plt

def buildLaplacianPyramid(img, levels, pyramid):
    if levels < 1:
        print("Levels should be larger than 1")
        return False
    pyramid = []
    currImg = img
    for i in range(levels):
        down = cv2.pyrDown(currImg)
        up = cv2.pyrUp(down, currImg)
