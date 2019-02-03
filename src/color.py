#!/usr/bin/env python3
import numpy as np
import cv2
from numpy.linalg import inv
import matplotlib.pyplot as plt

def rgb2ntsc(src):
    ret = src
    T = np.array([[1 , 1 , 1] , [0.956 , - 0.272 , - 1.106] , [0.621 , - 0.647 , 1.703]])
    T = inv(T)

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            ret[i,j,0] = src[i,j,0] * T[0,0] + src[i,j,1] * T[0,1] + src[i,j,2] * T[0,2]
            ret[i,j,1] = src[i,j,0] * T[1,0] + src[i,j,1] * T[1,1] + src[i,j,2] * T[1,2]
            ret[i,j,2] = src[i,j,0] * T[2,0] + src[i,j,1] * T[2,1] + src[i,j,2] * T[2,2]

    return ret

def ntsc2rgb(src):
    ret = src
    T = np.array([[1.0 , 0.956 , 0.621] , [1.0 , - 0.272 , - 0.647] , [1.0 , - 1.106 , 1.703]])
    T = T.T
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            ret[i,j,0] = src[i,j,0] * T[0,0] + src[i,j,1] * T[0,1] + src[i,j,2] * T[0,2]
            ret[i,j,1] = src[i,j,0] * T[1,0] + src[i,j,1] * T[1,1] + src[i,j,2] * T[1,2]
            ret[i,j,2] = src[i,j,0] * T[2,0] + src[i,j,1] * T[2,1] + src[i,j,2] * T[2,2]

    return ret

if __name__ == '__main__':
    img = cv2.imread('../images/test.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)
    new = rgb2ntsc(img)
    prev = ntsc2rgb(new)
    plt.imshow(new)
    plt.show()
    plt.imshow(prev)
    plt.show()



