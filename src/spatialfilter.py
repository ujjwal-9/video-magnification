#!/usr/bin/env python3

def gaussian(src, level):
	G = src.copy()
	gpA = [G]
	for i in range(level):
	    G = cv2.pyrDown(G)
	    gpA.append(G)
	return gpA

def laplacian(gpA, level):
	lpA = [gpA[level]]
	for i in xrange(5,0,-1):
	    GE = cv2.pyrUp(gpA[i])
	    L = cv2.subtract(gpA[i-1],GE)
	    lpA.append(L)
	return lpA