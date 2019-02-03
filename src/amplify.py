#!/usr/bin/env python3

delta = lambda_c/ 8.0 /( 1.0 +alpha)
# the factor to boost alpha above the bound
exaggeration_factor = 2.0
# compute the representative wavelength lambda
# for the lowest spatial frequency band of Laplacian pyramid
Lambda = sqrt (w*w + h*h)/ 3

#amplify the motion
def amplify(src, spatialType):
	if spatialType is 0:					# Motion Magnification
		curAlpha = Lambda/delta/8-1
		curAlpha *= exaggeration_factor
		if curLevel == levels or curLevel == 0:
			return src * 0
		else:
			return src*cv2.min(alpha, curAlpha)
	else:									# Color Magnification
		return src * alpha