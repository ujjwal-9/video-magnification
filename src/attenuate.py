#!/usr/bin/env python3

# attenuate I, Q channels
def attenuate(src, chromeAttenuation):
	planes = [src[:,:,0], src[:,:,1], src[:,:,2]]
	planes[1] = planes[1] * chromeAttenuation
	planes[2] = planes[2] * chromeAttenuation
	return planes