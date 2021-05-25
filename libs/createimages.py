

import cv2
import numpy as np
import time
import string
import random
from libs.utilities import *



FONTS = [cv2.FONT_HERSHEY_PLAIN,
	cv2.FONT_HERSHEY_COMPLEX,
	cv2.FONT_HERSHEY_TRIPLEX,
	cv2.FONT_HERSHEY_DUPLEX,
]


def getDataset(characters, size):

	outputs = [] #indexes in getOutputArray()
	inputs = [] #1024 elements per image

	possibleOutputs = characters

	for i in range(0, size):
		for j in range(len(possibleOutputs)):

			font = random.choice(FONTS)

			im = np.zeros((64,64,1))
			thickness = random.randint(2, 5)
			cv2.putText(im, possibleOutputs[j], (10,50), font, 2, (255), thickness, cv2.LINE_AA)

			#find the character area
			pts = cv2.findNonZero(im)
			ret = cv2.minAreaRect(pts)
			(cx,cy), (w,h), ang = ret

			#change letter angle randomly
			ang = 0
			ang += random.randint(-5, 5)
			M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
			im = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]))
			
			#get rid of rotation artifacts
			th, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY )
			
			im = getRoiAs32(im)

			outputs.append(j)
			inputs.append(im) #.flatten()

	#not needed, following line does the same better
	#####inputs = np.array(inputs, dtype=bool)
	inputs = np.divide(inputs, 255)
	outputs = np.eye(len(possibleOutputs))[outputs]

	in2 = []
	for i in inputs:
		in2.append([i])
	inputs = np.array(in2)
	return inputs, outputs

def smallDataset(size):
	characters = getLowerCase()
	return getDataset(characters, size)

def bigDataset(size):
	characters = getAll()
	return getDataset(characters, size)





