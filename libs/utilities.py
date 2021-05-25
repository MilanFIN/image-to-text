import cv2
import numpy as np
import string

def getAll():
	characters = string.ascii_letters + string.digits
	return list(characters)
def getLowerCase():
	characters = string.ascii_lowercase + string.digits
	return list(characters)


def numToAllChar(index):
	return getAll()[index] 

def numToLowerCase(index):
	return getLowerCase()[index] 


def getRoiAs32(letter):


	points = np.nonzero(letter)


	ys = points[0]
	xs = points[1]



	x2, x3, y2, y3 = min(xs), max(xs), min(ys), max(ys) 
	if (x2 == x3):
		x3 += 1

	letter = letter[y2:y3+1, x2:x3+1]

	#cv2.imshow('image',letter)
	#cv2.waitKey(1000)


	if (letter.size == 0):
		return letter
			
	letHeight = letter.shape[0]
	letWidth = letter.shape[1]
	
	if (letHeight > letWidth):
		scaleFactor = 32.0 /letHeight 
	else:
		scaleFactor = 32.0 /letWidth 
		

	goalWidth = int(scaleFactor*letWidth)
	goalHeight = int(scaleFactor*letHeight)
	if (goalWidth == 0): goalWidth = 1
	if (goalHeight == 0): goalHeight = 1
	if (goalWidth == 31): goalWidth = 32
	if (goalHeight == 31): goalHeight = 32

	letter = cv2.resize(letter, (goalWidth, goalHeight), interpolation = cv2.INTER_CUBIC)
	


	if (letHeight > letWidth):
		shortWidth = letter.shape[1]
		shortRemain = 32-shortWidth
		padLeft = int(shortRemain/2)
		padRight = padLeft
		if (padLeft + padRight + shortWidth < 32):#sum needs to be 32
			padRight += 1
		letter = cv2.copyMakeBorder(letter, 0, 0, padLeft, padRight, cv2.BORDER_CONSTANT, value=0)
	elif (letHeight < letWidth):
		shortHeight = letter.shape[0]
		shortRemain = 32-shortHeight
		padTop = int(shortRemain/2)
		padBot = padTop
		if (padTop + padBot + shortHeight < 32):#sum needs to be 32
			padBot += 1
		letter = cv2.copyMakeBorder(letter, padTop, padBot, 0,0, cv2.BORDER_CONSTANT, value=0)



	th, letter = cv2.threshold(letter, 127, 255, cv2.THRESH_BINARY)

	return letter