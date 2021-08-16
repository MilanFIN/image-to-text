import cv2
import numpy as np
import time
from libs.utilities import *
import torch
import sys


from libs.model import ImageNet


smallSize = len(getLowerCase())
bigSize = len(getAll())

#initialize ImageNet
smallModel = ImageNet(smallSize)
bigModel = ImageNet(bigSize)
#load existing weights
bigModel.load_state_dict(torch.load("./models/big.pt"))
bigModel.eval()

smallModel.load_state_dict(torch.load("./models/small.pt"))
smallModel.eval()


if (len(sys.argv) <= 1):
	print("please provide an imagename that is in images/ folder")
	sys.exit(0)
imageName = sys.argv[1]



## read image
img = cv2.imread("./images/"+imageName) #big.jpg



#convert to greyscale and threshold
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)





# find bounding rect around "text"
pts = cv2.findNonZero(threshed)
ret = cv2.minAreaRect(pts)

(cx,cy), (w,h), ang = ret


#if (img.shape[0] < img.shape[1]):
#	ang -= 90

if (h > w):
	ang += 90

if (abs(ang) > 90):
	ang = ang % 90


#rotate to be horizontal
M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
rotated = cv2.warpAffine(threshed, M, (img.shape[1], img.shape[0]))


#thresh to get rid of artifacts
th, rotated = cv2.threshold(rotated, 127, 255, cv2.THRESH_BINARY )



#cv2.imshow('image',rotated)
#cv2.waitKey(3000)


## find boundaries between lines of text
hist = cv2.reduce(rotated,1, cv2.REDUCE_AVG).reshape(-1)
th = 2
H,W = img.shape[:2]
uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]


lines = [] #lines with others than first letters of words
firstLines = [] #lines with first letters of words
spaces = {} #x,y:1 for each space


for i in range(0, len(uppers)):
	line = []
	firstLine = []

	y0 = uppers[i]
	y1 = lowers[i]
	y1+=5
	y0-=1

	#getting line
	roi = rotated[y0:y1, 0:W]

	points = np.nonzero(roi)
	ys = points[0]
	xs = points[1]
	x2, x3, y2, y3 = min(xs), max(xs), min(ys), max(ys) 
	roi = roi[y2:y3+1, x2:x3+1]


	if (roi.size == 0):
		continue



	#separating individual letters as they don't touch each other
	contours, hierarchy = cv2.findContours(roi, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
	contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])


	#calculate average distance between characters on the line, used for separating words
	previousEnd = 0
	cumulativeDistance = 0
	for contour in contours:
		[x,y,w,h] = cv2.boundingRect(contour)
		cumulativeDistance += x - previousEnd
		previousEnd = x+w
	averageDistance = cumulativeDistance / float(len(contours))


	#calculate average size of contour to be able to ignore false detections later
	#  (they are much smaller)
	cumulativeSize = 0
	for contour in contours:
		[x,y,w,h] = cv2.boundingRect(contour)
		cumulativeSize += w*h
	averageSize = cumulativeSize / float(len(contours))





	previousEnd = 0
	j = 0
	for contour in contours:

		[x,y,w,h] = cv2.boundingRect(contour)

		#ignoring contours which are less than x % of average on the line.
		# this will get rid of duplicate detections for letters such as i & j
		size = w*h
		if (size < 0.2 * averageSize):
			continue

		#figuring out the distance to the previous letter
		space = x - previousEnd
		previousEnd = x+w


		#take into account the entire height of the line
		height = roi.shape[0]
		letter = roi[0:height-1, x:x+w]
		points = np.nonzero(letter)

		ys = points[0]
		xs = points[1]


		if (len(ys) < 4): #less than 5 white pixels high, should probably ignore
			continue
		

		#crop the actual letter and scale to 32x32
		letter = getRoiAs32(letter)
		if (letter.size == 0):
			continue

		if (space > 2*averageDistance):
			#a new word
			spaces[(j, i)] = 1 #
			firstLine.append(letter)
		elif (j == 0):
			#first letter of line
			firstLine.append(letter)
		else:
			#all other letters
			line.append(letter)
		j += 1

	if (len(line) != 0 or len(firstLine) != 0):
		lines.append(line)
		firstLines.append(firstLine)



# going through the lines of text, getting corresponding
# predictions from the neural net
letters = lines
outputs = []
firstOutputs = []

for inputs in lines:
	if (len(inputs) == 0):
		outputs.append([])
		continue
	inputs = np.divide(inputs, 255)
	in2 = []
	for i in inputs:
		in2.append([i])
	inputs = np.array(in2)
	
	X = torch.from_numpy(inputs).type(torch.FloatTensor)

	out = smallModel(X)
	outputs.append(out.cpu().detach().numpy())


for inputs in firstLines:
	if (len(inputs) == 0):
		firstOutputs.append([])
		continue
	inputs = np.divide(inputs, 255)
	in2 = []
	for i in inputs:
		in2.append([i])
	inputs = np.array(in2)
	
	X = torch.from_numpy(inputs).type(torch.FloatTensor)

	out = bigModel(X)
	firstOutputs.append(out.cpu().detach().numpy())



#go through both first letters of words, spaces & other letters
#and append them to the screen in the correct order
for j in range(len(outputs)):
	out = outputs[j]
	outf = firstOutputs[j]
	line = ""
	big = 0
	small = 0
	i = 0
	while i < len(out) + len(outf):
		character = ""
		if ((i, j) in spaces):
			line += " "
			index = outf[big]
			charNumber = np.argmax(index)
			character = numToAllChar(charNumber) #numToAllChar
			big += 1
		elif (big == 0):
			index = outf[big]
			charNumber = np.argmax(index)
			character = numToAllChar(charNumber) #numToAllChar
			big += 1
		else:
			index = out[small]
			charNumber = np.argmax(index)
			character = numToLowerCase(charNumber) #numToAllChar
			small += 1
		i += 1
		line += character


	print(line)
