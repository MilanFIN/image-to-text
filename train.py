import torch
import torch.nn as nn
import torch.nn.functional as F


from libs.createimages import *

import sklearn.datasets
from sklearn.metrics import accuracy_score


import sys

from libs.model import ImageNet


def train(dataset, epochs, saveName):
	"""
	dataset = X:y pairs for image & one hot coding for label
	epochs = num of training epochs
	saveName, name of the model (without path or .pt ending)
	"""
	X, y = dataset
	#from_numpy takes a numpy element and returns torch.tensor
	X = torch.from_numpy(X).type(torch.FloatTensor)
	y = torch.from_numpy(y).type(torch.FloatTensor)
	#Initialize the model       
	model = ImageNet(len(y[0]))

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9) #lr=0.01


	for epoch in range(epochs):
		running_loss = 0.0
		# get the inputs; data is a list of [inputs, labels]

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = model(X)
		#https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216/2
		loss = criterion(outputs, torch.max(y, 1)[1])
		loss.backward()
		optimizer.step()

		running_loss = loss.item()
		print("epoch: ", epoch, "loss: ", running_loss)

	torch.save(model.state_dict(), "./models/"+saveName+".pt")

	#an example testing the accuracy for the first 62 characters in the dataset
	"""
	#
	testout = model(X[0:62])
	testlabels = y[0:62]

	for i in range(len(testout)):
		index = testout[i].cpu().detach().numpy()
		charNumber = np.argmax(index)
		actualNumber = np.argmax(testlabels[i])
		print(numToChar(actualNumber), numToChar(charNumber))
	"""





print("generating dataset for upper&lowercase characters")
dataset = bigDataset(650)
print("done")
print("training a model on upper&lowercase characters")
train(dataset, 100, "big")
print("finished training on upper&lowercase characters")

print("generating dataset for lowercase only")
dataset = smallDataset(650)
print("done")
print("training a model on lowercase characters")
train(dataset, 100, "small")
print("finished training on lowercase characters")






