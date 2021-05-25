import torch
import torch.nn as nn
import torch.nn.functional as F


#our class must extend nn.Module
class ImageNet(nn.Module):
	def __init__(self, labels):
		super(ImageNet,self).__init__()
		self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
		self.conv2 = nn.Conv2d(20, 40, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(40*5*5, 512)
		self.fc2 = nn.Linear(512, labels)
	def forward(self,x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 40*5*5) 
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		x = F.log_softmax(x, dim=1)
		return x
