import torch.nn as nn
import torch
import torch.nn.functional as F
from data.constants import *
from tqdm import tqdm


class ResidualBlock(nn.Module):
	def __init__(self, inchannel, outchannel, stride=1):
		super(ResidualBlock, self).__init__()
		self.left = nn.Sequential(
			nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(outchannel, track_running_stats = True),
			nn.ReLU(),
			nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(outchannel, track_running_stats = True))
		self.shortcut = nn.Sequential()

		if stride != 1 or inchannel != outchannel:
			self.shortcut = nn.Sequential(
				nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias = False),
				nn.BatchNorm2d(outchannel, track_running_stats = True),
				nn.ReLU())

	def forward(self, x):
		out = self.left(x)
		out += self.shortcut(x)
		return out

# ResNet
class ResNet(nn.Module): 
	def __init__(self, ResidualBlock, num_classes=62):
		super(ResNet, self).__init__()
		self.inchannel = 64
		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64, track_running_stats=True),
			nn.ReLU())
		# ResidualBlock basic
		# res34 3 4 6 3
		self.layer1 = self.make_layer(ResidualBlock, 64, 3, 1)
		self.layer2 = self.make_layer(ResidualBlock, 128, 4, 2)
		self.layer3 = self.make_layer(ResidualBlock, 256, 6, 2)
		self.layer4 = self.make_layer(ResidualBlock, 512, 3, 2)
		self.drop = nn.Dropout(0.5)
		self.rfc = nn.Sequential(nn.Linear(512, MAX_CAPTCHA*ALL_CHAR_SET_LEN))
		
	def make_layer(self, block, channels, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)  # strides = [1,1], to determine the stride for each layer in one residual block
		layers = []
		for stride in strides:
			layers.append(block(self.inchannel, channels, stride))
			self.inchannel = channels
		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x) # 100, 64, 96, 96
		x = self.layer1(x) # 100, 128, 48, 48
		x = self.layer2(x) # 100, 256, 24, 24
		x = self.layer3(x) # 100, 512, 12, 12
		x = self.layer4(x) # 100, 512, 1, 1
		x = nn.AdaptiveAvgPool2d(1)(x) # 100, 512
		x = x.view(-1, 512)
		x = self.drop(x)
		x = self.rfc(x) # 100, 248
		return x

# CNN Model (2 conv layer)
class CaptchaCNN(nn.Module):
	def __init__(self):
		super(CaptchaCNN, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.Dropout(0.5),  # drop 50% of the neuron
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.Dropout(0.5),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.layer3 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.Dropout(0.5),
			nn.ReLU(),
			nn.MaxPool2d(2))
		self.fc = nn.Sequential(
			nn.Linear((IMAGE_WIDTH//8)*(IMAGE_HEIGHT//8)*64, 1024),
			nn.Dropout(0.5),
			nn.ReLU())
		self.rfc = nn.Sequential(
			nn.Linear(1024, MAX_CAPTCHA*ALL_CHAR_SET_LEN))

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		out = self.rfc(out)
		return out
