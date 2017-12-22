import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

def conv_size(args):
	try:
		reduce
	except NameError:
		from functools import reduce
	
	out_size, num_filter = args
	_size = [out_size] + [2] * num_filter
	return reduce(lambda x, y: x // y, _size)

def gen_z(batch_size, z_dim, is_cuda, eval=False):
	z = torch.from_numpy(
		np.random.normal(size=(batch_size, z_dim)))

	z = Variable(z.float(), volatile=eval)
	if is_cuda:
		z = z.cuda()

	return z

class Generator(nn.Module):
	def __init__(self, out_h, out_w, channel_dims, z_dim=100):
		super().__init__()

		assert len(channel_dims) == 4, "length of channel dims should be 4"

		conv1_dim, conv2_dim, conv3_dim, conv4_dim = channel_dims
		conv1_h, conv2_h, conv3_h, conv4_h = map(conv_size, [(out_h, step) for step in [4 ,3 ,2 ,1]])
		conv1_w, conv2_w, conv3_w, conv4_w = map(conv_size, [(out_w, step) for step in [4 ,3 ,2 ,1]])

		self.fc = nn.Linear(z_dim, conv1_dim*conv1_h*conv1_w)
		self.deconvs = nn.Sequential(
				nn.BatchNorm2d(conv1_dim),
				nn.ReLU(),

				nn.ConvTranspose2d(conv1_dim, conv2_dim, kernel_size=4, stride=2, padding=1, bias=False),
				nn.BatchNorm2d(conv2_dim),
				nn.ReLU(),

				nn.ConvTranspose2d(conv2_dim, conv3_dim, kernel_size=4, stride=2, padding=1, bias=False),
				nn.BatchNorm2d(conv3_dim),
				nn.ReLU(),	

				nn.ConvTranspose2d(conv3_dim, conv4_dim, kernel_size=4, stride=2, padding=1, bias=False),
				nn.BatchNorm2d(conv4_dim),
				nn.ReLU(),			

				nn.ConvTranspose2d(conv4_dim, 3, kernel_size=4, stride=2, padding=1, bias=False),	
				nn.Tanh(),							
			)
		self.conv1_size = (conv1_dim, conv1_h, conv1_w)

		self._init_weight()

	def _init_weight(self):
		self.fc.weight.data.normal_(.0, 0.02)
		for layer in self.deconvs:
			if isinstance(layer, nn.ConvTranspose2d):
				layer.weight.data.normal_(.0, 0.02)
			if isinstance(layer, nn.BatchNorm2d):
				layer.weight.data.normal_(1., 0.02)
				layer.bias.data.fill_(0)

	def forward(self, z):
		out = self.fc(z)
		out = out.view(-1, *self.conv1_size)
		return self.deconvs(out)

class Discriminator(nn.Module):
	def __init__(self, out_h, out_w, channel_dims, relu_leak):
		super().__init__()

		assert len(channel_dims) == 4, "length of channel dims should be 4"

		conv4_dim, conv3_dim, conv2_dim, conv1_dim = channel_dims
		conv4_h, conv3_h, conv2_h, conv1_h = map(conv_size, [(out_h, step) for step in [4 ,3 ,2 ,1]])
		conv4_w, conv3_w, conv2_w, conv1_w = map(conv_size, [(out_w, step) for step in [4 ,3 ,2 ,1]])

		self.convs = nn.Sequential(
				nn.Conv2d(3, conv1_dim, kernel_size=4, stride=2, padding=1, bias=False),
				nn.LeakyReLU(relu_leak),

				nn.Conv2d(conv1_dim, conv2_dim, kernel_size=4, stride=2, padding=1, bias=False),
				nn.BatchNorm2d(conv2_dim),
				nn.LeakyReLU(relu_leak),	

				nn.Conv2d(conv2_dim, conv3_dim, kernel_size=4, stride=2, padding=1, bias=False),
				nn.BatchNorm2d(conv3_dim),
				nn.LeakyReLU(relu_leak),	

				nn.Conv2d(conv3_dim, conv4_dim, kernel_size=4, stride=2, padding=1, bias=False),	
				nn.BatchNorm2d(conv4_dim),
				nn.LeakyReLU(relu_leak),						
			)

		self.fc = nn.Linear(conv4_dim*conv4_h*conv4_w, 1)
		self.fc_dim = conv4_dim*conv4_h*conv4_w

		self._init_weight()

	def _init_weight(self):
		self.fc.weight.data.normal_(.0, 0.02)
		for layer in self.convs:
			if isinstance(layer, nn.Conv2d):
				layer.weight.data.normal_(.0, 0.02)
			if isinstance(layer, nn.BatchNorm2d):
				layer.weight.data.normal_(1., 0.02)
				layer.bias.data.fill_(0)

	def forward(self, input):
		out = self.convs(input)
		linear = self.fc(out.view(-1, self.fc_dim))
		return F.sigmoid(linear)


if __name__ == '__main__':
	from torch.autograd import Variable

	input = Variable(torch.randn((2, 100)))
	g = Generator(64, 64, [512,256,128,64])

	out = g(input)
	print(out)

	d = Discriminator(64, 64, [512,256,128,64], 0.2)
	out = d(out)
	print(out)

	z = gen_z(4, 100, True)
	print(z)
