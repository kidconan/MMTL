import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F

class CRNN(nn.Module):

	def __init__(self, out_dim, cnn_setting, rnn_setting, pooling_setting, dnn_setting, dropout_rate, cnn2rnn, rnn2dnn):
		super(CRNN, self).__init__()
		self.out_dim = out_dim

		self.cnn_amount = len(cnn_setting)
		self.cnn_blocks = nn.ModuleList()
		self.pooling_blocks = nn.ModuleList()
		for i in range(self.cnn_amount):
			in_chan, out_chan, kernel_size, stride, padding = cnn_setting[i]
			self.cnn_blocks.append(
				nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernel_size, stride=stride,
						  padding=padding)
			)
			kernel_size, stride = pooling_setting[i]
			self.pooling_blocks.append(
				nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
			)

		self.cnn2rnn = cnn2rnn

		self.rnn_amount = len(rnn_setting)
		self.rnn_blocks = nn.ModuleList()
		for i in range(self.rnn_amount):
			in_dim, hid_dim, layers, bool_bi = rnn_setting[i]
			self.rnn_blocks.append(
				nn.LSTM(input_size=in_dim, num_layers=layers, hidden_size=hid_dim, bidirectional=bool_bi,
						batch_first=True)
			)

		self.rnn2dnn = rnn2dnn

		self.dnn_dim = dnn_setting
		base_dim = rnn_setting[-1][1] * (1 + rnn_setting[-1][-1])
		self.dnn_layer1 = nn.Linear(base_dim, self.dnn_dim)
		self.drop_out = nn.Dropout(p=dropout_rate)
		self.dnn_layer2 = nn.Linear(self.dnn_dim, out_dim)
		self.transfer = nn.Linear(out_dim, out_dim)
		self.out_softmax = nn.Softmax(dim=1)

	def forward(self, feature, mode):

		if len(feature.size()) == 2:
			feature = (feature.unsqueeze(0)).unsqueeze(0)
		elif len(feature.size()) == 3:
			feature = feature.unsqueeze(1)
		else:
			print("Input Error: the feature should be H*W or C*H*W")
			raise ValueError

		out = feature.contiguous()
		for i in range(self.cnn_amount):
			out = self.cnn_blocks[i](out)
			out = F.relu(out)
			out = self.pooling_blocks[i](out)

		[B, C, H, W] = out.size()
		if self.cnn2rnn == 'concat':
			out = out.contiguous().view(B, -1, W)
		elif self.cnn2rnn == 'sum':
			out = out.sum(dim=1)
		elif self.cnn2rnn == 'avg':
			out = out.mean(dim=1)
		elif self.cnn2rnn == 'max':
			out = (out.max(dim=1))[0]
		else:
			raise ValueError

		out = out.contiguous().permute(0, 2, 1)

		for i in range(self.rnn_amount):
			out, (_, _) = self.rnn_blocks[i](out)

		if self.rnn2dnn == 'Avg':
			out = out.mean(dim=1).contiguous()
		elif self.rnn2dnn == 'Sum':
			out = out.sum(dim=1).contiguous()
		elif self.rnn2dnn == 'Max':
			out = (out.max(dim=1)[0]).contiguous()
		elif self.rnn2dnn == 'L-concat':
			out = (out[:, -1, :]).contiguous()
		elif self.rnn2dnn == 'FB-concat':
			B, T, D = out.size()
			device = out.get_device()
			tmp = torch.zeros(B, D).cuda(device=device)
			out = out.contiguous().view(B, T, 2, -1)
			for i in range(B):
				tmp[i, :] = torch.cat((out[i, 0, 1, :], out[i, -1, 0, :]), dim=0)
			out = tmp
		else:
			raise TypeError

		out = F.relu(self.dnn_layer1(out))
		out = self.drop_out(out)
		out = self.dnn_layer2(out)
		if mode == 'query':
			out = self.transfer(out)
		out = self.out_softmax(out)

		return out