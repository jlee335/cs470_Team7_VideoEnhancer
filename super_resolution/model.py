import os
import numpy as np
import time
from pathlib import Path
from PIL import Image
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import make_grid
from torchvision import transforms
from torch.autograd import Variable

import helper
# need to import crop function from helper.py


class Residual_Block(nn.Module):
    def __init__(self):
		super(Residual_Block, self).__init__()
		self.conv1 = nn.Conv2d(64,64,3,padding=1)
		self.conv2 = nn.Conv2d(64,64,3,padding=1)
		self.relu = nn.ReLU()

    def forward(self,x):
		rc = x
		x = self.conv1(x)
		x = self.relu(x)
		x = self.conv2(x)

		x = x + rc  # Think about it EDSR 논문 x0.1 
		return x

class ConvLeaky(nn.Module):
	# https://github.com/amanchadha/FRVSR-GAN/blob/master/FRVSRGAN_Models.py
    def __init__(self, in_dim, out_dim):
        super(ConvLeaky, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim,
                               kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        out = self.conv1(input)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        out = F.leaky_relu(out, 0.2)
        return out

class FNetBlock(nn.Module):
	# For Attempt1,2,3()
    def __init__(self, in_dim, out_dim, typ):
        super(FNetBlock, self).__init__()
        self.convleaky = ConvLeaky(in_dim, out_dim)
        if typ == "maxpool":
            self.final = lambda x: F.max_pool2d(x, kernel_size=2, ceil_mode=True)
        elif typ == "bilinear":
            self.final = lambda x: F.interpolate(x, scale_factor=2, mode="bilinear")
        else:
            raise Exception('Type does not match any of maxpool or bilinear')

    def forward(self, input):
        out = self.convleaky(input)
        out = self.final(out)
        return out

class FNet_Attemp1_2(nn.Module):
	# For Attemp2()
    # similar as FNet of FRVSR from https://arxiv.org/pdf/1801.04590.pdf
    # get input as concatted image of Xn-1, Xn, Xn+1 so in_dim is 9 as default
    def __init__(self, in_dim=9):
        super(FNet_Attemp1_2, self).__init__()
        self.encoder = nn.Sequential(
            FNetBlock(in_dim, 32, typ="maxpool",),
            FNetBlock(32, 64, typ="maxpool"),
            FNetBlock(64, 128, typ="maxpool")
        )
        self.decoder = nn.Sequential(
            FNetBlock(128, 256, typ="bilinear"),
            FNetBlock(256, 128, typ="bilinear"),
            FNetBlock(128, 64, typ="bilinear")
        )
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=2,
                               kernel_size=3, padding=1)

    def forward(self, input):
        #print(input.size())
        _, _ , tmp_x, tmp_y = input.size()

        output = self.encoder(input)
        output = self.decoder(output) 
        output = crop(output,0,0, tmp_x, tmp_y)
        output = self.conv1(output)
        output = F.leaky_relu(output, 0.2)
        output = self.conv2(output)
        output = torch.tanh(output)
        
        return output # return as 64 channel

class FNet_Attemp3(nn.Module):
	# For Attempt3()
    def __init__(self, in_dim=9):
        super(FNet_Attemp3, self).__init__()
        self.encoder = nn.Sequential(
            FNetBlock(in_dim, 32, typ="maxpool",),
            FNetBlock(32, 64, typ="maxpool"),
            FNetBlock(64, 128, typ="maxpool")
        )
        self.decoder = nn.Sequential(
            FNetBlock(128, 256, typ="bilinear"),
            FNetBlock(256, 128, typ="bilinear"),
            FNetBlock(128, 64, typ="bilinear")
        )
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=2,
                               kernel_size=3, padding=1)

    def forward(self, input):
        #print(input.size())
        _, _ , tmp_x, tmp_y = input.size()

        output = self.encoder(input)
        output = self.decoder(output) 
        output = crop(output,0,0, tmp_x, tmp_y)
        output = self.conv1(output)
        output = torch.tanh(output)
        
        return output # return as 64 channel

class Attempt1(nn.Module):
	#1st SRFlowNet
	# named as Final Project_v_0.2, not in final colab document
	# remain on Final Project_v_0.2
  	def __init__(self):
		super(Attempt1, self).__init__()
		self.conv_init = nn.Conv2d(6,64,3,padding = 1)
		self.relu = nn.ReLU()
		self.fnet = FNet_Attemp1_2()

		########################################

		self.res_thru = nn.Sequential(
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block()
		        )
		self.ct1 = nn.ConvTranspose2d(64,64,3,2)
		self.conv_img = nn.Conv2d(64,3,3,padding = 1)

 	def forward(self, input):
	    _, x, _ = input
	    f_x = torch.cat(input, dim=1)
	    curr = x
	    fs = self.fnet(f_x)
	    shape = x.shape
	    x_warp = optical_flow_warp(curr, fs) # x_warp: 3 X 3 X 200 X 200
	    x = torch.cat((x, x_warp), dim=1) # x_warp: 3 X 6 X 200 X 200
	    x = self.conv_init(x) 
	    x = self.relu(x)
	    residual = x
	    x = self.res_thru(x)
	    x += residual
	    x = self.ct1(x)
	    x = self.relu(x)


	    x = self.conv_img(x)

	    return crop(x,1,1,shape[2] * 2,shape[3] * 2)
    
class Attempt2(nn.Module):
	# WoongNet
	def __init__(self):
		super(Attempt2, self).__init__()
		self.conv_init = nn.Conv2d(6,64,3,padding = 1)
		self.relu = nn.ReLU()
		self.fnet = FNet_Attemp2()

		########################################

		self.res_thru = nn.Sequential(
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block()
		        )
		self.ct1 = nn.ConvTranspose2d(64,64,3,2)
		self.conv_skip = nn.Conv2d(3,64,3,padding = 1)
		self.conv_img = nn.Conv2d(64,3,3,padding = 1)

	def forward(self, input):
		_, x, _ = input
		f_x = torch.cat(input, dim=1)
		curr = x
		fs = self.fnet(f_x)
		shape = x.shape
		x_warp = optical_flow_warp(curr, fs) # x_warp: 3 X 3 X 200 X 200
		x = torch.cat((x, x_warp), dim=1) # x_warp: 3 X 6 X 200 X 200
		x = self.conv_init(x) 
		x = self.relu(x)
		residual = x
		x = self.res_thru(x)
		x += residual
		x = self.ct1(x)
		x= crop(x,1,1,shape[2] * 2,shape[3] * 2)
		x += self.conv_skip(F.interpolate(curr, scale_factor=2, mode="bilinear"))
		x = self.relu(x)
		x = self.conv_img(x)

		return x
	
class Attempt3(nn.Module):
	# SRFlowNet
	def __init__(self):
		super(Attempt3, self).__init__()
		self.conv_init = nn.Conv2d(3,64,3,padding = 1)
		self.relu = nn.ReLU()
		self.fnet = FNet_Attemp3()

		########################################

		self.res_thru = nn.Sequential(
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block(),
		          Residual_Block()
		        )
		self.ct1 = nn.ConvTranspose2d(64,64,3,2)
		self.conv_img = nn.Conv2d(64,3,3,padding = 1)

	def forward(self, input):
		_, x, _ = input
		f_x = torch.cat(input, dim=1)
		curr = x
		fs = self.fnet(f_x)
		shape = x.shape
		x = self.conv_init(x) 
		x = self.relu(x)
		x = x + fs
		residual = x
		x = self.res_thru(x)
		x += residual
		x = self.ct1(x)
		x = self.relu(x)
		x = self.conv_img(x)

		return crop(x,1,1,shape[2] * 2,shape[3] * 2)

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator,self).__init__()
		self.l_relu = nn.LeakyReLU()
		self.outer_conv = nn.Conv2d(3,64,3)

		self.conv_64c = nn.Conv2d(64,64,3,2)
		self.bn_64c = nn.BatchNorm2d(64)

		self.conv_128e = nn.Conv2d(64,128,3,1)
		self.bn_128e = nn.BatchNorm2d(128)

		self.conv_128c = nn.Conv2d(128,128,3,2)
		self.bn_128c = nn.BatchNorm2d(128)

		self.conv_256e = nn.Conv2d(128,256,3,1)
		self.bn_256e = nn.BatchNorm2d(256)

		self.conv_256c = nn.Conv2d(256,256,3,2)
		self.bn_256c = nn.BatchNorm2d(256)

		self.conv_512e = nn.Conv2d(256,512,3,1)
		self.bn_512e = nn.BatchNorm2d(512)

		self.conv_512c = nn.Conv2d(512,512,3,2)
		self.bn_512c = nn.BatchNorm2d(512)

		# B x 512 x H x W

		self.flatten = nn.AdaptiveAvgPool2d(1)
		self.lin1 = nn.Conv2d(512, 1024, kernel_size=1)
		self.lin2 = nn.Conv2d(1024, 1, kernel_size=1)
		self.sig = nn.Sigmoid()

	def forward(self,x):
		x = self.l_relu(self.outer_conv(x))
		x = self.l_relu(self.bn_64c(self.conv_64c(x)))
		x = self.l_relu(self.bn_128e(self.conv_128e(x)))
		x = self.l_relu(self.bn_128c(self.conv_128c(x)))
		x = self.l_relu(self.bn_256e(self.conv_256e(x)))
		x = self.l_relu(self.bn_256c(self.conv_256c(x)))
		x = self.l_relu(self.bn_512e(self.conv_512e(x)))
		x = self.l_relu(self.bn_512c(self.conv_512c(x)))

		x = self.flatten(x)
		x = self.l_relu(self.lin1(x))
		x = self.sig(self.lin2(x))
		return x


