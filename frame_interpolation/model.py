import os
import numpy as np
import time
from pathlib import Path
from PIL import Image
import cv2 as cv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import make_grid
from torchvision import transforms
from torch.autograd import Variable

from torchvision.models.vgg import VGG, vgg16, make_layers

import helper
# need to import crop function from helper.py


class ConvLeaky(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvLeaky, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim,
                               kernel_size=3, stride=1, padding=1)
        
        self.conv3 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim,
                               kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        out = self.conv1(input)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        out = F.leaky_relu(out, 0.2)
        out = self.conv3(out)
        out = F.leaky_relu(out, 0.2)
        return out

class FNetBlock(nn.Module):
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

class IFNet(nn.Module):
    def __init__(self, in_dim=6):
        super(IFNet, self).__init__()
        self.e64 =  FNetBlock(in_dim, 64, typ="maxpool")
        self.e128 =  FNetBlock(64, 128, typ="maxpool")
        self.e256 = FNetBlock(128, 256, typ="maxpool")
        self.e512 = FNetBlock(256, 512, typ="maxpool")
  
        self.d256 = FNetBlock(512, 256, typ="bilinear")
        self.d128  = FNetBlock(256, 128, typ="bilinear")
        self.d64  = FNetBlock(128, 64, typ="bilinear")
        self.d32  = FNetBlock(64, 32, typ="bilinear")

        self.conv1 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1)
      
        self.three = nn.Sequential(
          nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
          nn.ReLU()
        )
        self.tan = nn.Tanh()
    def forward(self, x):
        x = torch.cat(x,dim = 1)
        x = self.e64(x)
        r1 = x
        x = self.e128(x)
        r2 = x
        x = self.e256(x)
        r3 = x
        x = self.e512(x)
        x = self.d256(x)
        x += r3
        x = self.d128(x)
        x += r2
        x = self.d64(x)
        x += r1
        x = self.d32(x)
        x = self.three(x)
        x = self.conv1(x)
        x = self.tan(x)
        
        return x

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

class VGG_cutout(nn.Module):
    def __init__(self, original_model):
        super(VGG_cutout, self).__init__()
        self.features = nn.Sequential(nn.Sequential(*list(original_model.children())[0])[:27])
        
    def forward(self, x):
        x = self.features(x)
        return x

