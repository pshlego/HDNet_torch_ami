from pickle import TRUE
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
NUM_CH = [64,128,256,512,1024]
KER_SZ = 3 
def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes,kernel_size=KER_SZ, stride=stride,padding=padding, bias=False)
class conv_layer(nn.Module):
    def __init__(self, input_size, planes, stride=1, padding=1, bn=False):
        super(conv_layer, self).__init__()
        self.conv = conv3x3(input_size, planes, stride, padding=padding)
        self.bn=bn
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        if self.bn:
            out=self.bn1(out)
        out = self.relu(out)
        return out
class hourglass_refinement_1(nn.Module):
    def __init__(self,input_size,stride=1,bn=True):
        super(hourglass_refinement_1, self).__init__()
        self.c0=conv_layer(input_size,NUM_CH[0],stride,bn=bn)
        self.c1=conv_layer(NUM_CH[0], NUM_CH[0],stride,bn=bn)
        self.c2=conv_layer(NUM_CH[0], NUM_CH[0],stride,bn=bn)
        self.p0=nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3=conv_layer(NUM_CH[0],NUM_CH[1],stride,bn=bn)
        self.p1=nn.MaxPool2d(kernel_size=2, stride=2)
        self.c4=conv_layer(NUM_CH[1],NUM_CH[2],stride,bn=bn)
        self.p2=nn.MaxPool2d(kernel_size=2, stride=2)
        self.c5=conv_layer(NUM_CH[2],NUM_CH[3],stride,bn=bn)
        self.p3=nn.MaxPool2d(kernel_size=2, stride=2)
        self.c6=conv_layer(NUM_CH[3],NUM_CH[4],stride,bn=bn)
        self.c7=conv_layer(NUM_CH[4],NUM_CH[4],stride,bn=bn)
        self.c8=conv_layer(NUM_CH[4],NUM_CH[3],stride,bn=bn)
        self.up1=nn.Upsample(size=32, mode='bilinear')
        self.c9=conv_layer(NUM_CH[3]*2,NUM_CH[3],stride,bn=bn)
        self.c10=conv_layer(NUM_CH[3],NUM_CH[2],stride,bn=bn)
        self.up2=nn.Upsample(size=64, mode='bilinear')
        self.c11=conv_layer(NUM_CH[2]*2,NUM_CH[2],stride,bn=bn)
        self.c12=conv_layer(NUM_CH[2],NUM_CH[1],stride,bn=bn)
        self.up3=nn.Upsample(size=128, mode='bilinear')
        self.c13=conv_layer(NUM_CH[1]*2,NUM_CH[1],stride,bn=bn)
        self.c14=conv_layer(NUM_CH[1],NUM_CH[0],stride,bn=bn)
        self.up4=nn.Upsample(size=256, mode='bilinear')
        self.c15=conv_layer(NUM_CH[0]*2,NUM_CH[0],stride,bn=bn)
        self.c16=conv_layer(NUM_CH[0],NUM_CH[0],stride,bn=bn)
        self.c17=conv_layer(NUM_CH[0],1,stride,bn=bn)
        self.padding_size = (64,64,64,64)
    def forward(self, x):
        x = x.permute(0,3,1,2)
        out_0=self.c0(x) 
        out_1=self.c1(out_0)
        out_2=self.c2(out_1)
        out_3=self.p0(out_2)
        out_4=self.c3(out_3)
        out_5=self.p1(out_4)
        out_6=self.c4(out_5)
        out_7=self.p2(out_6)
        out_8=self.c5(out_7)
        out_9=self.p3(out_8)
        out_10=self.c6(out_9)
        out_11=self.c7(out_10)
        out_12=self.c8(out_11)
        out_13=self.up1(out_12)
        out_14=torch.cat([out_13,out_8],dim=1)
        out_15=self.c9(out_14)
        out_16=self.c10(out_15)
        out_17=self.up2(out_16)
        out_18=torch.cat([out_17,out_6],dim=1)
        out_19=self.c11(out_18)
        out_20=self.c12(out_19)
        out_21=self.up3(out_20)
        out_22=torch.cat([out_21,out_4],dim=1)
        out_23=self.c13(out_22)
        out_24=self.c14(out_23)
        out_25=self.up4(out_24)
        out_26=torch.cat([out_25,out_1],dim=1)
        out_27=self.c15(out_26)
        out_28=self.c16(out_27)
        stack_out_d=self.c17(out_28)
        stack_out_d = stack_out_d.permute(0,2,3,1)
        return stack_out_d
        
        
        
        