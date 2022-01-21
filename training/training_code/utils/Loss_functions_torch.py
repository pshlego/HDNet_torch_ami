import tensorflow as tf
# import tensorflow_graphics as tfg
import numpy as np
import skimage.data
from PIL import Image, ImageDraw, ImageFont
import math
from tensorflow.python.platform import gfile
import scipy.misc
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
# *****************************************************************************************************
def calc_loss_normal2_1(output, y_normal,z_refined):
    
    # gives mean angle error for given output tensor and its ref y
    output_mask = torch.abs(output) < 1e-5 #output의 절댓값이 10의 -5승보다 작으면 mask에 bool로 저장.
    output_no0 = torch.where(output_mask, 1e-5*torch.ones_like(output), output)   #위의 마스크가 true인 경우에는 10 -5승을 입력으로 넣어줌. false면 output의 값을 넣어줌.
    output_mag = torch.unsqueeze(torch.sqrt(torch.sum(output_no0**2,3)),-1) #위의 행렬 요소 제곱 -> 3이라는 축(4번째로 큰 축=1차원)으로 더해줌 최소 4차원-> 모든 요소에 root. -1은 가장 안쪽 차원 추가(가장 작은 차원)
    output_unit = torch.div(output_no0,output_mag)  #요소별 나눗셈 진행 ==> 말 그대로 normalize: 가장 낮은 차원의 요소들을 하나의 벡터로 보면 가장 낮은 차원의 벡터는 크기가 모두 1로 만들어줌.
                                                    #output이 작은 요소의 경우 입력을 10^-5로 통일해버림.
    z_mask = z_refined[...,0]   #가장 낮은 차원을 없애면서 가장 낮은 차원의 수들 중 가장 앞에 있는 수 추출.
    a11 = torch.masked_select(torch.sum(output_unit**2,3),z_mask)    #4차원 벡터의 각 요소 제곱합.제곱하고 reduce_sum하면 4번째 차원이 제거 되면서 모든 요소 1, 10^-5로 퉁친 부분은 값 커짐--> 여기다 z_mask를 씌우면 True인 부분만 남음
    a22 = torch.masked_select(torch.sum(y_normal**2,3),z_mask)  #4차원 벡터의 각 요소 제곱합.제곱하고 reduce_sum하면 4번째 차원이 제거되면서 모든 요소 ==> a11과 동일한 결과??? = y_normal의 가장 낮은 차원의 벡터의 크기가 1인 경우 && y-normal과 ouput_unit의 차원읻 동일하다면 같은 결과 나올 것.
    a12 = torch.masked_select(torch.sum(torch.mul(output_unit,y_normal),3),z_mask)    #제곱하는 대신 output_unit과 y_normal을 곱한다. 이후 제곱후 합을 진행. z_mask 씌워 a12.
#이해안됨!!!!! a11과 a22는 같은 위치에서 같은 1이라는 값을 갖지 않나?
    cos_angle = a12/(a11+0.00001)   #cos각을 값으로 갖는 행렬 cos 공식 대신 이것을 사용.
    loss = torch.mean(torch.acos(cos_angle))   #cos값들의 map에서 loss를 구함. 딱히 axis가 지정되지 않았으므로 모든 세타에 대해 평균을 구함. 
    return loss