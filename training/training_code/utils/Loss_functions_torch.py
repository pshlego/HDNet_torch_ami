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
import pdb
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
# *****************************************************************************************************
def calc_loss_1(output, y, z_r):  #y와 output의 행렬 shape가 같아야 할 것으로 보임, 결국 y와 output의 차이를 비교.
#tf.where(  bool type 텐서,   True일 때 출력값,   False일 때 출력값  )
    # y refine
    y_masked = torch.where(z_r, y, 0*torch.ones_like(y))  #bool 타입 z_r, 이 행렬에서 true값이 들어가 있는 곳은 y를 참조하고 false가 들어가 있는 곳은 0값으로 넣어버림.//tf.ones_like(y)는 y배열의 shape와 같은데 내용물이 1인 행렬
    y_masked_flat_refined = y_masked.view(-1, IMAGE_HEIGHT*IMAGE_WIDTH) #위 마스크된 결과 reshape할 것. 높이축은 임의로, 가로축 넓이는 이미지 입력 높이*너비
    
    # output refine
    o_masked = torch.where(z_r, output, 0*torch.ones_like(y)) #bool 타입z_r에 의해서 true인 부분은 output 넣고 false면 0넣음
    o_masked_flat_refined = o_masked.view(-1, IMAGE_HEIGHT*IMAGE_WIDTH) #위에서 만든 행렬의 모양을 변경, 너비를 높이*너비로 고정. (y_masked와 동일)
    
    # mask refine ==> binary하게 refine즉, 바이너리 마스크다.
    mask_one_refined = torch.where(z_r, torch.ones_like(y), 0*torch.ones_like(y))    #z_r true면 1, false면 0.
    mask_one_flat_refined = mask_one_refined.view(-1, IMAGE_HEIGHT*IMAGE_WIDTH) #마찬가지로, 너비를 높이*너비로 고정한다. (reshape)
    
    # num of pixels
    numOfPix = torch.sum(mask_one_flat_refined,1)   #mask_one_flat_refined를 행 단위로 sum, 즉, numOfPix의 각 행은 하나의 이미지 넓이에 대해서 1의 개수를 센다. 
    
    d = torch.sub(o_masked_flat_refined, y_masked_flat_refined)   #z_r이 true인 부분에 대해서는 output-y를 대입, false인 부분에 대해서는 0 대입.
    d_sum = torch.sum(d**2,1)   #위의 각 요소별 제곱을 한 후에 행 단위로 합침. 각 행별로 더하므로 
    
    cost = torch.mean(torch.div(d_sum, numOfPix))
    return cost
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
def calc_loss_d_refined_mask_1(output, y, z_refined,device):
    
    multiply = torch.tensor([IMAGE_HEIGHT*IMAGE_WIDTH]).to(device)  #상수 = 256*256
    
    # mask nonrefine
    mask_one = torch.where(z_refined, torch.ones_like(y), 0*torch.ones_like(y))  #true이면 1 false면 0
    mask_one_flat = mask_one.view(-1, IMAGE_HEIGHT*IMAGE_WIDTH) #위의 행렬을 reshape. 너비만 256*256으로 고정. (높이는 고정된 h라 하자)
    
    # y refine
    y_masked = torch.where(z_refined, y, 0*torch.ones_like(y))    #z_refined 기준으로 true면 y출력, false면 0
    y_masked_flat_refined = y_masked.view(-1, IMAGE_HEIGHT*IMAGE_WIDTH) #같은 모양으로 reshape. 너비를 256*256고정
    
    max_y = torch.max(y_masked_flat_refined,1)[0]  #axis 1에 대해 가장 큰 값을 추출. 즉, 너비 256*256당 1개의 max값 도출
    #pdb.set_trace()
    matrix_max_y = torch.transpose(max_y.repeat_interleave(multiply).view(multiply[0], max_y.size()[0]),0,1)   #tile=max_y를 복붙하는데 256*256 길이로 복붙. 각 픽셀마다 max_y 벡터 존재.
    #Max_y(h)가 동일한 형태로 256*256으로 배열되어 있던거를 reshape: [256*256, max_y벡터크기]
    #max_y(h)가 가로로 256*256개 늘어서 있는 모양
    
    # normalize depth
    output_flat = output.view(-1, IMAGE_HEIGHT*IMAGE_WIDTH) #output을 reshape: 너비를 256*256으로, 높이는 모름.(H=정해진 높이)
    output_flat_masked = torch.mul(output_flat, mask_one_flat)    #z_refined가 true면 reshape된 output 결과 그대로, false면 0
    
    output_max = torch.max(output_flat_masked,1)[0]   #output masked를 1차원에 대해 max값 취함. 
    matrix_max = torch.transpose(output_max.repeat_interleave(multiply).view( multiply[0], output_max.size()[0]),0,1)   #위 행렬을 256*256
    #multiply[0] = 행렬의 전체 요소 개수를 의미 (크기 의미)
    #output_max(H)가 가로로 256*256개 늘어서 있는 모양
    output_min = torch.min(output_flat_masked,1)[0]    #output masked를 1차원에 대해 min값 취함. 
    matrix_min = torch.transpose(output_min.repeat_interleave(multiply).view(multiply[0], output_min.size()[0]),0,1)
    #output_min(H)이 가로로 256*256개 늘어서 있는 모양
    output_unit_flat = torch.div(torch.sub(output_flat_masked,matrix_min),torch.sub(matrix_max,matrix_min))    #(원본-min)/(max-min)
    output_unit_flat = torch.mul(output_unit_flat,matrix_max_y)   #스스로를 max_y에 곱함. 
    
    # mask refine
    mask_one_refined = torch.where(z_refined, torch.ones_like(y), 0*torch.ones_like(y))  #다시, z_refined에 의해 true=1, false=0
    mask_one_flat_refined = mask_one_refined.view(-1, IMAGE_HEIGHT*IMAGE_WIDTH) #어떤 높이 H'가지는 넓이 256*256 되도록 위 행렬 reshape
    
    # output refine
    output_unit_masked_flat_refined = torch.mul(output_unit_flat, mask_one_flat_refined)  #두 개를 곱함.

    # y refine
    y_masked = torch.where(z_refined, y, 0*torch.ones_like(y))    #z_refined가 true=y, false=0
    y_masked_flat_refined = y_masked.view(-1, IMAGE_HEIGHT*IMAGE_WIDTH) #위 행렬을 너비 256*256되도록 reshape
    
    
    numOfPix = torch.sum(mask_one_flat_refined,1)   #H' 높이 가지는 이 행렬에 대해 1차원에 대해 합해줌.
    
    d = torch.sub(output_unit_masked_flat_refined, y_masked_flat_refined)
    a1 = torch.sum(d**2,1)
    a2 = torch.sum(d,1)
    a2 = a2**2

    cost = torch.mean(torch.div(a1, numOfPix) - (0.5 * torch.div(a2, numOfPix**2)))
    return cost