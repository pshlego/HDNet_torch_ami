## **********************  import **********************
from __future__ import absolute_import, division, print_function, unicode_literals#이건 파이썬 3에서 쓰던 문법을 파이썬 2에서 쓸수 있게 해주는 문법이다.
import pdb
import tensorflow as tf
import os.path
import os# 운영체제를 제어하는 모듈
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from os import path
import numpy as np# python에서 벡터, 행렬 등 수치 연산을 수행하는 선형대수 라이브러리
import skimage.data# skimage는 이미지 처리하기 위한 파이썬 라이브러리
from PIL import Image, ImageDraw, ImageFont# PIL은 파이썬 인터프리터에 다양한 이미지 처리와 그래픽 기능을 제공하는 라이브러리
import random
import scipy.misc# scipy에서 기타 함수 https://docs.scipy.org/doc/scipy/reference/misc.html
import math# 수학 관련 함수들이 들어있는 라이브러리
from utils.vector import cross
from utils.vector_torch import cross_1
from utils.hourglass_net_normal_singleStack import hourglass_normal_prediction
from utils.hourglass_net_normal_singleStack_torch import hourglass_normal_prediction_1
from utils.hourglass_net_depth_singleStack import hourglass_refinement
from utils.hourglass_net_depth_singleStack_torch import hourglass_refinement_1
from utils.IO import get_renderpeople_patch, get_camera, get_tiktok_patch, write_prediction, write_prediction_normal, save_prediction_png_normal#data의 input, output을 담당하는 함수들 import
#from utils.Loss_functions import calc_loss_normal2, calc_loss, calc_loss_d_refined_mask#loss function이 정의되어있는 함수들
from utils.Geometry_MB import dmap_to_nmap#depth를 normal로 바꿔주는 함수들 정의
from utils.Geometry_MB_torch import dmap_to_nmap_1#depth를 normal로 바꿔주는 함수들 정의
from utils.denspose_transform_functions import compute_dp_tr_3d_2d_loss2 #self-supervise할 때 필요한 warping을 통해 구현된 loss function
from utils.denspose_transform_functions_torch import compute_dp_tr_3d_2d_loss2_1 #self-supervise할 때 필요한 warping을 통해 구현된 loss function
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from functions import showOperation
from torchinfo import summary
model = hourglass_refinement_1(9)
refineNet_graph = tf.Graph()
with refineNet_graph.as_default():
    x1 = tf.placeholder(tf.float32, shape=(10, 256,256,9))
    with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):  #tf.variable_scope는 변수를 보다 쉽게 공유할 수 있도록 한다. hourglass_refinement에서 만들어진 variable들에 'hourglass_normal_prediction'라는 label을 붙인다.
        out2_1 = hourglass_refinement(x1,True)
total_parameters = 0
for variable in refineNet_graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
# shape is an array of tf.Dimension
    shape = variable.get_shape()
    print("Size of the matrix: {}".format(shape))
    print("How many dimensions it has: {}".format(len(shape)))
    variable_parameters = 1
    for dim in shape:
        #print("Dimension: {}".format(dim))
        variable_parameters *= dim.value
    print("Total number of elements in a matrix: {}".format(variable_parameters))
    print("---------------------------------------------")
    total_parameters += variable_parameters
print("Total number of parameters: {}". format(total_parameters))
summary(model, input_size=(10, 256, 256, 9))