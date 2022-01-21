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
os.environ["CUDA_VISIBLE_DEVICES"]="6"#6번 GPU를 씁니다.

# ********************** change your variables **********************
IMAGE_HEIGHT = 256#IMAGE의 HEIGHT는 256이고
IMAGE_WIDTH = 256#IMAGE의 WIDTH는 256이고
BATCH_SIZE = 1#여기서는 BATCH_SIZE를 8로 하겠습니다.

rp_path = "/home/ug_psh/AMILab/training_data/Tang_data"#Tang_data의 경로
tk_path = "/home/ug_psh/AMILab/training_data/tiktok_data"#tiktok_data의 경로
RP_image_range = range(0,188)#Tang_data의 개수는 188개이다.
origin1n, scaling1n, C1n, cen1n, K1n, Ki1n, M1n, R1n, Rt1n = get_camera(BATCH_SIZE,IMAGE_HEIGHT)#get_camera를 통해 다음과 같은 정보를 받아옴

#(X_1, X1, Y1, N1, Z1, DP1, Z1_3,frms) = get_renderpeople_patch(rp_path, BATCH_SIZE, RP_image_range, IMAGE_HEIGHT,IMAGE_WIDTH)
(X_1_tk, X1_tk, N1_tk, Z1_tk, DP1_tk, Z1_3_tk, X_2_tk, X2_tk, N2_tk, Z2_tk, DP2_tk, Z2_3_tk, i_r1_c1_r2_c2_tk, i_limit_tk, frms_tk, frms_neighbor_tk) = get_tiktok_patch(tk_path, BATCH_SIZE, IMAGE_HEIGHT,IMAGE_WIDTH)

# ********************** Hourglass_network **********************

#with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):
#   out2_1 = hourglass_refinement(tf.convert_to_tensor(X_1),True)
# with tf.variable_scope('hourglass_normal_prediction', reuse=tf.AUTO_REUSE):
#     out2_1_normal = hourglass_normal_prediction(tf.convert_to_tensor(X1),True)
#model = hourglass_refinement_1(torch.Tensor(X_1).size()[-1])
#out2_1_1 = model(torch.Tensor(X_1))
# model_1 = hourglass_normal_prediction_1(torch.Tensor(X1).size()[-1])
# out2_1_1_normal = model_1(torch.Tensor(X1))

# ********************** cross function **********************

# cross_result = cross(tf.convert_to_tensor(X1),tf.convert_to_tensor(N1))
# cross_result_1 = cross_1(torch.Tensor(X1),torch.Tensor(N1))

# ********************** Geometry_MB *************************

#nmap1 = dmap_to_nmap(tf.convert_to_tensor(out2_1_1.detach().numpy()), tf.convert_to_tensor(Rt1n,dtype=tf.float32), tf.convert_to_tensor(R1n,dtype=tf.float32), tf.convert_to_tensor(Ki1n,dtype=tf.float32), tf.convert_to_tensor(cen1n,dtype=tf.float32), tf.convert_to_tensor(Z1,dtype=tf.bool), tf.convert_to_tensor(origin1n,dtype=tf.float32), tf.convert_to_tensor(scaling1n,dtype=tf.float32))
#nmap1_1 = dmap_to_nmap_1(out2_1_1, torch.Tensor(Rt1n).type(torch.float32), torch.Tensor(R1n).type(torch.float32), torch.Tensor(Ki1n).type(torch.float32), torch.Tensor(cen1n).type(torch.float32), torch.Tensor(Z1).type(torch.bool), torch.Tensor(origin1n).type(torch.float32), torch.Tensor(scaling1n).type(torch.float32))

# ********************** denspose_transform_functions *************************

# with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):#tiktok dataset에서 i번째 frame의 (background가 흰색으로 적용된 RGB값, Normal estimator에 의해 predict된 normal값, UV 값)그리고 그거에 맞는 output depth를 얻는다.
#     out2_1_tk = hourglass_refinement(tf.convert_to_tensor(X_1_tk),True)
        
# with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):#tiktok dataset에서 j번째 frame의 (background가 흰색으로 적용된 RGB값, Normal estimator에 의해 predict된 normal값, UV 값)그리고 그거에 맞는 output depth를 얻는다.
#     out2_2_tk = hourglass_refinement(tf.convert_to_tensor(X_2_tk),True)
# model = hourglass_refinement_1(torch.Tensor(X_1_tk).size()[-1])
# out2_1_tk_1 = model(torch.Tensor(X_1_tk))
# out2_2_tk_1 = model(torch.Tensor(X_2_tk))
# C = tf.placeholder(tf.float32, shape=(3,4))
# K = tf.placeholder(tf.float32, shape=(3,3))
# C_1 = torch.empty(3,4).type(torch.float32)
# K_1 = torch.empty(3,3).type(torch.float32)

# loss3d,loss2d,PC2p,PC1_2 = compute_dp_tr_3d_2d_loss2(tf.convert_to_tensor(out2_1_tk_1.detach().numpy()),tf.convert_to_tensor(out2_2_tk_1.detach().numpy()),tf.convert_to_tensor(np.array(i_r1_c1_r2_c2_tk),dtype=tf.int32)[0,...],tf.convert_to_tensor(np.array(i_limit_tk),dtype=tf.int32)[0,...],C,tf.convert_to_tensor(R1n,dtype=tf.float32),tf.convert_to_tensor(Rt1n,dtype=tf.float32),tf.convert_to_tensor(cen1n,dtype=tf.float32),K,tf.convert_to_tensor(Ki1n,dtype=tf.float32),tf.convert_to_tensor(origin1n,dtype=tf.float32),tf.convert_to_tensor(scaling1n,dtype=tf.float32))

# loss3d_1,loss2d_1,PC2p_1,PC1_2_1 = compute_dp_tr_3d_2d_loss2_1(out2_1_tk_1,out2_2_tk_1,torch.Tensor(i_r1_c1_r2_c2_tk)[0,...].type(torch.int32),torch.Tensor(i_limit_tk)[0,...].type(torch.int32),C_1,torch.Tensor(R1n).type(torch.float32),torch.Tensor(Rt1n).type(torch.float32),torch.Tensor(cen1n).type(torch.float32),K_1,torch.Tensor(Ki1n).type(torch.float32),torch.Tensor(origin1n).type(torch.float32),torch.Tensor(scaling1n).type(torch.float32))
# pdb.set_trace()
