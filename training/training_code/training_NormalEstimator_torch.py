## **********************  import **********************
from __future__ import absolute_import, division, print_function, unicode_literals#이건 파이썬 3에서 쓰던 문법을 파이썬 2에서 쓸수 있게 해주는 문법이다.
import os.path
import os# 운영체제를 제어하는 모듈
from os import path
import numpy as np# python에서 벡터, 행렬 등 수치 연산을 수행하는 선형대수 라이브러리
import skimage.data# skimage는 이미지 처리하기 위한 파이썬 라이브러리
from PIL import Image, ImageDraw, ImageFont# PIL은 파이썬 인터프리터에 다양한 이미지 처리와 그래픽 기능을 제공하는 라이브러리
import random
import scipy.misc# scipy에서 기타 함수 https://docs.scipy.org/doc/scipy/reference/misc.html
import pdb
import math# 수학 관련 함수들이 들어있는 라이브러리
from utils.hourglass_net_normal_singleStack import hourglass_normal_prediction#depth estimator를 import한다.
from utils.hourglass_net_normal_singleStack_torch import hourglass_normal_prediction_1#depth estimator를 import한다.
from utils.IO import get_renderpeople_patch, get_camera, get_tiktok_patch, write_prediction, write_prediction_normal, save_prediction_png_normal#data의 input, output을 담당하는 함수들 import
from utils.Loss_functions_torch import calc_loss_normal2_1
from utils.Geometry_MB import dmap_to_nmap#depth를 normal로 바꿔주는 함수들 정의
from utils.denspose_transform_functions import compute_dp_tr_3d_2d_loss2 #self-supervise할 때 필요한 warping을 통해 구현된 loss function
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
print("You are using torch version ",torch.__version__)#당신은 torch version 몇을 쓰고 있습니다.
## ********************** Set gpu **********************
os.environ["CUDA_VISIBLE_DEVICES"]="6"#6번 GPU를 씁니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)  # 출력결과: cuda
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device())
## ********************** change your variables **********************
IMAGE_HEIGHT = 256#IMAGE의 HEIGHT는 256이고
IMAGE_WIDTH = 256#IMAGE의 WIDTH는 256이고
BATCH_SIZE = 8#여기서는 BATCH_SIZE를 8로 하겠습니다.
ITERATIONS = 100000000#이터레이션의 횟수
rp_path = "/home/ug_psh/AMILab/training_data/Tang_data"#Tang_data의 경로
RP_image_range = range(0,188)#Tang_data의 개수는 188개이다.
origin1n, scaling1n, C1n, cen1n, K1n, Ki1n, M1n, R1n, Rt1n = get_camera(BATCH_SIZE,IMAGE_HEIGHT)#get_camera를 통해 다음과 같은 정보를 받아옴
## **************************** define the network ****************************
model = hourglass_normal_prediction_1(3).to(device)
## ****************************optimizer****************************
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
##  ********************** initialize the network ********************** 
     
        
##  ********************** make the output folders ********************** 
ck_pnts_dir = "/home/ug_psh/AMILab/training_progress/pytorch/model/NormalEstimator"
log_dir = "/home/ug_psh/AMILab/training_progress/pytorch/"
Vis_dir_rp  = "/home/ug_psh/AMILab/training_progress/pytorch/visualization/NormalEstimator/Tang/"

if not path.exists(ck_pnts_dir):
    print("ck_pnts_dir created!")

    os.makedirs(ck_pnts_dir)

if not path.exists(Vis_dir_rp):
    print("Vis_dir created!")
    os.makedirs(Vis_dir_rp)
    
if (path.exists(log_dir+"trainLog.txt")):
    os.remove(log_dir+"trainLog.txt")
    
##  ********************** Run the training **********************     
for itr in range(ITERATIONS):
    (X_1, X1, Y1, N1, Z1, DP1, Z1_3,frms) = get_renderpeople_patch(rp_path, BATCH_SIZE, RP_image_range, IMAGE_HEIGHT,IMAGE_WIDTH)#renderpeople에서 GT를 가져온다.
    optimizer.zero_grad()
    out2_1_1_normal = model(torch.Tensor(X1).type(torch.float32).to(device))
    total_loss_n = calc_loss_normal2_1(out2_1_1_normal.type(torch.float32).to(device),torch.Tensor(N1).type(torch.float32).to(device),torch.Tensor(Z1).type(torch.bool).to(device))
    total_loss = total_loss_n.to(device)
    
    total_loss.backward()
    optimizer.step()
    
    if itr%10 == 0:
        f_err = open(log_dir+"trainLog.txt","a")
        f_err.write("%d %g\n" % (itr,total_loss))
        f_err.close()
        print("")
        print("iteration %3d, depth refinement training loss is %g." %(itr,  total_loss))
        
    if itr % 100 == 0:
        # visually compare the first sample in the batch between predicted and ground truth
        fidx = [int(frms[0])]
        output=out2_1_1_normal.cpu()
        write_prediction_normal(Vis_dir_rp,output.detach().numpy(),itr,fidx,Z1)
        save_prediction_png_normal (output[0,...].detach().numpy(),X1,Z1,Z1_3,Vis_dir_rp,itr,fidx)
        
    if itr % 10000 == 0 and itr != 0:
        torch.save(model.state_dict(), ck_pnts_dir +"/model_" + str(itr) + ".pt")#checkpoint만들기



