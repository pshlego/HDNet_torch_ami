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
from utils.hourglass_net_depth_singleStack_torch import hourglass_refinement_1
from utils.hourglass_net_normal_singleStack_torch import hourglass_normal_prediction_1#depth estimator를 import한다.
from utils.IO import get_renderpeople_patch, get_camera, get_tiktok_patch, write_prediction, write_prediction_normal, save_prediction_png_normal, save_prediction_png#data의 input, output을 담당하는 함수들 import
from utils.Loss_functions_torch import calc_loss_normal2_1, calc_loss_1, calc_loss_d_refined_mask_1
from utils.Geometry_MB_torch import dmap_to_nmap_1#depth를 normal로 바꿔주는 함수들 정의
from utils.denspose_transform_functions import compute_dp_tr_3d_2d_loss2 #self-supervise할 때 필요한 warping을 통해 구현된 loss function
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import warnings
import time
import wandb
import gc
wandb.init(project="training_Depthestimator_pytorch", entity="parksh0712")
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
BATCH_SIZE = 10#여기서는 BATCH_SIZE를 8로 하겠습니다.
ITERATIONS = 18*380#이터레이션의 횟수
LR = 0.001
wandb.config = {
  "IMAGE_HEIGHT" : IMAGE_HEIGHT,
  "IMAGE_WIDTH": IMAGE_WIDTH,
  "BATCH_SIZE": BATCH_SIZE,
  "ITERATIONS" : ITERATIONS,
  "Learning_Rate" : LR
}
rp_path = "/home/ug_psh/HDNet_torch_ami/training_data/Tang_data"#Tang_data의 경로
RP_image_range = range(0,188)#Tang_data의 개수는 188개이다.
origin1n, scaling1n, C1n, cen1n, K1n, Ki1n, M1n, R1n, Rt1n = get_camera(BATCH_SIZE,IMAGE_HEIGHT)#get_camera를 통해 다음과 같은 정보를 받아옴
origin1n = torch.Tensor(origin1n).type(torch.float32).to(device)
scaling1n = torch.Tensor(scaling1n).type(torch.float32).to(device)
C1n = torch.Tensor(C1n).type(torch.float32).to(device)
cen1n = torch.Tensor(cen1n).type(torch.float32).to(device)
K1n = torch.Tensor(K1n).type(torch.float32).to(device)
Ki1n = torch.Tensor(Ki1n).type(torch.float32).to(device)
M1n = torch.Tensor(M1n).type(torch.float32).to(device)
R1n = torch.Tensor(R1n).type(torch.float32).to(device)
Rt1n = torch.Tensor(Rt1n).type(torch.float32).to(device)
## **************************** define the network ****************************
model = hourglass_refinement_1(9).to(device)
## ****************************optimizer****************************
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-10)
##  ********************** make the output folders ********************** 
ck_pnts_dir = "/home/ug_psh/HDNet_torch_ami/training_progress/pytorch/model/DepthEstimator"
log_dir = "/home/ug_psh/HDNet_torch_ami/training_progress/pytorch/"
Vis_dir_rp  = "/home/ug_psh/HDNet_torch_ami/training_progress/pytorch/visualization/DepthEstimator/Tang/"

if not path.exists(ck_pnts_dir):
    print("ck_pnts_dir created!")
    os.makedirs(ck_pnts_dir)

if not path.exists(Vis_dir_rp):
    print("Vis_dir created!")
    os.makedirs(Vis_dir_rp)
    
if (path.exists(log_dir+"trainLog.txt")):
    os.remove(log_dir+"trainLog.txt")    
##  ********************** Run the training **********************
start_time_1 = time.time()
for itr in range(ITERATIONS):
    #start_time = time.time()
    (X_1, X1, Y1, N1, Z1, DP1, Z1_3,frms) = get_renderpeople_patch(rp_path, BATCH_SIZE, RP_image_range, IMAGE_HEIGHT,IMAGE_WIDTH)#renderpeople에서 GT를 가져온다.
    #print("get_render Time taken: %.2fs" % (time.time() - start_time))
    optimizer.zero_grad()
    X_1 = torch.Tensor(X_1).type(torch.float32).to(device)
    X1 = torch.Tensor(X1).type(torch.float32).to(device)
    Y1 = torch.Tensor(Y1).type(torch.float32).to(device)
    N1 = torch.Tensor(N1).type(torch.float32).to(device)
    Z1 = torch.Tensor(Z1).type(torch.bool).to(device)
    Z1_3 = torch.Tensor(Z1_3).type(torch.bool).to(device)
    #start_time = time.time()
    out2_1 = model(X_1)
    #print("model Time taken: %.2fs" % (time.time() - start_time))
    #start_time = time.time()
    #with torch.no_grad():
    nmap1 = dmap_to_nmap_1(out2_1, Rt1n, R1n, Ki1n, cen1n, Z1,origin1n, scaling1n,device).type(torch.float32)
    #print("dmap to nmap Time taken: %.2fs" % (time.time() - start_time))
    #start_time = time.time()
    total_loss1_d = calc_loss_1(out2_1,Y1,Z1)
    total_loss2_d = calc_loss_d_refined_mask_1(out2_1,Y1,Z1,device)
    total_loss_n = calc_loss_normal2_1(nmap1,N1,Z1)
    total_loss_rp = 2*total_loss1_d.to(device) + total_loss2_d.to(device) + total_loss_n.to(device)
    total_loss=total_loss_rp.to(device)
    #total_loss_rp=total_loss2_d
    #total_loss=total_loss_rp.to(device)
    #print("calc loss Time taken: %.2fs" % (time.time() - start_time))
    start_time = time.time()
    total_loss.backward()
    optimizer.step()
    #print("gradient descent Time taken: %.2fs" % (time.time() - start_time))
    gc.collect()
    torch.cuda.empty_cache()
    if itr%10 == 0:
        f_err = open(log_dir+"trainLog.txt","a")
        f_err.write("%d %g\n" % (itr,total_loss_rp.cpu().detach().numpy()))
        f_err.close()
        print("")
        print("iteration %3d, depth refinement training loss is %g." %(itr,  total_loss_rp.cpu().detach().numpy()))
        print("10 iter Time taken: %.2fs" % (time.time() - start_time_1))
        start_time_1 = time.time()
    if itr % 100 == 0:
        fidx = [int(frms[0])]
        prediction1=out2_1.cpu().detach().numpy()
        z1_cpu=Z1.cpu().detach().numpy();
        x1_cpu=X1.cpu().detach().numpy()
        z1_3_cpu=Z1_3.cpu().detach().numpy()
        nmap1_cpu=nmap1.cpu().detach().numpy()
        write_prediction(Vis_dir_rp,prediction1,itr,fidx,z1_cpu);
        write_prediction_normal(Vis_dir_rp,nmap1_cpu,itr,fidx,z1_cpu)
        save_prediction_png (prediction1[0,...,0],nmap1_cpu[0,...],x1_cpu,z1_cpu,z1_3_cpu,Vis_dir_rp,itr,fidx,0.99)      
    if itr % 1000 == 0 and itr != 0:
        torch.save(model.state_dict(), ck_pnts_dir + "/model_" + str(itr) + ".pt")#checkpoint만들기
        print("iteration %3d, checkpoint save." %(itr)) 
    if itr == (18*380-2):
        torch.save(model.state_dict(), ck_pnts_dir + "/model_" + str(itr) + ".pt")#checkpoint만들기
        print("iteration %3d, checkpoint save." %(itr))
    wandb.log({"loss": total_loss_rp.cpu().detach().numpy()})
    


