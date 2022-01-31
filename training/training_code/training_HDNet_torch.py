## **********************  import **********************
from __future__ import absolute_import, division, print_function, unicode_literals#이건 파이썬 3에서 쓰던 문법을 파이썬 2에서 쓸수 있게 해주는 문법이다.
import warnings
warnings.filterwarnings('ignore')
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
from utils.denspose_transform_functions_torch import compute_dp_tr_3d_2d_loss2_1
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import time
import wandb
import gc 
wandb.init(project="training_HDNet_pytorch", entity="parksh0712")
gc.collect()
torch.cuda.empty_cache()
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
BATCH_SIZE = 1#여기서는 BATCH_SIZE를 8로 하겠습니다.
ITERATIONS = 30*380#이터레이션의 횟수
LR = 0.001
wandb.config = {
  "IMAGE_HEIGHT" : IMAGE_HEIGHT,
  "IMAGE_WIDTH": IMAGE_WIDTH,
  "BATCH_SIZE": BATCH_SIZE,
  "ITERATIONS" : ITERATIONS,
  "Learning_Rate" : LR
}
pre_ck_pnts_dir = "/home/ug_psh/HDNet_torch_ami/model/torch/depth_prediction"
model_num = '1920000'
model_num_int = 1920000
rp_path = "/home/ug_psh/HDNet_torch_ami/training_data/Tang_data"#Tang_data의 경로
tk_path = "/home/ug_psh/HDNet_torch_ami/training_data/tiktok_data/"
RP_image_range = range(0,188)#Tang_data의 개수는 188개이다.
origin1n, scaling1n, C1n, cen1n, K1n, Ki1n, M1n, R1n, Rt1n = get_camera(BATCH_SIZE,IMAGE_HEIGHT)#get_camera를 통해 다음과 같은 정보를 받아옴
origin1n = torch.Tensor(origin1n).type(torch.float32).to(device)
scaling1n = torch.Tensor(scaling1n).type(torch.float32).to(device)
K = torch.rand([3,3]).type(torch.float32).to(device)
C = torch.rand([3,4]).type(torch.float32).to(device)
C1n = torch.Tensor(C1n).type(torch.float32).to(device)
cen1n = torch.Tensor(cen1n).type(torch.float32).to(device)
K1n = torch.Tensor(K1n).type(torch.float32).to(device)
Ki1n = torch.Tensor(Ki1n).type(torch.float32).to(device)
M1n = torch.Tensor(M1n).type(torch.float32).to(device)
R1n = torch.Tensor(R1n).type(torch.float32).to(device)
Rt1n = torch.Tensor(Rt1n).type(torch.float32).to(device)
## **************************** define the network ****************************
PATH = "/home/ug_psh/HDNet_torch_ami/training_progress/pytorch/model/DepthEstimator/"
model = hourglass_refinement_1(9).to(device)
model.load_state_dict(torch.load(PATH + 'model_6000.pt'))
print("Model restored.")
## ****************************optimizer****************************
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=0.1)
##  ********************** make the output folders ********************** 
ck_pnts_dir = "/home/ug_psh/HDNet_torch_ami/training_progress/pytorch/model/HDNet"
log_dir = "/home/ug_psh/HDNet_torch_ami/training_progress/pytorch/"
Vis_dir  = "/home/ug_psh/HDNet_torch_ami/training_progress/pytorch/visualization/HDNet/tiktok/"
Vis_dir_rp  = "/home/ug_psh/HDNet_torch_ami/training_progress/pytorch/visualization/HDNet/Tang/"

if not path.exists(ck_pnts_dir):
    print("ck_pnts_dir created!")

    os.makedirs(ck_pnts_dir)
    
if not path.exists(Vis_dir):
    print("Vis_dir created!")
    os.makedirs(Vis_dir)
    
if not path.exists(Vis_dir_rp):
    print("Vis_dir_rp created!")
    os.makedirs(Vis_dir_rp)
    
if (path.exists(log_dir+"trainLog.txt")):
    os.remove(log_dir+"trainLog.txt")    
##  ********************** Run the training ********************** 
start_time_1 = time.time()
for itr in range(ITERATIONS):
    (X_1, X1, Y1, N1, Z1, DP1, Z1_3,frms) = get_renderpeople_patch(rp_path, BATCH_SIZE, RP_image_range, IMAGE_HEIGHT,IMAGE_WIDTH)#renderpeople에서 GT를 가져온다.
    (X_1_tk, X1_tk, N1_tk, Z1_tk, DP1_tk, Z1_3_tk, X_2_tk, X2_tk, N2_tk, Z2_tk, DP2_tk, Z2_3_tk, i_r1_c1_r2_c2_tk, i_limit_tk, frms_tk, frms_neighbor_tk) = get_tiktok_patch(tk_path, BATCH_SIZE, IMAGE_HEIGHT,IMAGE_WIDTH)
    optimizer.zero_grad()
    X_1 = torch.Tensor(X_1).type(torch.float32).to(device)
    X1 = torch.Tensor(X1).type(torch.float32).to(device)
    Y1 = torch.Tensor(Y1).type(torch.float32).to(device)
    N1 = torch.Tensor(N1).type(torch.float32).to(device)
    Z1 = torch.Tensor(Z1).type(torch.bool).to(device)
    DP1 = torch.Tensor(DP1).type(torch.float32).to(device)
    Z1_3 = torch.Tensor(Z1_3).type(torch.bool).to(device)
    X_1_tk = torch.Tensor(X_1_tk).type(torch.float32).to(device)
    N1_tk = torch.Tensor(N1_tk).type(torch.float32).to(device)
    Z1_tk = torch.Tensor(Z1_tk).type(torch.bool).to(device)
    X_2_tk = torch.Tensor(X_2_tk).type(torch.float32).to(device)
    N2_tk = torch.Tensor(N2_tk).type(torch.float32).to(device)
    Z2_tk = torch.Tensor(Z2_tk).type(torch.bool).to(device)
    i_r1_c1_r2_c2_tk = torch.Tensor(i_r1_c1_r2_c2_tk).type(torch.int32).to(device)
    i_limit_tk = torch.Tensor(i_limit_tk).type(torch.int32).to(device)
    frms_tk = torch.Tensor(frms_tk).type(torch.float32).to(device)
    out2_1 = model(X_1)
    out2_1_tk = model(X_1_tk)
    out2_2_tk = model(X_2_tk)
##  ****************************Loss RP****************************
    nmap1 = dmap_to_nmap_1(out2_1, Rt1n, R1n, Ki1n, cen1n, Z1, origin1n, scaling1n,device).to(device)
    total_loss1_d = calc_loss_1(out2_1,Y1,Z1)
    total_loss2_d = calc_loss_d_refined_mask_1(out2_1,Y1,Z1,device)
    total_loss_n = calc_loss_normal2_1(nmap1,N1,Z1)
    total_loss_rp = 2*total_loss1_d + total_loss2_d + total_loss_n
##  ****************************Loss TK****************************
    nmap1_tk = dmap_to_nmap_1(out2_1_tk, Rt1n, R1n, Ki1n, cen1n, Z1_tk, origin1n, scaling1n, device).to(device)
    nmap2_tk = dmap_to_nmap_1(out2_2_tk, Rt1n, R1n, Ki1n, cen1n, Z2_tk, origin1n, scaling1n, device).to(device)
    total_loss_n_tk = calc_loss_normal2_1(nmap1_tk,N1_tk,Z1_tk)+calc_loss_normal2_1(nmap2_tk,N2_tk,Z2_tk)
    loss3d,loss2d,PC2p,PC1_2 = compute_dp_tr_3d_2d_loss2_1(out2_1_tk,out2_2_tk, i_r1_c1_r2_c2_tk[0,...],i_limit_tk[0,...],C,R1n,Rt1n,cen1n,K,Ki1n,origin1n,scaling1n,device)
    total_loss_tk = total_loss_n_tk + 5*loss3d
##  ****************************Loss all****************************
    total_loss=(total_loss_rp+total_loss_tk).to(device)
    total_loss.backward()
    optimizer.step()
    gc.collect()
    torch.cuda.empty_cache()
    if itr%10 == 0:
        f_err = open(log_dir+"trainLog.txt","a")
        f_err.write("%d %g\n" % (itr,total_loss_rp+total_loss_tk))
        f_err.close()
        print("")
        print("iteration %3d, depth refinement training loss is %g." %(itr,  (total_loss_rp+total_loss_tk)))
        print("10 iter Time taken: %.2fs" % (time.time() - start_time_1))
        start_time_1 = time.time()
    if itr % 100 == 0:
        prediction1=out2_1.detach().cpu().numpy()
        prediction1_tk=out2_1_tk.detach().cpu().numpy()
        fidx = [int(frms[0])]
        z1_cpu=Z1.detach().cpu().numpy();
        x1_cpu=X1.detach().cpu().numpy()
        z1_3_cpu=Z1_3.detach().cpu().numpy()
        nmap1_cpu=nmap1.detach().cpu().numpy()
        z1_tk_cpu=Z1_tk.detach().cpu().numpy()
        nmap1_tk_cpu=nmap1_tk.detach().cpu().numpy()
        write_prediction(Vis_dir_rp,prediction1,itr,fidx,z1_cpu);
        

        write_prediction_normal(Vis_dir_rp,nmap1_cpu,itr,fidx,z1_cpu)
        
        save_prediction_png (prediction1[0,...,0],nmap1_cpu[0,...],x1_cpu,z1_cpu,z1_3_cpu,Vis_dir_rp,itr,fidx,1)
        fidx = [int(frms_tk[0])]
        write_prediction(Vis_dir,prediction1_tk,itr,fidx,z1_tk_cpu);
        
        write_prediction_normal(Vis_dir,nmap1_tk_cpu,itr,fidx,z1_tk_cpu)
        save_prediction_png (prediction1_tk[0,...,0],nmap1_tk_cpu[0,...],X1_tk,z1_tk_cpu,Z1_3_tk,Vis_dir,itr,fidx,1)
        
    if itr % 1000 == 0 and itr != 0:
        torch.save(model.state_dict(), ck_pnts_dir + "/model_" + str(itr) + ".pt")#checkpoint만들기
        print("iteration %3d, checkpoint save." %(itr))
    if itr == (ITERATIONS-2):
        torch.save(model.state_dict(), ck_pnts_dir + "/model_" + str(itr) + ".pt")#checkpoint만들기
        print("iteration %3d, checkpoint save." %(itr))
    wandb.log({"loss": total_loss})



