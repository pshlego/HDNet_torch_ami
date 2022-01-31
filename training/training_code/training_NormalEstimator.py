## **********************  import **********************
from __future__ import absolute_import, division, print_function, unicode_literals#이건 파이썬 3에서 쓰던 문법을 파이썬 2에서 쓸수 있게 해주는 문법이다.

import tensorflow as tf# tensorflow import
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
from tensorflow.python.platform import gfile# open()이랑 같고, tensorflow용 파일 입출력 함수
import pdb
from utils.hourglass_net_normal_singleStack import hourglass_normal_prediction#depth estimator를 import한다.
from utils.IO import get_renderpeople_patch, get_camera, get_tiktok_patch, write_prediction, write_prediction_normal, save_prediction_png_normal#data의 input, output을 담당하는 함수들 import
from utils.Loss_functions import calc_loss_normal2, calc_loss, calc_loss_d_refined_mask#loss function이 정의되어있는 함수들
from utils.Geometry_MB import dmap_to_nmap#depth를 normal로 바꿔주는 함수들 정의
from utils.denspose_transform_functions import compute_dp_tr_3d_2d_loss2 #self-supervise할 때 필요한 warping을 통해 구현된 loss function
import time
import wandb
import gc
import pdb
wandb.init(project="training_NormalEstimator_tensorflow", entity="parksh0712")
print("You are using tensorflow version ",tf.VERSION)#당신은 tensorflow version 몇을 쓰고 있습니다.
os.environ["CUDA_VISIBLE_DEVICES"]="7"#7번 GPU를 씁니다.

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


## **************************** define the network ****************************
refineNet_graph = tf.Graph()#tensorflow는 dataflow-graph를 구성하고, graph의 일부를 session으로 구성해 실행시키는 방식이다.
with refineNet_graph.as_default():#with 구문을 이용하면 원하는 그래프와 연결할 수 있다. 그리고 .as_default()를 통해 이 그래프를 default graph로 지정한다.
    
    ## ****************************RENDERPEOPLE****************************
    #placeholder는 변수보다 더 기본적인 데이터 유형으로 초기 값이 필요하지 않은 상태로 graph를 만들 수 있도록 한다. 그래프는 데이터 유형 placeholder와 텐서만으로 저장된 값을 가지고 있지 않아도 무엇을 계산할 지 알고 있게 됩니다.
    x1 = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    n1 = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    z1 = tf.placeholder(tf.bool, shape=(None, 256,256,1))

    with tf.variable_scope('hourglass_normal_prediction', reuse=tf.AUTO_REUSE):  #tf.variable_scope는 변수를 보다 쉽게 공유할 수 있도록 한다. hourglass_refinement에서 만들어진 variable들에 'hourglass_normal_prediction'라는 label을 붙인다.
        out2 = hourglass_normal_prediction(x1,True)#hourglass 형태의 normal estimator의 결과를 out2에 저장한다. #out2=Batchsize x Image_Height x Image_Width x 3
    total_loss_n = calc_loss_normal2(out2,n1,z1)#surface normal의 GT와 normal estimator의 결과인 surface normal의 차이를 loss로 쓴다.
    total_loss = total_loss_n#그리고 그게 total loss이다.
    
    ## ****************************optimizer****************************
    train_step = tf.train.AdamOptimizer().minimize(total_loss)



##  ********************** initialize the network ********************** 
sess = tf.Session(graph=refineNet_graph)#graph를 초기화한다.
with sess.as_default():
    with refineNet_graph.as_default():
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
#         saver = tf.train.import_meta_graph(pre_ck_pnts_dir+'/model_'+model_num+'/model_'+model_num+'.ckpt.meta')
#         saver.restore(sess,pre_ck_pnts_dir+'/model_'+model_num+'/model_'+model_num+'.ckpt')
        print("Model restored.")
        
        
##  ********************** make the output folders ********************** 
ck_pnts_dir = "/home/ug_psh/HDNet_torch_ami/training_progress/tensorflow/model/NormalEstimator"
log_dir = "/home/ug_psh/HDNet_torch_ami/training_progress/tensorflow"
Vis_dir_rp  = "/home/ug_psh/HDNet_torch_ami/training_progress/tensorflow/visualization/NormalEstimator/Tang/"

if not gfile.Exists(ck_pnts_dir):
    print("ck_pnts_dir created!")
    gfile.MakeDirs(ck_pnts_dir)
    
if not gfile.Exists(Vis_dir_rp):
    print("Vis_dir created!")
    gfile.MakeDirs(Vis_dir_rp)
    
if (path.exists(log_dir+"trainLog.txt")):
    os.remove(log_dir+"trainLog.txt")
start_time = time.time()
##  ********************** Run the training **********************     
for itr in range(ITERATIONS):
    (X_1, X1, Y1, N1, 
     Z1, DP1, Z1_3,frms) = get_renderpeople_patch(rp_path, BATCH_SIZE, RP_image_range, 
                                                  IMAGE_HEIGHT,IMAGE_WIDTH)#renderpeople에서 GT를 가져온다.
    

    (_,loss_val,prediction1) = sess.run([train_step,total_loss,out2],
                                                  feed_dict={x1:X1,n1:N1,z1:Z1})#iteration마다 sess.run으로 graph를 실행시킨다.
    wandb.log({'loss': loss_val})    
    if itr%10 == 0:
        f_err = open(log_dir+"trainLog.txt","a")
        f_err.write("%d %g\n" % (itr,loss_val))
        f_err.close()
        print("")
        print("iteration %3d, normal prediction training loss is %g." %(itr,  loss_val))
        print("10 iter Time taken: %.2fs" % (time.time() - start_time))
        start_time = time.time()
    if itr % 100 == 0:
        # visually compare the first sample in the batch between predicted and ground truth
        fidx = [int(frms[0])]
        write_prediction_normal(Vis_dir_rp,prediction1,itr,fidx,Z1)
        save_prediction_png_normal (prediction1[0,...],X1,Z1,Z1_3,Vis_dir_rp,itr,fidx)
        
    if itr % 1000 == 0 and itr != 0:
        save_path = saver.save(sess,ck_pnts_dir+"/model_"+str(itr)+".ckpt")#checkpoint만들기
        print("iteration %3d, checkpoint save." %(itr))
    if itr == (18*380-2):
        save_path = saver.save(sess,ck_pnts_dir+"/model_"+str(itr)+".ckpt")#checkpoint만들기
        print("iteration %3d, checkpoint save." %(itr))

