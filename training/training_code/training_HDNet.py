## **********************  import **********************
from __future__ import absolute_import, division, print_function, unicode_literals#이건 파이썬 3에서 쓰던 문법을 파이썬 2에서 쓸수 있게 해주는 문법이다.
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf# tensorflow import
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
from tensorflow.python.platform import gfile# open()이랑 같고, tensorflow용 파일 입출력 함수

from utils.hourglass_net_depth_singleStack import hourglass_refinement#depth estimator를 import한다.
from utils.IO import get_renderpeople_patch, get_camera, get_tiktok_patch, write_prediction, write_prediction_normal, save_prediction_png#data의 input, output을 담당하는 함수들 import
from utils.Loss_functions import calc_loss_normal2, calc_loss, calc_loss_d_refined_mask#loss function이 정의되어있는 함수들
from utils.Geometry_MB import dmap_to_nmap#depth를 normal로 바꿔주는 함수들 정의
from utils.denspose_transform_functions import compute_dp_tr_3d_2d_loss2 #self-supervise할 때 필요한 warping을 통해 구현된 loss function

print("You are using tensorflow version ",tf.VERSION)#당신은 tensorflow version 몇을 쓰고 있습니다.
os.environ["CUDA_VISIBLE_DEVICES"]="6"#6번 GPU를 씁니다.

## ********************** change your variables **********************
IMAGE_HEIGHT = 256#IMAGE의 HEIGHT는 256이고
IMAGE_WIDTH = 256#IMAGE의 WIDTH는 256이고
BATCH_SIZE = 1#여기서는 BATCH_SIZE를 1로 하겠습니다.
ITERATIONS = 100000000#이터레이션의 횟수

pre_ck_pnts_dir = "/home/ug_psh/HDNet_torch_ami/model/tensorflow/depth_prediction"
model_num = '1920000'
model_num_int = 1920000

rp_path = "/home/ug_psh/HDNet_torch_ami/training_data/Tang_data"#Tang_data의 경로
tk_path = "/home/ug_psh/HDNet_torch_ami/training_data/tiktok_data/"#tiktok_data의 경로
RP_image_range = range(0,188)#Tang_data의 개수는 188개이다.
origin1n, scaling1n, C1n, cen1n, K1n, Ki1n, M1n, R1n, Rt1n = get_camera(BATCH_SIZE,IMAGE_HEIGHT)#get_camera를 통해 다음과 같은 정보를 받아옴

## **************************** define the network ****************************
refineNet_graph = tf.Graph()#tensorflow는 dataflow-graph를 구성하고, graph의 일부를 session으로 구성해 실행시키는 방식이다.
with refineNet_graph.as_default():#with 구문을 이용하면 원하는 그래프와 연결할 수 있다. 그리고 .as_default()를 통해 이 그래프를 default graph로 지정한다.
    
    ## ****************************RENDERPEOPLE****************************
    #placeholder는 변수보다 더 기본적인 데이터 유형으로 초기 값이 필요하지 않은 상태로 graph를 만들 수 있도록 한다. 그래프는 데이터 유형 placeholder와 텐서만으로 저장된 값을 가지고 있지 않아도 무엇을 계산할 지 알고 있게 됩니다.
    x1 = tf.placeholder(tf.float32, shape=(None, 256,256,9))
    y1 = tf.placeholder(tf.float32, shape=(None, 256,256,1))
    n1 = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    z1 = tf.placeholder(tf.bool, shape=(None, 256,256,1))
    
    ## ****************************tiktok****************************
    x1_tk = tf.placeholder(tf.float32, shape=(None, 256,256,9))
    n1_tk = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    z1_tk = tf.placeholder(tf.bool, shape=(None, 256,256,1))
    
    x2_tk = tf.placeholder(tf.float32, shape=(None, 256,256,9))
    n2_tk = tf.placeholder(tf.float32, shape=(None, 256,256,3))
    z2_tk = tf.placeholder(tf.bool, shape=(None, 256,256,1))
    
    i_r1_c1_r2_c2 = tf.placeholder(tf.int32, shape=(None, 25000,5))#i_r1_c1_r2_c2에서 i는 body part의 number이고, 우리가 j와 k번째 time instant의 frame을 비교한다고 했을 때 r1과 c1은 각각 j번째 frame의 correspondence의 row number와 column number를 의미한다. 그리고 r2와 c2는 k번째 frame의 correspondence의 row, column number이다.
    i_limit = tf.placeholder(tf.int32, shape=(None, 24,3))#i_limit은 첫번째 column이 24개의 body part의 number를 의미하고, 2번째, 3번째 column은 각각 해당 i의 i_r1_c1_r2_c2의 range이고, 만약 -1이 있다면 대응하는 body part가 보이지 않거나 correspondence들이 존재하지 않는다는 것이다.
    
    ## *****************************camera***********************************
    R = tf.placeholder(tf.float32, shape=(3,3))
    Rt = tf.placeholder(tf.float32, shape=(3,3))
    K = tf.placeholder(tf.float32, shape=(3,3))
    Ki = tf.placeholder(tf.float32, shape=(3,3))
    C = tf.placeholder(tf.float32, shape=(3,4))
    cen = tf.placeholder(tf.float32, shape=(3))
    origin = tf.placeholder(tf.float32, shape=(None, 2))
    scaling = tf.placeholder(tf.float32, shape=(None, 1))
    
    ## ****************************Network****************************#일단 scope의 label이 같기에 Network를 공유함을 알 수 있다.
    with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):#render people의 GT값을 넣는다.(background가 흰색으로 적용된 RGB값, normal값, UV 값) 그리고 그거에 맞는 output depth를 얻는다.
        out2_1 = hourglass_refinement(x1,True)
        
    with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):#tiktok dataset에서 i번째 frame의 (background가 흰색으로 적용된 RGB값, Normal estimator에 의해 predict된 normal값, UV 값)그리고 그거에 맞는 output depth를 얻는다.
        out2_1_tk = hourglass_refinement(x1_tk,True)
        
    with tf.variable_scope('hourglass_stack_fused_depth_prediction', reuse=tf.AUTO_REUSE):#tiktok dataset에서 j번째 frame의 (background가 흰색으로 적용된 RGB값, Normal estimator에 의해 predict된 normal값, UV 값)그리고 그거에 맞는 output depth를 얻는다.
        out2_2_tk = hourglass_refinement(x2_tk,True)

        
    ## ****************************Loss RP****************************
    
    nmap1 = dmap_to_nmap(out2_1, Rt, R, Ki, cen, z1, origin, scaling)# depthmap에서 나온 출력인 depth를 외적하여 surface normal로 만들었다.

    total_loss1_d = calc_loss(out2_1,y1,z1)#y1은 groundtruth depth이고, out2_1는 estimator의 결과로 나온 depth이다. 그리고 그 두 결과를 1 차원으로 비교한다.

    total_loss2_d = calc_loss_d_refined_mask(out2_1,y1,z1)#잘 모르겠는데, 위에 loss와 비슷하지만 살짝의 가공을 해서 만든 loss인 것 같다.

    total_loss_n = calc_loss_normal2(nmap1,n1,z1)#surface normal의 GT와 depth를 외적하여 만든 surface normal의 차이를 loss로 쓴다.

    total_loss_rp = 2*total_loss1_d + total_loss2_d + total_loss_n#renderpeople의 loss이다.
    
    ## ****************************Loss TK****************************
    
    nmap1_tk = dmap_to_nmap(out2_1_tk, Rt, R, Ki, cen, z1_tk, origin, scaling)# i번째 frame의 depth를 외적하여 surface normal로 만들었다.
    nmap2_tk = dmap_to_nmap(out2_2_tk, Rt, R, Ki, cen, z2_tk, origin, scaling)# j번째 frame의 depth를 외적하여 surface normal로 만들었다.

    total_loss_n_tk = calc_loss_normal2(nmap1_tk,n1_tk,z1_tk)+calc_loss_normal2(nmap2_tk,n2_tk,z2_tk)#일단 i번째의 n1_tk 즉, normal estimator에서 나온 결과와 nmap1_tk 즉, depth에 의해 유도된 normal을 calc_loss_normal2로 loss를 만들고 이를 j번째 frame에도 똑같이 적용해서 이 두 loss를 더한다.
    
    loss3d,loss2d,PC2p,PC1_2 = compute_dp_tr_3d_2d_loss2(out2_1_tk,out2_2_tk,
                                                         i_r1_c1_r2_c2[0,...],i_limit[0,...],
                                                         C,R,Rt,cen,K,Ki,origin,scaling)#그리고 i번째, j번째 frame의 depth값을 각각 사용해서도 loss를 구하는데, loss3d만 쓰이는 것 같다. 각 estimate된 depth에서 i번째 frame에서 3d reconstruction되고, warping function으로 warping된 후 j번째 frame의 해당 3D reconstruction point과의 차이를 제곱하여 loss를 구했다.

    total_loss_tk = total_loss_n_tk + 5*loss3d#Ls+5*Lw와 같은 것 같다.

    ## ****************************Loss all****************************
    total_loss = total_loss_rp+total_loss_tk
    
    ## ****************************optimizer****************************
    train_step = tf.train.AdamOptimizer(learning_rate=0.001,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=0.1,
                                        use_locking=False,
                                        name='Adam').minimize(total_loss)

##  ********************** initialize the network ********************** 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(graph=refineNet_graph, config=config)#graph를 초기화한다.
with sess.as_default():
    with refineNet_graph.as_default():
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
#        saver = tf.train.import_meta_graph(pre_ck_pnts_dir+'/model_'+model_num+'/model_'+model_num+'.ckpt.meta')
#        saver.restore(sess,pre_ck_pnts_dir+'/model_'+model_num+'/model_'+model_num+'.ckpt')
        print("Model restored.")
        
##  ********************** make the output folders ********************** 
ck_pnts_dir = "/home/ug_psh/HDNet_torch_ami/training_progress/tensorflow/model/DepthEstimator"
log_dir = "/home/ug_psh/HDNet_torch_ami/training_progress/tensorflow/"
Vis_dir  = "/home/ug_psh/HDNet_torch_ami/training_progress/tensorflow/visualization/HDNet/tiktok/"
Vis_dir_rp  = "/home/ug_psh/HDNet_torch_ami/training_progress/tensorflow/visualization/DepthEstimator/Tang/"

if not gfile.Exists(ck_pnts_dir):
    print("ck_pnts_dir created!")
    gfile.MakeDirs(ck_pnts_dir)

if not gfile.Exists(Vis_dir):
    print("Vis_dir created!")
    gfile.MakeDirs(Vis_dir)
    
if not gfile.Exists(Vis_dir_rp):
    print("Vis_dir_rp created!")
    gfile.MakeDirs(Vis_dir_rp)
    
if (path.exists(log_dir+"trainLog.txt")):
    os.remove(log_dir+"trainLog.txt")

    
##  ********************** Run the training **********************     
for itr in range(ITERATIONS):
    (X_1, X1, Y1, N1, 
     Z1, DP1, Z1_3,frms) = get_renderpeople_patch(rp_path, BATCH_SIZE, RP_image_range, 
                                                  IMAGE_HEIGHT,IMAGE_WIDTH)#renderpeople에서 GT를 가져온다.
    (X_1_tk, X1_tk, N1_tk, Z1_tk, DP1_tk, Z1_3_tk, 
     X_2_tk, X2_tk, N2_tk, Z2_tk, DP2_tk, Z2_3_tk, 
     i_r1_c1_r2_c2_tk, i_limit_tk, 
     frms_tk, frms_neighbor_tk) = get_tiktok_patch(tk_path, BATCH_SIZE, IMAGE_HEIGHT,IMAGE_WIDTH)#TikTok data에서 미리 정의한 correspondence라던지 i,j frame 쌍이라던지 그러한 요소들을 가져온다.


    (_,loss_val,prediction1,nmap1_pred, 
     prediction1_tk,nmap1_pred_tk,PC2pn,PC1_2n) = sess.run([train_step,total_loss,out2_1,
                                                           nmap1,out2_1_tk,nmap1_tk,PC2p,PC1_2],
                                                  feed_dict={x1:X_1,y1:Y1,n1:N1,z1:Z1,
                                                             Rt:Rt1n, Ki:Ki1n,cen:cen1n, R:R1n,
                                                             origin:origin1n,scaling:scaling1n,
                                                             x1_tk:X_1_tk,n1_tk:N1_tk,z1_tk:Z1_tk,
                                                             x2_tk:X_2_tk,n2_tk:N2_tk,z2_tk:Z2_tk,
                                                             i_r1_c1_r2_c2:i_r1_c1_r2_c2_tk,
                                                             i_limit:i_limit_tk})#위에서 정의한 graph를 실행한다.
    if itr%10 == 0:
        f_err = open(log_dir+"trainLog.txt","a")
        f_err.write("%d %g\n" % (itr,loss_val))
        f_err.close()
        print("")
        print("iteration %3d, depth refinement training loss is %g." %(itr,  loss_val))
        
    if itr % 100 == 0:
        # visually compare the first sample in the batch between predicted and ground truth
        fidx = [int(frms[0])]
        
        write_prediction(Vis_dir_rp,prediction1,itr,fidx,Z1);
        
        write_prediction_normal(Vis_dir_rp,nmap1_pred,itr,fidx,Z1)
        save_prediction_png (prediction1[0,...,0],nmap1_pred[0,...],X1,Z1,Z1_3,Vis_dir_rp,itr,fidx,1)
        fidx = [int(frms_tk[0])]
        write_prediction(Vis_dir,prediction1_tk,itr,fidx,Z1_tk);
        write_prediction_normal(Vis_dir,nmap1_pred_tk,itr,fidx,Z1_tk)
        save_prediction_png (prediction1_tk[0,...,0],nmap1_pred_tk[0,...],X1_tk,Z1_tk,Z1_3_tk,Vis_dir,itr,fidx,1)

    if itr % 10000 == 0 and itr != 0:
        save_path = saver.save(sess,ck_pnts_dir+"/model_"+str(itr)+"/model_"+str(itr)+".ckpt")






