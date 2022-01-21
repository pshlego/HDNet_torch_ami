import numpy as np      #numpy 라이브러리
import skimage.data     #이미지처리에 특화된 Python 이미지 라이브러리, Numpy배열로 동작해서 이미지 객체를 처리함/
from PIL import Image, ImageDraw, ImageFont # PIL은 파이썬 인터프리터에 다양한 이미지 처리와 그래픽 기능을 제공하는 라이브러리
import math # 수학 관련 함수들이 들어있는 라이브러리
import scipy.misc # scipy에서 기타 함수 https://docs.scipy.org/doc/scipy/reference/misc.html
import matplotlib.pyplot as plt
import os.path
import os
import pdb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(1)
#corr_mat, i_r1_c1_r2_c2.txt, i_limit.txt가 무엇인지는 https://github.com/yasaminjafarian/HDNet_TikTok/blob/main/training/README.md 에 있다.
## ******************************Write prediction************************************

def write_matrix_txt(a,filename):   #입력된 배열과 파일 이름을 가지고 행렬로 바꾼 뒤 txt파일로 저장해줌. test_data -> infer.out의 txt파일들이 이를 이용해 만들어진다.
    mat = np.matrix(a)  #a 입력으로 list입력들어오면 배열로 바꿔줌. a=[[0, 1, 2], [3, 4, 5]] 꼴 가능. 2x3 array
    with open(filename,'wb') as f:  #b는 파일을 binary mode로 연다. w는 쓰기모드 //with open(파일 경로, 모드) as 파일 객체: with as구문을 빠져나가게 되면 자동으로 close()함수 호출해 파일 닫음.
        for line in mat:            #mat의 element개수만큼 반복, line이 변수. 확실하지 않음.
            np.savetxt(f, line, fmt='%.5f')     #텍스트파일로 저장, f: 파일이름이나 파일handle, line: txt로 저장될 데이터. fmt:포맷.소수점 아래 5자리까지 표시.
            
def write_prediction(Vis_dir,prediction,i,idx,Z_r1):    #예측을 작성?
    image_tensor  = np.asarray(prediction)      #이미지 텐서. prediction으로 뭐가 들어오는지 파악 못함. asarray는 입력을 array로 바꿔주는 array와 동일 기능. but 데이터타입 옵션을 주면 동일한 데이터타입이어야 카피가 됨.
    num = 0  
    image = image_tensor[num,...,0] #이미지 = (0...0) 배열
    mask = Z_r1[num,...,0]>0        #
    write_matrix_txt(image,Vis_dir+"STEP%07d_frame%07d_DEPTH.txt" % (i,idx[0]))     #Vis_dir = test_data->infer_out폴더 (?), 이미지 배열을 
    
def write_prediction_normal(Vis_dir,prediction,i,idx,Z_r1):
    image_tensor  = np.asarray(prediction)  #prediction 이라는 내용의 배열 생성
    image = image_tensor[0,...] #이미지 = 이미지 텐서(배열)
    image_mag = np.expand_dims(np.sqrt(np.square(image).sum(axis=2)),-1)    #이미지_mag = root(이미지 각 배열요소 제곱하고 z축에 대해 합함.) y축에 대해 dimension 추가 -1, 1 동일 의미
    image_unit = np.divide(image,image_mag) #divide는 요소별 나눗셈 실행
    write_matrix_txt(image_unit[...,0],Vis_dir+"STEP%07d_frame%07d_NORMAL_1.txt" % (i,idx[0]))  #infer_out 폴더 안에 STEP 폴더? 없지 않나 %07d = 0을 7개 채우고 i, idx[0]을 오른쪽부터 채워넣음
    write_matrix_txt(image_unit[...,1],Vis_dir+"STEP%07d_frame%07d_NORMAL_2.txt" % (i,idx[0]))  #
    write_matrix_txt(image_unit[...,2],Vis_dir+"STEP%07d_frame%07d_NORMAL_3.txt" % (i,idx[0]))  #

# **********************************************************************************************************
def nmap_normalization(nmap_batch): #nmap normalization
    image_mag = np.expand_dims(np.sqrt(np.square(nmap_batch).sum(axis=2)),-1)#np.square()가 nmap_batch의 모든 요소를 제곱하고 axix=2로 sum을 해서 3 channel 값을 모두 더하고 np.axis_dims(,-1)로 한 차원 더 증가시킨다. heightxwidthx3->heightxwidth->heightxwidthx1로 바꾸는 것이다.
    image_unit = np.divide(nmap_batch,image_mag)#그리고 nmap_batch를 위에서 구한 image_mag로 나누어 normalization한다.
    return image_unit  #normalization한 결과를 ouput으로 한다.

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))#Image.new를 사용해서 RGB로 (im1.width + im2.width, im1.height)의 shape을 가진 Image인 dst를 만들고
    dst.paste(im1, (0, 0))#im1은 (0, 0)을 기준점으로 paste하고
    dst.paste(im2, (im1.width, 0))#im2은 (im1.width, 0)을 기준점으로 paste한다.
    return dst #그 결과를 출력한다.

def save_prediction_png (image,imagen,X,Z,Z3,Vis_dir,i,idx,perc):
    imagen=nmap_normalization(imagen)#이건 estimator에서 나온 값이 아니라 depth에서 추정한 normal값이다. 그리고 nmap_normalization에 넣어서 normalization한다.
    data_name = "salam"
    depth_map = image*Z[0,...,0]#image는 depth estimator의 결과이다.
    normal_map = imagen*Z3[0,...]#mask의 값이 저장된 matrix인 z3의 0번째 batch의 값을 imagen에 요소별로 곱하여 background의 normal 값을 0으로 없앤다.
    min_depth = np.amin(depth_map[depth_map>0])#np.amin은 배열의 최소값을 반환한다. 그래서 depth_map에서 0보다 큰 것 중 최솟값을 반환하여 min_depth에 저장한다.
    max_depth = np.amax(depth_map[depth_map>0])*perc#np.amax은 배열의 최대값을 반환한다. 그래서 depth_map에서 0보다 큰 것 중 최대값을 반환하여 max_depth에 저장한다.
    depth_map[depth_map < min_depth] = min_depth#depth_map에서 0보다 작은 것들을 min_depth로 채운다.
    depth_map[depth_map > max_depth] = max_depth#depth_map에서 max_depth보다 큰 것들을 max_depth로 채운다.
    normal_map_rgb = -1*normal_map#normal_map에 -1을 곱한다.
    normal_map_rgb[...,2] = -1*((normal_map[...,2]*2)+1)#xyz값중에 z값에 2를 곱하고 1을 더한 후 다시 -1을 씌워서 normal_map_rgb[...,2]=-2*normal_map(...,2)-1임을 의미한다.
    normal_map_rgb = np.reshape(normal_map_rgb, [256,256,3]);#그리고 256x256x3으로 reshape한다.
    normal_map_rgb = (((normal_map_rgb + 1) / 2) * 255).astype(np.uint8);#그리고 (normal_map_rgb+1)/2을 한 후 255를 곱하고 np.uint8이라는 type으로 만들어서  normal_map_rgb에 저장한다. normalization한 값을 0~254의 pixel 값으로 만든다.
    plt.imsave(Vis_dir+data_name+"_depth.png", depth_map, cmap="hot")#depth를 cmap="hot" 즉, heatmap의 방식으로 저장한다.
    plt.imsave(Vis_dir+data_name+"_normal.png", normal_map_rgb) #그리고 Vis_dir+data_name+_normal.png로 저장한다.
    d = np.array(scipy.misc.imread(Vis_dir+data_name+"_depth.png"),dtype='f')#그리고 위에서 저장한 png 파일을 다시 읽어서 d에 저장하고
    d = np.where(Z3[0,...]>0,d[...,0:3],255.0)#mask의 값이 0보다 큰 경우 d의 RGB를 저장하고 그렇지 않으면 255를 넣어서 흰색으로 만든다.
    n = np.array(scipy.misc.imread(Vis_dir+data_name+"_normal.png"),dtype='f')#그리고 위에서 저장한 png 파일을 다시 읽어서 n에 저장하고
    n = np.where(Z3[0,...]>0,n[...,0:3],255.0)#mask의 값이 0보다 큰 경우 n의 RGB를 저장하고 그렇지 않으면 255를 넣어서 흰색으로 만든다.
    final_im = get_concat_h(Image.fromarray(np.uint8(X[0,...])),Image.fromarray(np.uint8(d)))#Image.fromarray()는 numpy 배열을 PIL 이미지로 변환하는 함수이다. 따라서 batch color(X를 의미한다)의 0번째 batch의 이미지와 d을 나란이 concat한다.
    final_im = get_concat_h(final_im,Image.fromarray(np.uint8(n)))#Image.fromarray()는 numpy 배열을 PIL 이미지로 변환하는 함수이다. 따라서 batch color(X를 의미한다)의 0번째 batch의 이미지와 d을 나란이 concat한것에 또 n을 나란히 concat한다.
    final_im.save(Vis_dir+"STEP%07d_frame%07d_results.png" % (i,idx[0]))#그리고 다음과 같은 파일명으로 저장한다. i와 idx[0]은 각각 iteration 횟수와 frms이다.
    os.remove(Vis_dir+data_name+"_depth.png")#위에서 저장했던 png파일을 삭제한다.
    os.remove(Vis_dir+data_name+"_normal.png")#위에서 저장했던 png파일을 삭제한다.
    
def save_prediction_png_normal (imagen,X,Z,Z3,Vis_dir,i,idx):#
    imagen = nmap_normalization(imagen)#imagen은 normal estimator에서 나온 normal 값이다. 그리고 nmap_normalization에 넣어서 normalization한다.
    data_name = "salam"
    normal_map = imagen*Z3[0,...]#mask의 값이 저장된 matrix인 z3의 0번째 batch의 값을 imagen에 요소별로 곱하여 background의 normal 값을 0으로 없앤다.
    normal_map_rgb = -1*normal_map#그리고 -1을 곱한다.
    normal_map_rgb[...,2] = -1*((normal_map[...,2]*2)+1)#xyz값중에 z값에 2를 곱하고 1을 더한 후 다시 -1을 씌워서 normal_map_rgb[...,2]=-2*normal_map(...,2)-1임을 의미한다.
    normal_map_rgb = np.reshape(normal_map_rgb, [256,256,3]);#그리고 256x256x3으로 reshape한다.
    normal_map_rgb = (((normal_map_rgb + 1) / 2) * 255).astype(np.uint8);#그리고 (normal_map_rgb+1)/2을 한 후 255를 곱하고 np.uint8이라는 type으로 만들어서  normal_map_rgb에 저장한다. normalization한 값을 0~254의 pixel 값으로 만든다.
    plt.imsave(Vis_dir+data_name+"_normal.png", normal_map_rgb)#그리고 Vis_dir+data_name+_normal.png로 저장한다.
    n = np.array(scipy.misc.imread(Vis_dir+data_name+"_normal.png"),dtype='f')#그리고 위에서 저장한 png 파일을 다시 읽어서 n에 저장하고
    n = np.where(Z3[0,...]>0,n[...,0:3],255.0)#mask의 값이 0보다 큰 경우 n의 RGB를 저장하고 그렇지 않으면 255를 넣어서 흰색으로 만든다.
    final_im = get_concat_h(Image.fromarray(np.uint8(X[0,...])),Image.fromarray(np.uint8(n)))#Image.fromarray()는 numpy 배열을 PIL 이미지로 변환하는 함수이다. 따라서 batch color(X를 의미한다)의 0번째 batch의 이미지와 n을 나란이 concat한다.
    final_im.save(Vis_dir+"STEP%07d_frame%07d_results.png" % (i,idx[0]))#그리고 다음과 같은 파일명으로 저장한다. i와 idx[0]은 각각 iteration 횟수와 frms이다.
    os.remove(Vis_dir+data_name+"_normal.png")#위에서 저장했던 png파일을 삭제한다.

## *********************************Read RP trainign data**********************************************
    
max_corr_points = 25000
def read_renderpeople(rp_path, frms, IMAGE_HEIGHT,IMAGE_WIDTH):
    Bsize = len(frms)#batch_size를 의미한다.
    batch_densepose = []#일단 list를 선언해둔다.
    batch_color = []    
    batch_mask  = []
    batch_depth = []
    batch_normal = []
    for b in range(Bsize):#일단 batchsize만큼 for문을 돈다.
        cur_f = int(frms[b])#각 batch에 해당하는 frame 번호를 int로 바꿔서 cur_f에 저장한다.
        name = "%07d" %(cur_f)#그리고 0000012 뭐 이런식으로 name을 만든다.
        batch_densepose.append(scipy.misc.imread(rp_path +'/densepose/'+name+'.png'))#각 name에 맞는 densepose의 이미지를 scipy.misc.imread()로 읽어오고 위에서 미리 정의한 list인 batch_densepose값에 append한다.
        batch_color.append(scipy.misc.imread(rp_path +'/color_WO_bg/'+name+'.png'))#각 name에 맞는 back ground가 없는 이미지를 scipy.misc.imread()로 읽어오고 위에서 미리 정의한 list인 batch_color값에 append한다.
        batch_mask.append(scipy.misc.imread(rp_path +'/mask/'+name+'.png'))#각 name에 맞는 mask의 이미지를 scipy.misc.imread()로 읽어오고 위에서 미리 정의한 list인 batch_mask값에 append한다.
        batch_depth.append(np.genfromtxt(rp_path +'/depth/'+name+'.txt',delimiter=","))#각 name에 맞는 depth의 txt 파일을 np.genfromtxt()로 읽어오고 위에서 미리 정의한 list인 batch_depth값에 append한다.
        cur_normal = np.array(scipy.misc.imread(rp_path +'/normal_png/'+name+'.png'),dtype='f');# 그리고 scipy.misc.imread()를 사용해서 normal의 이미지를 가져오고, float로 저장한다.
        n1 = np.genfromtxt(rp_path +'/normal/'+name+'_1.txt',delimiter=",")#그리고 아마 해당 pixel 위치에 맞는 x, y, z normal vector의 방향을 txt로 저장해둔 것을 np.genfromtxt로 가져오고 n1,n2,n3에 저장한다.
        n2 = np.genfromtxt(rp_path +'/normal/'+name+'_2.txt',delimiter=",")#
        n3 = np.genfromtxt(rp_path +'/normal/'+name+'_3.txt',delimiter=",")#
        cur_normal[...,0] = n1;#cur_normal가 아마 height x width x 3일텐데 마지막 축에 0, 1, 2에 RGB값 대신 n1, n2, n3의 값을 저장하는 것 같다.
        cur_normal[...,1] = n2;
        cur_normal[...,2] = n3;
        batch_normal.append(cur_normal)# 그리고 batch별로 batch_normal에 append한다.
    
    batch_color = np.array(batch_color,dtype='f')#위에서 batch별로 append한 값을 np.array로 float형 배열로 바꾼다.
    batch_mask = np.array(batch_mask,dtype='f')#위에서 batch별로 append한 값을 np.array로 float형 배열로 바꾼다.
    batch_depth = np.array(batch_depth,dtype='f')#위에서 batch별로 append한 값을 np.array로 float형 배열로 바꾼다.
    batch_normal = np.array(batch_normal,dtype='f')#위에서 batch별로 append한 값을 np.array로 float형 배열로 바꾼다.
    batch_densepose = np.array(batch_densepose,dtype='f')#위에서 batch별로 append한 값을 np.array로 float형 배열로 바꾼다.
    
    X1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')# 사실 이걸 하는 의미를 잘 모르겠는데, 일단 for문으로 append한 matrix를 통일하는 것 같다.
    X1 = batch_color#X1에 batch_color를 넣는다.
    Y1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,1),dtype='f')
    Y1[...,0] = batch_depth#그리고 batch_depth의 값은 3차원이었고, 이를 4차원에 저장하기 위해 [...,0]을 사용한 것 같다.
    Z1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,1),dtype='b')# boolean으로 만든다.
    Z1[...,0] = batch_mask > 100# 100 이상의 값들만 true로 z1에 저장한다. 그리고 batch_mask의 값은 3차원이었고, 이를 4차원에 저장하기 위해 [...,0]을 사용한 것 같다.
    Z1[Y1<1.0] = False#그리고 mask의 값들 중에 depth가 너무 얕거나 깊은 point는 False를 넣는다.
    Z1[Y1>8.0] = False
    N1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    N1 = batch_normal
    DP1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')# 아마 UVW 좌표계이고, W는 항상 0일 것이다. https://blender.stackexchange.com/questions/79311/uv-maps-is-it-possible-to-use-the-third-channel-w
    DP1 = batch_densepose
    
    Z1_3 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='b')#mask의 값을 3개의 channel에 모두 다 저장을 한다. 즉, 그냥 1 channel이던 mask를 3 channel로 확장한다.
    Z1_3[...,0]=Z1[...,0]
    Z1_3[...,1]=Z1[...,0]
    Z1_3[...,2]=Z1[...,0]
    
    # make the image with white background.
    X1 = np.where(Z1_3,X1,np.ones_like(X1)*255.0)#true인 경우(mask에 해당하지 않으면) X1을 넣고, 만약 False라면 np.ones_like가 1로 이루어진 matrix이니 255를 곱해서 255를 넣는다.
    N1 = np.where(Z1_3,N1,np.zeros_like(N1))#그리고 true인 경우 (mask에 해당하지 않으면) N1을 넣고, False인 경우 np.zeros_like(N1)을 넣는데, 0을 넣는 것이다.
    Y1 = np.where(Z1,Y1,np.zeros_like(Y1))#그리고 true인 경우 (mask에 해당하지 않으면) Y1을 넣고, False인 경우 np.zeros_like(Y1)을 넣는데, 0을 넣는 것이다.
    
    # shift the depthmap to median 4.중앙값을 4로 가지도록 shift한다는 것 같다.
    Y2 = Y1#일단 Y1의 값을 Y2에 저장해둔다.
    for b in range(Bsize):#Batch size만큼 for문을 돌린다.
        yt = Y1[b,...]#일단 각 batch마다의 Y1 값을 
        yt_n0 = yt[yt>0]#depth가 양수인 경우의 값들만 yt_n0에 저장한다.
        med_yt = np.median(yt_n0)#그리고 np.median으로 yt_n0의 중앙값을 구하고
        yt = yt + 4 - med_yt#그 중앙값을 뺀 후 4를 더해서 4를 중앙값으로 만든다.
        Y2[b,...] = yt#그리고 다시 batch별로 Y2에 저장한다.
    Y1 = Y2
    
    Y1 = np.where(Z1,Y1,np.zeros_like(Y1))#그리고 true인 경우 (mask에 해당하면) Y1을 넣고, False인 경우 np.zeros_like(Y1)을 넣는데, 0을 넣는 것이다.
    
    X_1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,9),dtype='f')#그리고 Bsize x IMAGE_HEIGHT x IMAGE_WIDTH x 9 shape의 matrix를 0으로 만든다.

    X_1[...,0]=X1[...,0]#x_1에 backgound가 white인 값의 정보들을 넣는다. 
    X_1[...,1]=X1[...,1]
    X_1[...,2]=X1[...,2]
    X_1[...,3]=N1[...,0]
    X_1[...,4]=N1[...,1]
    X_1[...,5]=N1[...,2]
    X_1[...,6]=DP1[...,0]
    X_1[...,7]=DP1[...,1]
    X_1[...,8]=DP1[...,2]
    
    return X_1, X1, Y1, N1, Z1, DP1, Z1_3

def get_renderpeople_patch(rp_path,Bsize,image_nums,IMAGE_HEIGHT,IMAGE_WIDTH):#renderpeople(우리가 GT로 사용하기로 했던 data)의 patch를 가져오는 함수이다. image_nums로는 range(0,188)가 들어온다.
    num_of_ims = len(image_nums)#일단 기본적으로 image의 개수를 num_of_ims에 저장한다. num_of_mis = 189
    frms_nums = np.random.choice(num_of_ims, Bsize).tolist()#그리고 0~188중에 batch size만큼 무작위로 숫자를 고르고 tolist()로 고른 숫자를 리스트로 만든다.
    frms= []#일단 frms라는 list를 선언한다.
    for f in range(len(frms_nums)):#0~Bsize-1만큼 for문이 돌아간다.
        frms = frms + [image_nums[frms_nums[f]]]#그래서 각 batch마다 random하게 choice된 값을 frms에 넣은 것 같은데 사실 그냥 imgae_nums가 range 함수의 결과물이라서 frms_nums랑 같다.
    X_1, X1, Y1, N1, Z1, DP1, Z1_3 = read_renderpeople(rp_path, frms, IMAGE_HEIGHT,IMAGE_WIDTH)
    return X_1, X1, Y1, N1, Z1, DP1, Z1_3, frms

## *********************************Read Tk trainign data**********************************************
def read_tiktok(tk_path, frms, IMAGE_HEIGHT,IMAGE_WIDTH):#일단 "../training_data/tiktok_data/"와 corr_mat의 첫번째 열에 있는 값을 batch size만큼 random으로 뽑은 값인 frms가 input이고, IMAGE의 HEIGHT, IMAGE의 WIDTH를 input으로 받는다. 
    Bsize = len(frms)# frms가 Bsize만큼 random으로 뽑은 값이기 때문에 frms의 length는 Bsize이다. 
    batch_densepose = []# 일단 list를 선언해둔다.
    batch_color = []    
    batch_mask  = []
    batch_depth = []
    batch_normal = []
    for b in range(Bsize):# list(range(3))=[0, 1, 2]이기에 b는 0,..Bsize이다.
        cur_f = int(frms[b])#그리고 frms에서 해당하는 batch의 값을 cur_f에 int형으로 저장한다. frms의 각 값들은 frame의 index인 것 같다.
        name = "%07d" %(cur_f)#그리고 '00cur_f'값이 되어 name_i에 저장된다. 예를 들어 cur_f가 17634이면 name = '0017634'
        batch_densepose.append(scipy.misc.imread(tk_path +'/densepose/'+name+'.png'))#각 name에 맞는 densepose의 이미지를 scipy.misc.imread()로 읽어오고 위에서 미리 정의한 list인 batch_densepose값에 append한다.
        batch_color.append(scipy.misc.imread(tk_path +'/color_WO_bg/'+name+'.png'))#각 name에 맞는 back ground가 없는 이미지를 scipy.misc.imread()로 읽어오고 위에서 미리 정의한 list인 batch_color값에 append한다.
        batch_mask.append(scipy.misc.imread(tk_path +'/mask/'+name+'.png'))#각 name에 맞는 mask의 이미지를 scipy.misc.imread()로 읽어오고 위에서 미리 정의한 list인 batch_mask값에 append한다.
        cur_normal = np.array(scipy.misc.imread(tk_path +'/pred_normals_png/'+name+'.png'),dtype='f');# 그리고 scipy.misc.imread()를 사용해서 normal의 이미지를 가져오고, float로 저장한다.
        n1 = np.genfromtxt(tk_path +'/pred_normals/'+name+'_1.txt',delimiter=" ")#그리고 아마 해당 pixel 위치에 맞는 x, y, z normal vector의 방향을 txt로 저장해둔 것을 np.genfromtxt로 가져오고 n1,n2,n3에 저장한다.
        n2 = np.genfromtxt(tk_path +'/pred_normals/'+name+'_2.txt',delimiter=" ")#
        n3 = np.genfromtxt(tk_path +'/pred_normals/'+name+'_3.txt',delimiter=" ")#
        cur_normal[...,0] = n1;#cur_normal가 아마 height x width x 3일텐데 마지막 축에 0, 1, 2에 RGB값 대신 n1, n2, n3의 값을 저장하는 것 같다.
        cur_normal[...,1] = n2;
        cur_normal[...,2] = n3;
        batch_normal.append(cur_normal)# 그리고 batch별로 batch_normal에 append한다.
    batch_color = np.array(batch_color,dtype='f')#위에서 batch별로 append한 값을 np.array로 float형 배열로 바꾼다.
    batch_mask = np.array(batch_mask,dtype='f')#위에서 batch별로 append한 값을 np.array로 float형 배열로 바꾼다.
    batch_normal = np.array(batch_normal,dtype='f')#위에서 batch별로 append한 값을 np.array로 float형 배열로 바꾼다.
    batch_densepose = np.array(batch_densepose,dtype='f')#위에서 batch별로 append한 값을 np.array로 float형 배열로 바꾼다.
    
    X1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')# 사실 이걸 하는 의미를 잘 모르겠는데, 일단 for문으로 append한 matrix를 통일하는 것 같다.
    X1 = batch_color#X1에 batch_color를 넣는다.
    Z1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,1),dtype='b')# boolean으로 만든다.
    Z1[...,0] = batch_mask > 100 # 100 이상의 값들만 true로 z1에 저장한다. 그리고 batch_mask의 값은 3차원이었고, 이를 4차원에 저장하기 위해 [...,0]을 사용한 것 같다.
    N1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')
    N1 = batch_normal
    DP1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='f')# 아마 UVW 좌표계이고, W는 항상 0일 것이다. https://blender.stackexchange.com/questions/79311/uv-maps-is-it-possible-to-use-the-third-channel-w
    DP1 = batch_densepose
    
    Z1_3 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,3),dtype='b')#mask의 값을 3개의 channel에 모두 다 저장을 한다. 즉, 그냥 1 channel이던 mask를 3 channel로 확장한다.
    Z1_3[...,0]=Z1[...,0]
    Z1_3[...,1]=Z1[...,0]
    Z1_3[...,2]=Z1[...,0]
    
    # make the image with white background.
    X1 = np.where(Z1_3,X1,np.ones_like(X1)*255.0)#true인 경우(mask에 해당하지 않으면) X1을 넣고, 만약 false라면 np.ones_like가 1로 이루어진 matrix이니 255를 곱해서 255를 넣는다.
    N1 = np.where(Z1_3,N1,np.zeros_like(N1))#그리고 true인 경우 (mask에 해당하지 않으면) N1을 넣고, False인 경우 np.zeros_like(N1)을 넣는데, 0을 넣는 것이다.
    
    X_1 = np.zeros((Bsize,IMAGE_HEIGHT,IMAGE_WIDTH,9),dtype='f')#그리고 Bsize x IMAGE_HEIGHT x IMAGE_WIDTH x 9 shape의 matrix를 0으로 만든다.

    X_1[...,0]=X1[...,0]#x_1에 backgound가 white인 값의 정보들을 넣는다. 
    X_1[...,1]=X1[...,1]
    X_1[...,2]=X1[...,2]
    X_1[...,3]=N1[...,0]
    X_1[...,4]=N1[...,1]
    X_1[...,5]=N1[...,2]
    X_1[...,6]=DP1[...,0]
    X_1[...,7]=DP1[...,1]
    X_1[...,8]=DP1[...,2]
    
    return X_1, X1, N1, Z1, DP1, Z1_3

def read_correspondences_dp(f,fi,corr_path):#frm,nfrm을 각각 f와 fi로 받았다. 그리고 corr_path는 "../training_data/tiktok_data/"+'/correspondences/'이다.
    
    name_i = '%07d'%(f)# '00f'값이 되어 name_i에 저장된다. 예를 들어 f가 17634이면 name_i = '0017634'
    name_j = '%07d'%(fi)# '00fi'값이 되어 name_j에 저장된다. 예를 들어 fi가 17634이면 name_j = '0017934'
    
    i_r1_c1_r2_c2n = np.array(np.genfromtxt(corr_path+'corrs/'+name_i+'_'+name_j+'_i_r1_c1_r2_c2.txt',delimiter=","))# 그리고 /training_data/tiktok_data/correspondences/corrs/에 '0017634_0017934_i_r1_c1_r2_c2.txt'이런 식으로 저장한 txt 파일의 값을 ,를 기준으로 열을 나누고, enter 기준으로 행을 나눠서 matrix 형태로 i_r1_c1_r2_c2에 저장한다.
    i_r1_c1_r2_c2n = i_r1_c1_r2_c2n.astype('int')#astype은 type을 바꾸는 방법이다. 그래서 모든 matrix 내부의 요소의 값들을 int형으로 바꾼다.
    
    i_r1_c1_r2_c2n_f = np.zeros((max_corr_points,5),dtype='int')#최대 corr_points의 값은 25000이므로 25000x5이고, int형인 0으로 채워져있는 matrix를 만든다.
    i_r1_c1_r2_c2n_f[0:i_r1_c1_r2_c2n.shape[0],:]=i_r1_c1_r2_c2n# 그리고 i_r1_c1_r2_c2n의 값을 i_r1_c1_r2_c2n_f에 같은 위치에 저장한다. 그리고 공백은 0이다. 열은 꽉 차있고, 행이 비어있을 수 있다. 

    i_limitn = np.array(np.genfromtxt(corr_path+'corrs/'+name_i+'_'+name_j+'_i_limit.txt',delimiter=","))#그리고 '_i_limit.txt'에서 해당하는 name_i와 name_j의 i_limit을 ,를 기준으로 열을 나누고, enter 기준으로 행을 나눠서 matrix 형태로 i_limitn에 저장한다. 24x3이다.
    i_limitn = i_limitn.astype('int')#astype()으로 int형으로 바꾼다.
    return i_r1_c1_r2_c2n_f,i_limitn#그리고 output으로 i_r1_c1_r2_c2n_f,i_limitn을 내보낸다.

def get_tiktok_patch(tk_path, Bsize, IMAGE_HEIGHT,IMAGE_WIDTH):#titok data를 사용하려고 한다.
    corr_mat = np.genfromtxt(tk_path +'/correspondences/corr_mat.txt',delimiter=",")#79x5 matrix, np.genfromtxt(,delimiter=",")을 사용하여 "," 구분이 열이고, enter가 행으로 matrix를 받아온다.
    corr_path = tk_path +'/correspondences/'#"../training_data/tiktok_data/"+'/correspondences/'
    num_of_neighbors = np.shape(corr_mat)[1]-1#일단 열이 5이기에 5-1=4
    image_nums = corr_mat[:,0].tolist()#list로 바꿔주는 함수인 tolist() 덕분에 일단 corr_mat의 첫번째 열을 (79,) list로 image_nums에 저장한다.
    num_of_ims = len(image_nums)#len()은 리스트의 크기를 출력하는 함수이기에 79가 나온다.
    frms_nums = np.random.choice(num_of_ims, Bsize).tolist()#그리고 0~78중에 batch size만큼 무작위로 숫자를 고르고 tolis()로 고른 숫자를 리스트로 만든다.
    frms= []#일단 배열을 다 선언해둔다.
    frms_neighbor = []
    bi_r1_c1_r2_c2 = []
    bi_limit = []
    for f in range(len(frms_nums)):#list(range(3))=[0, 1, 2]이다. 따라서 range(batch_size)이기에 batch당 한번씩 for문이 돈다.
        row = frms_nums[f]#결국 corr_mat에서 받아온 matrix에서 무작위로 batch의 size만큼 뽑아낸 행의 위치 중 하나를 row에 저장한다. 
        frm = image_nums[row]#그리고 위에서 뽑은 행의 위치에 해당하는 corr_mat의 첫 번째 열의 값을 frm에 저장한다. ex)17634, 17934, ..
        frms = frms + [frm]#그리고 corr_mat의 첫번째 열, 위에서 뽑은 row의 위치에 있는 값을 frms에 이어서 붙여 저장한다.
        
        neighbor_choice = np.random.choice(num_of_neighbors, 1)[0] + 1#그리고 0,1,2,3 중에 하나를 random으로 뽑고 그 요소에 1을 더해서 neighbor_choice에 저장한다. 즉. 처음에 뽑은 0번째 행을 제외하고 1,2,3,4번째 행 중에 하나를 택하겠다는 것이다.
        nfrm = corr_mat[row,neighbor_choice]#위에서 뽑은 row에 해당하고, 윗줄에서 random으로 뽑은 숫자의 열에 위치하는 요소의 값을 nfrm에 저장한다.
        frms_neighbor = frms_neighbor + [nfrm]# 그리고 저장한 nfrm 값을 frms_neighbor에 저장한다. 즉, frms와 frms_neighbor은 행은 같고, 열은 다른 값이다.
        
        i_r1_c1_r2_c2n_f,i_limitn = read_correspondences_dp(frm,nfrm,corr_path)#그리고 frm과 nfrm, 그리고 corr_path를 보낸다. i_r1_c1_r2_c2n_f,i_limitn을 받아온다.
        bi_r1_c1_r2_c2.append(i_r1_c1_r2_c2n_f)#i_r1_c1_r2_c2n_f을 bi_r1_c1_r2_c2라는 matrix에 계속 append한다. 즉, 계속 받은 값을 이어붙인다.
        bi_limit.append(i_limitn)#i_limitn을 bi_limit라는 matrix에 계속 append한다. 즉, 계속 받은 값을 이어붙인다.
        
    i_r1_c1_r2_c2 = np.zeros((Bsize,max_corr_points,5),dtype='i')#int형 0으로 이루어진 Bsize x 25000 x 5인 matrix를 선언한다.
    i_r1_c1_r2_c2 = bi_r1_c1_r2_c2# 그리고 25000 x 5을 Bsize만큼 이어붙인 bi_r1_c1_r2_c2을 i_r1_c1_r2_c2에 저장한다.
    i_limit = np.zeros((Bsize,24,3),dtype='i')#int형 0으로 이루어진 Bsize x 24 x 3인 matrix를 선언한다.
    i_limit = bi_limit#그리고 24 x 3을 Bsize만큼 이어붙인 bi_limit을 i_limit에 저장한다.
    
    X_1, X1, N1, Z1, DP1, Z1_3 = read_tiktok(tk_path, frms, IMAGE_HEIGHT,IMAGE_WIDTH)#그리고 frms와 frms_neigbor에 해당하는 X_1, X1, N1, Z1, DP1, Z1_3를 출력한다.
    X_2, X2, N2, Z2, DP2, Z2_3 = read_tiktok(tk_path, frms_neighbor, IMAGE_HEIGHT,IMAGE_WIDTH)
    
    return X_1, X1, N1, Z1, DP1, Z1_3, X_2, X2, N2, Z2, DP2, Z2_3, i_r1_c1_r2_c2, i_limit, frms, frms_neighbor
    
## **************************************Get Camera**********************************************
def get_origin_scaling(bbs, IMAGE_HEIGHT):#bbs는 Bsizex4x2인 float형 0으로 이루어진 matrix이다. 그리고 IMAGE_HEIGHT를 입력으로 받는다.
    Bsz = np.shape(bbs)[0]#일단 첫번째 축의 크기를 Bsz에 저장하는데 그냥 batchsize다.
    batch_origin = []#빈 array를 선언한다.
    batch_scaling = []#빈 array를 선언한다.
    
    for i in range(Bsz):
        bb1_t = bbs[i,...] - 1#i번째 batch의 값을 저장한다. 4x2이다. 그리고 모든 요소에서 1을 뺀다.
        bbc1_t = bb1_t[2:4,0:3]#2x2이다. 4개의 행 중에 밑에 행 2개를 bbc1_t에 따로 저장한다.
        
        origin = np.multiply([bb1_t[1,0]-bbc1_t[1,0],bb1_t[0,0]-bbc1_t[0,0]],2)#[2*(bb1_t[1,0]-bbc1_t[1,0]),2*(bb1_t[0,0]-bbc1_t[0,0])]

        squareSize = np.maximum(bb1_t[0,1]-bb1_t[0,0]+1,bb1_t[1,1]-bb1_t[1,0]+1);#가장 큰 값을 정사각형의 한 변으로 한다.
        scaling = [np.multiply(np.true_divide(squareSize,IMAGE_HEIGHT),2)]#(squareSize/IMAGE_HEIGHT)*2
    
        batch_origin.append(origin)#append() 때문에 계속 [x, y]를 batch_origin에 넣어서 [[x,y],[],[],[]..]가 된다.
        batch_scaling.append(scaling)#[[],[],[],[],[]...]이것도 append 때문에 계속 list에 [scaling]을 넣는다.
    
    batch_origin = np.array(batch_origin,dtype='f')#각각을 다시 float형 array로 만들어서 저장한다.
    batch_scaling = np.array(batch_scaling,dtype='f')#
    
    O = np.zeros((Bsz,1,2),dtype='f')
    O = batch_origin
    
    S = np.zeros((Bsz,1),dtype='f')
    S = batch_scaling
    
    return O, S

def get_camera(Bsize,IMAGE_HEIGHT):#일단 IMAGE_HEIGHT와 batch size를 input으로 한다. batch size는 주로 1 또는 8이다.
    C1n = np.zeros((3,4),dtype='f')#data type float로 3x4인 0으로 이루어진 행렬을 만든다.
    C1n[0,0]=1#그리고 각 행렬의 (0,0),(1,1),(2,2)에 1을 넣는다.
    C1n[1,1]=1
    C1n[2,2]=1
    #       1 0 0 0
    # C1n = 0 1 0 0
    #       0 0 1 0
    R1n = np.zeros((3,3),dtype='f')#data type float로 3x3인 0으로 이루어진 행렬을 만든다.
    R1n[0,0]=1#그리고 각 행렬의 (0,0),(1,1),(2,2)에 1을 넣는다.
    R1n[1,1]=1
    R1n[2,2]=1
    
    Rt1n = R1n
    #       1 0 0
    # R1n = 0 1 0
    #       0 0 1
    K1n = np.zeros((3,3),dtype='f')#data type float로 3x3인 0으로 이루어진 행렬을 만든다.K1n은 camera intrinsic parameter이다.
    
    K1n[0,0]=1111.6
    K1n[1,1]=1111.6

    K1n[0,2]=960
    K1n[1,2]=540
    K1n[2,2]=1
    #       1111.6      0 960
    # R1n =      0 1111.6 540
    #            0      0   1
    M1n = np.matmul(np.matmul(K1n,R1n),C1n)#3x4 행렬인데, 마지막 column이 0이고, 앞의 3 column은 K1n과 같다.

    Ki1n = np.linalg.inv(K1n)#K1n matrix를 inverse 시킨 matrix인데, 우리가 3D point reconstruction할 때는 inverse가 필요하기에 inverse한것을 Ki1n에 저장하는 것 같다.
    
    cen1n = np.zeros((3),dtype='f')#1x3의 모든 요소가 float형 0으로 이루어진 matrix
    
    bbs1n_tmp = np.array([[25,477],[420,872],[1,453],[1,453]],dtype='f')#float형이고, 4x2인 matrix이다.흠..이건 bounding box인가?
    bbs1n_tmp = np.reshape(bbs1n_tmp,[1,4,2])#reshape으로 앞에 한 axis를 추가하여 1x4x2이다.
    
    bbs1n = np.zeros((Bsize,4,2),dtype='f')# Bsizex4x2인 float형 0으로 이루어진 matrix이다.
    for b in range(Bsize):#Bsizex4x2인데 각 batch마다 bbs1n_tmp를 모두 저장한다.
        bbs1n[b,...]=bbs1n_tmp[0,...]
           
    origin1n, scaling1n = get_origin_scaling(bbs1n, IMAGE_HEIGHT)
    
    return origin1n, scaling1n, C1n, cen1n, K1n, Ki1n, M1n, R1n, Rt1n