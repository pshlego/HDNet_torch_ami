import tensorflow as tf # tensorflow import
import numpy as np # python에서 벡터, 행렬 등 수치 연산을 수행하는 선형대수 라이브러리
import skimage.data # skimage는 이미지 처리하기 위한 파이썬 라이브러리
from PIL import Image, ImageDraw, ImageFont # PIL은 파이썬 인터프리터에 다양한 이미지 처리와 그래픽 기능을 제공하는 라이브러리
import math # 수학 관련 함수들이 들어있는 라이브러리
from tensorflow.python.platform import gfile # open()이랑 같고, tensorflow용 파일 입출력 함수
import scipy.misc # scipy에서 기타 함수 https://docs.scipy.org/doc/scipy/reference/misc.html
from utils.vector import cross #외적 함수

from functions import showOperation
IMAGE_HEIGHT = 256 # 이미지 가로 크기
IMAGE_WIDTH = 256 # 이미지 세로 크기

# ***************************************************************************************************** when batch_size=8
def depth2points3D_MB(output, Rt, Ki, cen, z_r, origin, scaling):
    
    # Rt and Ki are 3*3, cen is 1*3, z_r and output are B*256*256, origin is B*2, scaling is B*1
    # point3D is 3*N
    
    ones_mat = tf.cast(tf.ones_like(z_r), tf.bool)#8 x 256 x 256 x 1, tf.ones_like(z_r)은 z_r과 같은 형태의 모든 요소가 true인 matrix이고, tf.bool로 typecasting한다.z_r은 B*256*256인데 B가 batch의 수라고 추정된다.
    indices = tf.where(ones_mat) #  None x 4, tf.where(ones_mat)은 ones_mat에서 true인 부분의 위치를 출력해주는 함수이다. 그런데 모든 위치가 true이기에 모든 위치에 대한 index가 만들어진다.
#     indices = tf.where(z_r)
    
    good_output = tf.where(z_r, output, 1000000*tf.ones_like(output))#8 x 256 x 256 x 1,true인 부분은 ouput의 해당 위치의 값으로 채워지고, false인 부분은 1000000으로 채워진다.
    Dlambda = tf.gather_nd(good_output,indices) # None, tf.gather_nd(params, indices, name=None),indices에 따라 indices의 위치에 있는 값들을 good_output에서 모은다.

    num_of_points = tf.shape(Dlambda)[0] #None=524288, Dlambda의 행의 shape을 정수형 tensor로 반환.
    num_of_batches = tf.shape(z_r)[0] #B=8, z_r의 행의 shape를 정수형 tensor로 반환 
    num_of_points_in_each_batch = tf.cast(tf.divide(num_of_points,num_of_batches),tf.int32) #65536, point의 개수를 batch의 수로 나누고, int32로 typecasting한다. 즉, 한 batch에 있는 point의 개수를 뜻한다. N/B
    
    
    Dlambda_t = tf.reshape(Dlambda,[1,num_of_points]) # 1 x None, None x 1이었는데, transpose하여 1 x None로 만들었다.
    Dlambda3 = tf.concat([Dlambda_t,Dlambda_t],0) # 행으로 concat하여 2 x None으로 만듬
    Dlambda3 = tf.concat([Dlambda3,Dlambda_t],0) # 3 x None
        
    
    idx = tf.cast(indices, tf.float32)# None x 4의 indices를 float32로 typecasting했다.

    row_of_ones = tf.ones([1, num_of_points], tf.float32) # 1 x None의 요소가 모두 1로 이루어진 matrix
    
# dividing xy and the batch number    
    bxy = idx # None x 4
    b = tf.transpose(bxy[...,0]) # None, 첫번째 열을 선택 후 transpose했다.
    batches = tf.reshape(b,[1,num_of_points]) # 1 x None, reshape해서 batches에 저장했다.
    xy = tf.transpose(tf.reverse(bxy[...,1:3],[1])) # 2 x None, column 3개 중에 뒤에 2개(1,2)를 선택하고, tf.reverse로 [1] 즉, 열을 반전시킨다. 그리고 나서 transpose를 한다.
# tiling the scaling to match the data
    scaling2 = tf.reshape(scaling, [num_of_batches,1])# 8 x 1, scaling값을 [8,1]로 reshape한다.
    tiled_scaling = tf.tile(scaling2, [tf.constant(1),num_of_points_in_each_batch])#8 x None, tf.tile 함수는 주어진 텐서를 multiplies 만큼 이어붙이는 함수이다.
    scaling_row = tf.reshape(tiled_scaling,[1,num_of_points])#1XNone, tiled scaling을 또 reshape한다.   
    scaling_2_rows = tf.concat([scaling_row,scaling_row],0)#concat으로 이어붙여서 2xNone이 된다.

# scaling the input 
    scaled_xy = tf.multiply(xy, scaling_2_rows)#2xNone, tf.multifly는 각 요소별로 곱하는 것이다. scaling factor를 각 요소에 곱한다.

# dividing the origin 0 and origin 1 of the origin 
    origin0 = origin[...,0]#ixjxk에서 ixj는 모두 포함하고 k번째 index 기준으로 k번째 index가 0인 것을 가져오는 것
    origin0 = tf.reshape(origin0,[num_of_batches,1])#8 x 1, origin0을 [B,1]로 reshape한다.[[838]...]
    origin1 = origin[...,1]#ixjxk에서 ixj는 모두 포함하고 k번째 index 기준으로 k번째 index가 1인 것을 가져오는 것
    origin1 = tf.reshape(origin1,[num_of_batches,1])#8 x 1, origin1을 [B,1]로 reshape한다.[[48]...]
    
# tiling the origin0 to match the data
    tiled_origin0= tf.tile(origin0, [tf.constant(1),num_of_points_in_each_batch])#B x None/B, tf.tile 함수는 주어진 텐서를 multiplies 만큼 이어붙이는 함수이다.
    origin0_row = tf.reshape(tiled_origin0,[1,num_of_points])# tiled_origin0을 1 x None으로 reshape한다.
    
# tiling the origin1 to match the data    
    tiled_origin1= tf.tile(origin1, [tf.constant(1),num_of_points_in_each_batch]) #B x None/B, tf.tile 함수는 주어진 텐서를 multiplies 만큼 이어붙이는 함수이다.
    origin1_row = tf.reshape(tiled_origin1,[1,num_of_points]) # tiled_origin0을 1xNone으로 reshape한다.

# concatinating origin 0 and origin1 tiled 
    origin_2_rows = tf.concat([origin0_row,origin1_row],0)# concat으로 행으로 이어붙였다. 2xNone이다.
    
# computing the translated and scaled xy
    xy_translated_scaled = tf.add(scaled_xy ,origin_2_rows) # 2 x None, 둘이 각 요소끼리 더한다.
    
         
    xy1 = tf.concat([xy_translated_scaled,row_of_ones],0) #밑에가 모두 1인 3xNone matrix인데 생각해보면 이게 homogeneus representation인 것 같다.
    
    cen1 = tf.multiply(row_of_ones,cen[0])#1xNone인데 모든 요소가 cen[0]인 matrix
    cen2 = tf.multiply(row_of_ones,cen[1])#1xNone인데 모든 요소가 cen[1]인 matrix
    cen3 = tf.multiply(row_of_ones,cen[2])#1xNone인데 모든 요소가 cen[2]인 matrix
    
    cen_mat = tf.concat([cen1,cen2],0)
    cen_mat = tf.concat([cen_mat,cen3],0)# 3 x None, 결국 1번째 행은 cen[0], 결국 2번째 행은 cen[1], 결국 3번째 행은 cen[2]인 3xN인 center matrix를 만든다.
    
    Rt_Ki = tf.matmul(Rt,Ki)#3 x 3,Rt는 그냥 identity matrix이고, Ki가 K 카메라 intrinsic camera parameter의 inverse matrix이다.
    Rt_Ki_xy1 = tf.matmul(Rt_Ki,xy1)#3 x None, 이건 그냥 그 image 좌표 매트릭스랑 카메라 인트린식 좌표 매트릭스랑 곱한거 
    
    point3D = tf.add(tf.multiply(Dlambda3,Rt_Ki_xy1),cen_mat)#3 x None matrix이다. Dlamda3가 깊이인 것 같다. 그리거 모두 곱하고 cen_mat를 더해줘서 최종적으로 reconstruction한 3D point가 나온다.
    
    return point3D, batches#point3D는 3xN이다. 그리고  1 x None, indicies를 batches에 저장했던 것도 출력한다.

# *********************************************************************************************************

def dmap_to_nmap(Y, Rt, R, Ki, cen, Z, origin, scaling):#Y=Batchsize x Image_Height x Image_Width x 1
    BATCH_SIZE = tf.shape(Y)[0]#8, Y의 행의 구조를 1-d 정수형 텐서로 반환한다. Y의 행이 batchsize인가보다.
    IMAGE_HEIGHT = tf.shape(Y)[1]#256, Y의 열의 구조를 1-d 정수형 텐서로 반환한다. Y의 열이 image_height인가보다.
    
    p3d, batches = depth2points3D_MB(Y, Rt, Ki, cen, Z, origin, scaling)# p3d: 3 x None, batches: 1 x None, 위에서 사용한 3D reconstruction 함수인 depth2points3D_MB를 사용하여 3xN과 1xN인 출력을 얻었다.
    
    p3d_map1 = tf.reshape(p3d[0,...],[BATCH_SIZE,IMAGE_HEIGHT,IMAGE_HEIGHT,1])#원래 None=B*HEIGHT*WIDTH였는데, 3xNone에서 첫번째 행을 선택해서 1xNone을 BxHeightxWidthx1로 reshape한다.
    p3d_map1 = tf.where(Z,p3d_map1,tf.zeros_like(p3d_map1))#그리고 tf.where()을 사용해서 Z가 true이면 그 위치에 해당하는 p3d_map1의 값을 넣고, false이면 0을 채운다.
    p3d_map2 = tf.reshape(p3d[1,...],[BATCH_SIZE,IMAGE_HEIGHT,IMAGE_HEIGHT,1])#원래 None=B*HEIGHT*WIDTH였는데, 3xNone에서 두번째 행을 선택해서 1xNone을 BxHeightxWidthx1로 reshape한다.
    p3d_map2 = tf.where(Z,p3d_map2,tf.zeros_like(p3d_map2))#그리고 tf.where()을 사용해서 Z가 true이면 그 위치에 해당하는 p3d_map2의 값을 넣고, false이면 0을 채운다.
    p3d_map3 = tf.reshape(p3d[2,...],[BATCH_SIZE,IMAGE_HEIGHT,IMAGE_HEIGHT,1])#원래 None=B*HEIGHT*WIDTH였는데, 3xNone에서 세번째 행을 선택해서 1xNone을 BxHeightxWidthx1로 reshape한다.
    p3d_map3 = tf.where(Z,p3d_map3,tf.zeros_like(p3d_map3))#그리고 tf.where()을 사용해서 Z가 true이면 그 위치에 해당하는 p3d_map3의 값을 넣고, false이면 0을 채운다.

    pcmap = tf.concat([p3d_map1,p3d_map2],3)
    pcmap = tf.concat([pcmap,p3d_map3],3)#8 x 256 x 256 x 3, 그리고 위에서 구한 BxHeightxWidthx1의 형태인 p3d_map1, p3d_map2, p3d_map3을 4번째 축을 기준으로 이어 붙여서 BxHeightxWidthx3을 만든다.

    pcx_1 = tf.roll(pcmap, shift=-1, axis=2)#8 x 256 x 256 x 3, BxHeightxWidthx3에서 width 축이 왼쪽으로 shift한다.
    pcy1 = tf.roll(pcmap, shift=1, axis=1)#8 x 256 x 256 x 3, BxHeightxWidthx3에서 Height 축이 오른쪽으로 shift한다.

    pcx_1_pc = pcx_1 - pcmap;#8 x 256 x 256 x 3, 그리고 기존에 shift하기 전 matrix에서 뺀다.BxHeightxWidthx3
    pcy1_pc = pcy1 - pcmap; #8 x 256 x 256 x 3, 그리고 기존에 shift하기 전 matrix에서 뺀다.BxHeightxWidthx3

    new_normal_map = cross(pcx_1_pc, pcy1_pc)#BxHeightxWidthx3, 뺀 값들을 외적한다. BxHeightxWidthx3이다.
    n = new_normal_map#외적한 법선벡터를 n에 저장한다.
    output_mask = tf.abs(n) < 1e-5#BxHeightxWidthx3,tf.abs()는 절댓값을 return하는데, 절댓값이 법선 벡터의 크기가 0.00001보다 작으면 1이다. 
    output_no0 = tf.where(output_mask, 1e-5*tf.ones_like(n), n)#BxHeightxWidthx3, output_mask가 1인 위치에는 1e-5를 넣고, 0인 위치에는 n값을 넣는다. BxHeightxWidthx3
    output_mag = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(output_no0),3)),-1)#BxHeightxWidthx1, tf.square()로 인해 모든 요소가 제곱이 됐고, tf.reduce_sum()을 통해 4번재 축을 기준으로 다 더해져서 BxHeightxWidth이 되었다. 그리고 tf.sqrt로 근호를 씌우고,tf.expand_dims(,-1)으로 마지막 축을 하나 더 만들어서 BxHeightxWidthx1가 되었다.
    n = tf.truediv(output_no0,output_mag)#ouput_mag를 ouput_no0에 모두 나누었다. 그래서 shape은 BxHeightxWidthx3

    n1 = n[...,0]#마지막 축을 기준으로 0번째이기에 shape은 BxHeightxWidthx1이다.
    n1 = tf.reshape(n1,[1,BATCH_SIZE*IMAGE_HEIGHT*IMAGE_HEIGHT])#1 x 524288, 1xB*Height*Width로 reshape한다.
    n2 = n[...,1]#마지막 축을 기준으로 1번째이기에 shape은 BxHeightxWidthx1이다.
    n2 = tf.reshape(n2,[1,BATCH_SIZE*IMAGE_HEIGHT*IMAGE_HEIGHT])#1 x 524288, 1xB*Height*Width로 reshape한다.
    n3 = n[...,2]#마지막 축을 기준으로 2번째이기에 shape은 BxHeightxWidthx1이다.
    n3 = tf.reshape(n3,[1,BATCH_SIZE*IMAGE_HEIGHT*IMAGE_HEIGHT])#1 x 524288, 1xB*Height*Width로 reshape한다.

    n_vec_all = tf.concat([n1,n2],0)
    n_vec_all = tf.concat([n_vec_all,n3],0)#3 x 524288, 결국 concat해서 3xB*Height*Width가 된다.

    n_vec_all_rotated = tf.matmul(R,n_vec_all)#3 x 524288, rotated function이 3x3이기에 shape는 3xB*Height*Width가 된다.

    n1v = n_vec_all_rotated[0,...]#행을 기준으로 0번째 이기에 1xB*Height*Width가 된다.
    n1v = tf.reshape(n1v,[BATCH_SIZE,IMAGE_HEIGHT,IMAGE_HEIGHT,1])#8 x 256 x 256 x 1, BxHeightxWidthx1로 reshape한다.
    n2v = n_vec_all_rotated[1,...]#행을 기준으로 1번째 이기에 1xBxHeightxWidth가 된다.
    n2v = tf.reshape(n2v,[BATCH_SIZE,IMAGE_HEIGHT,IMAGE_HEIGHT,1])#8 x 256 x 256 x 1, BxHeightxWidthx1로 reshape한다.
    n3v = n_vec_all_rotated[2,...]#행을 기준으로 2번째 이기에 1xBxHeightxWidth가 된다.
    n3v = tf.reshape(n3v,[BATCH_SIZE,IMAGE_HEIGHT,IMAGE_HEIGHT,1])#8 x 256 x 256 x 1, BxHeightxWidthx1로 reshape한다.

    n_rotated = tf.concat([n1v,n2v],3)
    n_rotated = tf.concat([n_rotated,n3v],3)#8 x 256 x 256 x 3, 결국 concat해서 BxHeightxWidthx3이 된다.
    n = n_rotated#돌아간 결과를 n에 저장한다.
    output_mask = tf.abs(n) < 1e-5#8 x 256 x 256 x 3, tf.abs()는 절댓값을 return하는데, 절댓값이 법선 벡터의 크기가 0.00001보다 작으면 1이다.
    output_no0 = tf.where(output_mask, 1e-5*tf.ones_like(n), n)#8 x 256 x 256 x 3, output_mask가 1인 위치에는 1e-5를 넣고, 0인 위치에는 n값을 넣는다. BxHeightxWidthx3
    output_mag = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(output_no0),3)),-1)#8 x 256 x 256 x 1, tf.square()로 인해 모든 요소가 제곱이 됐고, tf.reduce_sum()을 통해 4번재 축을 기준으로 다 더해져서 BxHeightxWidth이 되었다. 그리고 tf.sqrt로 근호를 씌우고,tf.expand_dims(,-1)으로 마지막 축을 하나 더 만들어서 BxHeightxWidthx1가 되었다.
    n = tf.truediv(output_no0,output_mag)#ouput_mag를 ouput_no0에 모두 나누었다. 그래서 shape은 BxHeightxWidthx3
    
    return n#법선 벡터를 출력으로 내보낸다.
