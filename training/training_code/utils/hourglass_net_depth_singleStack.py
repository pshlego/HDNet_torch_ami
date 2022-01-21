import numpy as np # python에서 벡터, 행렬 등 수치 연산을 수행하는 선형대수 라이브러리
import tensorflow as tf # tensorflow import


VARIABLE_COUNTER = 0 # 변수 0으로 초기화


NUM_CH = [64,128,256,512,1024]#각 레이어의 채널 개수
KER_SZ = 3 #kernal size가 3이다. 즉, 연산을 수행할 때 윈도우의 크기

########## TABLE 1 ARCHITECTURE PARAMETERS ######### 이건 무엇인지 잘 모르겠다.
color_encode = "ABCDEFG"
block_in  = [128,128,128,128,256,256,256]
block_out = [64,128,128,256,256,256,128]
block_inter = [64,32,64,32,32,64,32]
block_conv1 = [1,1,1,1,1,1,1]
block_conv2 = [3,3,3,3,3,3,3]
block_conv3 = [7,5,7,5,5,7,5]
block_conv4 = [11,7,11,7,7,11,7]
####################### TABLE 1 END#################


def variable(name, shape, initializer,regularizer=None):
	global VARIABLE_COUNTER#variable의 개수를 세는 counter를 전역변수로 설정했다.
	with tf.device('/gpu:0'):#GPU를 할당하여 계산할 수 있고, ('/gpu:0')은 첫 번째 gpu를 사용한다는 것을 의미한다. 자세한 사항은 https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=cjh226&logNo=221303802004 참고
		VARIABLE_COUNTER += np.prod(np.array(shape))#np.prod는 array 내부 elements들의 곱이다. np.array는 리스트를 넣으면 배열로 반환해주는 함수이다.
		return tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer, dtype=tf.float32, trainable=True)#tf.get_variable는 입력된 이름의 변수를 생성하거나 반환합니다.


def conv_layer(input_tensor,name,kernel_size,output_channels,initializer=tf.contrib.layers.variance_scaling_initializer(),stride=1,bn=False,training=False,relu=True):
	input_channels = input_tensor.get_shape().as_list()[-1]#input tensor의 channel을 저장한다.
	with tf.variable_scope(name) as scope:# tf.variable_scope(name)는 tf.get_variable() 에 전달 된 이름의 네임스페이스를 관리하는데, tf.variable_scope(name)를 통해 변수에 대한 네임 스페이스를 푸시(pushes)하고 그 스페이스를 scope하고 한다.
		kernel = variable('weights', [kernel_size, kernel_size, input_channels, output_channels], initializer, regularizer=tf.contrib.layers.l2_regularizer(0.0005))#일단 variable 함수로 'weight'라는 이름을 가진 [kernel_size, kernel_size, input_channels, output_channels] shape의 변수를 만든다. 
		conv = tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding='SAME')# 보통 tf.nn.conv2d(input=[[batch, in_height, in_width, in_channels], filter=[filter_height, filter_width, in_channels, out_channels], strides=[1,stride,stride,1], padding = 'SAME' 또는 'VALID'. 패딩을 추가하는 공식의 차이. SAME은 출력 크기를 입력과 같게 유지.]
		biases = variable('biases', [output_channels], tf.constant_initializer(0.0))# variable 함수로 [ouput channel]의 shape으로 biases라는 이름을 가진 변수가 만들어진다.
		conv_layer = tf.nn.bias_add(conv, biases)# conv에 bias를 추가한다. 
		if bn:#bn이 true면
			conv_layer = batch_norm_layer(conv_layer,scope,training)#batch_norm_layer를 통해 batch norm을 구현
		if relu:#relu가 true면
			conv_layer = tf.nn.relu(conv_layer, name=scope.name)#Relu 통과
# 	print('Conv layer {0} -> {1}'.format(input_tensor.get_shape().as_list(),conv_layer.get_shape().as_list()))
	return conv_layer#Relu를 통과한 featuremap


def max_pooling(input_tensor,name,factor=2):
	pool = tf.nn.max_pool(input_tensor, ksize=[1, factor, factor, 1], strides=[1, factor, factor, 1], padding='SAME', name=name)#maxpool 함수로 ksize가 factorxfactor이고, strides도 마찬가지다.
# 	print('Pooling layer {0} -> {1}'.format(input_tensor.get_shape().as_list(),pool.get_shape().as_list()))
	return pool


def batch_norm_layer(input_tensor,scope,training):
	return tf.contrib.layers.batch_norm(input_tensor,scope=scope,is_training=training,decay=0.99)#batchnormalization 시켜주는 함수


def hourglass_refinement(netIN,training):#이제 위에 hourglass의 세부적인 부분을 구성하는 함수를 define했고, 본격적으로 hourglass 함수에 들어섰다.
	print('-'*30)#일단 줄을 친다 -------
	print('Hourglass Architecture')#hourglass에 들어섰다고 print
	print('-'*30)#마지막 줄을 친다 -------
	global VARIABLE_COUNTER#variable의 개수를 셀 전역 변수 위의 variable 함수가 선언될때마다 1씩 더해진다.
	VARIABLE_COUNTER = 0;#처음엔 당연히 0으로 초기화한다.
	layer_name_dict = {}#{}가 의미하는 것은 딕셔너리인데, Key와 Value를 한 쌍으로 갖는 자료형을 의미한다. key는 name을 의미하고, value는 여기서는 개수가 되는 것 같다.
	def layer_name(base_name):
		if base_name not in layer_name_dict:#만약 이 dict에 없었다면 value를 0으로 하여 쌍을 추가해준다.
			layer_name_dict[base_name] = 0
		layer_name_dict[base_name] += 1#그리고 해당 value에 1을 더한다.
		name = base_name + str(layer_name_dict[base_name])#그리고 출력되는 name은 예를들어 3번째 들어오는 conv면 conv3이런식으로 출력한다.
		return name


	bn = True#batch normalization 합니다~
	def hourglass_stack_fused_depth_prediction(stack_in):

		c0 = conv_layer(stack_in,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training) # 8 X 256 X 256 X 64
		c1 = conv_layer(c0,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)#8 X 256 X 256 X 64
		c2 = conv_layer(c1,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)#8 X 256 X 256 X 64

		p0 = max_pooling(c2,layer_name('pool'))#8 X 128 X 128 X 64
		c3 = conv_layer(p0,layer_name('conv'),KER_SZ,NUM_CH[1],bn=bn,training=training)#8 X 128 x 128 x 128

		p1 = max_pooling(c3,layer_name('pool'))#8 X 64 x 64 x 128
		c4 = conv_layer(p1,layer_name('conv'),KER_SZ,NUM_CH[2],bn=bn,training=training)#8 X 64 x 64 x 256

		p2 = max_pooling(c4,layer_name('pool'))#8 X 32 x 32 x 256
		c5 = conv_layer(p2,layer_name('conv'),KER_SZ,NUM_CH[3],bn=bn,training=training)#8 x 32 x 32 x 512

		p3 = max_pooling(c5,layer_name('pool'))#8 x 16 x 16 x 512
		c6 = conv_layer(p3,layer_name('conv'),KER_SZ,NUM_CH[4],bn=bn,training=training)#8 x 16 x 16 x 1024

		c7 = conv_layer(c6,layer_name('conv'),KER_SZ,NUM_CH[4],bn=bn,training=training)#8 x 16 x 16 x 1024
		c8 = conv_layer(c7,layer_name('conv'),KER_SZ,NUM_CH[3],bn=bn,training=training)#8 x 16 x 16 x 512
		r0 = tf.image.resize_images(c8,[c8.get_shape().as_list()[1]*2, c8.get_shape().as_list()[2]*2])#8 x 32 x 32 x 512
		cat0 = tf.concat([r0,c5],3)#8 x 32 x 32 x 1024
		c9 = conv_layer(cat0,layer_name('conv'),KER_SZ,NUM_CH[3],bn=bn,training=training)# 8 x 32 x 32 x 512
		c10 = conv_layer(c9,layer_name('conv'),KER_SZ,NUM_CH[2],bn=bn,training=training)# 8 x 32 x 32 x 256

		r1 = tf.image.resize_images(c10,[c10.get_shape().as_list()[1]*2, c10.get_shape().as_list()[2]*2])#8 x 64 x 64 x 256
		cat1 = tf.concat([r1,c4],3)#8 x 64 x 64 x 512

		c11 = conv_layer(cat1,layer_name('conv'),KER_SZ,NUM_CH[2],bn=bn,training=training)# 8 x 64 x 64 x 256
		c12 = conv_layer(c11,layer_name('conv'),KER_SZ,NUM_CH[1],bn=bn,training=training)#8 x 64 x 64 x 128

		r2 = tf.image.resize_images(c12,[c12.get_shape().as_list()[1]*2, c12.get_shape().as_list()[2]*2])#8 x 128 x 128 x 128
		cat2 = tf.concat([r2,c3],3)#8 x 128 x 128 x 256

		c13 = conv_layer(cat2,layer_name('conv'),KER_SZ,NUM_CH[1],bn=bn,training=training)#8 x 128 x 128 x 128
		c14 = conv_layer(c13,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)##8 x 128 x 128 x 64


		r3 = tf.image.resize_images(c14,[c14.get_shape().as_list()[1]*2, c14.get_shape().as_list()[2]*2])#8 x 256 x 256 x 64
		cat3 = tf.concat([r3,c2],3)#8 x 256 x 256 x 128

		c15 = conv_layer(cat3,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)#8 x 256 x 256 x 64
		c16 = conv_layer(c15,layer_name('conv'),KER_SZ,NUM_CH[0],bn=bn,training=training)#8 x 256 x 256 x 64
		stack_out_d = conv_layer(c16, layer_name('conv'), 1, 1, bn=False, training=training, relu=False)# #8 x 256 x 256 x 1
		return stack_out_d

	out0_d = hourglass_stack_fused_depth_prediction(netIN)#그래서 결국 netIN을 input으로 받았던 것인데 그걸 위에서 정의한 network 함수에 넣고 나온 featuremap을 최종적으로 out0_d에 저장해서 출력한다.
	return out0_d
