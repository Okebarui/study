
import tensorflow as tf
import re
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow.contrib.slim as slim
import time

tf.set_random_seed(777)  # reproducibility



# hyper parameters
flags = tf.app.flags
FLAGS = flags.FLAGS
FLAGS.image_size = 224
FLAGS.image_color = 3
FLAGS.num_classes = 2
FLAGS.batch_size = 100
FLAGS.learning_rate = 0.0001
FLAGS.training_epochs = 1000

#Stopsign
FLAGS.log_dir = 'Stopsign/log/'
FLAGS.svWeight_dir = 'Stopsign/weights/stop-ep'
FLAGS.training_file = 'Stopsign/training.csv'
FLAGS.testing_file = 'Stopsign/testing.csv'

'''

#Apple
FLAGS.log_dir = 'Apple/log/'
FLAGS.svWeight_dir = 'Apple/weights/stop-ep'
FLAGS.training_file = 'Apple/training.csv'
FLAGS.testing_file = 'Apple/testing.csv'
'''

def get_input_queue(csv_file_name, num_epochs = None):
	train_images = []
	train_labels = []
	for line in open(csv_file_name, 'r'):
		cols = re.split(',|\n', line)
		train_images.append(cols[0])
		train_labels.append(int(cols[2]))
	input_queue = tf.train.slice_input_producer([train_images, train_labels], num_epochs = num_epochs, shuffle = True)
    
	return input_queue

def read_data(input_queue):
	image_file = input_queue[0]
	label = input_queue[1]
    
	image = tf.image.decode_jpeg(tf.read_file(image_file), channels = FLAGS.image_color)
	image = tf.image.resize_images(image, [FLAGS.image_size, FLAGS.image_size])
    
	return image, label, image_file

def read_data_batch(csv_file_name, batch_size=FLAGS.batch_size):
	input_queue = get_input_queue(csv_file_name)
	image, label, file_name = read_data(input_queue)
	image = tf.reshape(image, [FLAGS.image_size, FLAGS.image_size, FLAGS.image_color])
	image = MinMaxScaler(image)
    
	batch_image, batch_label, batch_file = tf.train.batch([image, label, file_name], batch_size = batch_size)
	batch_file = tf.reshape(batch_file, [batch_size, 1])
    
	batch_label_one_hot = tf.one_hot(batch_label, FLAGS.num_classes)
    
	return batch_image, batch_label_one_hot, batch_file

def MinMaxScaler(data):
	numerator = data - np.min(data,0)
	denominator = np.max(data,0) - np.min(data,0)
	return data/255.0

#image load
def load_image(img_path):
	img = mpimg.imread(img_path)
	#converting shape from [224,224,3] to [1,224,224,3]
	x = np.expand_dims(img, axis=0)
	#converting RGB to BGR for VGG
	#x = x[:,:,:,::-1]
	return img

#CAM generator in results folder (after upsizng image 224x224)
def get_class_map(Fmap, Fweights, index, Num_classes):

	predweights = Fweights[:, Num_classes:Num_classes+1]
	camsum = np.zeros((224, 224))
	for j in range(512):
		camsum = camsum + np.maximum(predweights[j]*Fmap[index, :, :, j], 0) #batch index image only

	camavg = camsum / 512
	#passing through ReLu
	#camavg = np.maximum(camavg, 0)
	camavg = camavg / np.max(camavg)	

	#converting grayscale to 3-D
	camavg = np.expand_dims(camavg, axis=2)
	camavg = np.tile(camavg, [1,1,3])

	return camavg

# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
#keep_prob = tf.placeholder(tf.float32)


# input place holders
X = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, FLAGS.image_color])
Y = tf.placeholder(tf.float32, [None, FLAGS.num_classes])


'''
conv1 = slim.repeat(X, 2, slim.conv2d, 64, [3, 3], scope='conv1')
pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')
conv3 = slim.repeat(pool2, 3, slim.conv2d, 256, [3, 3], scope='conv3')
pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')
conv4 = slim.repeat(pool3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
pool4 = slim.max_pool2d(conv4, [2, 2], scope='pool4')
conv5 = slim.repeat(pool4, 3, slim.conv2d, 512, [3, 3], scope='conv5')


conv1 = slim.repeat(X, 2, slim.conv2d, 64, [3, 3], scope='conv1')
pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')
conv3 = slim.repeat(pool2, 4, slim.conv2d, 256, [3, 3], scope='conv3')
pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')
conv4 = slim.repeat(pool3, 4, slim.conv2d, 512, [3, 3], scope='conv4')
pool4 = slim.max_pool2d(conv4, [2, 2], scope='pool4')
conv5 = slim.repeat(pool4, 4, slim.conv2d, 512, [3, 3], scope='conv5')
'''

def vgg_19(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           fc_conv_padding='VALID'):

	with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
		with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn = tf.nn.relu, weights_initializer = tf.truncated_normal_initializer(0.0,0.01),
			weights_regularizer = slim.l2_regularizer(0.0005)):
			conv1 = slim.repeat(X, 2, slim.conv2d, 64, [3, 3], scope='conv1')
			pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
			conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
			pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')
			conv3 = slim.repeat(pool2, 3, slim.conv2d, 256, [3, 3], scope='conv3')
			pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')
			conv4 = slim.repeat(pool3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
			pool4 = slim.max_pool2d(conv4, [2, 2], scope='pool4')
			conv5 = slim.repeat(pool4, 3, slim.conv2d, 512, [3, 3], scope='conv5')
			conv_resized = tf.image.resize_bilinear(conv5, [FLAGS.image_size, FLAGS.image_size])
			# GAP layer
			gap = tf.reduce_mean(conv5, [1,2])

			#net = slim.avg_pool2d(net, [14, 14], scope='CAM')
			with tf.variable_scope("GAP"):
				gap_w = tf.get_variable("W", shape = [512, FLAGS.num_classes], initializer =
					 tf.truncated_normal_initializer(0.0,0.01))

			output = tf.matmul(gap, gap_w)
			weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv5')

	return conv_resized, output, gap, gap_w, weights


# define cost/loss & optimizer
conv_resized, output, gap, gap_w, weights = vgg_19(X, num_classes=FLAGS.num_classes, is_training=True)

image_batch, label_batch, file_batch = read_data_batch(FLAGS.training_file)
val_image_batch, val_label_batch, val_file_batch = read_data_batch(FLAGS.testing_file)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=label_batch))
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)

predict_val = tf.argmax(output, 1)
true_val = tf.argmax(label_batch, 1)
correct_prediction = tf.equal(predict_val, true_val)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#creatte a summary to monitor cost tensor
tf.summary.scalar("cost", cost)
#create a summary to monitor accuracy tensot
tf.summary.scalar("accuracy", accuracy)
#merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# initialize
with tf.Session() as sess:
	init_op = tf.global_variables_initializer() 
	saver = tf.train.Saver() # create saver to store training model into file
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	sess.run(init_op)
	#saver.restore(sess, 'Apple/weights/stop-ep-3') 

	#op to write logs to tensorboard
	summary_writer = tf.summary.FileWriter(FLAGS.log_dir, graph=tf.get_default_graph())

	# train my model
	print('Learning stared. It takes sometime.')

	total_batch_size = int(300 / FLAGS.batch_size)
	
	plt.figure()

	for epoch in range(FLAGS.training_epochs):
		avg_cost = 0
		avg_acc = 0

		for i in range(total_batch_size):
			s_t = time.time()
			batch_xs, batch_ys, batch_fi = sess.run([image_batch,label_batch, file_batch])
			feed_dict = {X: batch_xs, Y: batch_ys}

			we, c, _, summary, Fweights, Fmap = sess.run([gap, cost, optimizer, merged_summary_op, gap_w, conv_resized], 
							feed_dict=feed_dict)

			avg_cost += c / total_batch_size
			e_t = time.time()
	
			#write logs at every iteration
			summary_writer.add_summary(summary, epoch*total_batch_size+i)

			print('Data:', '%04d' % (FLAGS.batch_size * (i+1)), 'cost =', '{:.9f}'.format(avg_cost), 
				'Pr_time= %03d sec' % (e_t-s_t))

			#CAM generation
			camavg = get_class_map(Fmap=Fmap, Fweights=Fweights, index=0, Num_classes=np.argmax(batch_ys[0]))

			#save CAM-figure in results folder
			img = batch_xs[0].astype(float)	
			img /= img.max()
			new_img = img + 3*camavg
			new_img /= new_img.max()
			
			print(batch_fi)
			plt.imshow(new_img)
			plt.savefig('results/CAM_image{}_{}.png'.format(epoch+1, i))

		saver.save(sess, FLAGS.svWeight_dir, epoch+1) # save session

		for k in range(total_batch_size):
			batch_xs, batch_ys, batch_fi = sess.run([image_batch,label_batch, file_batch])
			feed_dict = {X: batch_xs, Y: batch_ys}
			cal_a, cal_cp = sess.run([accuracy, correct_prediction], feed_dict)
			avg_acc += cal_a / total_batch_size

		print('*** Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.5f}'.format(avg_cost), 'acc =','{:.3f} ***'.format(avg_acc))

	print('Learning Finished!')
    
	val_batch_xs, val_batch_ys = sess.run([val_image_batch, val_label_batch])
	feed_dict = {X: val_batch_xs, Y: val_batch_ys}
	val_a, val_cp = sess.run([accuracy, correct_prediction], feed_dict)
	print('Test Accuracy:', val_a)

	plt.close()


	coord.request_stop()
	coord.join(threads)




