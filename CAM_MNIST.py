"""
 Weakly Supervised Net (Global Average Pooling) with MNIST
 @Sungjoon Choi (sungjoon.choi@cpslab.snu.ac.kr
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('data/', one_hot=True)
trainimgs   = mnist.train.images
trainlabels = mnist.train.labels
testimgs    = mnist.test.images
testlabels  = mnist.test.labels

ntrain      = trainimgs.shape[0]
ntest       = testimgs.shape[0]
dim         = trainimgs.shape[1]
nout        = trainlabels.shape[1]

# Make multilayer perceptron 

weights = {
    'wc1' : tf.Variable(tf.random_normal([3, 3, 1, 128], stddev=0.1)), 
    'wc2' : tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.1)), 
    'out' : tf.Variable(tf.random_normal([256, 10], stddev=0.1))
}
biases = {
    'bc1' : tf.Variable(tf.random_normal([128], stddev=0.1)),
    'bc2' : tf.Variable(tf.random_normal([256], stddev=0.1)),
    'out' : tf.Variable(tf.random_normal([10], stddev=0.1))
}

def mlp(_X, _W, _b, _keepprob):
    # Reshape input
    _input_r = tf.reshape(_X, shape=[-1, 28, 28, 1])
    # Conv1 
    _conv1 = tf.nn.relu(
            tf.nn.bias_add(
                tf.nn.conv2d(_input_r, _W['wc1'], strides=[1, 1, 1, 1], padding='SAME')
            , _b['bc1'])
        )
    # Pool1
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # DropOut1
    _layer1 = tf.nn.dropout(_pool1, _keepprob)
    # Conv2
    _conv2 = tf.nn.relu(
            tf.nn.bias_add(
                tf.nn.conv2d(_layer1, _W['wc2'], strides=[1, 1, 1, 1], padding='SAME')
            , _b['bc2'])
        )
    # Pool2 (Global average pooling)
    _pool2 = tf.nn.avg_pool(_conv2, ksize=[1, 14, 14, 1], strides=[1, 14, 14, 1], padding='SAME')
    # DropOut2
    _layer2 = tf.nn.dropout(_pool2, _keepprob)
    # Vectorize
    _dense = tf.reshape(_layer2, [-1, _W['out'].get_shape().as_list()[0]])
    # FC1
    _out   = tf.nn.softmax(tf.add(tf.matmul(_dense, _W['out']), _b['out']))
    out = {
        'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'layer1': _layer1,
        'conv2': _conv2, 'pool2': _pool2, 'layer2': _layer2, 'dense': _dense, 
        'out': _out
    }
    return out



# Define Parameter
learning_rate   = 0.001
training_epochs = 1
batch_size      = 100
display_step    = 1

# Define Functions
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keepprob = tf.placeholder(tf.float32)
pred = mlp(x, weights, biases, keepprob)['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optm = tf.train.AdamOptimizer(learning_rate).minimize(cost)
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(corr, tf.float32))
init = tf.global_variables_initializer()
print ("Net Ready")

# Let's do it!
sess = tf.Session()
sess.run(init)

# Training cycle
for epoch in range(training_epochs):
    sum_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keepprob: 0.7})
        # Compute average loss
        curr_cost = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keepprob: 1.})
        sum_cost = sum_cost + curr_cost
    avg_cost = sum_cost / total_batch
    
    # Display logs per epoch step
    if epoch % display_step == 0:
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepprob: 1.})
        print ("Training accuracy: %.3f" % (train_acc))
        test_acc = sess.run(accr, feed_dict={x: testimgs, y: testlabels, keepprob: 1.})
        print ("Test accuracy: %.3f" % (test_acc))

print ("Optimization Finished!")

# Get Random Image
randidx = np.random.randint(ntest)
testimg = testimgs[randidx:randidx+1, :]

# Run Network 
inputimg = sess.run(mlp(x, weights, biases, keepprob)['input_r'],
                feed_dict={x: testimg, keepprob: 1.})
outval = sess.run(mlp(x, weights, biases, keepprob)['out'],
                feed_dict={x: testimg, keepprob: 1.})
camval = sess.run(mlp(x, weights, biases, keepprob)['conv2'],
                feed_dict={x: testimg, keepprob: 1.})
cweights = sess.run(weights['out'])

# Plot original Image 
plt.matshow(inputimg[0, :, :, 0], cmap=plt.get_cmap('gray'))
plt.title("Input image")
plt.colorbar()
plt.show()
    
# Plot class activation maps 
fig, axs = plt.subplots(2, 5, figsize=(15, 6))    
for i in range(10):
    predlabel   = np.argmax(outval)
    predweights = cweights[:, i:i+1]
    camsum = np.zeros((14, 14))
    for j in range(256):
        camsum = camsum + predweights[j]*camval[0, :, :, j]
    camavg = camsum / 256
    # Plot 
    im = axs[int(i/5)][i%5].matshow(camavg, cmap=plt.get_cmap('gray'))
    axs[int(i/5)][i%5].set_title(("[%d] prob is %.3f") % (i, outval[0, i]))
    plt.draw()

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

