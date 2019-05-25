#region packages
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#endregion

'''
This implements a neural network with no hidden layers,
So we have one neuron which takes the inputs,
matrix multiplies the weights, adds the bias and
pushes it through a softmax activation function
'''

#region public methods
def run():
    # this is a helper function built into TF and not the traditional way data is fed into TF
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Placeholder for 28 X 28 (=784) image data
    x = tf.placeholder(tf.float32, shape=[None, 784])

    # stores the predicted digit(0-9) class in one-hot encoded format. e.g. [0,1,0,0,0,0,0,0,0,0] for 1, [0,0,0,0,0,0,0,0,0,1] for 9
    y_ = tf.placeholder(tf.float32, [None, 10])

    # define weights and bias - these are what we update/train when optimising
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # define our inference model that will perform the prediction
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # softmax is used in multiclass NNs: 
    # https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax

    # review the shapes for cross entropy
    print("y shape: {0}".format(y.get_shape()))
    print("y_ shape: {0}".format(y_.get_shape()))

    # loss is cross entropy
    # sm_ce_wi_log used to 
    # https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
    # https://stackoverflow.com/questions/34240703/what-is-logits-softmax-and-softmax-cross-entropy-with-logits

    # reduce_mean used to get a single mean value from a multi-dimensional tensor
    # https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # each training step in gradient decent we want to minimize cross entropy
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # initialize the global variables
    init = tf.global_variables_initializer()

    # create an interactive session that can span multiple code blocks.  Don't 
    # forget to explicity close the session with sess.close()
    sess = tf.Session()

    # perform the initialization which is only the initialization of all global variables
    sess.run(init)

    # Perform 1000 training steps
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)    # get 100 random data points from the data. batch_xs = image, 
                                                            # batch_ys = digit(0-9) class
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # do the optimization with this data

    # Evaluate how well the model did. Do this by comparing the digit with the highest probability in 
    #    actual (y) and predicted (y_).
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print("Test Accuracy: {0}%".format(test_accuracy * 100.0))

    sess.close()
#endregion

run()