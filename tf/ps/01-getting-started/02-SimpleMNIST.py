#region packages
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#endregion

#region public methods
def run():
    mnist = get_mnist_data()

    x, y_ = __configure_placeholders()
    W, b = __configure_weights_and_bias()

    # define weights and bias
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # define our inference model
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # loss is cross entropy
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

#region data setup
def get_mnist_data():
    # Use the TF helper function retrieve the MNIST data
    return input_data.read_data_sets("MNIST_data/", one_hot=True)
#endregion

#region TF methods
def __configure_placeholders():
    # Placeholder for 28 X 28 (=784) image data
    x = tf.placeholder(tf.float32, shape=[None, 784])

    # element vector, containing the predicted digit(0-9) class in one-hot encoded format. 
    # e.g. [0,1,0,0,0,0,0,0,0,0] for 1, [0,0,0,0,0,0,0,0,0,1] for 9
    y_ = tf.placeholder(tf.float32, [None, 10])

    return x, y_


def __configure_weights_and_bias():
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    return W, b
#endregion

run()