#region packages
import tensorflow as tf
import numpy as np
#endregion

#region get_shape(), dtype, rank, reshape(), cast()
def common():
    x = tf.placeholder(tf.float32, shape=[None, 784])

    print(x.get_shape())
    print(x.dtype)
    print(tf.rank(x))


    values = [1, 2, 3, 4]
    result = tf.reshape(values, [2, 2])
    print(tf.Session().run(result))
    print(result.dtype)

    result2 = tf.cast(result, tf.float32)
    print(tf.Session().run(result2))
    print(result2.dtype)
#endregion

#region feeding a numpy array into TF
def feed_np_array_to_tf():
    x_input = np.array([[1,2,3],[4,5,6]])     # shape (2,3)
    W_input = np.array([[7,8,9]])             # shape (1,3)

    # Note: the second set of brackets are required for the 1 dimension,  
    #       otherwise the shape comes back as (3,1)

    # TODO: Does this work?
    # y_hat = tf.convert_to_tensor(np.array([[0.5, 1.5, 0.1],[2.2, 1.3, 1.7]]))

    x = tf.placeholder(tf.float32, [2, 3])
    W = tf.placeholder(tf.float32, [1, 3])
    
    
#endregion

#region reduce_mean
def reduce_mean_example():
    # https://www.dotnetperls.com/reduce-mean-tensorflow

    a = tf.constant([[100, 110], [10, 20], [1000, 1100]])

    # Use reduce_mean to compute the average (mean) across a dimension.
    b = tf.reduce_mean(a)
    c = tf.reduce_mean(a, axis=0)
    d = tf.reduce_mean(a, axis=1)

    session = tf.Session()

    print("INPUT")
    print(session.run(a))               
    print("REDUCE MEAN")
    print(session.run(b))               # 390
    print("REDUCE MEAN AXIS 0")
    print(session.run(c))               # [370 410]
    print("REDUCE MEAN AXIS 1")
    print(session.run(d))               # [ 105   15 1050]
#endregion

#region loss functions
def softmax_example():
    # https://www.dotnetperls.com/softmax-tensorflow

    # Takes an input vector and works out the probabilities (Adding up to 1)
    # of each of the values

    # Used in the final stage of a CNN. Softmax "squashes" values, removing outliers.

    # Use softmax on vector.
    x = [0., -1., 2., 3.]
    softmax_x = tf.nn.softmax(x)

    # Create 2D tensor and use soft max on the second dimension.
    y = [5., 4., 6., 7., 5.5, 6.5, 4.5, 4.]
    y_reshape = tf.reshape(y, [2, 2, 2])
    softmax_y = tf.nn.softmax(y_reshape, 1)

    session = tf.Session()
    print("X")
    print(x)                        
    print("SOFTMAX X")
    print(session.run(softmax_x))       # [ 0.03467109  0.01275478  0.25618663  0.69638747]
    print("Y")
    print(session.run(y_reshape))
    print("SOFTMAX Y")
    print(session.run(softmax_y))       # [[[ 0.26894143  0.04742587]
                                        #   [ 0.7310586   0.95257413]]
                                        #  [[ 0.7310586   0.92414182]
                                        #   [ 0.26894143  0.07585818]]]

#endregion


# softmax_cross_entropy_with_logits
    # This optimization minimises the cross entropy AND softmaxes after your last layer
    # It takes 2 inputs: logits (the output of the NN) and labels (of the labelled initial data)

    # https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
    # https://stackoverflow.com/questions/34240703/what-is-logits-softmax-and-softmax-cross-entropy-with-logits

    # logits:
        # - the function operates on the unscaled output of earlier layers
        # - the relative scale to understand the units is linear
        # - the sum of the inputs may not equal 1 as the values are not probabilities

    # softmax
        # Takes an input vector and works out the probabilities (Adding up to 1)
        # of each of the values

    # cross entropy
        # a summary metric: it sums across the elements. 
        # The output of tf.nn.softmax_cross_entropy_with_logits on a shape [2,5] tensor is of shape [2,1] 
        # (the first dimension is treated as the batch).

    # NOTES: This handles some tricky numerical edge cases

#endregion


#region matrix multiplication
def matmul_example():
    # https://www.dotnetperls.com/matmul-tensorflow
    # https://docs.w3cub.com/tensorflow~python/tf/matmul/

    a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
    b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
    c = tf.matmul(a, b)

    session = tf.Session()
    print(session.run(c))       # [[ 58,  64], [139, 154]]

    a = tf.constant(np.arange(1, 13, dtype=np.int32), shape=[2, 2, 3])
    b = tf.constant(np.arange(13, 25, dtype=np.int32), shape=[2, 3, 2])
    
    c = tf.matmul(a, b)
    print(session.run(c))

    d = a @ b @ [[10.], [11.]]                          # TypeError: Expected int32, got 10.0 of type 'float' instead.
    d = tf.matmul(tf.matmul(a, b), [[10.], [11.]])
    print(session.run(d))
# print(tf.matmul(x,W))



#endregion




#region Activation functions
'''
https://alexisalulema.com/2017/10/15/activation-functions-in-tensorflow/
'''


#endregion
