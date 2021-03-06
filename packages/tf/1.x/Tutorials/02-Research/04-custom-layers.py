'''
Custom layers
https://www.tensorflow.org/tutorials/eager/custom_layers

We recommend using tf.keras as a high-level API for building neural networks. 
That said, most TensorFlow APIs are usable with eager execution.

tf.keras - https://www.tensorflow.org/api_docs/python/tf/keras
'''
#region imports
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tf.enable_eager_execution()
#endregion

#region Layers: common sets of useful operations
'''
Layers: common sets of useful operations
https://www.tensorflow.org/tutorials/eager/custom_layers#layers_common_sets_of_useful_operations

Most of the time when writing code for machine learning models you want to operate at a higher level 
of abstraction than individual operations and manipulation of individual variables.

Many machine learning models are expressible as the composition and stacking of relatively simple 
layers, and TensorFlow provides both a set of many common layers as a well as easy ways for you to 
write your own application-specific layers either from scratch or as the composition of existing 
layers.

TensorFlow includes the full Keras API in the tf.keras package, and the Keras layers are very useful 
when building your own models.

Keras - https://keras.io/
'''
# In the tf.keras.layers package, layers are objects. To construct a layer,
# simply construct the object. Most layers take as a first argument the number
# of output dimensions / channels.
layer = tf.keras.layers.Dense(100)

# The number of input dimensions is often unnecessary, as it can be inferred
# the first time the layer is used, but it can be provided if you want to
# specify it manually, which is useful in some complex models.
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))

'''
The full list of pre-existing layers can be seen in the documentation. It includes: 

    Dense (a fully-connected layer), 
    Conv2D, 
    LSTM, 
    BatchNormalization, 
    Dropout, 
    and many others.

Keras layers documentation - https://www.tensorflow.org/api_docs/python/tf/keras/layers
'''
# To use a layer, simply call it.
layer(tf.zeros([10, 5]))                # tf.Tensor([[0. 0. ... 0. 0.]], shape=(10, 10), dtype=float32)

# Layers have many useful methods. For example, you can inspect all variables
# in a layer using `layer.variables` and trainable variables using
# `layer.trainable_variables`. In this case a fully-connected layer
# will have variables for weights and biases.
layer.variables

'''
    [<tf.Variable 'dense_1/kernel:0' shape=(5, 10) dtype=float32, numpy=
        array([[-0.16585675,  0.6189802 , -0.11172843, ... ,  0.2748283 ,  0.03758466]],
        dtype=float32)>, 
        
        <tf.Variable 'dense_1/bias:0' shape=(10,) dtype=float32, 
        numpy=array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)>]
'''

# The variables are also accessible through nice accessors
layer.kernel
'''
    <tf.Variable 'dense_1/kernel:0' shape=(5, 10) dtype=float32, numpy=
        array([[-0.3381335 ,  0.488115  , ... , -0.59715366, -0.39409754]],
        dtype=float32)>
'''

layer.bias
'''
    <tf.Variable 'dense_1/bias:0' shape=(10,) dtype=float32, 
        numpy=array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)>
'''
#endregion - Layers: common sets of useful operations

#region Implementing custom layers
'''
Implementing custom layers
https://www.tensorflow.org/tutorials/eager/custom_layers#implementing_custom_layers

The best way to implement your own layer is extending the tf.keras.Layer class and implementing: 
* __init__ , where you can do all input-independent initialization * build, where you know the 
shapes of the input tensors and can do the rest of the initialization * call, where you do the 
forward computation

Note that you don't have to wait until build is called to create your variables, you can also 
create them in __init__. However, the advantage of creating them in build is that it enables 
late variable creation based on the shape of the inputs the layer will operate on. On the other 
hand, creating variables in __init__ would mean that shapes required to create the variables 
will need to be explicitly specified.
'''

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                                        shape=[int(input_shape[-1]),
                                                   self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)

layer = MyDenseLayer(10)

print(layer(tf.zeros([10, 5])))
'''
    tf.Tensor([[0. 0. ... 0. 0.]], shape=(10, 10), dtype=float32)
'''

print(layer.trainable_variables)
'''
    [<tf.Variable 'my_dense_layer/kernel:0' shape=(5, 10) dtype=float32, numpy=
        array([[-0.6060415 , -0.1240592 , ... ,  0.09273005,  0.04697657]],
        dtype=float32)>]
'''

'''
Overall code is easier to read and maintain if it uses standard layers whenever possible, 
as other readers will be familiar with the behavior of standard layers. If you want to use 
a layer which is not present in tf.keras.layers or tf.contrib.layers, consider filing a 
github issue or, even better, sending us a pull request!

github issues - http://github.com/tensorflow/tensorflow/issues/new
'''
#endregion - Implementing custom layers

#region Models: composing layers
'''
Models: composing layers
https://www.tensorflow.org/tutorials/eager/custom_layers#models_composing_layers

Many interesting layer-like things in machine learning models are implemented by composing 
existing layers. For example, each residual block in a resnet is a composition of convolutions, 
batch normalizations, and a shortcut.

The main class used when creating a layer-like thing which contains other layers is 
tf.keras.Model. Implementing one is done by inheriting from tf.keras.Model.
'''
class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)

block = ResnetIdentityBlock(1, [1, 2, 3])
print(block(tf.zeros([1, 2, 3, 3])))

'''
tf.Tensor(
    [[[[0. 0. 0.]
       [0. 0. 0.]
       [0. 0. 0.]]

      [[0. 0. 0.]
       [0. 0. 0.]
       [0. 0. 0.]]]], 
       shape=(1, 2, 3, 3), dtype=float32
   )
'''

print([x.name for x in block.trainable_variables])

'''
[
    'resnet_identity_block/conv2d/kernel:0', 
    'resnet_identity_block/conv2d/bias:0', 
    'resnet_identity_block/batch_normalization/gamma:0', 
    'resnet_identity_block/batch_normalization/beta:0', 

    'resnet_identity_block/conv2d_1/kernel:0', 
    'resnet_identity_block/conv2d_1/bias:0', 
    'resnet_identity_block/batch_normalization_1/gamma:0', 
    'resnet_identity_block/batch_normalization_1/beta:0', 
    
    'resnet_identity_block/conv2d_2/kernel:0', 
    'resnet_identity_block/conv2d_2/bias:0', 
    'resnet_identity_block/batch_normalization_2/gamma:0', 
    'resnet_identity_block/batch_normalization_2/beta:0'
]
'''

'''
Much of the time, however, models which compose many layers simply call one layer 
after the other. This can be done in very little code using tf.keras.Sequential
'''
my_seq = tf.keras.Sequential([tf.keras.layers.Conv2D(1, (1, 1)),
                              tf.keras.layers.BatchNormalization(),
                              tf.keras.layers.Conv2D(2, 1, padding='same'),
                              tf.keras.layers.BatchNormalization(),
                              tf.keras.layers.Conv2D(3, (1, 1)),
                              tf.keras.layers.BatchNormalization()])

my_seq(tf.zeros([1, 2, 3, 3]))

'''
<tf.Tensor: id=514, shape=(1, 2, 3, 3), dtype=float32, numpy=
    array([[[[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]],

            [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]]]], dtype=float32)>
'''
#endregion - Models: composing layers

#region Next steps
'''
Now you can go back to the previous notebook and adapt the linear regression example to use 
layers and models to be better structured.
'''
#endregion - Next steps