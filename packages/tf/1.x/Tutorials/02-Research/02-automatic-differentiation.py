'''
Automatic differentiation and gradient tape
https://www.tensorflow.org/tutorials/eager/automatic_differentiation

In the previous tutorial we introduced Tensors and operations on them. In this tutorial we will 
cover automatic differentiation, a key technique for optimizing machine learning models.

automatic differentiation - https://en.wikipedia.org/wiki/Automatic_differentiation
'''
#region imports
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tf.enable_eager_execution()
#endregion - imports

#region Gradient tapes
'''
Gradient tapes
https://www.tensorflow.org/tutorials/eager/automatic_differentiation#gradient_tapes

TensorFlow provides the tf.GradientTape API for automatic differentiation - computing the gradient 
of a computation with respect to its input variables. Tensorflow "records" all operations executed 
inside the context of a tf.GradientTape onto a "tape". Tensorflow then uses that tape and the 
gradients associated with each recorded operation to compute the gradients of a "recorded" 
computation using reverse mode differentiation.

tf.GradientTape              - https://www.tensorflow.org/api_docs/python/tf/GradientTape
reverse mode differentiation - https://en.wikipedia.org/wiki/Automatic_differentiation

For example:
'''
x = tf.ones((2, 2))

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)        # 4.0 == 1 + 1 + 1 + 1
    z = tf.multiply(y, y)       # 16.0 == 4.0 * 4.0 == y * y == sum(x) * sum(x) == (1+1+1+1) * (1+1+1+1)

# Derivative of z with respect to the original input tensor x
dz_dx = t.gradient(z, x)        # [[8,8][8,8]] == TODO: I don't understand partial derivative of tensor x


for i in [0, 1]:
    for j in [0, 1]:
        print(dz_dx[i][j].numpy() == 8.0)
        assert dz_dx[i][j].numpy() == 8.0

'''
You can also request gradients of the output with respect to intermediate values computed during 
a "recorded" tf.GradientTape context.

tf.GradientTape - https://www.tensorflow.org/api_docs/python/tf/GradientTape
'''
x = tf.ones((2, 2))

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

# Use the tape to compute the derivative of z with respect to the
# intermediate value y.
dz_dy = t.gradient(z, y)
assert dz_dy.numpy() == 8.0

'''
By default, the resources held by a GradientTape are released as soon as GradientTape.gradient() method 
is called. To compute multiple gradients over the same computation, create a persistent gradient tape. 
This allows multiple calls to the gradient() method. as resources are released when the tape object is 
garbage collected. For example:
'''
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = x * x
    z = y * y
    
dz_dx = t.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
dy_dx = t.gradient(y, x)  # 6.0
del t  # Drop the reference to the tape

#endregion - Gradient tapes

#region Recording control flow
'''
Recording control flow
https://www.tensorflow.org/tutorials/eager/automatic_differentiation#recording_control_flow

Because tapes record operations as they are executed, Python control flow (using ifs and whiles 
for example) is naturally handled:
'''
def f(x, y):
    output = 1.0
    for i in range(y):
        if i > 1 and i < 5:
            output = tf.multiply(output, x)
    return output

# Assume y > 4, output will be as follows:
#   Loop #1: i = 1, output = 1
#   Loop #2: i = 2, output = 1 * x
#   Loop #3: i = 3, output = (1 * x) * x = x^2
#   Loop #4: i = 4, output = (x^2) * x   = x^3
#   Loop >4: output = x^3

def grad(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        out = f(x, y)
    return t.gradient(out, x)

# As per previous comments on f(x, y), grad would return:
#   x = 2; y = 6: x^3 with 3x^2 as the partial derivative, therefore 3(2)^2 == 12
#   x = 2; y = 5: x^3 with 3x^2 as the partial derivative, therefore 3(2)^2 == 12
#   x = 2; y = 4: x^2 with 2x as the partial derivative, therefore 2(2) == 4

x = tf.convert_to_tensor(2.0)

assert grad(x, 6).numpy() == 12.0       # True
assert grad(x, 5).numpy() == 12.0       # True
assert grad(x, 4).numpy() == 4.0        # True

#endregion - Recording control flow

#region Higher-order gradients
'''
Higher-order gradients
https://www.tensorflow.org/tutorials/eager/automatic_differentiation#higher-order_gradients

Operations inside of the GradientTape context manager are recorded for automatic differentiation. 
If gradients are computed in that context, then the gradient computation is recorded as well. As 
a result, the exact same API works for higher-order gradients as well. For example:
'''

x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0

with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
        y = x * x * x

    # Compute the gradient inside the 't' context manager
    # which means the gradient computation is differentiable as well.
    dy_dx = t2.gradient(y, x)

d2y_dx2 = t.gradient(dy_dx, x)

assert dy_dx.numpy() == 3.0         # True          3x^2
assert d2y_dx2.numpy() == 6.0       # True          6x
#endregion - Higher-order gradients

#region Next Steps
'''
Next Steps
https://www.tensorflow.org/tutorials/eager/automatic_differentiation#next_steps

In this tutorial we covered gradient computation in TensorFlow. With that we have enough of 
the primitives required to build and train neural networks.

See next script: 03-custom-training-basics.py
'''
#endregion - Next Steps