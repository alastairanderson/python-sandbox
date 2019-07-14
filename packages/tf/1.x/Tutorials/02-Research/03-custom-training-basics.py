'''
Custom training: basics
https://www.tensorflow.org/tutorials/eager/custom_training

In the previous tutorial (02-automatic-differentiation) we covered the TensorFlow APIs 
for automatic differentiation, a basic building block for machine learning. In this 
tutorial we will use the TensorFlow primitives introduced in the prior tutorials to do 
some simple machine learning.

TensorFlow also includes a higher-level neural networks API (tf.keras) which provides 
useful abstractions to reduce boilerplate. We strongly recommend those higher level APIs 
for people working with neural networks. However, in this short tutorial we cover neural 
network training from first principles to establish a strong foundation.

tf.keras - https://www.tensorflow.org/api_docs/python/tf/keras
'''
# region imports
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tf.enable_eager_execution()
#endregion imports

#region Variables
'''
Variables
https://www.tensorflow.org/tutorials/eager/custom_training#variables

Tensors in TensorFlow are immutable stateless objects. Machine learning models, however, 
need to have changing state: as your model trains, the same code to compute predictions 
should behave differently over time (hopefully with a lower loss!). To represent this 
state which needs to change over the course of your computation, you can choose to rely 
on the fact that Python is a stateful programming language:
'''
# Using python state
x = tf.zeros([10, 10])
x += 2                      # This is equivalent to x = x + 2, which does not mutate the original
                            # value of x

print(x)                    # tf.Tensor([[2. 2. ... 2. 2.]], shape=(10, 10), dtype=float32)

'''
TensorFlow, however, has stateful operations built in, and these are often more pleasant to 
use than low-level Python representations of your state. To represent weights in a model, 
for example, it's often convenient and efficient to use TensorFlow variables.

A Variable is an object which stores a value and, when used in a TensorFlow computation, will 
implicitly read from this stored value. There are operations (tf.assign_sub, tf.scatter_update, 
etc) which manipulate the value stored in a TensorFlow variable.

tf.assign_sub     - https://www.tensorflow.org/api_docs/python/tf/assign_sub
tf.scatter_update - https://www.tensorflow.org/api_docs/python/tf/scatter_update
'''
v = tf.Variable(1.0)
assert v.numpy() == 1.0         # True

# Re-assign the value
v.assign(3.0)
assert v.numpy() == 3.0         # True

# Use `v` in a TensorFlow operation like tf.square() and reassign
v.assign(tf.square(v))
assert v.numpy() == 9.0         # True

'''
Computations using Variables are automatically traced when computing gradients. For Variables 
representing embeddings TensorFlow will do sparse updates by default, which are more 
computation and memory efficient.

Using Variables is also a way to quickly let a reader of your code know that this piece of 
state is mutable.
'''
#endregion - Variables

#region Example: Fitting a linear model
'''
Example: Fitting a linear model
https://www.tensorflow.org/tutorials/eager/custom_training

Let's now put the few concepts we have so far ---Tensor, GradientTape, Variable --- to build 
and train a simple model. This typically involves a few steps:

    1. Define the model.
    2. Define a loss function.
    3. Obtain training data.
    4. Run through the training data and use an "optimizer" to adjust the variables to fit 
       the data.

In this tutorial, we'll walk through a trivial example of a simple linear model: 
    
    f(x) = x * W + b

which has two variables - W and b. Furthermore, we'll synthesize data such that a well trained 
model would have W = 3.0 and b = 2.0.
'''
#region Define the model
'''
Define the model
https://www.tensorflow.org/tutorials/eager/custom_training#define_the_model

Let's define a simple class to encapsulate the variables and the computation.

This simple class is linear regression, the equation of a straight line:
    y = mx + c

or re-written as:
    y = Wx + b
'''
class Model(object):
    def __init__(self):
        # Initialize variable to (5.0, 0.0)
        # In practice, these should be initialized to random values.
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b

model = Model()

assert model(3.0).numpy() == 15.0           # True

#endregion - Define the model

#region Define a loss function
'''
Define a loss function
https://www.tensorflow.org/tutorials/eager/custom_training#define_a_loss_function

A loss function measures how well the output of a model for a given input matches the 
desired output. Let's use the standard L2 loss.
'''
def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))

#endregion - Define a loss function

#region Obtain training data
'''
Obtain training data
https://www.tensorflow.org/tutorials/eager/custom_training#obtain_training_data

Let's synthesize the training data with some noise.
'''
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs  = tf.random_normal(shape=[NUM_EXAMPLES])
noise   = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

'''
Before we train the model let's visualize where the model stands right now. We'll plot 
the model's predictions in red and the training data in blue.
'''
import matplotlib.pyplot as plt

plt.scatter(inputs, outputs, c='b')             # training data
plt.scatter(inputs, model(inputs), c='r')       # predictions
plt.show()

print('Current loss: '),
print(loss(model(inputs), outputs).numpy())

#endregion - Obtain training data

#region Define a training loop
'''
Define a training loop
https://www.tensorflow.org/tutorials/eager/custom_training#define_a_training_loop

We now have our network and our training data. Let's train it, i.e., use the training data to 
update the model's variables (W and b) so that the loss goes down using gradient descent. 
There are many variants of the gradient descent scheme that are captured in tf.train.Optimizer 
implementations. We'd highly recommend using those implementations, but in the spirit of 
building from first principles, in this particular example we will implement the basic math 
ourselves.

Gradient descent   - https://en.wikipedia.org/wiki/Gradient_descent
tf.train.Optimizer - https://www.tensorflow.org/api_docs/python/tf/train/Optimizer
'''
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)

    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)

'''
Finally, let's repeatedly run through the training data and see how W and b evolve.
'''
model = Model()

# Collect the history of W-values and b-values to plot later
Ws, bs = [], []
epochs = range(10)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)

    train(model, inputs, outputs, learning_rate=0.1)
    print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
            (epoch, Ws[-1], bs[-1], current_loss))

# Let's plot it all
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true_b'])
plt.show()

#endregion - Define a training loop

#endregion - Example: Fitting a linear model

#region Next Steps
'''
In this tutorial we covered Variables and built and trained a simple linear model using the 
TensorFlow primitives discussed so far.

In theory, this is pretty much all you need to use TensorFlow for your machine learning research. 
In practice, particularly for neural networks, the higher level APIs like tf.keras will be much 
more convenient since it provides higher level building blocks (called "layers"), utilities to 
save and restore state, a suite of loss functions, a suite of optimization strategies etc.

tf.keras - https://www.tensorflow.org/api_docs/python/tf/keras
'''
#endregion - Next Steps