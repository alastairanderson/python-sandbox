'''
https://www.tensorflow.org/beta/tutorials/keras/basic_classification

NOTE: 'Beta' in URL - I expect this will change, and will need to update.

requirements.txt
setuptools==41.0.0
tensorflow==2.0.0-rc0
matplotlib==3.1.1

Note 1: Check https://github.com/tensorflow/tensorflow/releases for latest releases
Note 2: 2.0.0-rc0 ERROR: tb-nightly 1.15.0a20190806 has requirement setuptools>=41.0.0, but you'll have setuptools 40.8.0 which is incompatible.
pip install --upgrade setuptools==41.0.0
'''
#region imports
from __future__ import absolute_import, division, print_function, unicode_literals
'''
NOTE: This import allows for running the script as both Python 2 + 3
      https://python-future.org/quickstart.html

TODO: Try commenting this out once we have the script running in Python 3 environment
'''

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
#endregion - imports

#region Import the data
fashion_mnist = keras.datasets.fashion_mnist

# load_data() returns 2 tuples of <class 'numpy.ndarray'>
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#endregion Import the data

#region Explore the data
'''
Useful to understand the dimensions of the data we are working with

Typically this will involve many more steps around looking at the number of samples we can use
versus those we can't use, etc. Maybe even looking at biases in the data.
'''
print(train_images.shape)               # (60000, 28, 28)
print(len(train_labels))                # 60000
print(train_labels)                     # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
print(test_images.shape)                # (10000, 28, 28)
print(len(test_labels))                 # 10000
#endregion - Explore the data

#region Prepare the data

# Inspect the first image to get an idea of the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
# plt.show()

'''
Scale the data

Scale the images to between 0 and 1 before feeding it to a NN
The image above shows that the pixel values fall between 0 and 255
'''
train_images = train_images / 255.0
test_images = test_images / 255.0

'''
Verify the data is in the correct format after the scaling
'''
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

# plt.show()
#endregion - Prepare the data

#region Build the model


'''
Setup the layers

Define the layers of a NN, going from top-to-bottom is the equivalent of looking at a diagram 
going from left-to-right

Flatten - transforms the image from 28x28 2d array into a 1d 784 element array
Dense   - 

relu    - 
softmax - 
'''
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])



#endregion - Build the model