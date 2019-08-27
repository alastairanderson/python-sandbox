'''
https://www.tensorflow.org/tutorials/
https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb

requirements.txt
tensorflow==2.0.0-rc0

Note 1: Check https://github.com/tensorflow/tensorflow/releases for latest releases 
Note 2: 2.0.0-rc0 ERROR: tb-nightly 1.15.0a20190806 has requirement setuptools>=41.0.0, but you'll have setuptools 40.8.0 which is incompatible.

pip install --upgrade setuptools==41.0.0
'''
from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow
try:
  # %tensorflow_version only exists in Colab.
  import tensorflow.compat.v2 as tf
except Exception:
  pass

tf.enable_v2_behavior()

mnist = tf.keras.datasets.mnist

# type(x_train, y_train, x_test, y_test)  == <class 'numpy.ndarray'>
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Convert the samples from integers to floating point
x_train, x_test = x_train / 255.0, x_test / 255.0

# stack the sequential layers of the NN
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# specify an optimiser and cost/loss function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)