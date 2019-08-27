'''
https://www.tensorflow.org/beta/tutorials/keras/basic_classification

NOTE: 'Beta' in URL, change this once it goes live.

requirements.txt
tensorflow==2.0.0-rc0
matplotlib==3.1.1

Note 1: Check https://github.com/tensorflow/tensorflow/releases for latest releases 
Note 2: 2.0.0-rc0 ERROR: tb-nightly 1.15.0a20190806 has requirement setuptools>=41.0.0, but you'll have setuptools 40.8.0 which is incompatible.

pip install --upgrade setuptools==41.0.0
'''
#region imports
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
#endregion - imports