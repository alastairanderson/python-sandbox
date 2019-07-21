'''
    Build a linear model with Estimators
    https://www.tensorflow.org/tutorials/estimators/linear

    This tutorial uses the tf.estimator API in TensorFlow to solve a benchmark binary classification 
    problem. Estimators are TensorFlow's most scalable and production-oriented model type. For more 
    information see the Estimator guide.

    tf.estimator    - https://www.tensorflow.org/api_docs/python/tf/estimator
    Estimator guide - https://www.tensorflow.org/guide/estimators


    Overview
    https://www.tensorflow.org/tutorials/estimators/linear#overview

    Using census data which contains data a person's age, education, marital status, and occupation 
    (the features), we will try to predict whether or not the person earns more than 50,000 dollars 
    a year (the target label). We will train a logistic regression model that, given an individual's 
    information, outputs a number between 0 and 1 — this can be interpreted as the probability that 
    the individual has an annual income of over 50,000 dollars.

    Key Point: As a modeler and developer, think about how this data is used and the potential benefits 
    and harm a model's predictions can cause. A model like this could reinforce societal biases and 
    disparities. Is each feature relevant to the problem you want to solve or will it introduce bias? 
    For more information, read about ML fairness.

    ML fairness - https://developers.google.com/machine-learning/fairness-overview/
'''
from __future__ import absolute_import, division, print_function, unicode_literals

#region Setup
'''
    Setup
    https://www.tensorflow.org/tutorials/estimators/linear#setup

    pip install tensorflow
    pip install matplotlib
    pip install IPython
    pip install requests
    git clone --depth 1 https://github.com/tensorflow/models
'''
import tensorflow as tf
import tensorflow.feature_column as fc

import os
import sys

import matplotlib.pyplot as plt
from IPython.display import clear_output

tf.enable_eager_execution()                     # https://www.tensorflow.org/guide/eager
#endregion - Setup

#region Download the official implementation
'''
    Download the official implementation
    https://www.tensorflow.org/tutorials/estimators/linear#download_the_official_implementation

    We'll use the wide and deep model available in TensorFlow's model repository. Download the code, 
    add the root directory to your Python path, and jump to the wide_deep directory

    wide and deep model - https://github.com/tensorflow/models/tree/master/official/wide_deep/
    model repository    - https://github.com/tensorflow/models/

    Add the root directory of the repository to your Python path:
'''
models_path = os.path.join(os.getcwd(), 'models')

sys.path.append(models_path)

'''
    Download the dataset
'''
from official.wide_deep import census_dataset
from official.wide_deep import census_main

census_dataset.download("/tmp/census_data/")
#endregion - Download the official implementation

#region Command line usage
'''
Command line usage

'''
#endregion - Command line usage