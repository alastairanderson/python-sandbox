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
    pip install pandas
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
print(f"models_path: {models_path}")

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
https://www.tensorflow.org/tutorials/estimators/linear#command_line_usage

The repo includes a complete program for experimenting with this type of model.

To execute the tutorial code from the command line first add the path to tensorflow/models 
to your PYTHONPATH.
'''
#export PYTHONPATH=${PYTHONPATH}:"$(pwd)/models"
#running from python you need to set the `os.environ` or the subprocess will not see the directory.

if "PYTHONPATH" in os.environ:
    os.environ['PYTHONPATH'] += os.pathsep +  models_path
else:
    os.environ['PYTHONPATH'] = models_path

print(f"os.environ['PYTHONPATH']: {os.environ['PYTHONPATH']}")

# NOTE: I've added models_path to the .bash_profile file to access via the path in the terminal below
'''
    Use --help to see what command line options are available:
        
        $ python -m official.wide_deep.census_main --help

    Now run the model:

        $ python -m official.wide_deep.census_main --model_type=wide --train_epochs=2
'''
#endregion - Command line usage

#region Read the U.S. Census data
'''
    Read the U.S. Census data
    https://www.tensorflow.org/tutorials/estimators/linear#read_the_us_census_data

    This example uses the U.S Census Income Dataset from 1994 and 1995. We have provided the 
    census_dataset.py script to download the data and perform a little cleanup.

    Since the task is a binary classification problem, we'll construct a label column named 
    "label" whose value is 1 if the income is over 50K, and 0 otherwise. For reference, see 
    the input_fn in census_main.py.

    Let's look at the data to see which columns we can use to predict the target label:

    U.S Census Income Dataset - https://archive.ics.uci.edu/ml/datasets/Census+Income
    census_dataset.py - https://github.com/tensorflow/models/tree/master/official/wide_deep/census_dataset.py
    census_main.py    - https://github.com/tensorflow/models/tree/master/official/wide_deep/census_main.py

    At the command line, look at the data to see which columns we can use to predict the target label:

        $ ls  /tmp/census_data/
'''

train_file = "/tmp/census_data/adult.data"
test_file = "/tmp/census_data/adult.test"

'''
    pandas provides some convenient utilities for data analysis. Here's a list of columns available 
    in the Census Income dataset:

    pandas - https://pandas.pydata.org/
'''
import pandas

train_df = pandas.read_csv(train_file, header = None, names = census_dataset._CSV_COLUMNS)
test_df = pandas.read_csv(test_file, header = None, names = census_dataset._CSV_COLUMNS)

print(train_df.head())
'''
The columns are grouped into two types: categorical and continuous columns:

    A column is called categorical if its value can only be one of the categories in a finite set. 
    For example, the relationship status of a person (wife, husband, unmarried, etc.) or the 
    education level (high school, college, etc.) are categorical columns.

    A column is called continuous if its value can be any numerical value in a continuous range. 
    For example, the capital gain of a person (e.g. $14,084) is a continuous column.

'''
#endregion - Read the U.S. Census data

#region Converting Data into Tensors
'''
    Converting Data into Tensors
    https://www.tensorflow.org/tutorials/estimators/linear#converting_data_into_tensors

    When building a tf.estimator model, the input data is specified by using an input function (or 
    input_fn). This builder function returns a tf.data.Dataset of batches of (features-dict, label) 
    pairs. It is not called until it is passed to tf.estimator.Estimator methods such as train and 
    evaluate.

    The input builder function returns the following pair:

        1. features: A dict from feature names to Tensors or SparseTensors containing batches of 
        features.
        2. labels: A Tensor containing batches of labels.

    The keys of the features are used to configure the model's input layer.

    NOTE: The input function is called while constructing the TensorFlow graph, not while running the 
    graph. It is returning a representation of the input data as a sequence of TensorFlow graph 
    operations.

    tf.data.Dataset - https://www.tensorflow.org/api_docs/python/tf/data/Dataset

    For small problems like this, it's easy to make a tf.data.Dataset by slicing the pandas.DataFrame:
'''
def easy_input_function(df, label_key, num_epochs, shuffle, batch_size):
    label = df[label_key]
    ds = tf.data.Dataset.from_tensor_slices((dict(df),label))

    if shuffle:
        ds = ds.shuffle(10000)

    ds = ds.batch(batch_size).repeat(num_epochs)

    return ds

'''
    Since we have eager execution enabled, it's easy to inspect the resulting dataset:
'''
ds = easy_input_function(train_df, label_key='income_bracket', num_epochs=5, shuffle=True, batch_size=10)

for feature_batch, label_batch in ds.take(1):
    print('Some feature keys:', list(feature_batch.keys())[:5])
    print()
    print('A batch of Ages  :', feature_batch['age'])
    print()
    print('A batch of Labels:', label_batch )

'''
    Some feature keys: ['age', 'education', 'fnlwgt', 'hours_per_week', 'workclass']

    A batch of Ages  : tf.Tensor([35 20 46 29 42 40 46 33 36 62], shape=(10,), dtype=int32)

    A batch of Labels: tf.Tensor(
    [b'<=50K' b'<=50K' b'<=50K' b'<=50K' b'>50K' b'<=50K' b'>50K' b'<=50K'
    b'<=50K' b'<=50K'], shape=(10,), dtype=string)
'''

'''
    But this approach has severly-limited scalability. Larger datasets should be streamed from 
    disk. The census_dataset.input_fn provides an example of how to do this using tf.decode_csv 
    and tf.data.TextLineDataset:

    tf.decode_csv - https://www.tensorflow.org/api_docs/python/tf/io/decode_csv
    tf.data.TextLineDataset - https://www.tensorflow.org/api_docs/python/tf/data/TextLineDataset
'''
import inspect
print(inspect.getsource(census_dataset.input_fn))

'''
    def input_fn(data_file, num_epochs, shuffle, batch_size):
        """Generate an input function for the Estimator."""
        assert tf.gfile.Exists(data_file), (
            '%s not found. Please make sure you have run census_dataset.py and '
            'set the --data_dir argument to the correct path.' % data_file)

    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(data_file))
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        classes = tf.equal(labels, '>50K')  # binary classification
        return features, classes

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset
'''

'''
    This input_fn returns equivalent output:
'''
ds = census_dataset.input_fn(train_file, num_epochs=5, shuffle=True, batch_size=10)

for feature_batch, label_batch in ds.take(1):
    print('Feature keys:', list(feature_batch.keys())[:5])
    print()
    print('Age batch   :', feature_batch['age'])
    print()
    print('Label batch :', label_batch )

'''
    Feature keys: ['age', 'education', 'fnlwgt', 'hours_per_week', 'workclass']
    Age batch   : tf.Tensor([35 48 43 64 41 55 31 34 46 44], shape=(10,), dtype=int32)
    Label batch : tf.Tensor([False False  True False  True  True False  True  True  True], shape=(10,), dtype=bool)
'''

'''
    Because Estimators expect an input_fn that takes no arguments, we typically wrap configurable 
    input function into an obejct with the expected signature. For this notebook configure the 
    train_inpf to iterate over the data twice:
'''
import functools

train_inpf = functools.partial(census_dataset.input_fn, train_file, num_epochs=2, shuffle=True, batch_size=64)
test_inpf = functools.partial(census_dataset.input_fn, test_file, num_epochs=1, shuffle=False, batch_size=64)

#endregion - Converting Data into Tensors

#region Selecting and Engineering Features for the Model
'''
    Selecting and Engineering Features for the Model
    https://www.tensorflow.org/tutorials/estimators/linear#selecting_and_engineering_features_for_the_model

    Estimators use a system called feature columns to describe how the model should interpret each of the raw 
    input features. An Estimator expects a vector of numeric inputs, and feature columns describe how the model 
    should convert each feature.

    Selecting and crafting the right set of feature columns is key to learning an effective model. A feature 
    column can be either one of the raw inputs in the original features dict (a base feature column), or any 
    new columns created using transformations defined over one or multiple base columns (a derived feature 
    columns).

    A feature column is an abstract concept of any raw or derived variable that can be used to predict the 
    target label.

    feature columns - https://www.tensorflow.org/guide/feature_columns
'''
#region Base Feature Columns
'''
    Base Feature Columns
    https://www.tensorflow.org/tutorials/estimators/linear#base_feature_columns
'''
#region Numeric columns
'''
    Numeric columns
    https://www.tensorflow.org/tutorials/estimators/linear#numeric_columns

    The simplest feature_column is numeric_column. This indicates that a feature is a numeric value that 
    should be input to the model directly. For example:
'''
age = fc.numeric_column('age')


#endregion - Numeric columns
#endregion - Base Feature Columns



#endregion - Selecting and Engineering Features for the Model