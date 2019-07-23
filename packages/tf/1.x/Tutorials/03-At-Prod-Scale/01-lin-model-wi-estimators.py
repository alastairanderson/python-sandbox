'''
    Build a linear model with Estimators
    https://www.tensorflow.org/tutorials/estimators/linear

    This tutorial uses the tf.estimator API in TensorFlow to solve a benchmark binary classification 
    problem: https://www.tensorflow.org/api_docs/python/tf/estimator
    
    Estimators are TensorFlow's most scalable and production-oriented model type. 
    
    For more information see the Estimator guide: https://www.tensorflow.org/guide/estimators

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

tf.enable_eager_execution()             # https://www.tensorflow.org/guide/eager
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

'''
    The model will use the feature_column definitions to build the model input. You can inspect the 
    resulting output using the input_layer function:
'''
print(fc.input_layer(feature_batch, [age]).numpy())
'''
    [[18.]
    [45.]
    [23.]
    [17.]
    [32.]
    [38.]
    [37.]
    [54.]
    [25.]
    [71.]]

    The following will train and evaluate a model using only the age feature:
'''
classifier = tf.estimator.LinearClassifier(feature_columns=[age])
classifier.train(train_inpf)
result = classifier.evaluate(test_inpf)

clear_output()  # used for display in notebook
print(result)
'''
    {
        'precision': 0.1780822, 
        'label/mean': 0.23622628, 
        'accuracy_baseline': 0.76377374, 
        'prediction/mean': 0.23925366, 
        'auc': 0.6783024, 
        'loss': 33.404175, 
        'recall': 0.0033801352, 
        'average_loss': 0.5231905, 
        'auc_precision_recall': 0.31137863, 
        'accuracy': 0.7608869, 
        'global_step': 1018
    }

    Similarly, we can define a NumericColumn for each continuous feature column that we want to 
    use in the model:
'''
education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')

my_numeric_columns = [age,education_num, capital_gain, capital_loss, hours_per_week]

print(fc.input_layer(feature_batch, my_numeric_columns).numpy())
'''
    array([[3.500e+01, 0.000e+00, 0.000e+00, 1.100e+01, 4.000e+01],
        [4.800e+01, 0.000e+00, 0.000e+00, 5.000e+00, 4.000e+01],
            ...
        [4.400e+01, 4.386e+03, 0.000e+00, 1.400e+01, 4.000e+01]],
        dtype=float32)

    You could retrain a model on these features by changing the feature_columns argument 
    to the constructor:
'''
classifier = tf.estimator.LinearClassifier(feature_columns=my_numeric_columns)
classifier.train(train_inpf)

result = classifier.evaluate(test_inpf)

clear_output()      # used for display in notebook

for key,value in sorted(result.items()):
    print('%s: %s' % (key, value))
'''
    accuracy: 0.78391993
    accuracy_baseline: 0.76377374
    auc: 0.68254405
    auc_precision_recall: 0.5000775
    average_loss: 0.991552
    global_step: 1018
    label/mean: 0.23622628
    loss: 63.30768
    precision: 0.6261538
    prediction/mean: 0.21659191
    recall: 0.21164846
'''
#endregion - Numeric columns

#region Categorical columns
'''
    Categorical columns
    https://www.tensorflow.org/tutorials/estimators/linear#categorical_columns

    To define a feature column for a categorical feature, create a CategoricalColumn using one of 
    the tf.feature_column.categorical_column* functions.

    If you know the set of all possible feature values of a column—and there are only a few of 
    them — use categorical_column_with_vocabulary_list. Each key in the list is assigned an 
    auto-incremented ID starting from 0. For example, for the relationship column we can assign 
    the feature string Husband to an integer ID of 0 and "Not-in-family" to 1, etc.
'''
relationship = fc.categorical_column_with_vocabulary_list(
    'relationship',
    ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])

'''
    This creates a sparse one-hot vector from the raw input feature.

    The input_layer function we're using is designed for DNN models and expects dense inputs. 
    To demonstrate the categorical column we must wrap it in a tf.feature_column.indicator_column 
    to create the dense one-hot output (Linear Estimators can often skip this dense-step).

    indicator_column - https://www.tensorflow.org/api_docs/python/tf/feature_column/indicator_column

    NOTE: the other sparse-to-dense option is tf.feature_column.embedding_column.

    embedding_column - https://www.tensorflow.org/api_docs/python/tf/feature_column/embedding_column

    Run the input layer, configured with both the age and relationship columns:
'''
print(fc.input_layer(feature_batch, [age, fc.indicator_column(relationship)]))
'''
    <tf.Tensor: id=5100, shape=(10, 7), dtype=float32, numpy=
        array([[35.,  0.,  0.,  0.,  1.,  0.,  0.],
                [48.,  1.,  0.,  0.,  0.,  0.,  0.],
                ...
                [46.,  1.,  0.,  0.,  0.,  0.,  0.],
                [44.,  0.,  0.,  1.,  0.,  0.,  0.]], dtype=float32)>

    If we don't know the set of possible values in advance, use the categorical_column_with_hash_bucket 
    instead:
'''
occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000)

'''
    Here, each possible value in the feature column occupation is hashed to an integer ID as we 
    encounter them in training. The example batch has a few different occupations:
'''
for item in feature_batch['occupation'].numpy():
    print(item.decode())

'''
    Prof-specialty
    Machine-op-inspct
    Sales
    Craft-repair
    Craft-repair
    Sales
    Prof-specialty
    Prof-specialty
    Other-service
    Prof-specialty

    If we run input_layer with the hashed column, we see that the output shape is 
    (batch_size, hash_bucket_size):
'''
occupation_result = fc.input_layer(feature_batch, [fc.indicator_column(occupation)])

print(occupation_result.numpy().shape)
'''
    (10, 1000)

    It's easier to see the actual results if we take the tf.argmax over the hash_bucket_size dimension. 
    Notice how any duplicate occupations are mapped to the same pseudo-random index:
'''
print(tf.argmax(occupation_result, axis=1).numpy())
'''
    array([979, 911, 631, 466, 466, 631, 979, 979, 527, 979])

    NOTE: Hash collisions are unavoidable, but often have minimal impact on model quality. 
          The effect may be noticable if the hash buckets are being used to compress the input space. 
          See this notebook for a more visual example of the effect of these hash collisions.

          https://colab.research.google.com/github/tensorflow/models/blob/master/samples/outreach/blogs/housing_prices.ipynb

    No matter how we choose to define a SparseColumn, each feature string is mapped into an integer ID by 
    looking up a fixed mapping or by hashing. Under the hood, the LinearModel class is responsible for 
    managing the mapping and creating tf.Variable to store the model parameters (model weights) for each 
    feature ID. The model parameters are learned through the model training process described later.

        tf.Variable - https://www.tensorflow.org/api_docs/python/tf/Variable

    Let's do the similar trick to define the other categorical features:
'''



#endregion - Categorical columns

#endregion - Base Feature Columns



#endregion - Selecting and Engineering Features for the Model