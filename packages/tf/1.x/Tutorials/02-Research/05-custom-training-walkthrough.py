'''
    Custom training: walkthrough
    https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough

    This guide uses machine learning to categorize Iris flowers by species. It uses TensorFlow's 
    eager execution (https://www.tensorflow.org/guide/eager) to: 

        1. Build a model, 
        2. Train this model on example data, and 
        3. Use the model to make predictions about unknown data.


    TensorFlow programming
    https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#tensorflow_programming

    This guide uses these high-level TensorFlow concepts:

        Enable an eager execution development environment,   - https://www.tensorflow.org/guide/eager
        Import data with the Datasets API,                   - https://www.tensorflow.org/guide/datasets
        Build models and layers with TensorFlow's Keras API. - https://keras.io/getting-started/sequential-model-guide/

    This tutorial is structured like many TensorFlow programs:

        1. Import and parse the data sets.
        2. Select the type of model.
        3. Train the model.
        4. Evaluate the model's effectiveness.
        5. Use the trained model to make predictions.


    Setup program
    https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#setup_program

    Configure imports and eager execution
    https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#configure_imports_and_eager_execution

    Import the required Python modules—including TensorFlow—and enable eager execution for this program. 
    Eager execution makes TensorFlow evaluate operations immediately, returning concrete values instead of 
    creating a computational graph that is executed later. If you are used to a REPL or the python interactive 
    console, this feels familiar. Eager execution is available in Tensorlow >=1.8.

    Once eager execution is enabled, it cannot be disabled within the same program. See the eager execution guide 
    for more details.

    Graphs and Sessions - https://www.tensorflow.org/guide/graphs
    Eager Execution     - https://www.tensorflow.org/guide/eager
'''
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pyplot as plt

import tensorflow as tf

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

#region The Iris classification problem
'''
The Iris classification problem
https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#the_iris_classification_problem

Imagine you are a botanist seeking an automated way to categorize each Iris flower you find. Machine 
learning provides many algorithms to classify flowers statistically. For instance, a sophisticated 
machine learning program could classify flowers based on photographs. Our ambitions are more modest — 
we're going to classify Iris flowers based on the length and width measurements of their sepals and 
petals.

Links gives a more accurate description along with pictures
'''
#endregion - The Iris classification problem

#region Download the dataset
'''
Download the dataset
https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#download_the_dataset

Download the training dataset file using the tf.keras.utils.get_file function. This returns the 
file path of the downloaded file.

tf.keras.utils.get_file - https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file
'''
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))

#endregion - Download the dataset

#region Inspect the data
'''
Inspect the data
https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#inspect_the_data

This dataset, iris_training.csv, is a plain text file that stores tabular data formatted as 
comma-separated values (CSV). Use the head -n5 command to take a peak at the first five entries:

At the command-line:
    $head -n5 {train_dataset_fp}


From this view of the dataset, notice the following:

    The first line is a header containing information about the dataset:
        There are 120 total examples. Each example has four features and one of three possible label names.
    Subsequent rows are data records, one example per line, where:
        The first four fields are features: these are characteristics of an example. Here, the fields hold float numbers representing flower measurements.
        The last column is the label: this is the value we want to predict. For this dataset, it's an integer value of 0, 1, or 2 that corresponds to a flower name.

In code:
'''


#endregion - Inspect the data



