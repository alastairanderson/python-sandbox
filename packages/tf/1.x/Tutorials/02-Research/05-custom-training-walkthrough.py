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

    1. The first line is a header containing information about the dataset:
        - There are 120 total examples. Each example has four features and one of three possible label names.
    
    2. Subsequent rows are data records, one example per line, where:
        - The first four fields are features: these are characteristics of an example. Here, the fields hold 
          float numbers representing flower measurements.
        - The last column is the label: this is the value we want to predict. For this dataset, it's an 
          integer value of 0, 1, or 2 that corresponds to a flower name.

In code:
'''
# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))     # ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
print("Label: {}".format(label_name))           # species

'''
Each label is associated with string name (for example, "setosa"), but machine learning typically 
relies on numeric values. The label numbers are mapped to a named representation, such as:

    0: Iris setosa
    1: Iris versicolor
    2: Iris virginica

For more information about features and labels, see the ML Terminology section of the Machine Learning 
Crash Course - https://developers.google.com/machine-learning/crash-course/framing/ml-terminology
'''
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

#endregion - Inspect the data

#region Create a tf.data.Dataset
'''
Create a tf.data.Dataset
https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#create_a_tfdatadataset

TensorFlow's Dataset API handles many common cases for loading data into a model. This is a high-level API 
for reading data and transforming it into a form used for training. See the Datasets Quick Start guide for 
more information.

Dataset API                - https://www.tensorflow.org/guide/datasets
Datasets Quick Start guide - https://www.tensorflow.org/get_started/datasets_quickstart

Since the dataset is a CSV-formatted text file, use the make_csv_dataset function to parse the data into a 
suitable format. Since this function generates data for training models, the default behavior is to shuffle 
the data (shuffle=True, shuffle_buffer_size=10000), and repeat the dataset forever (num_epochs=None). We 
also set the batch_size parameter.

make_csv_dataset - https://www.tensorflow.org/api_docs/python/tf/contrib/data/make_csv_dataset
                 - https://www.tensorflow.org/api_docs/python/tf/data/experimental/make_csv_dataset

batch_size       - https://developers.google.com/machine-learning/glossary/#batch_size
'''
batch_size = 32

'''
NOTE: tf.contrib contains experimental code, and will be removed in TF 2.0
'''

train_dataset = tf.data.experimental.make_csv_dataset(train_dataset_fp,
                                                      batch_size,
                                                      column_names=column_names,
                                                      label_name=label_name,
                                                      num_epochs=1)

# type(train_dataset) == <class 'tensorflow.python.data.ops.dataset_ops.DatasetV1Adapter'>

'''
The make_csv_dataset function returns a tf.data.Dataset of (features, label) pairs, where features is a 
dictionary: {'feature_name': value}

tf.data.Dataset - https://www.tensorflow.org/api_docs/python/tf/data/Dataset

With eager execution enabled, these Dataset objects are iterable. Let's look at a batch of features:
'''
features, labels = next(iter(train_dataset))

print(features)
'''
    OrderedDict([('sepal_length', <tf.Tensor: id=65, shape=(32,), dtype=float32, numpy=
        array([4.8, 5.5, 6.5, 4.8, 4.9, 6.8, 7. , 6.3, 6.5, 6.2, 5.3, 6.4, 5.9,
            5.1, 6. , 5.8, 5.8, 6.2, 4.7, 5.6, 6.5, 5. , 6.9, 7.2, 5. , 6.7,
            6.1, 5.7, 6.9, 5. , 6.4, 5. ], dtype=float32)>), 
            ('sepal_width', <tf.Tensor: id=66, shape=(32,), dtype=float32, numpy=
        array([3.4, 2.4, 3. , 3. , 3.1, 3.2, 3.2, 2.7, 3. , 3.4, 3.7, 2.8, 3. ,
            3.7, 2.2, 2.7, 4. , 2.8, 3.2, 2.7, 2.8, 3.2, 3.1, 3.6, 2. , 3.1,
            3. , 2.8, 3.2, 3. , 3.2, 3.4], dtype=float32)>), 
            ('petal_length', <tf.Tensor: id=63, shape=(32,), dtype=float32, numpy=
        array([1.6, 3.8, 5.5, 1.4, 1.5, 5.9, 4.7, 4.9, 5.8, 5.4, 1.5, 5.6, 5.1,
            1.5, 5. , 5.1, 1.2, 4.8, 1.6, 4.2, 4.6, 1.2, 4.9, 6.1, 3.5, 5.6,
            4.9, 4.5, 5.7, 1.6, 5.3, 1.5], dtype=float32)>), 
            ('petal_width', <tf.Tensor: id=64, shape=(32,), dtype=float32, numpy=
        array([0.2, 1.1, 1.8, 0.3, 0.1, 2.3, 1.4, 1.8, 2.2, 2.3, 0.2, 2.2, 1.8,
            0.4, 1.5, 1.9, 0.2, 1.8, 0.2, 1.3, 1.5, 0.2, 1.5, 2.5, 1. , 2.4,
            1.8, 1.3, 2.3, 0.2, 2.3, 0.2], dtype=float32)>)])
'''

'''
Notice that like-features are grouped together, or batched. Each example row's fields are appended 
to the corresponding feature array. Change the batch_size to set the number of examples stored in 
these feature arrays.

You can start to see some clusters by plotting a few features from the batch:
'''
plt.scatter(features['petal_length'].numpy(),
            features['sepal_length'].numpy(),
            c=labels.numpy(),
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()

'''
To simplify the model building step, create a function to repackage the features dictionary into 
a single array with shape: (batch_size, num_features).

This function uses the tf.stack method which takes values from a list of tensors and creates a 
combined tensor at the specified dimension.

tf.stack - https://www.tensorflow.org/api_docs/python/tf/stack
'''
def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

'''
Then use the tf.data.Dataset.map method to pack the features of each (features,label) pair into 
the training dataset:

tf.data.Dataset.map - https://www.tensorflow.org/api_docs/python/tf/data/dataset/map
'''
train_dataset = train_dataset.map(pack_features_vector)

'''
The features element of the Dataset are now arrays with shape (batch_size, num_features). 
Let's look at the first few examples:
'''
features, labels = next(iter(train_dataset))

print(features[:5])

'''
    tf.Tensor(
        [[7.7 3.8 6.7 2.2]
         [5.7 3.  4.2 1.2]
         [5.4 3.7 1.5 0.2]
         [5.4 3.9 1.3 0.4]
         [6.6 3.  4.4 1.4]], shape=(5, 4), dtype=float32)
'''
#endregion - Create a tf.data.Dataset

#region Select the type of model

#endregion - Select the type of model

