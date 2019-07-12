'''
https://www.tensorflow.org/tutorials/keras/basic_regression

Regression: predict fuel efficiency

In a regression problem, we aim to predict the output of a continuous value, like a price or a probability. 
Contrast this with a classification problem, where we aim to select a class from a list of classes (for 
example, where a picture contains an apple or an orange, recognizing which fruit is in the picture).

This example uses the classic Auto MPG Dataset and builds a model to predict the fuel efficiency of 
late-1970s and early 1980s automobiles. To do this, we'll provide the model with a description of 
many automobiles from that time period. This description includes attributes like: 
cylinders, displacement, horsepower, and weight.

Auto MPG Dataset
https://archive.ics.uci.edu/ml/datasets/auto+mpg

tf.keras API
https://www.tensorflow.org/api_docs/python/tf/keras

Keras Guide
https://www.tensorflow.org/guide/keras

# requirements.txt
tensorflow                  # https://pypi.org/project/tensorflow/
matplotlib                  # https://pypi.org/project/matplotlib/
seaborn                     # https://pypi.org/project/seaborn/
pandas                      # https://pypi.org/project/pandas/

# Using seaborn for pairplot
'''

#region imports
from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
#endregion - imports

#region Settings
SHOW_PLOTS = True
#endregion - Settings

#region Retrieve the data
'''
Get the data
https://www.tensorflow.org/tutorials/keras/basic_regression#get_the_data

The Auto MPG dataset available here: https://archive.ics.uci.edu/ml/
'''
dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path

'''
Import it using pandas
'''
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values = "?", comment='\t',
                          sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
# type(dataset)                     # <class 'pandas.core.frame.DataFrame'>
print(dataset.tail())
#endregion - Retrieve the data

#region Clean the data
'''
Clean the data
https://www.tensorflow.org/tutorials/keras/basic_regression#clean_the_data

The dataset contains a few unknown values.
'''
print(dataset.isna().sum())

'''
For simplicity in this tutorial just drop this data
'''
dataset = dataset.dropna()

'''
The "Origin" column is really categorical, not numeric. So convert that to a one-hot

NOTE: There are easier ways to do this!!
'''
origin = dataset.pop('Origin')

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
print(dataset.tail())
#endregion - Clean the data

#region Prepare the data
'''
Split the data into train and test
https://www.tensorflow.org/tutorials/keras/basic_regression#split_the_data_into_train_and_test

Now split the dataset into a training set and a test set.
We will use the test set in the final evaluation of our model.
'''
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# type(train_dataset, test_dataset)         # <class 'pandas.core.frame.DataFrame'> 

'''
Inspect the data
https://www.tensorflow.org/tutorials/keras/basic_regression#inspect_the_data

Have a quick look at the joint distribution of a few pairs of columns from the training set.
'''
if SHOW_PLOTS:
    sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
    plt.show()

'''
Also look at the overall statistics:
'''
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

'''
Split features from labels
https://www.tensorflow.org/tutorials/keras/basic_regression#split_features_from_labels

Separate the target value, or "label", from the features. This label is the value that you will 
train the model to predict.
'''
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

'''
Normalize the data
https://www.tensorflow.org/tutorials/keras/basic_regression#normalize_the_data

Look again at the train_stats block above and note how different the ranges of each feature are.

It is good practice to normalize features that use different scales and ranges. Although the model 
might converge without feature normalization, it makes training more difficult, and it makes the 
resulting model dependent on the choice of units used in the input. 

Note: Although we intentionally generate these statistics from only the training dataset, these 
statistics will also be used to normalize the test dataset. We need to do that to project the test 
dataset into the same distribution that the model has been trained on.

This normalized data is what we will use to train the model.
'''
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

'''
Caution: The statistics used to normalize the inputs here (mean and standard deviation) need to be 
applied to any other data that is fed to the model, along with the one-hot encoding that we did 
earlier. That includes the test set as well as live data when the model is used in production.
'''
#endregion - Prepare the data

#region Build the model
'''
Build the model
https://www.tensorflow.org/tutorials/keras/basic_regression#build_the_model

Let's build our model. Here, we'll use a Sequential model with two densely connected hidden layers, and 
an output layer that returns a single, continuous value. The model building steps are wrapped in a 
function, build_model, since we'll create a second model, later on.
'''
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    '''
    https://keras.io/optimizers/
    '''
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

model = build_model()
#endregion - Build the model

#region Inspect the model
'''
Inspect the model
https://www.tensorflow.org/tutorials/keras/basic_regression#inspect_the_model

Use the .summary method to print a simple description of the model
'''
model.summary()

'''
Now try out the model. Take a batch of 10 examples from the training data and call model.predict on it.
'''
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)
#endregion - Inspect the model

#region Train the model
'''
Train the model
https://www.tensorflow.org/tutorials/keras/basic_regression#train_the_model

Train the model for 1000 epochs, and record the training and validation accuracy in the history object
'''
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

'''
The number of epochs is a hyperparameter that defines the number times that the learning algorithm 
will work through the entire training dataset

https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
'''
EPOCHS = 1000

history = model.fit(normed_train_data, train_labels,
                    epochs=EPOCHS, validation_split = 0.2, verbose=0,
                    callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
            label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
            label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()

plot_history(history)

'''
This graph shows little improvement, or even degradation in the validation error after about 100 epochs. 
Let's update the model.fit call to automatically stop training when the validation score doesn't improve. 
We'll use an EarlyStopping callback that tests a training condition for every epoch. If a set amount of 
epochs elapses without showing improvement, then automatically stop the training.

You can learn more about this callback here:
https://www.tensorflow.org/versions/master/api_docs/python/tf/keras/callbacks/EarlyStopping
'''

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

'''
The graph shows that on the validation set, the average error is usually around +/- 2 MPG. 
Is this good? We'll leave that decision up to you.

Let's see how well the model generalizes by using the test set, which we did not use when 
training the model. This tells us how well we can expect the model to predict when we use 
it in the real world.
'''
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
#endregion - Train the model

#region Make predictions
'''
Make predictions
https://www.tensorflow.org/tutorials/keras/basic_regression#make_predictions

Finally, predict MPG values using data in the testing set:
'''
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

'''
It looks like our model predicts reasonably well. Let's take a look at the 
error distribution.
'''

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()

'''
It's not quite gaussian, but we might expect that because the number of samples 
is very small.
'''
#endregion - Make predictions

#region Conclusions
'''
Conclusions
https://www.tensorflow.org/tutorials/keras/basic_regression#conclusion

This example introduced a few techniques to handle a regression problem.

    - Mean Squared Error (MSE) is a common loss function used for regression problems (different loss 
      functions are used for classification problems).

    - Similarly, evaluation metrics used for regression differ from classification. A common regression 
      metric is Mean Absolute Error (MAE).

    - When numeric input data features have values with different ranges, each feature should be scaled 
      independently to the same range.

    - If there is not much training data, one technique is to prefer a small network with few hidden 
      layers to avoid overfitting.
    
    - Early stopping is a useful technique to prevent overfitting.
'''
#endregion - Conclusions