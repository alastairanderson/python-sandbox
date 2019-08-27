'''
https://www.tensorflow.org/tutorials/keras/basic_classification

requirements.txt
tensorflow==1.14.0
matplotlib==3.1.1
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

#region Settings
SHOW_PLOTS = False
#endregion - Settings

#region Retrieve the data
'''
Import the Fashion MNIST dataset
https://www.tensorflow.org/tutorials/keras/basic_classification#import_the_fashion_mnist_dataset

NOTE: First time you call load_data() an internet connection is required to pull the data from the internet
'''
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# print(type(train_images))       # <class 'numpy.ndarray'>
# print(type(train_labels))       # <class 'numpy.ndarray'>
# print(type(test_images))        # <class 'numpy.ndarray'>
# print(type(test_labels))        # <class 'numpy.ndarray'>

# the class names are not included with the dataset, store them here to use later when plotting the images
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#endregion - Retrieve the data

#region Explore the data
'''
Explore the data
https://www.tensorflow.org/tutorials/keras/basic_classification#explore_the_data
'''
print(train_images.shape)               # (60000, 28, 28)
print(len(train_labels))                # 60000
print(train_labels)                     # array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)
print(test_images.shape)                # (10000, 28, 28)
print(len(test_labels))                 # 10000
#endregion - Explore the data

#region Prepare the data
'''
Preprocess the data
https://www.tensorflow.org/tutorials/keras/basic_classification#preprocess_the_data

The data must be preprocessed before training the network. If you inspect the first image in the training set, 
you will see that the pixel values fall in the range of 0 to 255:
'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
if SHOW_PLOTS:
    plt.show()

'''
We scale these values to a range of 0 to 1 before feeding to the neural network model. For this, we divide 
the values by 255. It's important that the training set and the testing set are preprocessed in the same way:
'''
train_images = train_images / 255.0
test_images = test_images / 255.0

'''
Display the first 25 images from the training set and display the class name below each image. Verify that 
the data is in the correct format and we're ready to build and train the network.
'''
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
if SHOW_PLOTS:
    plt.show()
#endregion - Prepare the data

#region Build the model
'''
Build the model
https://www.tensorflow.org/tutorials/keras/basic_classification#build_the_model

Building the neural network requires configuring the layers of the model, then compiling the model.

Setup the layers
https://www.tensorflow.org/tutorials/keras/basic_classification#setup_the_layers

- The basic building block of a neural network is the layer. 
- Layers extract representations from the data fed into them. 
- (And, hopefully,) these representations are more meaningful for the problem at hand.

Most of deep learning consists of chaining together simple layers. 
Most layers, like tf.keras.layers.Dense, have parameters that are learned during training.

https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
'''
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

'''
Note: Sequential sets up the steps of the different layers of the NN.

https://www.tensorflow.org/api_docs/python/tf/keras/Sequential

The first layer in this network, tf.keras.layers.Flatten, transforms the format of the images from 
a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels. 

Think of this layer as unstacking rows of pixels in the image and lining them up. 
This layer has no parameters to learn; it only reformats the data.

https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten

After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers. 
These are densely-connected, or fully-connected, neural layers. 

The first Dense layer has 128 nodes (or neurons). 
The second (and last) layer is a 10-node softmax layer—this returns an array of 10 probability scores that sum to 1. 
Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.
'''
#endregion - Build the model

#region Compile the model
'''
Compile the model
https://www.tensorflow.org/tutorials/keras/basic_classification#compile_the_model

Before the model is ready for training, it needs a few more settings. 
These are added during the model's compile step:

- Loss function — This measures how accurate the model is during training. We want to minimize this 
                  function to "steer" the model in the right direction.

- Optimizer     — This is how the model is updated based on the data it sees and its loss function.

- Metrics       — Used to monitor the training and testing steps. The following example uses accuracy, 
                  the fraction of the images that are correctly classified.
'''
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#endregion - Compile the model

#region Train the model
'''
Train the model
https://www.tensorflow.org/tutorials/keras/basic_classification#train_the_model

Training the neural network model requires the following steps:

    1. Feed the training data to the model—in this example, the train_images and train_labels arrays.
    2. The model learns to associate images and labels.
    3. We ask the model to make predictions about a test set—in this example, the test_images array. 
       We verify that the predictions match the labels from the test_labels array.

To start training, call the model.fit method—the model is "fit" to the training data:
'''
model.fit(train_images, train_labels, epochs=5)

'''
As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of 
about 0.88 (or 88%) on the training data.
'''
#endregion - Train the model

#region Evaluate accuracy
'''
Evaluate accuracy
https://www.tensorflow.org/tutorials/keras/basic_classification#evaluate_accuracy

Compare how the model performs on the test dataset:
'''
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

'''
the accuracy on the test dataset is a little less than the accuracy on the training dataset. 
This gap between training accuracy and test accuracy is an example of overfitting. 
Overfitting is when a machine learning model performs worse on new data than on their training data.
'''
#endregion - Evaluate accuracy

#region Make predictions
'''
Make predictions
https://www.tensorflow.org/tutorials/keras/basic_classification#make_predictions

With the model trained, we can use it to make predictions about some images.
'''
predictions = model.predict(test_images)

'''
Here, the model has predicted the label for each image in the testing set. Let's take a look at 
the first prediction:
'''
print(predictions[0])

'''
A prediction is an array of 10 numbers. These describe the "confidence" of the model that the image 
corresponds to each of the 10 different articles of clothing. We can see which label has the highest 
confidence value:
'''
print(np.argmax(predictions[0]))            # 9 - Boot

'''
So the model is most confident that this image is an ankle boot, or class_names[9]. And we can check 
the test label to see this is correct:
'''
print(test_labels[0])                       # 9 - boot

'''
We can graph this to look at the full set of 10 class predictions
'''
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
  
    plt.imshow(img, cmap=plt.cm.binary)
  
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
  
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
        100*np.max(predictions_array),
        class_names[true_label]),
        color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
  
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

'''
look at the 0th image, predictions, and prediction array.
'''
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
if SHOW_PLOTS:
    plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
if SHOW_PLOTS:
    plt.show()

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
if SHOW_PLOTS:
    plt.show()

'''
Finally, use the trained model to make a prediction about a single image.
'''
img = test_images[0]
print(img.shape)                    # (28, 28)

'''
tf.keras models are optimized to make predictions on a batch, or collection, of examples at once. 
So even though we're using a single image, we need to add it to a list:
'''
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)                    # (1, 28, 28)

'''
Now predict the image:
'''
predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
if SHOW_PLOTS:
    plt.show()

'''
model.predict returns a list of lists, one for each image in the batch of data. 
Grab the predictions for our (only) image in the batch:
'''
prediction_result = np.argmax(predictions_single[0])
print(prediction_result)
#endregion - Make predictions