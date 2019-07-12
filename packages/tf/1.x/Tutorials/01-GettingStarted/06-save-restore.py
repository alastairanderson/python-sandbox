'''
Save and restore models
https://www.tensorflow.org/tutorials/keras/save_and_restore_models

Model progress can be saved during—and after—training. This means a model can resume where it left 
off and avoid long training times. Saving also means you can share your model and others can recreate 
your work. When publishing research models and techniques, most machine learning practitioners share:

    - code to create the model, and
    - the trained weights, or parameters, for the model

Sharing this data helps others understand how the model works and try it themselves with new data.

Caution: Be careful with untrusted code — TensorFlow models are code. 
See Using TensorFlow Securely for details: https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md


Options
https://www.tensorflow.org/tutorials/keras/save_and_restore_models#options

There are different ways to save TensorFlow models—depending on the API you're using. 
This guide uses tf.keras, a high-level API to build and train models in TensorFlow. 
For other approaches, see the TensorFlow Save and Restore 
(https://www.tensorflow.org/guide/saved_model) guide or Saving in eager 
(https://www.tensorflow.org/guide/eager#object-based_saving).


Installs and imports
https://www.tensorflow.org/tutorials/keras/save_and_restore_models#installs_and_imports

Install and import TensorFlow and dependencies:

pip install h5py pyyaml
pip install tf_nightly
'''
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)


#region Retrieve the data
'''
Get an example dataset
https://www.tensorflow.org/tutorials/keras/save_and_restore_models#get_an_example_dataset

We'll use the MNIST dataset to train our model to demonstrate saving weights. 
To speed up these demonstration runs, only use the first 1000 examples:

http://yann.lecun.com/exdb/mnist/
'''
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
#endregion - Retrieve the data

#region Define a model
'''
Define a model
https://www.tensorflow.org/tutorials/keras/save_and_restore_models#define_a_model

Build a simple model we'll use to demonstrate saving and loading weights
'''
# Returns a short sequential model
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model

# Create a basic model instance
model = create_model()
model.summary()
#endregion - Define a model

#region Save checkpoints during training
'''
Save checkpoints during training
https://www.tensorflow.org/tutorials/keras/save_and_restore_models#save_checkpoints_during_training

The primary use case is to automatically save checkpoints during and at the end of training. 
This way you can use a trained model without having to retrain it, or pick-up training where you 
left of—in case the training process was interrupted.

tf.keras.callbacks.ModelCheckpoint is a callback that performs this task. The callback takes a couple 
of arguments to configure checkpointing.

https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
'''
#region Checkpoint callback usage
'''
Checkpoint callback usage
https://www.tensorflow.org/tutorials/keras/save_and_restore_models#checkpoint_callback_usage

Train the model and pass it the ModelCheckpoint callback
'''
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()

model.fit(train_images, train_labels,  epochs = 10,
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback])  # pass callback to training

'''
# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.

This creates a single collection of TensorFlow checkpoint files that are updated at 
the end of each epoch

Check the 'training_1' directory for the files
'''

'''
Create a new, untrained model. When restoring a model from only weights, you must 
have a model with the same architecture as the original model. Since it's the same 
model architecture, we can share weights despite that it's a different instance of 
the model.

Now rebuild a fresh, untrained model, and evaluate it on the test set. An untrained 
model will perform at chance levels (~10% accuracy):
'''
model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

'''
Then load the weights from the checkpoint, and re-evaluate:
'''
model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
#endregion - Checkpoint callback usage

#region Checkpoint callback options
'''
Checkpoint callback options
https://www.tensorflow.org/tutorials/keras/save_and_restore_models#checkpoint_callback_options

The callback provides several options to give the resulting checkpoints unique names, and adjust 
the checkpointing frequency.

Train a new model, and save uniquely named checkpoints once every 5-epochs:
'''
# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_images, train_labels,
          epochs = 50, callbacks = [cp_callback],
          validation_data = (test_images,test_labels),
          verbose=0)

'''
Review the contents of the ./training_2 folder

We can retrieve the latest checkpoint generated with the below:
'''

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)
'''
Note: the default tensorflow format only saves the 5 most recent checkpoints.


To test, reset the model and load the latest checkpoint:
'''
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
#endregion - Checkpoint callback options

#region What are these files?
'''
What are these files?
https://www.tensorflow.org/tutorials/keras/save_and_restore_models#what_are_these_files

The above code stores the weights to a collection of checkpoint-formatted files that contain 
only the trained weights in a binary format. Checkpoints contain: * One or more shards that 
contain your model's weights. * An index file that indicates which weights are stored in a 
which shard.

If you are only training a model on a single machine, you'll have one shard with the suffix: 
.data-00000-of-00001

https://www.tensorflow.org/guide/saved_model#save_and_restore_variables
"TensorFlow saves variables in binary checkpoint files that map variable names to tensor values."
'''
#endregion - What are these files?

#region Manually save weights
'''
Manually save weights
https://www.tensorflow.org/tutorials/keras/save_and_restore_models#manually_save_weights

Above you saw how to load the weights into a model.

Manually saving the weights is just as simple, use the Model.save_weights method.
https://www.tensorflow.org/api_docs/python/tf/keras/Model#save_weights
'''
# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
#endregion - Manually save weights
#endregion - Save checkpoints during training

#region Save the entire model
'''
Save the entire model
https://www.tensorflow.org/tutorials/keras/save_and_restore_models#save_the_entire_model

The entire model can be saved to a file that contains the weight values, the model's configuration, 
and even the optimizer's configuration (depends on set up). This allows you to checkpoint a model 
and resume training later—from the exact same state—without access to the original code.

Saving a fully-functional model is very useful—you can load them in TensorFlow.js (HDF5, Saved 
Model) and then train and run them in web browsers, or convert them to run on mobile devices using 
TensorFlow Lite (HDF5, Saved Model)

TF.js
https://js.tensorflow.org/tutorials/import-keras.html
https://js.tensorflow.org/tutorials/import-saved-model.html

TF Lite
https://www.tensorflow.org/lite/convert/python_api#exporting_a_tfkeras_file_
https://www.tensorflow.org/lite/convert/python_api#exporting_a_savedmodel_
'''
#region As an HDF5 file
'''
As an HDF5 file
https://www.tensorflow.org/tutorials/keras/save_and_restore_models#as_an_hdf5_file
https://en.wikipedia.org/wiki/Hierarchical_Data_Format

Keras provides a basic save format using the HDF5 standard. For our purposes, the saved model 
can be treated as a single binary blob.
'''
model = create_model()

model.fit(train_images, train_labels, epochs=5)

# Save entire model to a HDF5 file
model.save('my_model.h5')

'''
Now recreate the model from that file:
'''
# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

'''
Check its accuracy:
'''
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

'''
This technique saves everything:

    - The weight values
    - The model's configuration(architecture)
    - The optimizer configuration

Keras saves models by inspecting the architecture. Currently, it is not able to save 
TensorFlow optimizers (from tf.train). When using those you will need to re-compile the 
model after loading, and you will lose the state of the optimizer.
'''
#endregion - As an HDF5 file

#region As a saved_model
'''
As a saved_model
https://www.tensorflow.org/tutorials/keras/save_and_restore_models#as_a_saved_model

Caution: This method of saving a tf.keras model is experimental and may change in future versions.

Build a fresh model:
'''
model = create_model()
model.fit(train_images, train_labels, epochs=5)

'''
Create a saved_model:
'''
import time

saved_model_path = "./saved_models/"+str(int(time.time()))
tf.contrib.saved_model.save_keras_model(model, saved_model_path)

'''
Have a look in the ./saved_models directory
'''
new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
new_model.summary()

'''
Run the restored model.
'''
# The model has to be compiled before evaluating.
# This step is not required if the saved model is only being deployed.

new_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

# Evaluate the restored model.
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
#endregion - As a saved_model
#endregion - Save the entire model

#region What's Next
'''
What's Next
https://www.tensorflow.org/tutorials/keras/save_and_restore_models#whats_next

That was a quick guide to saving and loading in with tf.keras.

    The tf.keras guide shows more about saving and loading models with tf.keras.
    See Saving in eager for saving during eager execution.
    The Save and Restore guide has low-level details about TensorFlow saving.

tf.keras         - https://www.tensorflow.org/api_docs/python/tf/keras
tf.keras guide   - https://www.tensorflow.org/guide/keras
Saving in eager  - https://www.tensorflow.org/guide/eager#object_based_saving
Save and Restore - https://www.tensorflow.org/guide/saved_model

'''
#endregion - What's Next
