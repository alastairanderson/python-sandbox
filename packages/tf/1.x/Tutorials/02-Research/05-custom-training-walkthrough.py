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
'''
    Why model?
    https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#why_model

    A model is a relationship between features and the label. For the Iris classification problem, 
    the model defines the relationship between the sepal and petal measurements and the predicted 
    Iris species. Some simple models can be described with a few lines of algebra, but complex 
    machine learning models have a large number of parameters that are difficult to summarize.

    model - https://developers.google.com/machine-learning/crash-course/glossary#model

    Could you determine the relationship between the four features and the Iris species without 
    using machine learning? That is, could you use traditional programming techniques (for example, 
    a lot of conditional statements) to create a model? Perhaps—if you analyzed the dataset long 
    enough to determine the relationships between petal and sepal measurements to a particular 
    species. And this becomes difficult—maybe impossible—on more complicated datasets. A good 
    machine learning approach determines the model for you. If you feed enough representative 
    examples into the right machine learning model type, the program will figure out the 
    relationships for you.


    Select the model
    https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#select_the_model

    We need to select the kind of model to train. There are many types of models and picking a 
    good one takes experience. This tutorial uses a neural network to solve the Iris classification 
    problem. Neural networks can find complex relationships between features and the label. It is a 
    highly-structured graph, organized into one or more hidden layers. Each hidden layer consists 
    of one or more neurons. There are several categories of neural networks and this program uses a 
    dense, or fully-connected neural network: the neurons in one layer receive input connections 
    from every neuron in the previous layer. For example, Figure 2 illustrates a dense neural 
    network consisting of an input layer, two hidden layers, and an output layer:

    [See link for a diagram]

    When the model from Figure 2 is trained and fed an unlabeled example, it yields three predictions: 
    the likelihood that this flower is the given Iris species. This prediction is called inference. 
    For this example, the sum of the output predictions is 1.0. In Figure 2, this prediction breaks 
    down as: 0.02 for Iris setosa, 0.95 for Iris versicolor, and 0.03 for Iris virginica. This means 
    that the model predicts—with 95% probability—that an unlabeled example flower is an Iris 
    versicolor.
'''
#endregion - Select the type of model

#region Create a model using Keras
'''
    Create a model using Keras
    https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#create_a_model_using_keras

    The TensorFlow tf.keras API is the preferred way to create models and layers. This makes it easy to 
    build models and experiment while Keras handles the complexity of connecting everything together.

    The tf.keras.Sequential model is a linear stack of layers. Its constructor takes a list of layer 
    instances, in this case, two Dense layers with 10 nodes each, and an output layer with 3 nodes 
    representing our label predictions. The first layer's input_shape parameter corresponds to the 
    number of features from the dataset, and is required.

    Dense - https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
'''
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
])

'''
    The activation function determines the output shape of each node in the layer. These 
    non-linearities are important — without them the model would be equivalent to a single layer. 
    There are many available activations, but ReLU is common for hidden layers.

    activation - https://developers.google.com/machine-learning/crash-course/glossary#activation_function
    available activations - https://www.tensorflow.org/api_docs/python/tf/keras/activations
    ReLU - https://developers.google.com/machine-learning/crash-course/glossary#ReLU

    The ideal number of hidden layers and neurons depends on the problem and the dataset. Like many 
    aspects of machine learning, picking the best shape of the neural network requires a mixture of 
    knowledge and experimentation. As a rule of thumb, increasing the number of hidden layers and 
    neurons typically creates a more powerful model, which requires more data to train effectively.
'''
#endregion - Create a model using Keras

#region Using the model
'''
    Using the model
    https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#using_the_model

    Let's have a quick look at what this model does to a batch of features:
'''
predictions = model(features)
predictions[:5]
'''
    <tf.Tensor: id=206, shape=(5, 3), dtype=float32, numpy=
        array([[ 0.29601657,  5.3627486 ,  0.3017386 ],
            [ 0.25280747,  3.552406  ,  0.15006669],
            [ 0.39407727,  1.9884037 , -0.13551341],
            [ 0.41629514,  1.8554846 , -0.15992008],
            [ 0.3298331 ,  3.8514755 ,  0.12291278]], dtype=float32)>
'''

'''
    Here, each example returns a logit for each class.

    logit - https://developers.google.com/machine-learning/crash-course/glossary#logits

    To convert these logits to a probability for each class, use the softmax function:

    softmax - https://developers.google.com/machine-learning/crash-course/glossary#softmax
'''
tf.nn.softmax(predictions[:5])
'''
    <tf.Tensor: id=212, shape=(5, 3), dtype=float32, numpy=
        array([[0.0062243 , 0.9875157 , 0.00626001],
            [0.03447786, 0.93441063, 0.03111147],
            [0.1535189 , 0.7560821 , 0.09039897],
            [0.1730314 , 0.7297212 , 0.09724736],
            [0.02804809, 0.9491464 , 0.02280547]], dtype=float32)>
'''

'''
    Taking the tf.argmax across classes gives us the predicted class index. But, the model hasn't 
    been trained yet, so these aren't good predictions.

    tf.argmax - https://www.tensorflow.org/api_docs/python/tf/math/argmax
'''
print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))

'''
    Prediction: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
        Labels: [2 1 0 0 1 1 1 2 2 0 0 0 1 0 1 2 0 2 0 0 2 1 1 2 1 0 1 2 2 1 1 1]
'''
#endregion - Using the model

#region Train the model
'''
    Train the model
    https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#train_the_model

    Training is the stage of machine learning when the model is gradually optimized, or the model learns 
    the dataset. The goal is to learn enough about the structure of the training dataset to make predictions 
    about unseen data. If you learn too much about the training dataset, then the predictions only work for 
    the data it has seen and will not be generalizable. This problem is called overfitting—it's like 
    memorizing the answers instead of understanding how to solve a problem.

    The Iris classification problem is an example of supervised machine learning: the model is trained from 
    examples that contain labels. In unsupervised machine learning, the examples don't contain labels. 
    Instead, the model typically finds patterns among the features.

    Training - https://developers.google.com/machine-learning/crash-course/glossary#training
    Overfitting - https://developers.google.com/machine-learning/crash-course/glossary#overfitting
    Supervised ML - https://developers.google.com/machine-learning/glossary/#supervised_machine_learning
    Unsupervised ML - https://developers.google.com/machine-learning/glossary/#unsupervised_machine_learning
'''
#region Define the loss and gradient function
'''
    Define the loss and gradient function
    https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#define_the_loss_and_gradient_function

    Both training and evaluation stages need to calculate the model's loss. This measures how off a model's 
    predictions are from the desired label, in other words, how bad the model is performing. We want to 
    minimize, or optimize, this value.

    loss - https://developers.google.com/machine-learning/crash-course/glossary#loss

    Our model will calculate its loss using the tf.keras.losses.categorical_crossentropy function which takes 
    the model's class probability predictions and the desired label, and returns the average loss across the 
    examples.

    categorical_crossentropy - https://www.tensorflow.org/api_docs/python/tf/losses/sparse_softmax_cross_entropy
'''
def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


l = loss(model, features, labels)
print("Loss test: {}".format(l))

'''
    Use the tf.GradientTape context to calculate the gradients used to optimize our model. For more examples of 
    this, see the eager execution guide.

    tf.GradientTape       - https://www.tensorflow.org/api_docs/python/tf/GradientTape
    gradients             - https://developers.google.com/machine-learning/crash-course/glossary#gradient
    eager execution guide - https://www.tensorflow.org/guide/eager
'''
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

#endregion - Define the loss and gradient function

#region Create an optimizer
'''
    Create an optimizer
    https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#create_an_optimizer

    An optimizer applies the computed gradients to the model's variables to minimize the loss function. 
    You can think of the loss function as a curved surface (see Figure 3) and we want to find its lowest 
    point by walking around. The gradients point in the direction of steepest ascent—so we'll travel the 
    opposite way and move down the hill. By iteratively calculating the loss and gradient for each batch, 
    we'll adjust the model during training. Gradually, the model will find the best combination of weights 
    and bias to minimize loss. And the lower the loss, the better the model's predictions.

    optimiser - https://developers.google.com/machine-learning/crash-course/glossary#optimizer

    TensorFlow has many optimization algorithms available for training. This model uses the 
    tf.train.GradientDescentOptimizer that implements the stochastic gradient descent (SGD) algorithm. 
    The learning_rate sets the step size to take for each iteration down the hill. This is a hyperparameter 
    that you'll commonly adjust to achieve better results.

    optimization algorithms  - https://www.tensorflow.org/api_guides/python/train
    GradientDescentOptimizer - https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer
    stochastic gradient descent - https://developers.google.com/machine-learning/crash-course/glossary#gradient_descent

    Setup the optimizer and the global_step counter:
'''
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

global_step = tf.Variable(0)

'''
    Use this to calculate a single optimization step:
'''
loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(global_step.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)

print("Step: {},         Loss: {}".format(global_step.numpy(),
                                          loss(model, features, labels).numpy()))
#endregion - Create an optimizer

#region Training loop
'''
    Training loop
    https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#training_loop

    With all the pieces in place, the model is ready for training! A training loop feeds the 
    dataset examples into the model to help it make better predictions. The following code 
    block sets up these training steps:

        1. Iterate each epoch. An epoch is one pass through the dataset.
        2. Within an epoch, iterate over each example in the training Dataset grabbing its 
        features (x) and label (y).
        3. Using the example's features, make a prediction and compare it with the label. 
        Measure the inaccuracy of the prediction and use that to calculate the model's loss 
        and gradients.
        4. Use an optimizer to update the model's variables.
        5. Keep track of some stats for visualization.
        6. Repeat for each epoch.

    The num_epochs variable is the number of times to loop over the dataset collection. 
    Counter-intuitively, training a model longer does not guarantee a better model. num_epochs 
    is a hyperparameter that you can tune. Choosing the right number usually requires both 
    experience and experimentation.
'''
## Note: Rerunning this cell uses the same model variables

from tensorflow import contrib
tfe = contrib.eager

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    # Training loop - using batches of 32
    for x, y in train_dataset:
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                global_step)

        # Track progress
        epoch_loss_avg(loss_value)  # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))
'''
    Epoch 000: Loss: 1.253, Accuracy: 52.500%
    Epoch 050: Loss: 0.481, Accuracy: 85.833%
    Epoch 100: Loss: 0.335, Accuracy: 93.333%
    Epoch 150: Loss: 0.244, Accuracy: 91.667%
    Epoch 200: Loss: 0.188, Accuracy: 95.000%
'''
#endregion - Training loop

#region Visualize the loss function over time
'''
    Visualize the loss function over time
    https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#visualize_the_loss_function_over_time

    While it's helpful to print out the model's training progress, it's often more helpful to see this progress. 
    TensorBoard is a nice visualization tool that is packaged with TensorFlow, but we can create basic charts 
    using the matplotlib module.

    Interpreting these charts takes some experience, but you really want to see the loss go down and the accuracy 
    go up.

    TensorBoard - https://www.tensorflow.org/guide/summaries_and_tensorboard
'''
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()
#endregion - Visualize the loss function over time
#endregion - Train the model

#region Evaluate the model's effectiveness
'''
    Evaluate the model's effectiveness
    https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#evaluate_the_models_effectiveness

    Now that the model is trained, we can get some statistics on its performance.

    Evaluating means determining how effectively the model makes predictions. To determine the model's 
    effectiveness at Iris classification, pass some sepal and petal measurements to the model and ask 
    the model to predict what Iris species they represent. Then compare the model's prediction against 
    the actual label. For example, a model that picked the correct species on half the input examples 
    has an accuracy of 0.5. Figure 4 shows a slightly more effective model, getting 4 out of 5 predictions 
    correct at 80% accuracy:

    accuracy - https://developers.google.com/machine-learning/glossary/#accuracy
'''

#region Setup the test dataset
'''
    Setup the test dataset
    https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#setup_the_test_dataset

    Evaluating the model is similar to training the model. The biggest difference is the examples come 
    from a separate test set rather than the training set. To fairly assess a model's effectiveness, the 
    examples used to evaluate a model must be different from the examples used to train the model.

    The setup for the test Dataset is similar to the setup for training Dataset. Download the CSV text 
    file and parse that values, then give it a little shuffle:

    test set - https://developers.google.com/machine-learning/crash-course/glossary#test_set
'''
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)

test_dataset = tf.contrib.data.make_csv_dataset(test_fp,
                                                batch_size,
                                                column_names=column_names,
                                                label_name='species',
                                                num_epochs=1,
                                                shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)

#endregion - Setup the test dataset

#region Evaluate the model on the test dataset
'''
    Evaluate the model on the test dataset
    https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#evaluate_the_model_on_the_test_dataset

    Unlike the training stage, the model only evaluates a single epoch of the test data. In the following code cell, 
    we iterate over each example in the test set and compare the model's prediction against the actual label. This 
    is used to measure the model's accuracy across the entire test set.

    epoch - https://developers.google.com/machine-learning/glossary/#epoch
'''
test_accuracy = tfe.metrics.Accuracy()

for (x, y) in test_dataset:
    logits = model(x)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
'''
    Test set accuracy: 96.667%

    We can see on the last batch, for example, the model is usually correct:
'''
tf.stack([y,prediction],axis=1)

'''
    <tf.Tensor: id=105687, shape=(30, 2), dtype=int32, numpy=
        array([[1, 1],
               [2, 2],
               [0, 0],
               [1, 1],
               [1, 1],
               [1, 1], 
               ...
               [1, 1]], dtype=int32)>
'''
#endregion - Evaluate the model on the test dataset

#region Use the trained model to make predictions
'''
    Use the trained model to make predictions
    https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough#use_the_trained_model_to_make_predictions

    We've trained a model and "proven" that it's good—but not perfect—at classifying Iris species. Now let's use the 
    trained model to make some predictions on unlabeled examples; that is, on examples that contain features but not 
    a label.

    In real-life, the unlabeled examples could come from lots of different sources including apps, CSV files, and 
    data feeds. For now, we're going to manually provide three unlabeled examples to predict their labels. Recall, 
    the label numbers are mapped to a named representation as:

        0: Iris setosa
        1: Iris versicolor
        2: Iris virginica

    unlabeled examples - https://developers.google.com/machine-learning/glossary/#unlabeled_example
'''
predict_dataset = tf.convert_to_tensor([
    [5.1, 3.3, 1.7, 0.5,],
    [5.9, 3.0, 4.2, 1.5,],
    [6.9, 3.1, 5.4, 2.1]
])

predictions = model(predict_dataset)

for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = class_names[class_idx]
    print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))

'''
Example 0 prediction: Iris setosa (97.6%)
Example 1 prediction: Iris versicolor (89.3%)
Example 2 prediction: Iris virginica (50.1%)
'''

#endregion - Use the trained model to make predictions
#endregion - Evaluate the model's effectiveness