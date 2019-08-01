<<INTRO
    Predicting Income with the Census Income Dataset
    https://github.com/tensorflow/models/tree/master/official/wide_deep

    The Census Income Data Set contains over 48,000 samples with attributes including age, occupation, 
    education, and income (a binary label, either >50K or <=50K). The dataset is split into roughly 
    32,000 training and 16,000 testing samples.

    https://archive.ics.uci.edu/ml/datasets/Census+Income

    Here, we use the wide and deep model to predict the income labels. The wide model is able to 
    memorize interactions with data with a large number of features but not able to generalize these 
    learned interactions on new data. The deep model generalizes well but is unable to learn exceptions 
    within the data. The wide and deep model combines the two models and is able to generalize while 
    learning exceptions.

    https://research.googleblog.com/2016/06/wide-deep-learning-better-together-with.html

    For the purposes of this example code, the Census Income Data Set was chosen to allow the model to 
    train in a reasonable amount of time. You'll notice that the deep model performs almost as well as 
    the wide and deep model on this dataset. The wide and deep model truly shines on larger data sets 
    with high-cardinality features, where each feature has millions/billions of unique possible values 
    (which is the specialty of the wide model).

    Finally, a key point. As a modeler and developer, think about how this dataset is used and the 
    potential benefits and harm a model's predictions can cause. A model like this could reinforce 
    societal biases and disparities. Is a feature relevant to the problem you want to solve, or will 
    it introduce bias? For more information, read about ML fairness.

    https://developers.google.com/machine-learning/fairness-overview/

    ------------------------------------------------------------------------------------------------

    The code sample in this directory uses the high level tf.estimator.Estimator API. This API is 
    great for fast iteration and quickly adapting models to your own datasets without major code 
    overhauls. It allows you to move from single-worker training to distributed training, and it 
    makes it easy to export model binaries for prediction.

    The input function for the Estimator uses tf.contrib.data.TextLineDataset, which creates a Dataset 
    object. The Dataset API makes it easy to apply transformations (map, batch, shuffle, etc.) to the 
    data: https://www.tensorflow.org/guide/datasets

    The Estimator and Dataset APIs are both highly encouraged for fast development and efficient 
    training.

INTRO

<<RUNNING_THE_CODE
    Running the code
    ----------------

        First make sure you've added the models folder to your Python path; otherwise you may encounter an 
        error like 
            
            ImportError: No module named official.wide_deep.

        https://github.com/tensorflow/models/blob/master/official/#running-the-models

        Note: Alternatively, if running from the command line, you can just navigate to the models directory
        and run the scripts from there ./models/official/wide_deep

    Setup
    -----

        The Census Income Data Set that this sample uses for training is hosted by the UC Irvine Machine Learning 
        Repository. We have provided a script that downloads and cleans the necessary files.

        Census Income Data Set - https://archive.ics.uci.edu/ml/datasets/Census+Income
        UC Irvine Machine Learning Repository - https://archive.ics.uci.edu/ml/datasets/

            $ python census_dataset.py

        This will download the files to /tmp/census_data. To change the directory, set the --data_dir flag.
    
    Training
    --------

        You can run the code locally as follows:

            $ python census_main.py

        The model is saved to /tmp/census_model by default, which can be changed using the --model_dir flag.

        To run the wide or deep-only models, set the --model_type flag to wide or deep. Other flags are 
        configurable as well; see census_main.py for details.

        The final accuracy should be over 83% with any of the three model types.

        You can also experiment with -inter and -intra flag to explore inter/intra op parallelism for 
        potential better performance as follows:

            $ python census_main.py --inter=<int> --intra=<int>

        Note the above optional inter/intra op does not affect model accuracy. These are TensorFlow 
        framework configurations that only affect execution time. 

    TensorBoard

        Run TensorBoard to inspect the details about the graph and training progression.

            tensorboard --logdir=/tmp/census_model --host localhost --port 8088


RUNNING_THE_CODE

<<INFERENCE_WITH_SAVEDMODEL

    You can export the model into Tensorflow SavedModel format by using the argument --export_dir:
    https://www.tensorflow.org/guide/saved_model

        python census_main.py --export_dir /tmp/wide_deep_saved_model

INFERENCE_WITH_SAVEDMODEL
