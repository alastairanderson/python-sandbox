<<COMMENT
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

COMMENT

<<RUNNINGTHECODE
    Running the code
    https://github.com/tensorflow/models/tree/master/official/wide_deep#running-the-code

    First make sure you've added the models folder to your Python path; otherwise you may encounter an 
    error like 
        
        ImportError: No module named official.wide_deep.

    https://github.com/tensorflow/models/blob/master/official/#running-the-models



RUNNINGTHECODE
