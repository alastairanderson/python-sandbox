'''
Towards Data Science - Random Forest in Python
https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

requirements.txt
pandas
matplotlib

'''
#region imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
#endregion - imports

#region Retrieve the data
features = pd.read_csv('./data/seattle-temps.csv')
print(features.head(5))

'''
year:    2016 for all data points
month:   number for month of the year
day:     number for day of the year
week:    day of the week as a character string
temp_2:  max temperature 2 days prior
temp_1:  max temperature 1 day prior
average: historical average max temperature
actual:  max temperature measurement
friend:  your friend’s prediction, a random number between 20 below the average and 20 above the average
'''
#endregion - Retrieve the data

#region Explore/Clean the data
print('The shape of our features is:', features.shape)      # (348, 9)

'''
To identify anomalies, compute the summary stats for a cursory glance
'''
print(features.describe())

'''
There are not any data points that immediately appear as anomalous and no zeros in any of 
the measurement columns

TODO: Chart the data to view it - Not highest priority right now/
'''
#endregion - Explore/Clean the data

#region Prepare the data
'''
Perform 
'''
features = pd.get_dummies(features)             # One-hot encode all columns with a text value

# print(features.iloc[:,5:].head(5))            # Display first 5 rows of last 12 columns

labels = np.array(features['actual'])           # separate labels into np array
features = features.drop('actual', axis = 1)    # drop labels
feature_list = list(features.columns)           # retrieve features
features = np.array(features)                   # convert to np array


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, 
                                                                            test_size = 0.25, 
                                                                            random_state = 42)

print('Training Features Shape:', train_features.shape)     # Training Features Shape: (261, 14)
print('Training Labels Shape:', train_labels.shape)         # Training Labels Shape: (261,)
print('Testing Features Shape:', test_features.shape)       # Testing Features Shape: (87, 14)
print('Testing Labels Shape:', test_labels.shape)           # Testing Labels Shape: (87,)
#endregion - Prepare the data

#region Establish a baseline
'''
Establish a baseline

Before we can make and evaluate predictions, we need to establish a baseline, a sensible measure 
that we hope to beat with our model. If our model cannot improve upon the baseline, then it will 
be a failure and we should try a different model or admit that machine learning is not right for 
our problem.

The baseline prediction for our case can be the historical max temperature averages. In other words, 
our baseline is the error we would get if we simply predicted the average max temperature for all 
days.
'''
baseline_preds = test_features[:, feature_list.index('average')]    # The baseline predictions are the historical averages
baseline_errors = abs(baseline_preds - test_labels)                 # Baseline errors, and display average baseline error

print(f'Avg. baseline err: {round(np.mean(baseline_errors), 2)} degrees')    # 5.06 degrees.
#endregion - Establish a baseline

