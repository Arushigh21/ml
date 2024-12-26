# -*- coding: utf-8 -*-
"""Random Forest Classifier



#### Data Preparation
"""

# Load the dataset.
# Import the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Load the dataset.
file_path = 'glass-types.csv'
df = pd.read_csv(file_path, header = None)

# # Drop the 0th column as it contains only the serial numbers.
df.drop(columns = 0, inplace = True)

# A Python list containing the suitable column headers as string values. Also, create a Python dictionary as described above.
column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']

# Required Python dictionary.
columns_dict = {}
for i in df.columns:
  columns_dict[i] = column_headers[i - 1]

# Rename the columns.
df.rename(columns_dict, axis = 1, inplace = True)

# Display the first five rows of the data-frame.
print(df.head(), "\n")

# Get the information about the dataset.
print(df.info(), "\n")

# Get the count of each glass-type sample in the dataset.
print(df['GlassType'].value_counts(), "\n")

# Get the percentage of each glass-type sample in the dataset.
round(df['GlassType'].value_counts() * 100 / df.shape[0], 2)

"""Through percentages, we can clearly see the imbalance in the dataset.

---

#### Preliminary Model Building
"""

# Create separate data-frames for training and testing the model.
from sklearn.model_selection import train_test_split

# Creating the features data-frame holding all the columns except the last column
x = df.iloc[:, :-1]
print(f"First five rows of the features data-frame:\n{x.head()}\n")

# Creating the target series that holds last column 'RainTomorrow'
y = df['GlassType']
print(f"First five rows of the GlassType column:\n{y.head()}")

# Splitting the train and test sets using the 'train_test_split()' function.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

"""---

####  Building Random Forest Classifier Model
"""

#  Build a random forest classifier model to predict different glass-types.
# Import the 'RandomForestClassifier' module.
from sklearn.ensemble import RandomForestClassifier

# Create an object of the 'RandomForestClassifier' class and store it in the 'rf_clf' variable.
rf_clf = RandomForestClassifier()

# Call the 'fit()' function on the 'RandomForestClassifier' object with 'x_train' and 'y_train' as inputs.
rf_clf.fit(x_train, y_train)

# Call the 'score()' function with 'x_train' and 'y_train' as inputs to check the accuracy score of the model.
rf_clf.score(x_train, y_train)

"""So the accuracy score obtained is 100% on the train set. Let's make a confusion matrix and print the f1-scores."""

#  Make predictions on the train set and print the count of each of the classes predicted.
rf_y_train_pred = pd.Series(rf_clf.predict(x_train))
rf_y_train_pred.value_counts()

"""So all the classes have been identified which is expected because the accuracy score is 1."""

#  Apply the 'SMOTE()' function to balance the training data.

# Import the 'SMOTE' module from the 'imblearn.over_sampling' library.
from imblearn.over_sampling import SMOTE

# Call the 'SMOTE()' function and store it in the 'smote' variable.
smote = SMOTE(sampling_strategy = "all", random_state = 42)

# Call the 'fit_sample()' function with 'x_train' and 'y_train' datasets as inputs.
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

#  Create the confusion matrix between the actual and the predicted values for the train set.
from sklearn.metrics import confusion_matrix, classification_report
labels = pd.Series(y_train_resampled).sort_values(ascending = True).unique()
rf_train_conf_matrix = confusion_matrix(y_train, rf_y_train_pred)

# Create a Pandas DataFrame object for the confusion matrix created above labelled with the classes.
rf_train_cm_df = pd.DataFrame(rf_train_conf_matrix, columns = labels, index = labels)

# Create a heatmap for the confusion matrix.
plt.figure(figsize = (10, 5), dpi = 96)
sns.heatmap(rf_train_cm_df, annot = True)
plt.show()

"""The above confusion matrix contains only true positive values. So we don't need to calculate the f1-scores because they all will be 1 for all the glass-types. So let's create a confusion matrix and calculate f1-scores for the test set."""

# Create the confusion matrix between the actual and predicted values for the test set.
rf_y_test_pred = pd.Series(rf_clf.predict(x_test))

rf_test_conf_matrix = confusion_matrix(y_test, rf_y_test_pred)

# Create a Pandas DataFrame object for the confusion matrix created above labelled with the classes.
rf_test_cm_df = pd.DataFrame(rf_test_conf_matrix, columns = labels, index = labels)

# Create a heatmap for the confusion matrix.
plt.figure(figsize = (10, 5), dpi = 96)
sns.heatmap(rf_test_cm_df, annot = True)
plt.show()

"""From the confusion matrix, we can see that there is some misclassification. Let's print the f1-scores for the test set values."""

# Print the classification report for the test set.
print(classification_report(y_test, rf_y_test_pred))

"""So the f1-score is low only for class `3` maybe because it is a minority class. The random forest classifier is working well for all other classes on the non-resampled dataset because for all the labels, the f1-scores are greater than 0.50.

Let's see how it performs when we build it again on the resampled train set.
"""

#  Build a random forest classifier model on the resampled train set.
rf_clf_res = RandomForestClassifier()

# Call the 'fit()' function on the 'RandomForestClassifier' object with 'x_train' and 'y_train' as inputs.
rf_clf_res.fit(x_train_resampled, y_train_resampled)

# Call the 'score()' function with 'x_train' and 'y_train' as inputs to check the accuracy score of the model.
rf_clf_res.score(x_train_resampled, y_train_resampled)

"""The accuracy score is almost 100% in this case."""

# Create a confusion matrix on the test set directly.
# Get the predicted labels on the test set obtained from the RFC model built on the resample train set.
rf_y_test_pred_res = pd.Series(rf_clf_res.predict(x_test))

# Create a confusion matrix.
rf_test_conf_matrix_res = confusion_matrix(y_test, rf_y_test_pred_res)

# Create a Pandas DataFrame object for the confusion matrix created above and labelled with the classes.
rf_test_cm_df_res = pd.DataFrame(rf_test_conf_matrix_res, columns = labels, index = labels)

# Create a heatmap for the confusion matrix.
plt.figure(figsize = (10, 5), dpi = 96)
sns.heatmap(rf_test_cm_df_res, annot = True)
plt.show()

"""From the above confusion matrix, we can see that there is some misclassification of the classes for the test set.

Let's print the classification report to see if there is any further improvement in the f1-scores especially for class `3`.
"""

# Print the classification report for the test set.
print(classification_report(y_test, rf_y_test_pred_res))

"""From the classification report, we can see that the f1-scores have decreased probably because of oversampling in the case of the random forest classifier based multi-class classification model. Hence, the previous random forest classifier model (without oversampling) is the most accurate one for this problem statement.



"""