# -*- coding: utf-8 -*-
"""NaiveBayesClassifier



---
"""

#  Import the required modules and load the game play dataset. Also, display the first five rows.

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# Read the csv file and create the DataFrame
df_play = pd.read_csv('play.csv')
df_play.head()

#  Calculate Prior Probabilities

# Obtain total 'Yes' and 'No' values for 'Play'
num_yes = (df_play['Play'] == "Yes").sum()
num_no = (df_play['Play'] == "No").sum()

# Calculate prior probabilities from Yes and No occurrences in the 'Play' Column
P_Yes = num_yes / df_play.shape[0]
P_No = num_no / df_play.shape[0]

# Print the probability
print("P(Yes): ", P_Yes)
print("P(No): ", P_No)

#  Calculate Conditional Probability or Likelihood for 'Outlook'
P_Sunny_Yes = len(df_play[(df_play['Outlook'] == "Sunny") & (df_play['Play'] == "Yes")]) / num_yes
P_Sunny_No = len(df_play[(df_play['Outlook'] == "Sunny") & (df_play['Play'] == "No")]) / num_no
print("P(Sunny|Yes):", P_Sunny_Yes)
print("P(Sunny|No):", P_Sunny_No)

#  Calculate Conditional Probability or Likelihood for 'Temperature'
P_Cool_Yes = len(df_play[(df_play['Temperature'] == "Cool") & (df_play['Play'] == "Yes")]) / num_yes
P_Cool_No = len(df_play[(df_play['Temperature'] == "Cool") & (df_play['Play'] == "No")]) / num_no
print("P(Cool|Yes):", P_Cool_Yes)
print("P(Cool|No):", P_Cool_No)

#  Calculate Conditional Probability or Likelihood for 'Humidity'
P_High_Yes = len(df_play[(df_play['Humidity'] == "High") & (df_play['Play'] == "Yes")]) / num_yes
P_High_No = len(df_play[(df_play['Humidity'] == "High") & (df_play['Play'] == "No")]) / num_no
print("P(High|Yes):", P_High_Yes)
print("P(High|No):", P_High_No)

#  Calculate Conditional Probability or Likelihood for 'Wind'
P_Strong_Yes = len(df_play[(df_play['Wind'] == "Strong") & (df_play['Play'] == "Yes")]) / num_yes
P_Strong_No = len(df_play[(df_play['Wind'] == "Strong") & (df_play['Play'] == "No")]) / num_no
print("P(Strong|Yes):", P_Strong_Yes)
print("P(Strong|No):", P_Strong_No)

"""Now that we have obtained the likelihood for each features, we need to obtain final value of $P(X|c_1)$ and $P(X|c_2)$. For this, simply multiply the likelihood values of all the features, as the Naive Bayes Classifier assumes that the feature values are independent of one another.

Thus,

\begin{align}
P(X|c_1): P(\text {Sunny, Cool, High, Strong}|\text {Yes}) &= P(Sunny|Yes) \times P(Cool|Yes) \times P(High|Yes) \times P(Strong|Yes) \\ &=\frac {2}{9} \times \frac{3}{9} \times \frac{3}{9} \times \frac{3}{9}
\end{align}

\begin{align}
P(X|c_2): P(\text {Sunny, Cool, High, Strong}|\text {No}) &= P(Sunny|No) \times P(Cool|No) \times P(High|No) \times P(Strong|No) \\ &=\frac {3}{5} \times \frac{1}{5} \times \frac{4}{5} \times \frac{3}{5}
\end{align}

Let us perform this calculation using Python.


"""

#  Calculate final likelihood or conditional probabilities for 'Yes' and 'No' values
P_X_c1 = P_Sunny_Yes * P_Cool_Yes * P_High_Yes * P_Strong_Yes
P_X_c2 = P_Sunny_No * P_Cool_No * P_High_No * P_Strong_No
print("P(X|c1):", P_X_c1)
print("P(X|c2):", P_X_c2)

#  Evaluate Game Play Probabilities Using Bayes' Theorem

# Probability for having a game
P_c1_X = P_X_c1 * P_Yes
print("Probability of having a game : ", P_c1_X)

# Probability for not having a game
P_c2_X = P_X_c2 * P_No
print("Probability of not having a game : ", P_c2_X)

"""From the output, you may observe that:   
 $$P(c_1 | \text{X}) < P(c_2 | \text{X})$$ i.e.
$$P(Yes | \text{X}) < P(No | \text{X})$$

**Answer:** Since the probability for **not** having a game is higher when compared with probability for having a game, we predict that the outcome for day  15 is **No**.

| Day | Outlook | Temperature | Humidity | Wind | Play |
| -- | -- | -- | -- | -- | --|
| 15 | Sunny | Cool | High | Strong | <font color = red> **No** </font> |

**Summarising the steps of Naive Bayes Classifier:**
1. Calculate prior probability $P(c_i)$ for each class.
2. Calculate likelihood for all the feature values $P(X|c_i)$.
3. Calculate $P(X|c_i) \times P(c_i)$ for each class. The class $c_i$ for which $P(X|c_i) \times P(c_i)$ is maximum would be the predicted class label.

---

####  Naive Bayes Classifier Using `sklearn.naive_bayes`

Let us determine the target label for the above gameplay dataset using scikit-learn module of Python.

Before, let us display the total number of rows, features, data types of columns (features) and check for any missing values in the dataset.
"""

#  Apply the 'info()' function on the 'df_play' DataFrame.
df_play.info()

"""All the columns of the dataset are categorical. However, to use Naive Bayes Classifier, these features must be converted to integer values.

Let us perform label encoding on the DataFrame using the steps given below:     

1. Import `LabelEncoder` class from `sklearn.preprocessing` module.

2. Create an object of `LabelEncoder` class say `label`.

3. Initiate a `for` loop to iterate through all the columns of the DataFrame. In this loop, call the `fit_transform()` function and pass column values of the DataFrame.

4. Print the DataFrame to verify label encoding.
"""

#  Encode the categorical values

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
for column in df_play.columns:
  df_play[column] = label.fit_transform(df_play[column])

df_play

"""When label encoding is applied to this column, the labels would be arranged in alphabetical order and a unique index is assigned to each label starting from `0`. For example, in the feature column: `Outlook`: `Overcast` label would be encoded as `0`, `Rain` would be `1`, `Sunny` would be `2` and so on.

Also, note that the target label `Yes` is replaced with `1` and `No` is replaced with `0` by the label encoder.

Before we proceed with the Classifier design,
let's create separate DataFrames for features and the target column.

1. Create a `features_df` DataFrame by dropping the `target` column from the original DataFrame.   

2. Create a `target_df` DataFrame consisting of target values from the original DataFrame.


Also, print the shape of the features and target DataFrames.
"""

#  Create separate DataFrames for feature and target

features_df = df_play.drop('Play', axis = 1)
target_df = df_play['Play']

print(features_df.shape)
print(target_df.shape)

"""Next, split the DataFrames into train/test sets and display the shape of each set."""

#  Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(features_df, target_df, test_size = 0.3,
                                                    random_state = 2)

# Print the shape of train and test sets.
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

"""Next, let's create a Naive Bayes Classifier model. In the previous class, we used `GaussianNB` Classifier which is suitable specifically for continuous features. However, the features of our dataset contains discrete values.

Thus, we will use `CategoricalNB` Classifier which is another variant of Naive Bayes Classifier suitable for classification problems having discrete features.

To construct a Classifier using `CategoricalNB` Classifier, follow the steps given below:

1. Import the required library which contains methods and attributes to design a Naive Bayes Classifier.

  ```python
  from sklearn.naive_bayes import CategoricalNB
  ```
2. Create an object (say `nb_clf`) of the `CategoricalNB()` constructor.

4. Call the `fit()` function on the above constructor with train features and target variables as inputs.
"""

# Implement Naive Bayes Classifier

# Import the required library
from sklearn.naive_bayes import CategoricalNB

# Model the NB Classifier
nb_clf = CategoricalNB()
nb_clf.fit(X_train, y_train)

# Predict the train and test sets
y_train_predict_nb = nb_clf.predict(X_train)
y_test_predict_nb = nb_clf.predict(X_test)

# Evaluate the accuracy scores
print('Accuracy on the training set: {:.2f}'.format(nb_clf.score(X_train, y_train)))
print('Accuracy on the test set: {:.2f}'.format(nb_clf.score(X_test, y_test)))

"""Now the Classifier model has been trained. Even from a small dataset of merely 14 data points and after splitting them into train/test with 9 and 5 datapoints, we observe that  the Naive Bayes Classifier returns an exceptional accuracy of 0.89 and 0.80 for the train and test sets. Hence, Naive Bayes Classifier performs well even for small datasets.


Let us use the Classifier to predict the game play for the $15^{th}$ day. To recall weather conditions for day 15 are given as:

| Day | Outlook | Temperature | Humidity | Wind | Play |
| -- | -- | -- | -- | -- | --|
| 15 | Sunny | Cool | High | Strong | ❓ |

Since we have encoded the features using label encoder, let pass the encoded values for features while performing prediction.

Hence, the feature set will become:

| Day | Outlook | Temperature | Humidity | Wind | Play |
| -- | -- | -- | -- | -- | --|
| 15 | 1| 0 | 0 | 0 | ❓ |

<br>

To predict the outcome of the game based on the features, call the `predict()` function using Classifier object and pass the two-dimensional array `[1, 0, 0, 0]`  as input to the predict the outcome for day 15.
"""

# Predict the outcome for Day 15
nb_clf.predict([[1, 0, 0, 0]])

"""The outcome has been predicted as `array([0])` which means the predicted value for target is `No`, which matches with  prediction performed without using scikit-learn library in the previous activity.


**Takeaway points:** Naive Bayes Classifier can give fast results with very high accuracy when:

1. When the features as independent i.e. Naive assumptions are true (although this happens rarely with real-world data.)

2. When we have well-separated categories, especially when model is not very complex.

3. For very high-dimensional data, which can be well reduced by some feature engineering method and the model is not very complex.

4. For categorical features,  `CategoricalNB`  Naive Bayes Classifier model is advised and for continuous features `GaussianNB` Naive Bayes Classifier model is advised.

**Advantages:** Naive Bayes Classifier has several advantages:

1. They are extremely fast for both training and making predictions.

2. They provide easy to understand probability based predictions.

3. They have very few (if any) tunable parameters.

---
"""