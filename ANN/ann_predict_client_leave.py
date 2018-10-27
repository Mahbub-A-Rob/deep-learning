"""
Predict if a customer will leave the bank
"""
# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values



# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_Country = LabelEncoder()
X[:, 1] = labelencoder_X_Country.fit_transform(X[:, 1]) # Take country column which is 1st column

labelencoder_X_Gender = LabelEncoder()
X[:, 2] = labelencoder_X_Gender.fit_transform(X[:, 2]) 


onehotencoder = OneHotEncoder(categorical_features = [1]) # Take only country column to crate dummy variables.
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:] # Drop first column of dummy variables to avoid dummy variable trap


# Splitting the dataset into the Training set and Test set
# Note: use model_selection instead of cross validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Apply ANN
#import keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# Improve ANN
from keras.layers import Dropout # To avoid overfitting, underfitting

# Initialising ANN
classifier = Sequential()


# Add input layer and the first hidden layer
# Output dim is replaces by units in API 2
# 11 input + 1 output = 12, so 12/2=6, so Six nodes in the hidden layer
# Rectifier for hidden layer, sigmoid for output layer
# API 2 update
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
classifier.add(Dropout(p = 0.1)) # Use 0.1 to .4


# Add second hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
classifier.add(Dropout(p = 0.1)) # Use same as above

# Add output layer
# Only one output so, units = 1
# Activation Function = sigmoid

classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    
# Compiling the ANN
# Optimizer = the algorithm you want to use
# loss function within stachastic gradient
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])

# Fitting ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



"""# Predict if the following customer will leave the bank
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60,000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50,000

"""
"""
Use a horizontal array using double [[]]
Use same order as in the X or in the dataset
1. Figure out what is the country France
Use Dataset and X and check side by side
"""
"""
We need to scale same as the train
so, we will use sc
"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > .5) # Result: false, means The customer will not leave the bank

# The bais variance trade off
# We want a model that has more accuracy and not has much variance in different run
"""
We need to run our model on different test sets. 
k-fold cross validation will fix this variance problem
It will split the training set by 10 folds
It will check using 9 fold and 1 folds using 10 iterations 
and 10 combinaiton
"""
# Evaluating the ANN using k-fold cross validation
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense


# We need a function
def build_classifier():
    # Initialising ANN
    classifier = Sequential()
    classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])
    classifier.fit(X_train, y_train, batch_size=10, epochs=100) 
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5, n_jobs=None)
mean = accuracies.mean()
variance = accuracies.std()

