# Import needed modules

####################################################################################################################################################
import warnings
import random
import os
import sys # For using command line arguments

# Stop tensor flow from flooding the terminal with varius unimportant messages 
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # Ignore INFO and WARNING

warnings.filterwarnings("ignore") # ignore warnings about deprecation of current modules, as they are working best

# Decision Tree, Random Forest and Naive Bayes for generated data
import pandas as pd

# TODO: create a dataframe with the results
from pandas.core.frame import DataFrame
import numpy as np # For mathematical operations

# Classifiers
# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Neural Network
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler # For scaling the data
from sklearn.model_selection import GridSearchCV, train_test_split

# Metrics for classifiers result
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# For png image of the tree
from six import StringIO
import pydotplus
from IPython.core.display import Image
from imblearn.over_sampling import ADASYN

####################################################################################################################################################

dataset = str(sys.argv[1]) # What dataset was loaded

classifierUsed = str(sys.argv[2]) # What classifying algorithm will be used

trainDataset_size = float(sys.argv[3]) # How much of the data set is going to be used for training the model(s)

# Load data file
creditCardTransactions = pd.read_csv(dataset)

# Experimental - store data for each algorithm
results = []

print("Data shape:")
print(creditCardTransactions.shape)


print(f"Number of samples under each target value :- \n {creditCardTransactions['Class'].value_counts()}")

print(f"Dataset info :- \n {creditCardTransactions.info()}")

# Reduce the wide range of Amount column
# Standardization is used to remove the mean and sclae to unit variance
# 68% of the values lie in between (-1, 1)
creditCardTransactions['Amount'] = StandardScaler().fit_transform(creditCardTransactions['Amount'].values.reshape(-1,1))

# Drop the not needed columns
# creditCardTransactions = creditCardTransactions.drop(['Amount'], axis=1)
creditCardTransactions = creditCardTransactions.drop(['Time'], axis=1)

# Retrieve the features and the Class column from the dataset
# Gets all the records (lines)
X = creditCardTransactions.iloc[:, creditCardTransactions.columns != 'Class']
y = creditCardTransactions.iloc[:, creditCardTransactions.columns == 'Class']

# or
# X = creditCardTransactions.drop(['Class'], axis=1) # the features
# y = creditCardTransactions[['Class']] # the class

# Splitting dataset into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=trainDataset_size, random_state=0)

print("Train input data set size: ", X_train.shape)
print("Train ouput data set size: ", X_test.shape)
print("Test input data set size: ", y_train.shape)
print("Test ouput data set size: ", y_test.shape)

metrics_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']

# Defining the methods for the algorithms

# Logistic Regression
def logisticRegression_classifier(X_train, y_train, X_test, y_test):
    # initialize object for Logistic Regression classifier
    lr_classifier = LogisticRegression()

    # train model by using fit method
    print("Model training with Logistic Regression started...")
    
    lr_classifier.fit(X_train, y_train.values.ravel())
    
    print("Model training with Logistic Regression completed.")

    print(f'Accuracy of model on test dataset :- {lr_classifier.score(X_test, y_test)}')
    
    print("Predicting results with Logistic Regression started...")

    # Predict result using test dataset
    y_pred = lr_classifier.predict(X_test)

    print("Predicting results with Logistic Regression completed.")

    # confusion matrix
    print(f"Confusion Matrix :- \n {confusion_matrix(y_test, y_pred)}")
    
    # classification report for f1-score
    print(f"Classification Report :- \n {classification_report(y_test, y_pred)}")

    print(f"AROC score :- \n {roc_auc_score(y_test, y_pred)}")


def neuralNetwork_classifier(X_train, y_train, X_test, y_test):
    # Initialising the ANN
    # increase max_iter if the model doesn't converge
    # also hidden_layer_sizes (tuple, length = n_layers - 2), alpha (0.0001 as default), solver (adam as default)
    
    # mlp_classifier = MLPClassifier(hidden_layer_sizes=(29, 24, 24, 29), random_state=5)
    mlp_classifier = MLPClassifier()
    
    print("Model training with Neural Network is starting...")
    
    print("Shape of X_train is: ", X_train.shape)
    print("Shape of y_train is ", y_train.shape)

    # Fitting the ANN to the Training set
    mlp_classifier.fit(X_train, y_train)

    print("Model training with Neural Network completed.")

    # Predicting the Test set results
    y_pred = mlp_classifier.predict(X_test)

    y_pred = (y_pred > 0.5)

    print(f'The accuracy of the Neural Network was: {mlp_classifier.score(X_test, y_test)}')

    # confusion matrix
    print(f"Confusion Matrix :- \n {confusion_matrix(y_test, y_pred)}")
    
    # classification report for f1-score
    print(f"Classification Report :- \n {classification_report(y_test, y_pred)}")

    print(f"AROC score :- \n {roc_auc_score(y_test, y_pred)}")

##################### SAMPLING TECHNIQUES ########################

# Accuracy is high but f1 not
# We must apply some sampling techniques

class_val = creditCardTransactions['Class'].value_counts()
print(f"Number of samples for each class: -\n {class_val}")

# Split both classes into separate variables
non_fraud = class_val[0]
fraud = class_val[1]

print(f"Non-fraudulent numbers: - {non_fraud}")
print(f"Fraudulent numbers: - {fraud}")

### TEST ###
fraud_ind = np.array(creditCardTransactions[creditCardTransactions.Class == 1].index)
normal_ind = creditCardTransactions[creditCardTransactions.Class == 0].index
### TEST ###

# Take indexes of both classes and randomly choose Class-0 samples indexes
# That are equal to the number of Class-1 samples.
nonfraud_indecies = creditCardTransactions[creditCardTransactions['Class'] == 0].index
fraud_indecies = creditCardTransactions[creditCardTransactions['Class'] == 1].index

# Take random samples from non-frauds equal to frauds samples
random_normal_indeces = np.array(np.random.choice(nonfraud_indecies, fraud, replace=False))
# random_normal_indeces = np.array(random_normal_indeces)

# Combine both classes indexes and extract all features of gathered indexes
# Concatenate both indices of fraud and non fraud
under_sample_indices = np.concatenate([fraud_indecies, random_normal_indeces])

# Extract all features from the whole data for undersampling indices only
under_sample_data = creditCardTransactions.iloc[under_sample_indices, :]

# Divide under sampling data to all features & target
X_undersample_data = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
y_undersample_data = under_sample_data.iloc[:, under_sample_data.columns == 'Class']

# X_undersample_data = under_sample_data.drop(['Class'], axis=1)
# y_undersample_data = under_sample_data[['Class']]

# Split dataset to train and test datasets as before
X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(
                                                                X_undersample_data, 
                                                                y_undersample_data, 
                                                                train_size=trainDataset_size,
                                                                random_state=0)

# # Oversampling
# # SMOTE (Synthetic Minority Over-sampling TEchnique) - it looks at the feature space for the minority class data points and considers its k nearest neighbours - doesn't work
# X_oversample, y_oversample = SMOTE(sampling_strategy='auto', random_state=7).fit_sample(X, y)
# # print(f"X_RESAMPLE: {X_oversample.shape}")
# # print(Counter(y_oversample))

# oversampled_train = pd.concat([pd.DataFrame(y_oversample), pd.DataFrame(X_oversample)], axis=1)
# normalized_df = pd.concat([X_undersample_data, y_undersample_data])
# oversampled_train.columns = normalized_df.columns


# Oversampling
ada = ADASYN(sampling_strategy='minority', random_state=6)
X_oversample, y_oversample = ada.fit_resample(X, y)

X_train_oversample, X_test_oversample, y_train_oversample, y_test_oversample = train_test_split(X_oversample, y_oversample, train_size=trainDataset_size, random_state=7)


# X_train_OS_sample, y_train_OS_sample, X_test_OS_sample, y_test_OS_sample = train_test_split(
#                                                                 X_oversample,
#                                                                 y_oversample,
#                                                                 train_size=trainDataset_size,
#                                                                 random_state=0)


message = "Unbalanced data detected! Applying sampling techniques and running again."


if (str(sys.argv[2]) == "LOGISTIC_REGRESSION"):
    logisticRegression_classifier(X_train, y_train, X_test, y_test)

    print(message)

    logisticRegression_classifier(X_train_sample, y_train_sample, X_test_sample, y_test_sample)

    print("OVERSAMPLING")

    logisticRegression_classifier(X_train_oversample, X_test_oversample, y_train_oversample, y_test_oversample)

elif (str(sys.argv[2]) == "NEURAL_NETWORK"):
    neuralNetwork_classifier(X_train, y_train, X_test, y_test)

    print(message)

    neuralNetwork_classifier(X_train_sample, y_train_sample, X_test_sample, y_test_sample)

    print("OVERSAMPLING")

    neuralNetwork_classifier(X_train_oversample, X_test_oversample, y_train_oversample, y_test_oversample)

elif (str(sys.argv[2]) == "ALL"):
    print("###### Before sampling: ######")
    logisticRegression_classifier(X_train, y_train, X_test, y_test)
    
    print()
    print("###### After sampling: ######")
    logisticRegression_classifier(X_train_sample, y_train_sample, X_test_sample, y_test_sample)

    print("###### Before sampling: ######")
    neuralNetwork_classifier(X_train, y_train, X_test, y_test)

    print()
    print("###### After sampling: ######")
    neuralNetwork_classifier(X_train_sample, y_train_sample, X_test_sample, y_test_sample)

