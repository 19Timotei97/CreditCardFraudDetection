# Import needed modules

####################################################################################################################################################
import warnings
import random
import os

# Stop tensor flow from flooding the terminal with varius unimportant messages 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # Ignore INFO and WARNING

warnings.filterwarnings("ignore") # ignore warnings about deprecation of current modules, as they are working best

# Decision Tree, Random Forest and Naive Bayes for generated data
import pandas as pd
# TODO: create a dataframe with the results
from pandas.core.frame import DataFrame

import numpy as np # For mathematical operations
import sys # For using command line arguments
import uuid

# Classifiers
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Neural Network
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler # For scaling and encoding the data
from sklearn.model_selection import GridSearchCV, train_test_split

# Metrics for classifiers result
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# For png image of the tree
import pydotplus
from IPython.core.display import Image
from six import StringIO

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

print(f"Number of samples under each target value :- \n {creditCardTransactions['is_fraud'].value_counts()}")

# DATA PREPROCESSING
fraud_creditCardTransactions = DataFrame()

columns_to_be_removed = []
# unnecessary_columns = ['cc_num', 'unix_time', 'street', 'job', 'state', 'last']
unnecessary_columns = ['cc_num', 'unix_time', 'trans_date_trans_time', 'job']
necessary_columns = ['trans_num', 'merchant', 'category', 'city', 'first', 'dob']

# Drop unnecessary columns or skip if the column is not present
for (columnName, _) in creditCardTransactions.iteritems():
    if columnName in unnecessary_columns:
        columns_to_be_removed.append(columnName)

# Drop unnecessary columns
try:
    fraud_creditCardTransactions = creditCardTransactions.drop(columns_to_be_removed, axis=1)
except KeyError:
    print("The dataset is not correct!")
    sys.exit(1)

def factorize(dataSet):
    # Factorize all columns which don't have numerical type
    # Obtainin a numeric representation of columns that are not numerical, impossible to skip or work with categorical data
    dataSet['gender'] = dataSet['gender'].map({'F':0, 'M':1})

    # ohe.fit_transform(dataSet[['trans_date_trans_time']]) # doesn't work as intended, it would create an even bigger data set: new feature for each of the values, with the appropiate label...
    dataSet['trans_num'], _ = pd.factorize(dataSet['trans_num'])
    dataSet['merchant'], _ = pd.factorize(dataSet['merchant'])
    dataSet['street'], _ = pd.factorize(dataSet['street'])
    dataSet['state'], _ = pd.factorize(dataSet['state'])
    dataSet['last'], _ = pd.factorize(dataSet['last'])
    dataSet['category'], _ = pd.factorize(dataSet['category'])   
    dataSet['city'], _ = pd.factorize(dataSet['city'])
    dataSet['first'], _ = pd.factorize(dataSet['first'])
    dataSet['dob'], _ = pd.factorize(dataSet['dob'])

factorize(fraud_creditCardTransactions)

print(fraud_creditCardTransactions.shape)
print(f"Dataset info :- \n {fraud_creditCardTransactions.info()}")

X = fraud_creditCardTransactions.drop(['is_fraud'], axis=1) # the features
y = fraud_creditCardTransactions[['is_fraud']] # the class

# Splitting dataset into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=trainDataset_size, random_state=0)

print("Train input data set size: ", X_train.shape)
print("Train ouput data set size: ", X_test.shape)
print("Test input data set size: ", y_train.shape)
print("Test ouput data set size: ", y_test.shape)

metrics_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']

# Defining the methods for the algorithms

# Decision Tree
def decisionTree_classifier(X_train, y_train, X_test, y_test):
    # initialize object for DecisionTreeClassifier class
    # default params:
    # random_state=None, criterion='gini', splitter='best', min_samples_leaf=1, min_samples_split=2
    dt_classifier = DecisionTreeClassifier(criterion='entropy')
    
    print("Model training with Decision Tree started...")

    # Train the model by using fit method
    dt_classifier.fit(X_train, y_train.values.ravel())
    
    print("Model training with Decision Tree completed.")
    
    print(f'Accuracy of model on test dataset :- {dt_classifier.score(X_test, y_test)}')
    
    print("Predicting results with Decision Tree started...")

    # Predict result using test dataset
    y_pred = dt_classifier.predict(X_test)
    
    print("Predicting results with Decision Tree finished.")
    
    # confusion matrix
    print(f"Confusion Matrix :- \n {confusion_matrix(y_test, y_pred)}")
    
    # Classification report for f1-score
    print(f"Classification Report :- \n {classification_report(y_test, y_pred)}")

    # Area under ROC curve
    print(f"AROC score :- \n {roc_auc_score(y_test, y_pred)}")

    dot = StringIO()
    
    export_graphviz(dt_classifier, out_file=dot, filled=True, rounded=True, special_characters=True, feature_names=X_train.columns, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot.getvalue())
    graph.write_png("decisionTree_fraudDet_generated.png")
    Image(graph.create_png())

    decisionTree_classifier_grid(dt_classifier, X_train, y_train, X_test, y_test)


###########################################################################################################################

# Decision Tree after pruning
def decisionTree_classifier_grid(decisionTreeModel, X_train, y_train, X_test, y_test):
    # Good to know but it doesn't affect too much the accuracy and the other metrics
    print("Best parameters for pruning in the scope of:")

    param_grid = {
        'max_depth': [5, 15, 25, 35, 45],
        'min_samples_leaf': [2, 4, 8],
        'max_leaf_nodes': [10, 30, 50]
    }

    # k-folds cross validation splits the dataset into chunks so that we have k different training and test sets
    # this is mainly used to get the best possible values for the metrics, the real example of how the algorithm classifies is when the whole set is used
    
    # cv is by default 5
    gs1 = GridSearchCV(decisionTreeModel, param_grid, scoring='f1')
    print("1) Maximizing f1-score")

    gs1.fit(X_test, y_test)

    diction1 = gs1.best_params_

    print(diction1)

    ####################

    print("Building a decision tree with the best parameters")

    dt_classifier_pruned_f1 = DecisionTreeClassifier(max_depth=diction1['max_depth'], max_leaf_nodes=diction1['max_leaf_nodes'], min_samples_leaf=diction1['min_samples_leaf'])
    
    print("Model training with a pruned Decision Tree started...")

    # Train the model by using fit method
    dt_classifier_pruned_f1.fit(X_train, y_train.values.ravel())
    
    print("Model training with a pruned Decision Tree completed.")
    
    print(f'Accuracy of model on test dataset :- {dt_classifier_pruned_f1.score(X_test, y_test)}')
    
    print("Predicting results with a pruned Decision Tree started...")

    # Predict result using test dataset
    y_pred_pruned = dt_classifier_pruned_f1.predict(X_test)
    
    print("Predicting results with a pruned Decision Tree finished.")
    
    # confusion matrix
    print(f"Confusion Matrix :- \n {confusion_matrix(y_test, y_pred_pruned)}")
    
    # Classification report for f1-score
    print(f"Classification Report :- \n {classification_report(y_test, y_pred_pruned)}")

    # Area under ROC curve
    print(f"AROC score :- \n {roc_auc_score(y_test, y_pred_pruned)}")

    dot = StringIO()
    
    export_graphviz(dt_classifier_pruned_f1, out_file=dot, filled=True, rounded=True, special_characters=True, feature_names=X_test.columns, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot.getvalue())
    graph.write_png("decisionTree_fraudDet_generated_pruned_f1.png")
    Image(graph.create_png())

    ####################
    
    gs2 = GridSearchCV(decisionTreeModel, param_grid, scoring='roc_auc', cv=5)
    print("2) Maximizing ROC AUC")

    gs2.fit(X_test, y_test)

    diction2 = gs2.best_params_
    
    print(diction2)

    print("Building a decision tree with the best parameters")

    dt_classifier_pruned_AROC = DecisionTreeClassifier(max_depth=diction2['max_depth'], max_leaf_nodes=diction2['max_leaf_nodes'], min_samples_leaf=diction2['min_samples_leaf'])
    
    print("Model training with a pruned Decision Tree started...")

    # Train the model by using fit method
    dt_classifier_pruned_AROC.fit(X_train, y_train.values.ravel())
    
    print("Model training with a pruned Decision Tree completed.")
    
    print(f'Accuracy of model on test dataset :- {dt_classifier_pruned_AROC.score(X_test, y_test)}')
    
    print("Predicting results with a pruned Decision Tree started...")

    # Predict result using test dataset
    y_pred_pruned = dt_classifier_pruned_AROC.predict(X_test)
    
    print("Predicting results with a pruned Decision Tree finished.")
    
    # confusion matrix
    print(f"Confusion Matrix :- \n {confusion_matrix(y_test, y_pred_pruned)}")
    
    # Classification report for f1-score
    print(f"Classification Report :- \n {classification_report(y_test, y_pred_pruned)}")

    # Area under ROC curve
    print(f"AROC score :- \n {roc_auc_score(y_test, y_pred_pruned)}")

    dot_AROC = StringIO()
    
    export_graphviz(dt_classifier_pruned_AROC, out_file=dot_AROC, filled=True, rounded=True, special_characters=True, feature_names=X_test.columns, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_AROC.getvalue())
    graph.write_png("decisionTree_fraudDet_generated_pruned_roc_auc.png")
    Image(graph.create_png())

###########################################################################################################################


# Random Forest
def randomForest_classifier(X_train, y_train, X_test, y_test):
     # initialize object for RandomForestClassifier class
    rf_classifier = RandomForestClassifier(n_estimators=50)

    # train model by using fit method
    print("Model training with Random Forest started...")

    # Train the model by using fit method
    rf_classifier.fit(X_train, y_train.values.ravel())

    print("Model training with Random Forest completed.")
    
    print(f'Accuracy of model on test dataset :- {rf_classifier.score(X_test, y_test)}')
    
    print("Predicting results with Random Forest started...")

    # Predict result using test dataset
    y_pred = rf_classifier.predict(X_test)

    print("Predicting results with Random Forest completed.")
        
    # Confusion matrix
    print(f"Confusion Matrix :- \n {confusion_matrix(y_test, y_pred)}")
        
    # Classification report for f1-score
    print(f"Classification Report :- \n {classification_report(y_test, y_pred)}")

    # Area under ROC curve
    print(f"AROC score :- \n {roc_auc_score(y_test, y_pred)}")

    dot1 = StringIO()
    
    export_graphviz(rf_classifier.estimators_[0], out_file=dot1, filled=True, rounded=True, special_characters=True, feature_names=X_test.columns, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot1.getvalue())
    graph.write_png("randomForest_first_fraudDet_generated.png")
    Image(graph.create_png())

    dot2 = StringIO()
    
    export_graphviz(rf_classifier.estimators_[24], out_file=dot2, filled=True, rounded=True, special_characters=True, feature_names=X_test.columns, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot2.getvalue())
    graph.write_png("randomForest_middle_fraudDet_generated.png")
    Image(graph.create_png())

    dot3 = StringIO()
    
    export_graphviz(rf_classifier.estimators_[24], out_file=dot3, filled=True, rounded=True, special_characters=True, feature_names=X_test.columns, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot3.getvalue())
    graph.write_png("randomForest_last_fraudDet_generated.png")
    Image(graph.create_png())

    randomForest_classifier_grid(rf_classifier, X_train, y_train, X_test, y_test)

#######################################################################################################################################################################################

# Random Forest with best parameters
def randomForest_classifier_grid(randomForestModel, X_train, y_train, X_test, y_test):
      # Good to know but it doesn't affect too much the accuracy and the other metrics
    print("Best parameters for RF in the scope of:")

    param_grid = {
        'n_estimators': [10, 30, 50, 100],
        'max_features': [3, 5, 10, 15]
    }

    # k-folds cross validation splits the dataset into chunks so that we have k different training and test sets
    # this is mainly used to get the best possible values for the metrics, the real example of how the algorithm classifies is when the whole set is used
    
    # cv is by default 5
    gs1 = GridSearchCV(randomForestModel, param_grid, scoring='f1')
    print("1) Maximizing f1-score")
    
    gs1.fit(X_test, y_test)

    diction1 = gs1.best_params_

    print(diction1)

    print("Building a random forest with the best parameters for f1-score")

    rf_classifier_pruned_f1 = RandomForestClassifier(n_estimators=diction1['n_estimators'], max_features=diction1['max_features'])
    
    print("Model training with a custom Random Forest for f1-score started...")

    # Train the model by using fit method
    rf_classifier_pruned_f1.fit(X_train, y_train.values.ravel())
    
    print("Model training with custom Random Forest for f1-score completed.")
    
    print(f'Accuracy of model on test dataset :- {rf_classifier_pruned_f1.score(X_test, y_test)}')
    
    print("Predicting results with a custom Random Forest for f1-score started...")

    # Predict result using test dataset
    y_pred_pruned = rf_classifier_pruned_f1.predict(X_test)
    
    print("Predicting results with a custom Random Forest for f1-score finished.")
    
    # confusion matrix
    print(f"Confusion Matrix :- \n {confusion_matrix(y_test, y_pred_pruned)}")
    
    # Classification report for f1-score
    print(f"Classification Report :- \n {classification_report(y_test, y_pred_pruned)}")

    # Area under ROC curve
    print(f"AROC score :- \n {roc_auc_score(y_test, y_pred_pruned)}")

    dot_f1 = StringIO()
    
    export_graphviz(rf_classifier_pruned_f1.estimators_[len(rf_classifier_pruned_f1.estimators_)//2], 
                                    out_file=dot_f1, filled=True, 
                                    rounded=True, special_characters=True,
                                    feature_names=X_test.columns, class_names=['0', '1'])
    
    graph = pydotplus.graph_from_dot_data(dot_f1.getvalue())
    graph.write_png("randomForest_middle_fraudDet_generated_f1.png")
    Image(graph.create_png())

    ###############################################################################################

    gs2 = GridSearchCV(randomForestModel, param_grid, scoring='roc_auc', cv=5)
    print("2) Maximizing ROC AUC")

    gs2.fit(X_test, y_test)

    diction2 = gs1.best_params_

    print(diction2)

    print("Building a random forest with the best parameters for AROC")

    rf_classifier_pruned_AROC = RandomForestClassifier(n_estimators=diction2['n_estimators'], max_features=diction2['max_features'])
    
    print("Model training with a custom Random Forest for AROC started...")

    # Train the model by using fit method
    rf_classifier_pruned_AROC.fit(X_train, y_train.values.ravel())
    
    print("Model training with custom Random Forest for AROC completed.")
    
    print(f'Accuracy of model on test dataset :- {rf_classifier_pruned_AROC.score(X_test, y_test)}')
    
    print("Predicting results with a custom Random Forest for AROC started...")
 
    # Predict result using test dataset
    y_pred_pruned = rf_classifier_pruned_AROC.predict(X_test)
    
    print("Predicting results with a custom Random Forest for AROC finished.")
    
    # confusion matrix
    print(f"Confusion Matrix :- \n {confusion_matrix(y_test, y_pred_pruned)}")
    
    # Classification report for f1-score
    print(f"Classification Report :- \n {classification_report(y_test, y_pred_pruned)}")

    # Area under ROC curve
    print(f"AROC score :- \n {roc_auc_score(y_test, y_pred_pruned)}")

    dot_AROC = StringIO()
    
    export_graphviz(rf_classifier_pruned_AROC.estimators_[len(rf_classifier_pruned_AROC.estimators_)//2], 
                                    out_file=dot_AROC, filled=True, 
                                    rounded=True, special_characters=True,
                                    feature_names=X_test.columns, class_names=['0', '1'])

    graph = pydotplus.graph_from_dot_data(dot_AROC.getvalue())
    graph.write_png("randomForest_middle_fraudDet_generated_roc_auc.png")
    Image(graph.create_png())


#######################################################################################################################################################################################

# Naive Bayes
def naiveBayes_classifier(X_train, y_train, X_test, y_test):
    # initialize object for Naive Bayes classifier
    nb_classifier = GaussianNB()

    # train model by using fit method
    print("Model training with Naive Bayes started...")
    
    nb_classifier.fit(X_train, y_train.values.ravel())
    
    print("Model training with Naive Bayes completed.")

    print(f'Accuracy of model on test dataset :- {nb_classifier.score(X_test, y_test)}')
    
    print("Predicting results with Naive Bayes started...")

    # Predict result using test dataset
    y_pred = nb_classifier.predict(X_test)

    print("Predicting results with Naive Bayes completed.")

    # confusion matrix
    print(f"Confusion Matrix :- \n {confusion_matrix(y_test, y_pred)}")
    
    # classification report for f1-score
    print(f"Classification Report :- \n {classification_report(y_test, y_pred)}")

    print(f"AROC score :- \n {roc_auc_score(y_test, y_pred)}")

def neuralNetwork_classifier(X_train, y_train, X_test, y_test):
    # Initialising the ANN
    classifier = Sequential()

    """
    # For some reason, the SAME NN used for PCA is not yielding the same results as in the presentation?
    # Why?
    # I got 0 ideas


    # Adding the input layer and the first hidden layer
    # classifier.add(Dense(units = 18, activation = 'relu', input_dim = 18))
    classifier.add(Dense(units = 80, activation = 'relu', input_dim = 18))


    # Adding the second hidden layer
    classifier.add(Dense(units = 24, activation = 'relu')) 
    
    # Adding a dropout layer to prevent overfitting
    classifier.add(tf.keras.layers.Dropout(0.5))

    # Adding the third hidden layer
    classifier.add(Dense(units = 20, activation = 'relu')) 

    # And the fourth hidden layer
    classifier.add(Dense(units = 24, activation = 'relu'))

    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    """

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 80, kernel_initializer = 'uniform', activation = 'relu', input_dim = 18))

    # Adding the second hidden layer
    classifier.add(Dense(units = 80, kernel_initializer = 'uniform', activation = 'relu'))

    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    print("Model training with Neural Network is starting...")
    
    print("Shape of X_train is: ", X_train.shape)
    print("Shape of y_train is ", y_train.shape)

    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size = 20, epochs = 100)

    print("Model training with Neural Network completed.")

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    y_pred = (y_pred > 0.5)

    print(f'The accuracy of the Neural Network was: {classifier.evaluate(X_test, y_test)}')

    # Store other important metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # confusion matrix
    print(f"Confusion Matrix :- \n {confusion_matrix(y_test, y_pred)}")
    
    # classification report for f1-score
    print(f"Classification Report :- \n {classification_report(y_test, y_pred)}")

    print(f"AROC score :- \n {roc_auc_score(y_test, y_pred)}")

    results_neural_network = pd.DataFrame([['Neural Network', acc, prec, rec, f1]],
               columns = metrics_cols)
    
    results.append(results_neural_network)

##################### SAMPLING TECHNIQUES ########################

# Accuracy is high but f1 and the other metrics not so
# We must apply some sampling techniques

class_val = fraud_creditCardTransactions['is_fraud'].value_counts()
# print(f"Number of samples for each class: -\n {class_val}")

# Split both classes into separate variables
non_fraud = class_val[0]
fraud = class_val[1]

print(f"Non-fraudulent numbers: - {non_fraud}")
print(f"Fraudulent numbers: - {fraud}")

# Take indexes of both classes and randomly choose Class-0 samples indexes
# That are equal to the number of Class-1 samples.
nonfraud_indecies = fraud_creditCardTransactions[fraud_creditCardTransactions['is_fraud'] == 0].index
fraud_indecies = fraud_creditCardTransactions[fraud_creditCardTransactions['is_fraud'] == 1].index

# Take random samples from non-frauds equal to frauds samples
random_normal_indeces = np.random.choice(nonfraud_indecies, fraud, replace=False)
random_normal_indeces = np.array(random_normal_indeces)

# Combine both classes indexes and extract all features of gathered indexes
# Concatenate both indices of fraud and non fraud
under_sample_indices = np.concatenate([fraud_indecies, random_normal_indeces])

# Extract all features from the whole data for undersampling indices only
under_sample_data = fraud_creditCardTransactions.iloc[under_sample_indices, :]

# Divide under sampling data to all features & target
X_undersample_data = under_sample_data.drop(['is_fraud'], axis=1)
y_undersample_data = under_sample_data[['is_fraud']]

# Split dataset to train and test datasets as before
X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(
                                                                X_undersample_data, 
                                                                y_undersample_data, 
                                                                train_size=trainDataset_size,
                                                                random_state=0)


message = "Unbalanced data detected! Applying sampling techniques and running again."

if (str(sys.argv[2]) == "DECISION_TREE"):
    decisionTree_classifier(X_train, y_train, X_test, y_test) 
  
    print(message)
  
    decisionTree_classifier(X_train_sample, y_train_sample, X_test_sample, y_test_sample)

elif (str(sys.argv[2]) == "RANDOM_FOREST"):
    randomForest_classifier(X_train, y_train, X_test, y_test)

    print(message)

    randomForest_classifier(X_train_sample, y_train_sample, X_test_sample, y_test_sample)

elif (str(sys.argv[2]) == "NAIVE_BAYES"):
    naiveBayes_classifier(X_train, y_train, X_test, y_test)

    print(message)

    naiveBayes_classifier(X_train_sample, y_train_sample, X_test_sample, y_test_sample)

elif (str(sys.argv[2]) == "NEURAL_NETWORK"):
    neuralNetwork_classifier(X_train, y_train, X_test, y_test)

    print(message)

    neuralNetwork_classifier(X_train_sample, y_train_sample, X_test_sample, y_test_sample)

elif (str(sys.argv[2]) == "ALL"):
    decisionTree_classifier(X_train, y_train, X_test, y_test)
  
    print(message)
  
    decisionTree_classifier(X_train_sample, y_train_sample, X_test_sample, y_test_sample)

    print("###### Before sampling: ######")
    randomForest_classifier(X_train, y_train, X_test, y_test)
    
    print()
    print("###### After sampling: ######")
    randomForest_classifier(X_train_sample, y_train_sample, X_test_sample, y_test_sample)

    print("###### Before sampling: ######")
    naiveBayes_classifier(X_train, y_train, X_test, y_test)
    
    print()
    print("###### After sampling: ######")
    naiveBayes_classifier(X_train_sample, y_train_sample, X_test_sample, y_test_sample)

    print("###### Before sampling: ######")
    neuralNetwork_classifier(X_train, y_train, X_test, y_test)

    print()
    print("###### After sampling: ######")
    neuralNetwork_classifier(X_train_sample, y_train_sample, X_test_sample, y_test_sample)