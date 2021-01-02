# Import needed modules

####################################################################################################################################################
import warnings
import random
import os
import sys # For using command line arguments

# Stop tensor flow from flooding the terminal with varius unimportant messages 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # Ignore INFO and WARNING

warnings.filterwarnings("ignore") # ignore warnings about deprecation of current modules, as they are working best

# Decision Tree, Random Forest and Naive Bayes for generated data
import pandas as pd

# TODO: create a dataframe with the results
from pandas.core.frame import DataFrame
import numpy as np # For mathematical operations
from imblearn.over_sampling import ADASYN

# Classifiers
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Neural Network
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler # For scaling the data
from sklearn.model_selection import GridSearchCV, train_test_split

# Metrics for classifiers result
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# For png image of the tree
from six import StringIO
import pydotplus
from IPython.core.display import Image
from collections import Counter

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

# Drop the not needed column - specifies the time difference between the first and the n transaction
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

# Defining the methods for the algorithms

# Decision Tree
def decisionTree_classifier(X_train, y_train, X_test, y_test, pruning):
    # initialize object for DecisionTreeClassifier class
    # default params:
    # random_state=None, criterion='gini', splitter='best', min_samples_leaf=1, min_samples_split=2
    # dt_classifier = DecisionTreeClassifier(criterion='entropy') - little to no difference if either gini or entropy is used
    dt_classifier = DecisionTreeClassifier()
    
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
    
    export_graphviz(dt_classifier, out_file=dot, filled=True, rounded=True, special_characters=True, feature_names=X_test.columns, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot.getvalue())
    graph.write_png("decisionTree_fraudDet_PCA.png")
    Image(graph.create_png())

    if (pruning):
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

    print("Building a decision tree with the best parameters for f1 score")

    dt_classifier_prunedf1 = DecisionTreeClassifier(max_depth=diction1['max_depth'], max_leaf_nodes=diction1['max_leaf_nodes'], min_samples_leaf=diction1['min_samples_leaf'])
    
    print("Model training with a pruned Decision Tree for f1 score started...")

    # Train the model by using fit method
    dt_classifier_prunedf1.fit(X_train, y_train.values.ravel())
    
    print("Model training with a pruned Decision Tree for f1 score completed.")
    
    print(f'Accuracy of model on test dataset :- {dt_classifier_prunedf1.score(X_test, y_test)}')
    
    print("Predicting results with a pruned Decision Tree started...")

    # Predict result using test dataset
    y_pred_pruned_f1 = dt_classifier_prunedf1.predict(X_test)
    
    print("Predicting results with a pruned Decision Tree for f1 score finished.")
    
    # confusion matrix
    print(f"Confusion Matrix:- \n {confusion_matrix(y_test, y_pred_pruned_f1)}")
    
    # Classification report for f1-score
    print(f"Classification Report:- \n {classification_report(y_test, y_pred_pruned_f1)}")

    # Area under ROC curve
    print(f"AROC score:- \n {roc_auc_score(y_test, y_pred_pruned_f1)}")

    dot_f1 = StringIO()
    
    export_graphviz(dt_classifier_prunedf1, out_file=dot_f1, filled=True, rounded=True, special_characters=True, feature_names=X_test.columns, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_f1.getvalue())
    graph.write_png("decisionTree_fraudDet_PCA_pruned_f1.png")
    Image(graph.create_png())

    ##########################################################################

    print("Best parameters for pruning in the scope of:")

    gs2 = GridSearchCV(decisionTreeModel, param_grid, scoring='roc_auc', cv=5)
    print("2) Maximizing ROC AUC")
    
    gs2.fit(X_test, y_test)    
    diction2 = gs2.best_params_

    print("Building a decision tree with the best parameters for AROC.")

    dt_classifier_prunedAROC = DecisionTreeClassifier(max_depth=diction2['max_depth'], max_leaf_nodes=diction2['max_leaf_nodes'], min_samples_leaf=diction2['min_samples_leaf'])
    
    print("Model training with a pruned Decision Tree for AROC started...")

    # Train the model by using fit method
    dt_classifier_prunedAROC.fit(X_train, y_train.values.ravel())
    
    print("Model training with a pruned Decision Tree for AROC completed.")
    
    print(f'Accuracy of model on test dataset :- {dt_classifier_prunedAROC.score(X_test, y_test)}')
    
    print("Predicting results with a pruned Decision Tree started...")

    # Predict result using test dataset
    y_pred_pruned_AROC = dt_classifier_prunedAROC.predict(X_test)
    
    print("Predicting results with a pruned Decision Tree for AROC finished.")
    
    # confusion matrix
    print(f"Confusion Matrix :- \n {confusion_matrix(y_test, y_pred_pruned_AROC)}")
    
    # Classification report for f1-score
    print(f"Classification Report :- \n {classification_report(y_test, y_pred_pruned_AROC)}")

    # Area under ROC curve
    print(f"AROC score :- \n {roc_auc_score(y_test, y_pred_pruned_AROC)}")

    dot_AROC = StringIO()
    
    export_graphviz(dt_classifier_prunedAROC, out_file=dot_AROC, filled=True, rounded=True, special_characters=True, feature_names=X_test.columns, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_AROC.getvalue())
    graph.write_png("decisionTree_fraudDet_PCA_pruned_roc_auc.png")
    Image(graph.create_png())


###########################################################################################################################

# Random Forest
def randomForest_classifier(X_train, y_train, X_test, y_test, customize):
     # initialize object for RandomForestClassifier class, default gini criterion
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
    graph.write_png("randomForest_first_fraudDet_PCA.png")
    Image(graph.create_png())

    dot2 = StringIO()
    
    export_graphviz(rf_classifier.estimators_[24], out_file=dot2, filled=True, rounded=True, special_characters=True, feature_names=X_test.columns, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot2.getvalue())
    graph.write_png("randomForest_middle_fraudDet_PCA.png")
    Image(graph.create_png())

    dot3 = StringIO()
    
    export_graphviz(rf_classifier.estimators_[24], out_file=dot3, filled=True, rounded=True, special_characters=True, feature_names=X_test.columns, class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot3.getvalue())
    graph.write_png("randomForest_last_fraudDet_PCA.png")
    Image(graph.create_png())

    if (customize):
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

    ####################

    print("Building a random forest with the best parameters for f1-score")

    rf_classifier_prunedf1 = RandomForestClassifier(n_estimators=diction1['n_estimators'], max_features=diction1['max_features'])
    
    print("Model training with a custom Random Forest for f1-score started...")

    # Train the model by using fit method
    rf_classifier_prunedf1.fit(X_train, y_train.values.ravel())
    
    print("Model training with custom Random Forest for f1-score completed.")
    
    print(f'Accuracy of model on test dataset :- {rf_classifier_prunedf1.score(X_test, y_test)}')
    
    print("Predicting results with a custom Random Forest for f1-score started...")

    # Predict result using test dataset
    y_pred_pruned = rf_classifier_prunedf1.predict(X_test)
    
    print("Predicting results with a custom Random Forest for f1-score finished.")
    
    # confusion matrix
    print(f"Confusion Matrix :- \n {confusion_matrix(y_test, y_pred_pruned)}")
    
    # Classification report for f1-score
    print(f"Classification Report :- \n {classification_report(y_test, y_pred_pruned)}")

    # Area under ROC curve
    print(f"AROC score :- \n {roc_auc_score(y_test, y_pred_pruned)}")

    dot_f1 = StringIO()
    
    export_graphviz(rf_classifier_prunedf1.estimators_[len(rf_classifier_prunedf1.estimators_)//2], 
                        out_file=dot_f1, filled=True, 
                        rounded=True, special_characters=True, 
                        feature_names=X_test.columns, class_names=['0', '1'])



    graph = pydotplus.graph_from_dot_data(dot_f1.getvalue())
    graph.write_png("randomForest_middle_fraudDet_pca_f1.png")
    Image(graph.create_png())


    gs2 = GridSearchCV(randomForestModel, param_grid, scoring='roc_auc', cv=5)
    print("2) Maximizing ROC AUC")

    gs2.fit(X_test, y_test)

    diction2 = gs2.best_params_
    print(diction2)

    print("Building a random forest with the best parameters for AROC")

    rf_classifier_prunedAROC = RandomForestClassifier(n_estimators=diction2['n_estimators'], max_features=diction2['max_features'])
    
    print("Model training with a custom Random Forest for f1-score started...")

    # Train the model by using fit method
    rf_classifier_prunedAROC.fit(X_train, y_train.values.ravel())
    
    print("Model training with custom Random Forest for f1-score completed.")
    
    print(f'Accuracy of model on test dataset :- {rf_classifier_prunedAROC.score(X_test, y_test)}')
    
    print("Predicting results with a custom Random Forest for f1-score started...")

    # Predict result using test dataset
    y_pred_pruned = rf_classifier_prunedAROC.predict(X_test)
    
    print("Predicting results with a custom Random Forest for f1-score finished.")
    
    # confusion matrix
    print(f"Confusion Matrix :- \n {confusion_matrix(y_test, y_pred_pruned)}")
    
    # Classification report for f1-score
    print(f"Classification Report :- \n {classification_report(y_test, y_pred_pruned)}")

    # Area under ROC curve
    print(f"AROC score :- \n {roc_auc_score(y_test, y_pred_pruned)}")

    dot_AROC = StringIO()
    
    export_graphviz(rf_classifier_prunedAROC.estimators_[len(rf_classifier_prunedAROC.estimators_)//2], 
                        out_file=dot_AROC, filled=True, 
                        rounded=True, special_characters=True, 
                        feature_names=X_test.columns, class_names=['0', '1'])

    graph = pydotplus.graph_from_dot_data(dot_AROC.getvalue())
    graph.write_png("randomForest_middle_fraudDet_pca_roc_auc.png")
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

    # Adding the input layer and the first hidden layer
    # input_dim = number of input variables, units = the number of nodes / neurons in each layer
    # Rectified Linear Unit - ReLU - activation function for the hidden layers
    # Performs better than Sigmoid and Hyperbolic Tangent functions
    classifier.add(Dense(units = 29, activation = 'relu', input_dim = 29))

    # Adding the second hidden layer
    # classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 24, activation = 'relu')) 
    
    classifier.add(tf.keras.layers.Dropout(0.5))

    # Adding the third hidden layer
    classifier.add(Dense(units = 20, activation = 'relu')) 

    # And the fourth hidden layer
    classifier.add(Dense(units = 24, activation = 'relu'))

    # Adding the output layer
    # classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))

    # Compiling the ANN
    # binary cross entropy is used because the real distribution of fraudulous / non-frauduluous transactions is not known at the beginning
    # because using a different distribution than the real one, the cross entropy will have a bigger value than the entropy
    # adam - extension to stochastic gradient descent, derived from Adaptive Moment Estimation, Adam optimization is a stochastic gradient method that is based on adaptive estimation of first and second order moments
    # is used to update network weights
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    print("Model training with Neural Network is starting...")
    
    print("Shape of X_train is: ", X_train.shape)
    print("Shape of y_train is ", y_train.shape)

    # Fitting the ANN to the Training set
    # epoch - one time that the entire training set through the network
    # batch - number of samples to iterate through before updating the internal model parameters
    classifier.fit(X_train, y_train, batch_size = 20, epochs = 100)

    print("Model training with Neural Network completed.")

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    y_pred = (y_pred > 0.5)

    print(f'The accuracy of the Neural Network was: {classifier.evaluate(X_test, y_test)}')

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

# Oversampling - this doesnt work also, will investigate
# ada = ADASYN(sampling_strategy='minority', random_state=6)
# X_oversample, y_oversample = ada.fit_resample(X, y)

# X_train_oversample, X_test_oversample, y_train_oversample, y_test_oversample = train_test_split(X_oversample, y_oversample, train_size=trainDataset_size, random_state=7)

# print("OVERSAMPLING")
# decisionTree_classifier(X_train_oversample, X_test_oversample, y_train_oversample, y_test_oversample, False) 

# # SMOTE (Synthetic Minority Over-sampling TEchnique) - it looks at the feature space for the minority class data points and considers its k nearest neighbours - doesn't work
# X_oversample, y_oversample = SMOTE(sampling_strategy='auto', random_state=7).fit_sample(X, y)
# # print(f"X_RESAMPLE: {X_oversample.shape}")
# # print(Counter(y_oversample))

# oversampled_train = pd.concat([pd.DataFrame(y_oversample), pd.DataFrame(X_oversample)], axis=1)
# normalized_df = pd.concat([X_undersample_data, y_undersample_data])
# oversampled_train.columns = normalized_df.columns

# X_train_OS_sample, y_train_OS_sample, X_test_OS_sample, y_test_OS_sample = train_test_split(
#                                                                 X_oversample,
#                                                                 y_oversample,
#                                                                 train_size=trainDataset_size,
#                                                                 random_state=0)


message = "Unbalanced data detected! Applying sampling techniques and running again."


if (str(sys.argv[2]) == "DECISION_TREE"):
    decisionTree_classifier(X_train, y_train, X_test, y_test, False)
  
    print(message)
    
    decisionTree_classifier(X_train_sample, y_train_sample, X_test_sample, y_test_sample, True)

    print()    

elif (str(sys.argv[2]) == "RANDOM_FOREST"):
    randomForest_classifier(X_train, y_train, X_test, y_test, False)

    print(message)

    randomForest_classifier(X_train_sample, y_train_sample, X_test_sample, y_test_sample, True)

    print()

elif (str(sys.argv[2]) == "NAIVE_BAYES"):
    naiveBayes_classifier(X_train, y_train, X_test, y_test)

    print(message)

    naiveBayes_classifier(X_train_sample, y_train_sample, X_test_sample, y_test_sample)

elif (str(sys.argv[2]) == "NEURAL_NETWORK"):
    neuralNetwork_classifier(X_train, y_train, X_test, y_test)

    print(message)

    neuralNetwork_classifier(X_train_sample, y_train_sample, X_test_sample, y_test_sample)

elif (str(sys.argv[2]) == "ALL"):   
    decisionTree_classifier(X_train, y_train, X_test, y_test, False)
  
    print(message)
  
    decisionTree_classifier(X_train_sample, y_train_sample, X_test_sample, y_test_sample, True)

    print("###### Before sampling: ######")
    randomForest_classifier(X_train, y_train, X_test, y_test, False)
    
    print()
    print("###### After sampling: ######")
    randomForest_classifier(X_train_sample, y_train_sample, X_test_sample, y_test_sample, True)

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

