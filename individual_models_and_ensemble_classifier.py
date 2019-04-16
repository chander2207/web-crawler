# importing the libraries
import pandas as pd
import numpy as np
import time
import statistics
import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# importing the dataset
Training_dataset = pd.read_csv('lab4-train.csv')
Testing_dataset = pd.read_csv('lab4-test.csv')

# splitting the data
X_train = Training_dataset.iloc[:,0:4].values
X_test = Testing_dataset.iloc[:,0:4].values
y_train = Training_dataset.iloc[:,4].values
y_test = Testing_dataset.iloc[:,4].values

# Feature scaling
# from sklearn.preprocessing import StandardScaler
# # sc = StandardScaler()
# # X_train = sc.fit_transform(X_train)
# # X_test = sc.fit_transform(X_test)

# Fitting Random Forest Classification to the Training set
# Random state makes sure to replicate same output on same input
# n_estimator tells about the number of trees algorithm builds
# n_jobs tells how many processor to use.
from sklearn.ensemble import RandomForestClassifier
rfclassifier = RandomForestClassifier(n_estimators=15, max_depth=6, min_samples_leaf=8, criterion='entropy', n_jobs=2, random_state=40)
rfclassifier.fit(X_train,y_train)

# Predicting to the Test set results
y_pred_rfc = rfclassifier.predict(X_test)

# Making the confusion Matrix
cm_rfc = confusion_matrix(y_test,y_pred_rfc)
print("\nRandom Forest Classifier Confusion Matrix")
print(cm_rfc)

# Calculating the overall accuracy
accuracy_rfc = accuracy_score(y_test,y_pred_rfc)
print("\nRandom Forest Accuracy: {0}".format(accuracy_rfc*100))

# ------------------------------------------------------------
# AdaBoost Classifier
# This classifier uses decision tree as the default classifier and as these
# trees are so short and contain only one decision for classification, they are
# often called decision stumps.
from sklearn.ensemble import AdaBoostClassifier
abc_classifier = AdaBoostClassifier(n_estimators=10, random_state=0)
abc_classifier.fit(X_train,y_train)

# Predicting to the Test set results
y_pred_abc = abc_classifier.predict(X_test)

# Making the confusion Matrix
cm_abc = confusion_matrix(y_test,y_pred_abc)
print("\nAdaBoost Classifier Confusion Matrix")
print(cm_abc)

# Calculating the overall accuracy
accuracy_abc = accuracy_score(y_test,y_pred_abc)
print("\nAdaBoost Accuracy: {0}".format(accuracy_abc*100))

# --------------------------------------------------------------
# Neural Network
# fix random seed for reproducibility
np.random.seed(7)

from keras.models import Sequential
from keras.layers import Dense

start_time = time.time()
#Initializing the ANN
nn_classifier = Sequential()

#Adding the input layer and the first hidden layer
nn_classifier.add(Dense(activation="relu", input_dim=4, units=10, kernel_initializer="uniform"))

# Adding the second hidden layer
# nn_classifier.add(Dense(activation="relu", units=4, kernel_initializer="uniform"))

# Adding the third hidden layer
# nn_classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

# Adding the output layer
nn_classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# compiling the ANN
nn_classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training data
nn_classifier.fit(X_train, y_train, batch_size = 20, epochs = 20)
print("--- %s seconds ---" % (time.time() - start_time))

# predicting the test set result
y_pred_nn = nn_classifier.predict(X_test)
y_pred_nn = (y_pred_nn > 0.5).astype(int)
y_pred_nn = np.hstack(y_pred_nn)

# Making the confusion Matrix
cm_nn = confusion_matrix(y_test,y_pred_nn)
print("\nNeural Network Classifier Confusion Matrix")
print(cm_nn)

# Calculating the overall accuracy
accuracy_nn = accuracy_score(y_test,y_pred_nn)
print("\nNeural Network Accuracy: {0}".format(accuracy_nn*100))

# --------------------------------------------------------------
# K-NN Classifier
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=8, p=2, metric='minkowski')
knn_classifier.fit(X_train,y_train)

# Predicting to the Test set results
y_pred_knn = knn_classifier.predict(X_test)

# Making the confusion Matrix
cm_knn = confusion_matrix(y_test,y_pred_knn)
print("\nK-NN Classifier Confusion Matrix")
print(cm_knn)

# Calculating the overall accuracy
accuracy_knn = accuracy_score(y_test,y_pred_knn)
print("\nK-NN Accuracy: {0}".format(accuracy_knn*100))

# --------------------------------------------------------------
# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(solver='lbfgs', random_state=0)
lr_classifier.fit(X_train,y_train)

# Predicting to the Test set results
y_pred_lr = lr_classifier.predict(X_test)

# Making the confusion Matrix
cm_lr = confusion_matrix(y_test,y_pred_lr)
print("\nLogistic Regression Classifier Confusion Matrix")
print(cm_lr)

# Calculating the overall accuracy
accuracy_lr = accuracy_score(y_test,y_pred_lr)
print("\nLogistic Regression Accuracy: {0}".format(accuracy_lr*100))

# --------------------------------------------------------------
# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train,y_train)

# Predicting to the Test set results
y_pred_nb = nb_classifier.predict(X_test)

# Making the confusion Matrix
cm_nb = confusion_matrix(y_test,y_pred_lr)
print("\nNaive Bayes Classifier Confusion Matrix")
print(cm_nb)

# Calculating the overall accuracy
accuracy_nb = accuracy_score(y_test,y_pred_nb)
print("\nNaive Bayes Accuracy: {0}".format(accuracy_nb*100))

# --------------------------------------------------------------
# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc_classifier = DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=20, random_state=10)
dtc_classifier.fit(X_train,y_train)

# Predicting to the Test set results
y_pred_dtc = dtc_classifier.predict(X_test)

# Making the confusion Matrix
cm_dtc = confusion_matrix(y_test,y_pred_dtc)
print("\nDecision Tree Classifier Confusion Matrix")
print(cm_dtc)

# Calculating the overall accuracy
accuracy_dtc = accuracy_score(y_test,y_pred_dtc)
print("\nDecision Tree Accuracy: {0}".format(accuracy_dtc*100))

# ---------------------------------------------------------------
# Ensemble Learning unweighted majority vote for 5 models
y_pred_uw8_5model = np.array([])
for i in range (0,len(X_test)):
    y_pred_uw8_5model = np.append(y_pred_uw8_5model,statistics.mode([y_pred_nn[i],y_pred_knn[i],y_pred_lr[i],y_pred_nb[i],y_pred_dtc[i]]))

# Calculating the overall accuracy
accuracy_ens_uw8_5model = accuracy_score(y_test,y_pred_uw8_5model)
print("\nEnsemble Unweighted Accuracy for 5 models: {0}".format(accuracy_ens_uw8_5model*100))

# ---------------------------------------------------------------
# Ensemble Learning weighted majority vote for 5 models
n_clf_list = []
clf_pred_list =[]
clf_and_weight_tuple_list =[]
y_pred_w8_5model = np.array([])
sum_1 = 0
sum_0 = 0
clf_nn_weight = 4
clf_knn_weight = 1
clf_lr_weight = 2
clf_nb_weight = 3
clf_dt_weight = 5
cumulative_weight =(clf_dt_weight+clf_knn_weight+clf_lr_weight+clf_nb_weight+clf_nn_weight)

# normalized weight
n_clf_nn = round(clf_nn_weight/cumulative_weight,2)
n_clf_knn = round(clf_knn_weight/cumulative_weight,2)
n_clf_lr = round(clf_lr_weight/cumulative_weight,2)
n_clf_nb = round(clf_nb_weight/cumulative_weight,2)
n_clf_dt = round(clf_dt_weight/cumulative_weight,2)
# ncw = n_clf_dt + n_clf_knn + n_clf_lr + n_clf_nb + n_clf_nn

# Creating a list of the normalized weight of classifier
n_clf_list = [n_clf_nn,n_clf_knn,n_clf_lr,n_clf_nb,n_clf_dt]
print("\nnormalized weight for 5 models : {0} ".format(n_clf_list))

# Looping through all the instances of the test set to create ensemble predictor
for i in range (0,len(X_test)):
    clf_pred_list = [y_pred_nn[i],y_pred_knn[i],y_pred_lr[i],y_pred_nb[i],y_pred_dtc[i]]
    clf_and_weight_tuple_list = list(zip(clf_pred_list,n_clf_list))
    # print(clf_and_weight_tuple_list)
    # To find the weight, we are adding the classifier weight corresponding to 1 or 0
    # and picks the value with highest weight
    for x,y in clf_and_weight_tuple_list:
        if x == 1:
            sum_1 += y
        else:
            sum_0 += y
    # print("sum_1: {0} and sum_0: {1}".format(round(sum_1,2),round(sum_0,2)))
    if sum_1 > sum_0:
        y_pred_w8_5model = np.append(y_pred_w8_5model,1)
    else:
        y_pred_w8_5model = np.append(y_pred_w8_5model,0)
    sum_1 = 0
    sum_0 = 0

# Calculating the overall accuracy
accuracy_ens_w8_5model = accuracy_score(y_test,y_pred_w8_5model)
print("\nEnsemble Weighted Accuracy for 5 models: {0}".format(accuracy_ens_w8_5model*100))

# ---------------------------------------------------------------
# Ensemble Learning unweighted majority vote for 7 models
y_pred_uw8_7model = np.array([])
for i in range (0,len(X_test)):
    y_pred_uw8_7model = np.append(y_pred_uw8_7model,statistics.mode([y_pred_nn[i],y_pred_knn[i],y_pred_lr[i],y_pred_nb[i],y_pred_dtc[i],y_pred_abc[i],y_pred_rfc[i]]))

# Calculating the overall accuracy
accuracy_ens_uw8_7model = accuracy_score(y_test,y_pred_uw8_7model)
print("\nEnsemble Unweighted Accuracy for 7 models: {0}".format(accuracy_ens_uw8_7model*100))

# ---------------------------------------------------------------
# Ensemble Learning weighted majority vote for 7 models
n_clf_list_7m = []
clf_pred_list_7m =[]
clf_and_weight_tuple_list_7m =[]
y_pred_w8_7model = np.array([])
sum_1 = 0
sum_0 = 0
clf_nn_weight = 4
clf_knn_weight = 2
clf_lr_weight = 5
clf_nb_weight = 3
clf_dt_weight = 1
clf_rf_weight = 20
clf_ab_weight = 10
cumulative_weight =(clf_dt_weight+clf_knn_weight+clf_lr_weight+clf_nb_weight+clf_nn_weight+clf_rf_weight+clf_ab_weight)

# normalized weight
n_clf_nn = round(clf_nn_weight/cumulative_weight,2)
n_clf_knn = round(clf_knn_weight/cumulative_weight,2)
n_clf_lr = round(clf_lr_weight/cumulative_weight,2)
n_clf_nb = round(clf_nb_weight/cumulative_weight,2)
n_clf_dt = round(clf_dt_weight/cumulative_weight,2)
n_clf_rf = round(clf_rf_weight/cumulative_weight,2)
n_clf_ab = round(clf_ab_weight/cumulative_weight,2)
ncw = n_clf_dt + n_clf_knn + n_clf_lr + n_clf_nb + n_clf_nn + n_clf_rf + n_clf_ab

# Creating a list of the normalized weight of classifier
n_clf_list_7m = [n_clf_nn,n_clf_knn,n_clf_lr,n_clf_nb,n_clf_dt,n_clf_rf,n_clf_ab]
print("\nnormalized_weight : {}".format(n_clf_list_7m))

# Looping through all the instances of the test set to create ensemble predictor
for i in range (0,len(X_test)):
    clf_pred_list_7m = [y_pred_nn[i],y_pred_knn[i],y_pred_lr[i],y_pred_nb[i],y_pred_dtc[i],y_pred_rfc[i],y_pred_abc[i]]
    clf_and_weight_tuple_list_7m = list(zip(clf_pred_list_7m,n_clf_list_7m))
    # print(clf_and_weight_tuple_list)
    # To find the weight of binary values, we are adding the classifier weight corresponding to 1 or 0
    # and picks the value with highest weight
    for x,y in clf_and_weight_tuple_list_7m:
        if x == 1:
            sum_1 += y
        else:
            sum_0 += y
    # print("sum_1: {0} and sum_0: {1}".format(round(sum_1,2),round(sum_0,2)))
    if sum_1 > sum_0:
        y_pred_w8_7model = np.append(y_pred_w8_7model,1)
    else:
        y_pred_w8_7model = np.append(y_pred_w8_7model,0)
    sum_1 = 0
    sum_0 = 0

# Calculating the overall accuracy
accuracy_ens_w8_7model = accuracy_score(y_test,y_pred_w8_7model)
print("\nEnsemble Weighted Accuracy for 7 models: {0}".format(accuracy_ens_w8_7model*100))