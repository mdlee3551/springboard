#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from sklearn.decomposition import PCA
from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.externals.six import StringIO
import plotly.graph_objs as go

#visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from IPython.display import Image
import pydotplus
from sklearn.tree import export_graphviz

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline
import matplotlib.cm as cm

# for the neural network
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dense, Dropout

data = pd.read_csv("data.csv")
print(data.dtypes)
print(data.head())
print(data.shape)
data_set = data.iloc[:,1:32]
data_set.head()
data_set.info()
data_set.diagnosis.replace(to_replace = dict(M = 1, B = 0), inplace = True)
#fit a linear regression model and store the prediction

feature = list(data_set.columns[1:32])
X = data_set[feature]
y = data_set.diagnosis



# normalizing input data into ranges between 0 and 1
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

logreg = LogisticRegression(C=100 ,solver='lbfgs')  

logreg.fit(X_train, y_train)
diagnosis_pred_class_log = logreg.predict(X_test)
 
# for model performance
#confusion matrix
cm = metrics.confusion_matrix(y_test, diagnosis_pred_class_log)
print("confusion matrix:\n",cm)

print("Logistic Regression Classification Report:\n", metrics.classification_report(y_test, diagnosis_pred_class_log))

# SVM 

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

cm1 = metrics.confusion_matrix (y_test, y_pred)
print("SVM confusion matrix:\n",cm1)
print("SVM Classification Report:\n", metrics.classification_report(y_test, y_pred))


# data input with no scaling.

feature = list(data_set.columns[1:32])
X = data_set[feature]
y = data_set.diagnosis

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

#decision tree

decision_tree = DecisionTreeClassifier(max_depth=2)
decision_tree.fit(X_train,y_train)


# decision tree visualization
dot_data = StringIO()  
export_graphviz(decision_tree, out_file=dot_data,  
                    feature_names=X_train.columns.tolist(),  
                    filled=True, rounded=True,  
                    special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# measure testing errors 
depths = range(1,11)
train_rmse, test_rmse = [],[]
for depth in depths:
    decision_tree = DecisionTreeClassifier(max_depth=depth,random_state=10)
    decision_tree.fit(X_train,y_train)
    curr_train_rmse = np.sqrt(mean_squared_error(y_train,decision_tree.predict(X_train)))
    curr_test_rmse = np.sqrt(mean_squared_error(y_test,decision_tree.predict(X_test)))
    print("Decision Tree Train/Test RMSE:",curr_train_rmse," ",curr_test_rmse)
    train_rmse.append(curr_train_rmse)
    test_rmse.append(curr_test_rmse)
sns.mpl.pyplot.plot(depths,train_rmse,label='train_rmse')
sns.mpl.pyplot.plot(depths,test_rmse,label='test_rmse')
sns.mpl.pyplot.xlabel("maximum tree depth")
sns.mpl.pyplot.ylabel("rmse - lower is better")
sns.mpl.pyplot.legend()


#  The lowest test error occurs for a tree of max depth 9 but max_depth =7 gave the best results 

best_decision_tree = DecisionTreeClassifier(max_depth=7)
best_decision_tree.fit(X_train,y_train)


pd.DataFrame({'feature':feature, 'importance':best_decision_tree.feature_importances_})

dot_data2 = StringIO()  
export_graphviz(best_decision_tree, out_file=dot_data2,  
                    feature_names=X_train.columns.tolist(),  
                    filled=True, rounded=True,  
                    special_characters=True)  
graph_best = pydotplus.graph_from_dot_data(dot_data2.getvalue())  
Image(graph_best.create_png())  


# use fitted model to make predictions on testing data
y_pred = best_decision_tree.predict(X_test)
y_pred

cm = metrics.confusion_matrix(y_test, y_pred)
print("confusion matrix:\n",cm)

print(" Classification Report:\n", metrics.classification_report(y_test, y_pred))

#Random forest
rf = RandomForestClassifier(n_estimators=500, bootstrap=True, oob_score=True, random_state=123)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

cm2 = metrics.confusion_matrix(y_test, y_pred_rf)
print("confusion matrix:\n",cm2)

print("Classification Report:\n", metrics.classification_report(y_test, y_pred_rf))

#Tuning Random Forests
# list of values to try for n_estimators
estimator_range = range(20, 500, 20)

# list to store the average RMSE for each value of n_estimators
RMSE_scores = []

for estimator in estimator_range:
    rfreg = RandomForestClassifier(n_estimators=estimator, bootstrap=True, oob_score=True, random_state=1)
    rfreg.fit(X_train,y_train)
    preds = rfreg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test,preds))
    RMSE_scores.append(rmse)
    
# plot n_estimators (x-axis) versus RMSE (y-axis)
sns.mpl.pyplot.plot(estimator_range, RMSE_scores)
sns.mpl.pyplot.xlabel('n_estimators')
sns.mpl.pyplot.ylabel('RMSE (lower is better)')


#  The lowest test error occurs for n_estimators 50 

rf = RandomForestClassifier(n_estimators=50, bootstrap=True, oob_score=True, random_state=123)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest RMSE:",np.sqrt(mean_squared_error(y_test,y_pred_rf)))

cm2 = metrics.confusion_matrix(y_test, y_pred_rf)
print("confusion matrix:\n",cm2)

print("Classification Report:\n", metrics.classification_report(y_test, y_pred_rf))

#Gradient Boosting

# Fit regression model
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01}
clf = GradientBoostingClassifier(**params)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)

cm2 = metrics.confusion_matrix(y_test, y_pred)
print("confusion matrix:\n",cm2)
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))

#PCA analysis
target_pca = data_set['diagnosis']
data_pca = data_set.drop('diagnosis', axis=1)

target_pca = pd.DataFrame(target_pca)

#To make a PCA, normalize data is essential
X_pca = data_pca.values
X_std = StandardScaler().fit_transform(X_pca)

pca = PCA(svd_solver='full')
pca_std = pca.fit(X_std, target_pca).transform(X_std)

pca_std = pd.DataFrame(pca_std)
pca_std = pca_std.merge(target_pca, left_index = True, right_index = True, how = 'left')
pca_std['diagnosis'] = pca_std['diagnosis'].replace({1:'malignant',0:'benign'})


pca = PCA(n_components = 2)
pca_std = pca.fit(X_std, target_pca).transform(X_std)
pca_std = pd.DataFrame(pca_std,columns = ['First Principal Component','Second Principal Component'])
pca_std = pca_std.merge(target_pca, left_index = True, right_index = True, how = 'left')
pca_std['diagnosis'] = pca_std['diagnosis'].replace({1:'malignant',0:'benign'})


def pca_scatter(target,color) :
    tracer = go.Scatter(x = pca_std[pca_std['diagnosis'] == target]['First Principal Component'] ,
                        y = pca_std[pca_std['diagnosis'] == target]['Second Principal Component'],
                        name = target, mode = 'markers',
                        marker = dict(color = color,line = dict(width = 0))
                       )
    return tracer
layout = go.Layout(dict(title = 'PCA Scatter plot ',
                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                     title = 'First Principal Component',
                                     zerolinewidth=1,ticklen=5,gridwidth=2),
                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                     title = 'Second Principal Component',
                                     zerolinewidth=1,ticklen=5,gridwidth=2),
                        height = 800
                       ))
trace1 = pca_scatter('malignant','darkblue')
trace2 = pca_scatter('benign','darkorange')
plots = [trace2,trace1]
fig = go.Figure(data = plots,layout = layout)
fig.show()


# building neural network

X = data_set.iloc[:,1:].values
y = data_set.iloc[:,0].values

labelencoder_X = LabelEncoder()
y = labelencoder_X.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.1, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialising the ANN
classifier = Sequential()
classifier.add(Dense(activation="relu", input_dim=30, units=16, kernel_initializer="uniform"))
# Adding dropout to prevent overfitting
classifier.add(Dropout(rate=0.1))
# Adding the second hidden layer
classifier.add(Dense(activation="relu", units=16, kernel_initializer="uniform"))
# Adding dropout to prevent overfitting
classifier.add(Dropout(rate=0.1))
# Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=100, epochs=150)
# Long scroll ahead but worth
# The batch size and number of epochs have been set using trial and error. Still looking for more efficient ways. Open to suggestions. 

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
cm2 = metrics.confusion_matrix(y_test, y_pred)
print("confusion matrix:\n",cm2)
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))

