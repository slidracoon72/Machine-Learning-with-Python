import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

my_data = pd.read_csv("drug200.csv", delimiter=",")

column_count = my_data['Drug'].value_counts()
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

# converting M to
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

Y = X.astype(float)

y = my_data["Drug"]

from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(X_trainset,y_trainset)

#testing
predTree = drugTree.predict(X_testset)
print("Predicted values: ")
print (predTree[0:5])
print("_______________")
print("Actual values: ")
print (y_testset[0:5])

from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: %.3f" % metrics.accuracy_score(y_testset, predTree))

# For visualization, see Jupyter

