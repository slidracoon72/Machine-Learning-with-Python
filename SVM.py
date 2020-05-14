import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

cell_df = pd.read_csv("cell_samples.csv")
cell_df.head()

# distribution of the classes based on Clump thickness and Uniformity of cell size
# class=4: Malignant, class=2: benign
ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

print("Old")
print(cell_df.dtypes)

# converting BareNuc[object] to BareNuc[int]
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
print("New")
print(cell_df.dtypes)

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

from sklearn import svm
# using RBF (Radial Basis Function) as kernel function for kernelling; ie, to increase the dimension
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 
yhat = clf.predict(X_test)

print("________________________________________")
print("Kernel = RBF")
#evaluation
from sklearn.metrics import f1_score
print("Avg F1-score : %.4f" % f1_score(y_test, yhat, average='weighted'))

from sklearn.metrics import jaccard_similarity_score
print("Jaccard Index : %.4f" % jaccard_similarity_score(y_test, yhat))
print("________________________________________")

print("Kernel = Linear")
clf2 = svm.SVC(kernel='linear')
clf2.fit(X_train, y_train) 
yhat2 = clf2.predict(X_test)
print("Avg F1-score: %.4f" % f1_score(y_test, yhat2, average='weighted'))
print("Jaccard score: %.4f" % jaccard_similarity_score(y_test, yhat2))
print("________________________________________")