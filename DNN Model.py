#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB # or any other NB model
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB # or any other NB model
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')
import sklearn
from sklearn import tree
from sklearn.utils import resample
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
##from google.colab import drive
##drive.mount('/content/drive')
##df= pd.read_csv("/content/drive/MyDrive/Customer Churn Prediction/lastFinalLast.csv")
df= pd.read_csv("./dataset/CCP-CBE-data.csv")
df.rename(columns={'INACTIVE_MARKER': 'Churn'}, inplace=True)
le = LabelEncoder()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv1D,BatchNormalization,Dropout
from keras.layers import Conv2D, Dense, MaxPool1D, Flatten, Input
model = Sequential()
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X_balance, y_balance, epochs=80, batch_size= 40)
#y_pred_dnn=model.predict(X_test)
# evaluate the keras model
_, accuracy = model.evaluate(X_balance, y_balance)
print('Accuracy: %.2f' % (accuracy*100))
ypred = model.predict(X_test)
print(ypred) 
ypred_lis = []
for i in ypred:
    if i>0.5:
        ypred_lis.append(1)
    else:
        ypred_lis.append(0)
print(ypred_lis)

conf_mat = confusion_matrix(y_test, ypred_lis)
print(conf_mat)
print("Val Accuracy: ", accuracy_score(ypred_lis, y_test))
data = {'orignal_churn':y_test, 'predicted_churn':ypred_lis}
df_check = pd.DataFrame(data)
df_check.tail(20)
print(confusion_matrix(y_test,ypred_lis))
print(classification_report(y_test,ypred_lis))

cmm = confusion_matrix(y_test, ypred_lis)
modello = 'DNN HeatMap'
cmap = ['green', 'blue', 'red', 'black','brown', 'yellow']
plt.figure(figsize= (10,8))
plt.title(modello)
sns.heatmap(cmm, annot=True,fmt='.7g',cmap=cmap ,cbar= True, robust=True ,linewidths=2.0)
print("Val Accuracy: ", accuracy_score(ypred_lis, y_test))
conf_mat = confusion_matrix(y_test, ypred_lis)
print(conf_mat)
print(confusion_matrix(y_test,ypred_lis))
print(classification_report(y_test,ypred_lis))
plt.savefig('cff.DNN.png', bbox_inches='tight', pad_inches=0.0)

print("Accuracy:", metrics.accuracy_score(y_test, ypred_lis))
print("Precision:", metrics.precision_score(y_test, ypred_lis, average='weighted'))
print("Recall:", metrics.recall_score(y_test, ypred_lis, average='weighted'))
print("F1 Score:", metrics.f1_score(y_test, ypred_lis, average='weighted'))

