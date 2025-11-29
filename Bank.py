import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

data=pd.read_csv('bank-full.csv')
data.columns=('age','job','marital','education','default','balance','housing')
print(data.head())
print(data.describe())
print(data.info())

label_encoder=LabelEncoder()

data['age']=label_encoder.fit_transform(data['age'])
data['job']=label_encoder.fit_transform(data['job'])
data['marital']=label_encoder.fit_transform(data['marital'])
data['education']=label_encoder.fit_transform(data['education'])
data['default']=label_encoder.fit_transform(data['default'])
data['balance']=label_encoder.fit_transform(data['balance'])
data['housing']=label_encoder.fit_transform(data['housing'])

X=data[['age','job','marital','education','default','balance']]
Y=data['housing']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)
y_pred=classifier.predict(X_test)

matrix=confusion_matrix(Y_test, y_pred)
sns.heatmap(matrix,annot=True,fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.show()
print(classification_report(Y_test, y_pred))