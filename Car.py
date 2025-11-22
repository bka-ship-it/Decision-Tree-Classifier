import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

data=pd.read_csv('car.csv')
data.columns=('sales','maintenance','doors','persons','boot_space','safety','class')
print(data.head())
print(data.describe())
print(data.info())

label_encoder=LabelEncoder()

data['sales']=label_encoder.fit_transform(data['sales'])
data['boot_space']=label_encoder.fit_transform(data['boot_space'])
data['maintenance']=label_encoder.fit_transform(data['maintenance'])
data['doors']=label_encoder.fit_transform(data['doors'])
data['safety']=label_encoder.fit_transform(data['safety'])
data['class']=label_encoder.fit_transform(data['class'])
data['persons']=label_encoder.fit_transform(data['persons'])

X=data[['sales','maintenance','doors','persons','boot_space','safety']]
Y=data['class']
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