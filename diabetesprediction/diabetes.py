import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
#Data Collection and analysis 
diabetes_dataset=pd.read_csv("diabetes.csv")
diabetes_dataset.shape
diabetes_dataset.describe()
diabetes_dataset['Outcome'].value_counts()

X=diabetes_dataset.drop(columns='Outcome',axis=1)
Y=diabetes_dataset['Outcome']

# Standardization

scaler=StandardScaler()
scaler.fit(X)
standerdized_data=scaler.transform(X)
X=standerdized_data

#Splitting data in test and train and test

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

#training

classifier=svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)

#Evaluation

X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print(f"Accuracy Score of the test data: {test_data_accuracy}")

#predictive system

input_data=(4,110,92,0,0,37.6,0.191,30)
input_data_as_numpy_array=np.asarray(input_data)

#reshape
input_data_reshape=input_data_as_numpy_array.reshape(1,-1)
std_data=scaler.transform(input_data_reshape)
print(std_data)
prediction=classifier.prdict(std_data)
print(prediction)

if(prediction[0]==0):
    print("The person is not diabetic")

else:
    print("The person is diabetic")

