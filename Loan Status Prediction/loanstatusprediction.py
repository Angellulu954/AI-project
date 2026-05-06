#dependencies 
import numpy as np
import pandas as pd
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#Data Collection and Processing 
loan_dataset=pd.read_csv('train_u6lujuX_CVtuZ9i (1).csv')

#dropping the missing values 

loan_dataset=loan_dataset.dropna()
loan_dataset.replace({"Loan_Status":{'N':0, 'Y':1}},inplace=True)
print(loan_dataset.head())

#making all the three+ value in the original dataset (dependents) to be 4(generalized)
loan_dataset=loan_dataset.replace(to_replace='3+', value=4)

# Data Visualization

#sns.countplot(x='Education' ,hue='Loan_Status', data=loan_dataset )
#plt.show()

#sns.countplot(x='Gender', hue='Loan_Status',data=loan_dataset)
#plt.show()


#Convert categorical columns to numerical values 
loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0} ,'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)
#print(loan_dataset.head())


#Separating and removing unecessary loan id 
# Force the columns to be numeric types
loan_dataset['Loan_Status'] = loan_dataset['Loan_Status'].astype(int)
loan_dataset['Dependents'] = loan_dataset['Dependents'].astype(int)


X=loan_dataset.drop(columns=['Loan_ID','Loan_Status'])
Y=loan_dataset['Loan_Status']


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)

# Training the Model 

classifier=svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)

#Train data accuracy

X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print(f"Accuracy on training data={training_data_accuracy}")

#Test data accuracy

X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print(f"Accuracy on test data={test_data_accuracy}")







