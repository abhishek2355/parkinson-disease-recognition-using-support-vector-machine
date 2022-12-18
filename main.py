# Importing the Dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score




#  Data Collection and Analysis 

# loading the data from csv files to a pandas Dataframe
parkinsons_data = pd.read_csv('C:/Users/Abhishek Raut/Desktop/Mini Project/parkinsons.csv')

# Printing the first 5 rows of the dataframe
print(parkinsons_data.head())

# number of the rows and columns in the dataset
print(parkinsons_data.shape)


# getting more information about the dataset
print(parkinsons_data.info())

# Checking for missing values in each columns
print(parkinsons_data.isnull)

# Getting some statistical measures about the dataset
print(parkinsons_data.describe())

# Distribution of targets variables Status
print(parkinsons_data['status'].value_counts())

# ------------------------------ 0--> Not having parkinsons and 1--> having parkinsons


# grouping the data based on the target variables 
print(parkinsons_data.groupby('status').mean())





# Data preprocessing
# sepreting the features and targetting
x= parkinsons_data.drop(columns=['name','status'], axis=1)
y= parkinsons_data['status']

print(x)
print(y)

# Spliting the data to training data and Test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
print(x.shape,x_train.shape,x_test.shape)


# Data Standardization 
scaler = StandardScaler()
print(scaler.fit(x_train))
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train)


# Model Training
################ Support vector machines Model #######################################
model = svm.SVC(kernel='linear')

# training the svm model with the training data
model.fit(x_train,y_train)

# Evaluating Model

# 1. Accuracy Score
# ---- Accuracy score on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print("Accuracy score of training data:",training_data_accuracy)

#  ----- Accuracy score on the test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)
print("Accuracy score of testting data:",test_data_accuracy)


# Bulding a predictive System 
input_data=(242.85200,255.03400,227.91100,0.00225,0.000009,0.00117,0.00139,0.00350,0.01494,0.13400,0.00847,0.00879,0.01014,0.02542,0.00476,25.03200,0.431285,0.638928,-6.995820,0.102083,2.365800,0.102706)

# changing input data to numpy array
input_data_as_numpy_array = np.asarray(input_data) 

# reshape the numpy array
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

# Standardize the data
std_data = scaler.transform(input_data_reshape)

prediction = model.predict(std_data)

final_msg=""
if prediction[0]==0:
    final_msg="The person does not have parkinsons"
else:
    final_msg="The person haveing parkinsons."



from flask import Flask, render_template
app = Flask(__name__)
@app.route('/')
def home():
    return render_template("home.html",msg=final_msg)
app.run()