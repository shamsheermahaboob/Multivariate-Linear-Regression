# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:

Step1
Read the input dataset containing multiple input variables and the output variable.

Step2
Initialize the regression coefficients and learning rate.

Step3
Calculate the predicted output using the linear regression equation.

Step4
Adjust the coefficients to minimize the prediction error.

Step5
Use the final model to predict the output for new data.

## Program:
```
Developed by:Shamsheer Banu M
Register number:212225040400
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model,metrics
from sklearn.model_selection import train_test_split
housing=datasets.fetch_california_housing()
x=housing.data
y=housing.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=1)
reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
print('Coefficients:',reg.coef_)
print('Variance score:{}'.format(reg.score(x_test,y_test)))
plt.style.use('fivethirtyeight')
plt.scatter(reg.predict(x_train),reg.predict(x_train)-y_train,color="green",s=10,label='Train data')
plt.scatter(reg.predict(x_test),reg.predict(x_test)-y_test,color="blue",s=10,label='Test data')
plt.hlines(y=0,xmin=0,xmax=50,linewidth=2)
plt.legend(loc='upper right')
plt.title('residual Errors')
plt.show()









```
## Output:


<img width="643" height="828" alt="image" src="https://github.com/user-attachments/assets/6fc2a09f-d9f3-4e5e-a6fb-573967264d63" />


## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
