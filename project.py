                                #HEART ABNORMALITY DETECTION USING MACHINE LEARNING TECHNQUES(LINEAR REGRESSION)

#importing packages
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#assigning dataset to variable
heart_data = pd.read_csv(r'heart.csv')
heart_data.hist()
Target=pd.DataFrame(heart_data.target,columns=['target'])

# two variable for regression model
X=heart_data
Y=Target
lm=LinearRegression()
model=lm.fit(X,Y)
print("Intercept:",f'alpha={model.intercept_}')
print("Coefficent:",f'beta={model.coef_}')
Ye=model.predict(X) #print(Ye)
result_y = []
for i in Ye:
	result_y.append(abs(int(round(i[0]))))
target = list(pd.Series(Y['target']))
"""print(result_y.count(1),result_y.count(0))
print(target.count(1),target.count(0))"""
Y1=Y.to_numpy()
E=np.mean(Y1-Ye)
MSE=E**2
print("MSE",MSE)  #mean squared error

# plot
print(Ye.shape)
print(Y1.shape)

plt.figure(figsize=(12,6))
plt.scatter(Y1,np.arange(0,len(Y)),color='red') #predicted
plt.title('Estimated')
plt.xlabel('Target')
plt.ylabel('No.of samples')
plt.legend(['Estimated output data','Estimated Linear regression model',  ])

plt.figure(figsize=(12,6))
plt.scatter(Ye,np.arange(0,len(Y)),color='blue')    #actual
plt.title('Actual')
plt.xlabel('Target')
plt.ylabel('No.of samples')
plt.legend(['Actual output data','Estimated Linear regression model',  ])
plt.show()

#testing phase
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y)
model = lm.fit(X_train, y_train)
pred = model.predict(X_test)
result_y = []
for i in pred:
	result_y.append(abs(int(round(i[0]))))   #rounding,int,absolute,list of list
target = list(pd.Series(y_test['target']))
print("Target obtained by test phase:",result_y)
print("Target obtained by regression:",target)
print("Target obtained by test phase:",result_y.count(1),result_y.count(0))
print("Target obtained by regression:",target.count(1),target.count(0))

#findng accuracy
lm.fit(X_train,y_train)
r2_score = lm.score(X_test,y_test)
print("Accuracy:",r2_score*100,'%')


