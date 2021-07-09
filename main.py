#Import libraries
import math
import pandas as pd
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Import dataset 
dataset = pd.read_csv('kc_house_data.csv')
dataset = dataset.drop(['sqft_lot','zipcode', 'sqft_lot15'], axis = 1)    #After experimentation, it was found that these features did not contribute much to the model (R squared value), 
#hence to reduce computation, these features were dropped
x = dataset.iloc[:, 1:19].values
y = dataset.iloc[:, 0].values

#Splitting data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#Multiple Linear Regression
model = LinearRegression()
model.fit(x_train,y_train)

#Predictions
y_pred_test = model.predict(x_test)
y_pred_train = model.predict(x_train)

#Evaluation Metrics
print('\nR squared(Train) =',round(model.score(x_train, y_train), 4))
print('R squared(Test) =',round(model.score(x_test, y_test), 4))
print('Adjusted R squared =',round(1-(1-r2_score(y_test, y_pred_test))*((len(x_test)-1)/(len(x_test)-len(x_test[0])-1)), 3))

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
print('\nMean Squared Error(Train) =',round(mse_train, 3))
print('Mean Squared Error(Test) =',round(mse_test, 3))

print('\nRoot Mean Squared Error(Train) =', round(math.sqrt(mse_train), 3))
print('Root Mean Squared Error(Test) =', round(math.sqrt(mse_test), 3))

print('\nMean Absolute Error(Train) =', round(mean_absolute_error(y_train, y_pred_train), 3))
print('Mean Absolute Error(Test) =', round(mean_absolute_error(y_test, y_pred_test), 3))


print('\nIntercept = ', model.intercept_)
print('\nCoefficient =', model.coef_)
