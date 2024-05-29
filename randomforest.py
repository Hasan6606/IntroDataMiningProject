import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('supermarket_sales.csv')
# Drop unnecessary columns
data = data.drop(['Invoice ID', 'Time', 'Payment', 'cogs', 'gross margin percentage', 'gross income', 'Rating'], axis=1)

# Convert categorical variables to numerical variables
data['Branch'] = pd.factorize(data['Branch'])[0]
data['City'] = pd.factorize(data['City'])[0]
data['Customer type'] = pd.factorize(data['Customer type'])[0]
data['Gender'] = pd.factorize(data['Gender'])[0]
data['Product line'] = pd.factorize(data['Product line'])[0]

# Convert date column to numerical format
data['Date'] = pd.to_datetime(data['Date']).astype('int64')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.drop(['Total'], axis=1), data['Total'], test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Train random forest regressor model
model_random_forest = RandomForestRegressor()
model_random_forest.fit(X_train, y_train)
y_pred_random_forest = model_random_forest.predict(X_test)
mse_random_forest = mean_squared_error(y_test, y_pred_random_forest)
print('Random Forest Regressor MSE:', mse_random_forest)
#The value 1.9201959365872643 is the mean squared error (MSE) of the predictions made by the Random Forest Regressor model on the test dataset. MSE is a commonly used metric for evaluating the performance of regression models. It measures the average squared difference between the predicted values and the actual values in the test dataset. A lower MSE value indicates better performance of the model. In this case, the MSE value of 1.9201959365872643 indicates that the Random Forest Regressor model has an average squared error of approximately 1.92 between its predictions and the actual values in the test dataset. This value can be used to compare the performance of different regression models and to evaluate the effectiveness of the Random Forest Regressor model in predicting the target variable.

# Assuming new_data is your new data

# Predict the total amount spent by the new customers
# Calculate the interquartile range (IQR)
Q1 = np.percentile(y_test, 25)
Q3 = np.percentile(y_test, 75)
IQR = Q3 - Q1

# Identify the outliers
outlier_mask = ~((y_test >= Q1 - 1.5 * IQR) & (y_test <= Q3 + 1.5 * IQR))

# Create the graph
plt.figure(figsize=(10, 6))

# Plot the normal data points
plt.scatter(range(len(y_test)), y_test[~outlier_mask], color='blue', label='Normal Data')

# Plot the outlier data points
plt.scatter(np.where(outlier_mask)[0], y_test[outlier_mask], color='red', label='Outliers')

# Add title and labels
plt.title('Outlier Detection and Visualization')
plt.xlabel('Index')
plt.ylabel('Total')

# Add legend
plt.legend()

# Show the graph
plt.show()