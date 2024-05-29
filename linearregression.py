import pandas as pd
import matplotlib.pyplot as plt
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Train linear regression model
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)
y_pred_linear = model_linear.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
print('Linear Regression MSE:', mse_linear)


# Assuming you have a model that makes predictions on the test set
from sklearn.linear_model import LinearRegression

# Assume X_train, y_train, X_test, and y_test are already defined

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Now you can use the model to make predictions
y_pred = model.predict(X_test)

# Create the scatter plot to compare predictions with actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Comparison of Predictions with Actual Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()
# Assuming y_test and y_pred are already defined

# Define the range of values considered "normal"
normal_range = (y_test.mean() - 2 * y_test.std(), y_test.mean() + 2 * y_test.std())

# Create a boolean mask to filter out the outlier values
outlier_mask = ~((y_test >= normal_range[0]) & (y_test <= normal_range[1]))

# Create the scatter plot with different colors for normal values and outliers
plt.scatter(y_test[outlier_mask], y_pred[outlier_mask], color='red', label='Outliers')
plt.scatter(y_test[~outlier_mask], y_pred[~outlier_mask], color='blue', label='Normal Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Comparison of Predictions with Actual Values')
plt.legend()
plt.show()