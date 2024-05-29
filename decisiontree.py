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
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Train decision tree regressor model
model_decision_tree = DecisionTreeRegressor()
model_decision_tree.fit(X_train, y_train)
y_pred_decision_tree = model_decision_tree.predict(X_test)
mse_decision_tree = mean_squared_error(y_test, y_pred_decision_tree)
print('Decision Tree Regressor MSE:', mse_decision_tree)


# Group the data by gender and calculate the total amount spent
gender_spending = data.groupby('Gender')['Total'].sum()

# Create the graph
plt.figure(figsize=(8, 6))

# Plot the bar chart
plt.bar(gender_spending.index, gender_spending.values)

# Add title and labels
plt.title('Total Amount Spent by Gender')
plt.xlabel('Gender')
plt.ylabel('Total Amount Spent')

# Show the graph
plt.show()
# Calculate the interquartile range (IQR)
Q1 = np.percentile(data['Total'], 25)
Q3 = np.percentile(data['Total'], 75)
IQR = Q3 - Q1

# Identify the outliers
outlier_mask = ~((data['Total'] >= Q1 - 1.5 * IQR) & (data['Total'] <= Q3 + 1.5 * IQR))

# Create the graph
plt.figure(figsize=(10, 6))

# Plot the normal data points
plt.scatter(data[~outlier_mask].index, data[~outlier_mask]['Total'], color='blue', label='Normal Data')

# Plot the outlier data points
plt.scatter(data[outlier_mask].index, data[outlier_mask]['Total'], color='red', label='Outliers')

# Add title and labels
plt.title('Outlier Detection and Visualization')
plt.xlabel('Index')
plt.ylabel('Total')

# Add legend
plt.legend()

# Show the graph
plt.show()