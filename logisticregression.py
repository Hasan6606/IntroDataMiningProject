import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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

# Set threshold value for binary classification
threshold = 1000
data['Total'] = data['Total'].apply(lambda x: 1 if x > threshold else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop(['Total'], axis=1), data['Total'], test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)



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