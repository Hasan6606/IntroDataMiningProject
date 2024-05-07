import pandas as pd  #2.0.2 version was used
import matplotlib.pyplot as plt #3.7.1 version was used
import numpy as np #1.23.1 version was used

class DataFilterDisplays:
    def __init__(self, fileName):
        self.data = pd.read_csv(fileName)

    # Other methods...

    def salesTrends(self):
        # Extracting month and year from the 'Date' column
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data['YearMonth'] = self.data['Date'].dt.to_period('M')

        # Grouping sales by month
        monthly_sales = self.data.groupby('YearMonth')['Total'].sum()

        # Visualizing sales trends
        plt.figure(figsize=(10, 6))
        plt.plot(monthly_sales.index.astype(str), monthly_sales.values, marker='o')
        plt.title('Monthly Sales Trends')
        plt.xlabel('Year-Month')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def marketingCampaignPerformance(self):
        # Average rating by product line
        avg_rating = self.data.groupby('Product line')['Rating'].mean().sort_values(ascending=False)

        # Visualizing average rating
        plt.figure(figsize=(10, 6))
        avg_rating.plot(kind='bar', color='skyblue')
        plt.title('Average Rating by Product Line')
        plt.xlabel('Product Line')
        plt.ylabel('Average Rating')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    def filterCityMarketAnalysis(self):
        # Group by city and calculate the average gross margin percentage
        city_avg_margin = self.data.groupby('City')['gross margin percentage'].mean()

        # Plotting
        plt.figure(figsize=(10, 6))
        city_avg_margin.plot(kind='bar', color='skyblue')
        plt.title('Average Gross Margin Percentage by City')
        plt.xlabel('City')
        plt.ylabel('Average Gross Margin Percentage')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def filterCustomerTypeAnalysis(self):
        # Group by city, customer type, and gender to get customer counts
        customer_counts = self.data.groupby(['City', 'Customer type', 'Gender']).size().reset_index(name='Count')

        # Visualizing
        plt.figure(figsize=(12, 6))
        for customer_type in customer_counts['Customer type'].unique():
            for gender in customer_counts['Gender'].unique():
                filtered_data = customer_counts[(customer_counts['Customer type'] == customer_type) & (customer_counts['Gender'] == gender)]
                plt.bar(filtered_data['City'], filtered_data['Count'], label=f'{customer_type} - {gender}')

        plt.xlabel('City')
        plt.ylabel('Number of Customers')
        plt.title('Number of Customers by City, Customer Type, and Gender')
        plt.xticks(rotation=45)
        plt.legend(title='Customer Type - Gender')
        plt.tight_layout()
        plt.show()

# Usage
dataDisplays = DataFilterDisplays(r"C:\Users\Kaan\Downloads\supermarket_sales.csv")
dataDisplays.filterCityMarketAnalysis()
dataDisplays.filterCustomerTypeAnalysis()
dataDisplays.salesTrends()
dataDisplays.marketingCampaignPerformance()