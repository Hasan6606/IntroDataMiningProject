import pandas as pd  #2.0.2 version was used
import matplotlib.pyplot as plt #3.7.1 version was used
import numpy as np #1.23.1 version was used

#This data set have analyzed abaout supermarket anlaysis and Customer Behavior Optimization.
#ıt have  attributes which are called "Invoice ID,Branch,City,Customer type,Gender,Product line,Unit price,Quantity,Tax 5%,Total,Date,Time,Payment,cogs,gross margin percentage,gross income,Rating"
#The attributes have ordered as regularly, ıf you want to open dataset for looking attributes. It could not necessary.

#This class have filtered sub dataset for algorthim parameters.
class DataFilterDisplays:
    def __init__(self, fileName):
        self.data = pd.read_csv(fileName)

    def filterCityMarketAnalysis(self):
        # It is Attributes of 'City', Gross Margin Percentage' ve 'Payment' group and count progresss.
         #Using Value: City, Gross Margin Percentage
        filteredData=self.data[['City', 'gross margin percentage']]
        city=filteredData['City'].tolist()
        margins=filteredData['gross margin percentage']
        plt.figure(figsize=(10,6))
        plt.bar(city,margins)
        plt.xlabel("City Name")
        plt.ylabel("Gross Margin Percentage value")
        plt.title(f'Gross Margin Percentage value for the all cities')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)  
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        plt.tight_layout()
        plt.show()
        
        
    def filterCustomerTypeAnalysis(self):
      # It is Attributes of 'Customer type', 'Gender' ve 'Payment' group and count progresss.
      #Using Value: Customer Type, Gender, City
        customer_counts = self.data.groupby(['City', 'Customer type', 'Gender']).size().reset_index(name='Count')

        # visualizing the data have displayed using Matplotlib library
        #It has analyzed two layer. 
        #The first layer is anlayzed between customer type and  count of customer.
        #The second layer is anlayzed between gender and  count of customer.
       
        plt.figure(figsize=(12, 6))
        for customer_type in customer_counts['Customer type'].unique():
            for gender in customer_counts['Gender'].unique():
                filtered_data = customer_counts[(customer_counts['Customer type'] == customer_type) & (customer_counts['Gender'] == gender)]
                plt.bar(filtered_data['City'], filtered_data['Count'], label=f'{customer_type} - {gender}')

        plt.xlabel('Cities')
        plt.ylabel('Number of Customer')
        plt.title('The number of customers in each city by referanced values of gender and customer type')
        plt.xticks(rotation=45)
        plt.legend(title='Customer Type - Gender')
        plt.tight_layout()
        plt.show()
     
    def customer_payment_analyze(self):
      # It is Attributes of 'Customer payment type', 'Gender' 'Customer Type' 'Payment' group and count progresss.
      #Using Value: Customer Type, Gender, Payment
        customer_table = self.data.groupby(['Gender', 'Customer type', 'Payment']).size().reset_index(name='Count')
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Gender', y='Count', hue='Customer type', data=customer_table, ci=None)
        plt.title('Customer Analyze by Gender, Customer Type, and Payment')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.legend(title='Customer Type')
        plt.tight_layout()
        plt.show()

    def product_analyze(self):
        # It is Attributes of 'Customer Type', 'City' 'Product line' group and count progresss.
        #Using Value: Customer Type, City, Product line
        product_table = self.data.groupby(['Product line', 'City', 'Customer type']).size().reset_index(name='Count')
        plt.figure(figsize=(12, 8))
        sns.countplot(x='Product line', hue='Customer type', data=product_table)
        plt.title('Product Analyze by Product Line, City, and Customer Type')
        plt.xlabel('Product Line')
        plt.ylabel('Count')
        plt.legend(title='Customer Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
   
        
# defined data set as documentation
dataDisplays = DataFilterDisplays("supermarket_sales.csv")

#Defined the return is that which method will run 
result=dataDisplays.filterCustomerTypeAnalysis()
#Displated result
print(result)





