# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Data.csv')
dataset = dataset.drop(['Unnamed: 0'], axis=1)
#PREPROCESSING DATA
#Counting the number of Nan values
dataset.isna().sum()
#The below is the output we get
'''
Name                    0
Location                0
Year                    0
Kilometers_Driven       0
Fuel_Type               0
Transmission            0
Owner_Type              0
Mileage                 2            -> 7th index
Engine                 36
Power                  36
Seats                  42
New_Price            5195
Price                   0
dtype: int64
'''
#Adjusting the unknown values in the dataset
dataset['Engine'].fillna(value='unknown',inplace=True)
dataset['Mileage'].fillna(value='unknown',inplace=True)
dataset['Power'].fillna(value='unknown',inplace=True)
dataset['Seats'].fillna(value=dataset['Seats'].mean(), inplace=True)

#now let's see the unkown values again
dataset.isna().sum()
'''
Name                    0
Location                0
Year                    0
Kilometers_Driven       0
Fuel_Type               0
Transmission            0
Owner_Type              0
Mileage                 0
Engine                  0
Power                   0
Seats                   0
New_Price            5195
Price                   0
dtype: int64
'''

#As New_Price has many unknown values let's drop the whole column.
dataset = dataset.drop(['New_Price'], axis = 1)

#Now our data is clean so we can proceed to further steps
#Dividing the data into two variables X-> Independent and Y-> Dependent
X = pd.DataFrame(dataset.iloc[:, :-1].values)
X.columns = dataset.columns[:-1]        #Setting the columns for the X
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#EXPLORATORY DATA ANALYSIS
X_train["Name"].value_counts()
make_train = X_train["Name"].str.split(" ", expand=True)
make_test  = X_test["Name"].str.split(" ", expand=True)

#I will add a Manufacturer column to the dataset as the sale depends mostly on the Manufacturer.
X_train["Manufacturer"] = make_train[0]
X_test["Manufacturer"] = make_test[0]

#I will drop the Name column as we don't need that
X_train.drop("Name", axis = 1)
X_test.drop("Name", axis = 1)

#Manufacturers VS Number of cars plotting
plt.figure(figsize = (14, 10))
plot = sns.countplot( x = "Manufacturer", data = X_train)
plt.xticks(rotation = 90)
#Annotating the plot
for p in plot.patches:
    plot.annotate(p.get_height(), 
                        (p.get_x() + p.get_width() / 2.0, 
                         p.get_height()), 
                        ha = 'center', 
                        va = 'center', 
                        xytext = (0, 5),
                        textcoords = 'offset points')

plt.title("Count of cars based on manufacturers")
plt.xlabel("Manufacturer")
plt.ylabel("Count of cars")
plt.show()  

