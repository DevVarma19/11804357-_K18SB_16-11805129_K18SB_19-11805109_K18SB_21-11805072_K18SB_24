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

X_train["Name"].value_counts()
make_train = X_train["Name"].str.split(" ", expand=True)
make_test  = X_test["Name"].str.split(" ", expand=True)

#I will add a Manufacturer column to the dataset as the sale depends mostly on the Manufacturer.
X_train["Manufacturer"] = make_train[0]
X_test["Manufacturer"] = make_test[0]

#Extracting the numerical values of Milage, Engine, Power
milage_train = X_train["Mileage"].str.split(" ", expand=True)
milage_test = X_test["Mileage"].str.split(" ", expand=True)
X_train["Mileage"] = pd.to_numeric(milage_train[0], errors = 'coerce')
X_test["Mileage"] = pd.to_numeric(milage_test[0], errors = 'coerce')

engine_train = X_train["Engine"].str.split(" ", expand=True)
engine_test  = X_test["Engine"].str.split(" ", expand=True)
X_train["Engine"] = pd.to_numeric(engine_train[0], errors = 'coerce')
X_test["Engine"]  = pd.to_numeric(engine_test[0], errors = 'coerce')

power_train = X_train["Power"].str.split(" ", expand=True)
power_test  = X_test["Power"].str.split(" ", expand=True)
X_train["Power"] = pd.to_numeric(power_train[0], errors = 'coerce')
X_test["Power"]  = pd.to_numeric(power_test[0], errors = 'coerce') 

#Replacing the Null values with Mean of all the values in that column
X_train["Engine"].fillna(X_train["Engine"].astype("float64").mean(), inplace = True)
X_test["Engine"].fillna(X_train["Engine"].astype("float64").mean(), inplace = True)

X_train["Mileage"].fillna(X_train["Mileage"].astype("float64").mean(), inplace = True)
X_test["Mileage"].fillna(X_train["Mileage"].astype("float64").mean(), inplace = True)

X_train["Power"].fillna(X_train["Power"].astype("float64").mean(), inplace = True)
X_test["Power"].fillna(X_train["Power"].astype("float64").mean(), inplace = True)

X_train["Seats"].fillna(X_train["Seats"].astype("float64").mean(), inplace = True)
X_test["Seats"].fillna(X_train["Seats"].astype("float64").mean(), inplace = True)

#Dropping all the unnecessary columns in the dataset
#Name is no longer required as we are considering the Manufacturer as the sole base.
X_train = X_train.drop("Name", axis = 1)
X_test = X_test.drop("Name", axis = 1)
#Location should not determine the price of the car.
X_train = X_train.drop("Location", axis = 1)
X_test = X_test.drop("Location", axis = 1)
#Year has no significance unless we consider how old the given car is
X_train = X_train.drop("Year", axis = 1)
X_test = X_test.drop("Year", axis = 1)



#EXPLORATORY DATA ANALYSIS
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

#Categorical Data
#let's create dummy columns for categorical columns before we begin training.
X_train = pd.get_dummies(X_train,
                         columns = ["Manufacturer", "Fuel_Type", "Transmission", "Owner_Type"],
                         drop_first = True)
X_test = pd.get_dummies(X_test,
                         columns = ["Manufacturer", "Fuel_Type", "Transmission", "Owner_Type"],
                         drop_first = True)
#It might be possible that the dummy column creation would be different in test and train data, thus, I'd fill in all missing columns with zeros
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[X_train.columns]

#Feature Scaling the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = y_train.reshape(-1, 1)
y_train = sc_y.fit_transform(y_train)

X_train.sum()
y_train.sum()
#Fitting simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#predicting the Test set results 
y_pred = regressor.predict(X_test)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(LinearRegression(), X_train, y_train, cv=5, scoring='r2')
print(scores)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
'''
    Output :- 0.730265013874368
'''
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100)
rf.fit(X_train, y_train.ravel())
y_pred = rf.predict(X_test)
r2_score(y_test, y_pred)





