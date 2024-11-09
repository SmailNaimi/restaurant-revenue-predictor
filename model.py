import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
import pickle

# Load the dataset
# This dataset contains information about restaurant customers, menu pricing, and revenue.
df = pd.read_csv('Restaurant_revenue.csv')

# ## Exploratory Data Analysis
# Displaying the first few rows of the dataset
df.head()

# Checking for missing values in the dataset
df.isnull().sum()

# Checking the shape (number of rows and columns) of the dataset
df.shape

# Display dataset information such as column types and non-null values
df.info()

# Display summary statistics of numerical and categorical columns
df.describe(include='all')

# ### Insights
# The following insights were derived from the dataset:
# - Japanese cuisine is the most ordered by customers.
# - On average, 53 customers arrive at the restaurant, with a range of approximately +-(26).
# - Average customer spending is about 29.47, with a range of approximately +-(11.47).
# - Average monthly revenue is around 268.72, with a range of approximately +-(104).

# Checking for duplicate rows in the dataset
df.duplicated().sum()

# Grouping the dataset by 'Cuisine_Type' and calculating the mean, min, max, and standard deviation for selected columns
df.groupby("Cuisine_Type")["Menu_Price","Marketing_Spend","Average_Customer_Spending","Monthly_Revenue","Reviews"].agg(["mean","min","max","std"]).round(2)

# Convert 'Cuisine_Type' from categorical to numerical using a mapping
df['Cuisine_Type'] = df['Cuisine_Type'].map({'Japanese': 0, 'American': 1, 'Mexican': 2, 'Italian': 3})

# Displaying the updated dataset
df.head()

# #### Checking for outliers
# Loop through each column and display a boxplot to visually identify any outliers
for col in df.columns:
    if col != 'Promotions' and col != 'Cuisine_Type':  # Skip 'Promotions' and 'Cuisine_Type'
        sns.boxplot(df[col])
        plt.xlabel(col)
        plt.ylabel('count')
        plt.show()

# Remove outliers from 'Monthly_Revenue' column using the IQR method
Q1 = df['Monthly_Revenue'].quantile(0.25)
Q3 = df['Monthly_Revenue'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Filtering the dataframe to remove rows with outliers
df = df[(df['Monthly_Revenue'] >= lower_bound) & (df['Monthly_Revenue'] <= upper_bound)]

# ### Histograms for Qualitative Variables
# Plotting the distribution of qualitative variables using pie charts
qualitative_vars=['Promotions','Cuisine_Type']
for col in qualitative_vars:
    plt.figure(figsize=(3,3))
    df[col].value_counts().plot(kind="pie") # Plotting a pie chart for each qualitative variable
    plt.title(col)
    plt.show()

# Plotting the distribution of numerical variables using histograms
for col in df.columns:
    if col != 'Promotions' and col != 'Cuisine_Type':
        sns.histplot(df[col])
        plt.xlabel(col)
        plt.ylabel('count')
        plt.show()

# Plotting a heatmap of the correlation matrix to visualize relationships between variables
corr=df.corr()
sns.heatmap(corr, cmap ='coolwarm', annot = True)

# Insights derived from correlation analysis:
# - Number of Customers and Monthly Revenue are highly correlated.
# - Menu Price and Marketing Spend are also highly correlated with Monthly Revenue. The reason for this correlation is that increased marketing and higher menu prices attract more customers and increase monthly revenue.

# ### MODEL
# Preparing data for machine learning model training

# Splitting dataset into features (X) and target variable (y)
X = df.drop(["Monthly_Revenue"],axis=1)  # Drop 'Monthly_Revenue' as it's the target variable
y =df[["Monthly_Revenue"]]  # Target variable

# Displaying the first few rows of features (X)
X.head()

# Displaying the first few rows of target variable (y)
y.head()

# Normalizing the features
# Separating binary and non-binary columns for proper normalization
binary_data = X["Promotions"]
non_binary_data = X.drop(["Promotions"],axis=1)
scaler = StandardScaler()
nb_scaled = scaler.fit_transform(non_binary_data)  # Scaling the non-binary features

# Creating a DataFrame for the normalized non-binary features
nb_scaled=pd.DataFrame(nb_scaled,columns=non_binary_data.columns,index=non_binary_data.index)

# Combining scaled non-binary features with the binary column
# Final feature set
df_final=pd.concat([nb_scaled,binary_data],axis=1)
df_final

# Splitting the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Displaying the first three rows of the training set
X_train.head(3)

# Initializing a linear regression model and fitting it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Displaying the model's coefficients and intercept
print(X_train.columns)
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")

# Making predictions on the test set
y_pred = model.predict(X_test)

# Making predictions on the training set
y_pred_train_model = model.predict(X_train)

# Calculating R-squared scores for both training and testing sets
r2_score_model_test = r2_score(y_test, y_pred)
r2_score_model_train = r2_score(y_train, y_pred_train_model)

# Printing the performance metrics for training and testing sets
print(f"The Train R2 score is: {r2_score(y_train, y_pred_train_model)}")
print(f"The RMSE score for Train data is: {mean_squared_error(y_train, y_pred_train_model, squared=False)}")

print("--" * 50)

print(f"The Test R2 score is: {r2_score(y_test, y_pred)}")
print(f"The RMSE score for Test data is: {mean_squared_error(y_test, y_pred, squared=False)}")

# Plotting a scatter plot of actual vs predicted values for the test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='orange', label='Predicted Value')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--', label='Ideal Fit')  # Ideal fit reference line
plt.xlabel('Actual Value (y_test)')
plt.ylabel('Predicted Value (y_pred)')
plt.title('Scatter Plot of Actual vs Predicted Values')
plt.legend()
plt.grid()
plt.show()

# Saving the scaler to a file for future use
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Saving the trained model to a file for future use
pickle.dump(model, open('model.pkl', 'wb'))
