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

df = pd.read_csv('Restaurant_revenue.csv')

# ## Explatory Data Analysis

df.head()

df.isnull().sum()

df.shape




df.info()

df.describe(include='all')


# ### Insights 

# Japanese cuisine is the most ordered by the customers.
# 
# On an average 53 customers arrives in the restaurant with a range of +-(26).
# 
# Average customer spending is about 29.47 with a range of +-(11.47).
# 
# Average Monthly Revenue is around 268.72 with a range of +-(104).

# In[8]:


df.duplicated().sum()


# In[9]:


df.groupby("Cuisine_Type")[["Menu_Price","Marketing_Spend","Average_Customer_Spending","Monthly_Revenue","Reviews"]].agg(["mean","min","max","std"]).round(2) #we can group more specifically to see detailed


# In[10]:


# Convert 'Cuisine_Type' from categorical to numerical using a mapping
df['Cuisine_Type'] = df['Cuisine_Type'].map({'Japanese': 0, 'American': 1, 'Mexican': 2, 'Italian': 3})

df.head()


# #### Checking for outliers

# In[11]:


for col in df.columns:
    if col != 'Promotions' and col != 'Cuisine_Type':
        sns.boxplot(df[col])
        plt.xlabel(col)
        plt.ylabel('count')
        plt.show()


# In[12]:


#remove outliers from monthly revenue
Q1 = df['Monthly_Revenue'].quantile(0.25)
Q3 = df['Monthly_Revenue'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Monthly_Revenue'] >= lower_bound) & (df['Monthly_Revenue'] <= upper_bound)]


# ### Histplots

# In[13]:


#qualitative variables
qualitative_vars=['Promotions','Cuisine_Type']
for col in qualitative_vars:
    plt.figure(figsize=(3,3))
    df[col].value_counts().plot(kind="pie") # ou : sns.countplot(x=col, data=data)
    plt.title(col)
    plt.show()


# In[14]:


for col in df.columns:
    if col != 'Promotions' and col != 'Cuisine_Type':
        sns.histplot(df[col])
        plt.xlabel(col)
        plt.ylabel('count')
        plt.show()


# In[16]:


#correlation matrix.
corr=df.corr()
sns.heatmap(corr, cmap ='coolwarm', annot = True)


# Number of Customers and Monthly Revenue are highly correlated
# 
# Also Menu Price and Marketing spend are highly correlated with Monthly Revenue. The reason for this correlation is if you make your brand appear more on everywhere, more customer will come and more revenue will be made also higher the menu price higher the monthly revenue

# #### MODEL

# In[17]:


X = df.drop(["Monthly_Revenue"],axis=1)
y =df[["Monthly_Revenue"]]


# In[18]:


X.head()


# In[19]:


y.head()


# In[20]:


#normalization
binary_data=X["Promotions"]
non_binary_data=X.drop(["Promotions"],axis=1)
scaler = StandardScaler()
nb_scaled = scaler.fit_transform(non_binary_data)


# In[21]:


#dataframe
nb_scaled=pd.DataFrame(nb_scaled,columns=non_binary_data.columns,index=non_binary_data.index)


# In[22]:


#final_df
df_final=pd.concat([nb_scaled,binary_data],axis=1)
df_final


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train.head(3)


# In[24]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[25]:


#model coefs
print(X_train.columns)
print(f"les coefficients du modèle:{model.coef_}")
print(f"la constante:{model.intercept_}")


# In[26]:


y_pred = model.predict(X_test)
y_pred


# In[27]:


y_pred_train_model = model.predict(X_train)


# In[39]:


r2_score_model_test = r2_score(y_test,y_pred)
r2_score_model_train = r2_score(y_train,y_pred_train_model)


# In[41]:


#performance scores for data train
print(f"The Train r2 score is: {r2_score(y_train, y_pred_train_model)}")
print(f"The RMSE score for Train data is: {mean_squared_error(y_train, y_pred_train_model, squared=False)}")

print("--" * 50)

#performance scores for data test
print(f"The Test r2 score is: {r2_score(y_test, y_pred)}")
print(f"The RMSE score for Test data is: {mean_squared_error(y_test, y_pred, squared=False)}")


# In[37]:


#plot y_test vs y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='orange', label='Predicted Value')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--', label='Ideal Fit')  # Ligne de référence idéale
plt.xlabel('Actual Value (y_test)')
plt.ylabel('Predicted Value (y_pred)')
plt.title('Scatter Plot of Actual vs Predicted Values')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


#saving scaler
pickle.dump(scaler, open('scaler.pkl', 'wb'))

#saving model
pickle.dump(model, open('model.pkl', 'wb'))

