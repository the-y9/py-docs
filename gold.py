#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
data = pd.read_excel(r"C:\Users\dell\Desktop\py docs\Prices.xlsx", sheet_name=1)
data = data.drop([0,1,2,3])
# print(data.info())
# print(data.head(5))

df = data[['Unnamed: 2','Unnamed: 9']]
df = df.rename(columns={'Unnamed: 2': 'Date', 'Unnamed: 9': 'INR'})
df = df.drop([4])
df.reset_index(drop=True, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df.dropna(axis=0, how='any', inplace=True)
df.head()
# %%
plt.scatter(df['Year'],df['INR'])
plt.show()
#%%
from sklearn.model_selection import train_test_split
train, test = train_test_split(df[['Year','INR']], test_size=0.2, random_state=42)
xtrain, ytrain = train.iloc[:,0], train.iloc[:,1] 
xtest, ytest = test.iloc[:,0], test.iloc[:,1]
xtrain.shape, ytrain.shape, xtest.shape, ytest.shape
plt.scatter(xtrain,ytrain)
plt.show()
#%%
from sklearn.preprocessing import StandardScaler
xs = StandardScaler()
ys = StandardScaler()
xtrain_s = xs.fit_transform(xtrain.values.reshape(-1, 1))
ytrain_s = ys.fit_transform(ytrain.values.reshape(-1, 1))
plt.scatter(xtrain_s,ytrain_s)
plt.show()
# %%
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(xtrain_s)

# %%
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(xtrain_s,ytrain_s)
print(f"coef: {lr.coef_}")
print(f"intercept: {lr.intercept_}")

y_pred = lr.predict(ytrain_s)
plt.scatter(xtrain_s,ytrain_s)
plt.plot(xtrain_s,y_pred, '-r')
plt.show()
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_excel(r"C:\Users\dell\Desktop\py docs\Prices.xlsx", sheet_name=1)
data = data.drop([0,1,2,3])

# Prepare the data
df = data[['Unnamed: 2','Unnamed: 9']]
df = df.rename(columns={'Unnamed: 2': 'Date', 'Unnamed: 9': 'INR'})
df = df.drop([4])
df.reset_index(drop=True, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df.dropna(axis=0, how='any', inplace=True)

# Split the data into training and testing sets
X = df['Year'].values.reshape(-1, 1)
y = df['INR'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create polynomial features
poly = PolynomialFeatures(degree=1)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_poly)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Plot the results
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Year')
plt.ylabel('INR')
plt.legend()
plt.show()