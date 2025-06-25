# %%
# load data
import pandas as pd

data = pd.read_csv('Fish.csv')
df = data.copy()
# dispaly ten randomly selected samples
df.sample(10)
# %%
# view data information
df.info()
# %%
# check for missing values
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
# %%
# calculate the correlation matrix
df.corr(numeric_only=True)
# %%
# visualize the correlation matrix
import seaborn as sns
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='YlGnBu')
# %%
# correlation plot
sns.pairplot(df, kind='scatter', hue='Species')
# %%
# describe numerical data, where .T is used to transpose the output
df.describe().T
# %%
# target: Weight of the fish
# features: all other numerical columns (Length1, Length2, Length3, Height, Width)
y = df['Weight']
X = df.iloc[:, 2:]
# %%
# split the dataset into training and testing sets, then display their shapes
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print('X_train: {}'.format(np.shape(X_train)))
print('y_train: {}'.format(np.shape(y_train)))
print('X_test: {}'.format(np.shape(X_test)))
print('y_test: {}'.format(np.shape(y_test)))
# %%
# fit a linear regression model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
# %%
# display the model's intercept and coefficients
print('Model intercept: ', reg.intercept_)
print('Model coefficients: ', reg.coef_)
# %%
# output the regression equation
print('y = {} + {} * X1 + {} * X2 + {} * X3 + {} * X4 + {} * X5'.format(reg.intercept_, reg.coef_[0], reg.coef_[1], reg.coef_[2], reg.coef_[3], reg.coef_[4]))
# %%
# evaluate the model
from sklearn.metrics import r2_score

r2_score(y_train, reg.predict(X_train))
# %%
# k-fold cross-validation
from sklearn.model_selection import cross_val_score

cross_val_score(reg, X_train, y_train, cv=10, scoring='r2').mean()
# %%
# predict the test set and calculate the R^2 score
y_pred = reg.predict(X_test)
r2_score(y_test, y_pred)
# %%
