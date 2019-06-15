# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

dataset = pd.read_csv('tc20171021.csv', error_bad_lines=False)
X = dataset.iloc[:,2:4].values
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                    test_size = 1/3, 
                                    random_state = 0)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression(n_jobs=1)
lin_reg.fit(X_train, y_train)

coefficients = lin_reg.coef_

y_pred = lin_reg.predict(X_test)

# build the optimal model usig backward elimination
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((1216250,1)).astype(int), values=X, axis=1)
X_opt = X[:, [0, 1]]
lin_reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
lin_reg_OLS.summary()