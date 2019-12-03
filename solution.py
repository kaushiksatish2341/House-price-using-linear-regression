import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge as linreg
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler

#scalar=MinMaxScaler()


a=pd.read_csv('train.csv')
a=a.fillna(method='ffill')
#print(a)

b=pd.read_csv('test.csv')
b=b.fillna(method='ffill')

X=a[['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold']]

y=a['SalePrice']

X1=b[['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold']]



X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=23)

#scalar.fit(X_train)
#X_train_scaled=scalar.transform(X_train)
#X_test_scaled=scalar.transform(X_test)

lr=linreg(alpha=20.0).fit(X_train,y_train)

#print('Coefficient: ',lr.coef_)
#print('Intercept: ',lr.intercept_)





print('R-squared score(training):{:.3f}'.format(lr.score(X_train,y_train)))
print('R-squared score(test):{:.3f}'.format(lr.score(X_test,y_test)))


print(lr.predict(X1))


