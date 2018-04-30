#==============================================
#House Prices -regression problem
#==============================================




#Reference
#==============================================
#https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
#http://erica-tan.me/housePrice.html
#https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
#https://www.hackerearth.com/practice/machine-learning/machine-learning-projects/python-project/tutorial/


# data analysis , wrangling and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm , skew
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV ,Ridge ,Lasso ,LassoCV
import lightgbm as lgb


#fill missing data depending on its type by a certain value or by another column value in thesame row
def fill_missing_data(frame, column, value = 'None', re_column='None'):
    filtered = frame[column].isnull()
    if re_column == 'None':
        frame.loc[filtered, column] = value
    else:
        frame.loc[filtered, column] = frame.loc[filtered, re_column]

#Root Mean Squered Error(distance between predicted & actual value)
def rmse(y_test,y_pred):
      return np.sqrt(mean_squared_error(y_test,y_pred))


#==============================================
#Preprocessing Data 
#==============================================
     
# train data
dataset_train = pd.read_csv('train.csv')
# sumbit data 
dataset_test = pd.read_csv('test.csv')

#important variables
y_train = dataset_train.SalePrice.values
train_ID = dataset_train['Id']
test_ID = dataset_test['Id']
dataset_train.drop("Id", axis = 1, inplace = True)
dataset_test.drop("Id", axis = 1, inplace = True)

#============================================================================================
#General study (train data)
#============================================================================================
print (dataset_train.columns)
print (dataset_train.columns.values)
dataset_train.isnull().sum()
dataset_train.head()
dataset_train.tail()
dataset_train.info()
dataset_train.describe()
dataset_train.describe(include=['O'])



#============================================================================================
#univariavle study (train data)
#============================================================================================

dataset_train.SalePrice.describe()

#skewness and kurtosis
print("Skewness: %f" % dataset_train['SalePrice'].skew())
print("Kurtosis: %f" % dataset_train['SalePrice'].kurt())

#histogram  plot
plt.hist(y_train, color='blue')
plt.show()
#or 
sns.distplot(dataset_train['SalePrice'], fit=norm);
fig = plt.figure()

# normal probability plot
res = stats.probplot(dataset_train['SalePrice'], plot=plt)

#Do normalization for the goal
y_train= np.log(y_train)

#============================================================================================
# Multivariate study (train data)
#============================================================================================

# Method one :pick selected features
#scatter plot grlivarea/saleprice  //linear relationship.
data = pd.concat([dataset_train['SalePrice'],dataset_train['GrLivArea']],axis=1)
data.plot.scatter(x='GrLivArea' , y='SalePrice',ylim=(0,800000));

#scatter plot totalbsmtsf/saleprice
data = pd.concat([dataset_train['SalePrice'],dataset_train['TotalBsmtSF']],axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice',ylim=(0,800000));

#box plot overallqual/saleprice
data = pd.concat([dataset_train['SalePrice'], dataset_train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);

#box plot YearBuilt/saleprice
data = pd.concat([dataset_train['SalePrice'], dataset_train['YearBuilt']], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x='YearBuilt', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);

#Method two : take all the features using
#1- correlation matrix (heat map)
corrmat = dataset_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8,square=True);

#2- saleprice correlation matrix (zoomed heat map)
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index
cm = np.corrcoef(dataset_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#10 top correlated features
#5 with positive effect
print (corrmat['SalePrice'].sort_values(ascending=False)[:5], '\n')
#5 with negative effect
print (corrmat['SalePrice'].sort_values(ascending=False)[-5:])

''' top 10 +
SalePrice       
OverallQual     
GrLivArea       TotRmsAbvGrd
GarageCars      GarageArea
GarageArea      --(repeated feature)
TotalBsmtSF     1stFlrSF
1stFlrSF        --(repeated feature)
FullBath        
TotRmsAbvGrd    --(repeated feature)
YearBuilt 
'''

#scatterplot for top 10 without repeated features (features with same correlation %)
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(dataset_train[cols], size = 2.5) 
plt.show();

#============================================================================================
# Basic cleaning (train data , test data)
#============================================================================================

# 1- outliars (train data)
#================================================
#i - Univariate analysis
#standardizing data
saleprice_scaled = StandardScaler().fit_transform(dataset_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled [saleprice_scaled[:,0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

#ii- bivariate analysis for continous numerical(continous numerical important feature from corr)
#scatter plot grlivarea/saleprice 
data = pd.concat([dataset_train['SalePrice'],dataset_train['GrLivArea']],axis=1)
data.plot.scatter(x='GrLivArea' , y='SalePrice',ylim=(0,800000));
#deleting points (outliars values)
dataset_train = dataset_train.drop(dataset_train[dataset_train['GrLivArea'] >4000].index)

#bivariate analysis saleprice/TotalBsmtSF
data = pd.concat([dataset_train['SalePrice'],dataset_train['TotalBsmtSF']],axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice',ylim=(0,800000));
# we can consder above > 3000 is outliar but I suppose it's not worth it


# 2- missing data (train data , test data)
#================================================
#to apply process once at test and train 
ntrain = dataset_train.shape[0]
ntest = dataset_test.shape[0]
all_data = pd.concat((dataset_train, dataset_test)).reset_index(drop=True)



# how many features are missing values 
all_data.columns[all_data.isnull().any()]
total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(35)

#method one: drop columns with high null values
'''
dataset_train = dataset_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
#drop the row with null value 
dataset_train  = dataset_train.drop(dataset_train.loc[dataset_train['Electrical'].isnull()].index)
dataset_train.isnull().sum().max() #just checking that there's no missing data missing...
'''

#method two : fill with values 
#PoolQC
filtered = (all_data["PoolQC"].isnull())
all_data.loc[filtered,["PoolQC","PoolArea"]]
all_data["PoolQC"].unique()
all_data["PoolQC"].isnull().sum()
fill_missing_data(all_data, "PoolQC", "None")

#GarageType / GarageYrBlt / GarageFinish / GarageCars / GarageArea / GarageQual / GarageCond
garage_cols=["GarageType","GarageYrBlt","GarageFinish","GarageCars","GarageArea","GarageQual","GarageCond"]
filtered = all_data[garage_cols].isnull().apply(lambda x: any(x), axis=1)
all_data.loc[filtered, garage_cols]
for column in garage_cols:
    if all_data.dtypes[column] == "object":
        fill_missing_data(all_data, column, "None")
    else:
        fill_missing_data(all_data, column, 0)

#Electrical    
all_data.Electrical.value_counts()
fill_missing_data(all_data, "Electrical", all_data.Electrical.dropna().mode()[0])

#BsmtQual / BsmtCond / BsmtExposure / BsmtFinType1 / BsmtFinType2
basement_cols=["BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","BsmtFinSF1","BsmtFinSF2","BsmtFullBath", "BsmtHalfBath" ,"TotalBsmtSF","BsmtUnfSF"]
filtered = all_data[basement_cols].isnull().apply(lambda x: any(x), axis=1)
all_data.loc[filtered, basement_cols]

# fill rows that have only one value in one coulm is missing
all_data.loc[[947,2346], "BsmtExposure"] = all_data.BsmtExposure.dropna().mode()[0]
all_data.loc[[2215,2216], "BsmtQual"] = all_data.BsmtQual.dropna().mode()[0]
all_data.loc[[2183,2522], "BsmtCond"] = all_data.BsmtCond.dropna().mode()[0]

grouped = all_data.groupby("BsmtFinType2")
grouped = grouped["BsmtFinSF2"].agg(np.mean)
all_data[["BsmtFinType2","BsmtFinSF2"]].groupby(['BsmtFinType2'],as_index=False).mean().sort_values(by='BsmtFinSF2',ascending=False)
all_data.loc[332, "BsmtFinType2"] = "LwQ"

#fill remaining rows
for column in basement_cols:
    if all_data.dtypes[column] == "object":
        fill_missing_data(all_data, column, "None")
    else:
        fill_missing_data(all_data, column, 0)


#MasVnrType / MasVnrArea
filtered = (all_data["MasVnrType"].isnull()) | (all_data["MasVnrArea"].isnull())
all_data.loc[filtered,["MasVnrType","MasVnrArea"]]

# fill rows that have only one value in one coulm is missing
all_data.loc[2608,"MasVnrType"]= all_data.MasVnrType.dropna().mode()[0]

#fill remaining rows
fill_missing_data(all_data, "MasVnrType", "None")
fill_missing_data(all_data, "MasVnrArea", 0)     

#LotFrontage
'''all_data[["LotFrontage","Neighborhood"]].groupby(['Neighborhood'],as_index=False).mean().sort_values(by='LotFrontage',ascending=False)
filtered = all_data["LotFrontage"].isnull()
grouped = all_data.groupby("Neighborhood")
grouped = grouped["LotFrontage"].agg(np.mean)
all_data.loc[filtered,"LotFrontage"] = all_data.loc[filtered,"Neighborhood"].map(lambda neighbor : grouped[neighbor])'''
# or 
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median())) 

#Fence / MiscFeature
filtered = (all_data["Fence"].isnull())
all_data.loc[filtered,"Fence"]
all_data.Fence.unique()
fill_missing_data(all_data, "Fence", "None")
fill_missing_data(all_data, "MiscFeature", "None")

#FireplaceQu
filtered = (all_data["Fireplaces"] > 0) & (all_data["FireplaceQu"].isnull())
len(all_data.loc[filtered,["Fireplaces","FireplaceQu"]])
fill_missing_data(all_data, "FireplaceQu", "None")

#Alley
fill_missing_data(all_data, "Alley", "None")

#MSZoning
all_data['MSZoning']=all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

#Utilities
all_data.Utilities.value_counts()
#there is one value duplicated alot and only 1-3 rows with diffrent value so that means this coulmns does not effect on the goal
all_data = all_data.drop(['Utilities'], axis=1)

#Functional
all_data.Functional.value_counts()
all_data['Functional']=all_data['Functional'].fillna(all_data['Functional'].mode()[0])

#Exterior2nd /Exterior1st
all_data.Exterior2nd.unique()
all_data.Exterior1st.unique()
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])


#KitchenQual
all_data.KitchenQual.unique()
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])


#SaleType 
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


y_train = all_data.SalePrice.values[:ntrain]
all_data = all_data.drop(['SalePrice'], axis=1)
# 3- convert to categorized var (train data , test data ) &
# 4- Engineering Features (create new features)
#================================================
# (numerical , categorized)
numeric_features = all_data.select_dtypes(include=[np.number])
categoricals = all_data.select_dtypes(exclude=[np.number])


#there are 5 numerical and also they are catogrical 
#so convert them to string and categorized to be able to apply get_dummies()
#MSSubClass / OverallQual  / OverallCond
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['OverallQual'] = all_data['OverallQual'].astype(str)
#Year and month sold are transformed into categorized features.(consider it is a number catigrical)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

# now we can do dummies for all string categorical vars 
# but we will do engineering features first 

# we can create new features, in 3 ways :
# 1- Simplifications of existing features

# by : categorized vars contain severl options which are ordered information meaning ex : g , vg , Ex (reduce them)
# How : Encode some categorized features as ordered numbers then reduce these options to 3 options for example by maping manually
# example do the replace:
# map 1,2 to 1 
# map 3,4 to 2
# map 5 to 3 

#categorized features with orderd meaning
cols_1 = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond' ,'GarageFinish', 'Street',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1','Fence', 
        'BsmtFinType2',  'LandSlope','LotShape', 'PavedDrive', 'Alley', 'OverallCond','OverallQual')
cols_2 = ('Functional', 'CentralAir', 'BsmtExposure')
                            

# Encoding features :
'''from sklearn.preprocessing import LabelEncoder
for c in cols:
    labelencoder_X = LabelEncoder()
    all_data[c] = labelencoder_X.fit_transform(all_data[c])
'''
# but the issue here is the values in labelencoder will be assigned to codes randomly 
#i wanna give the code "EX" the largest value 5

# soultion: 
# replace codes with values manually insted of use labelencoder with desired order 
for c in cols_2:
    print (c)
    print (all_data[c].unique())                                         
for c in cols_1:
    all_data[c] = all_data[c].replace( {"None":0,"No" : 0,"N" : 0,
                                      "Po" : 1,"Grvl" : 1, "Unf" : 1,"MnWw":1, "P" : 1,"Sev" : 1,"IR3" : 1,
                                      "Fa" : 2,"RFn":2,"Pave" : 2,"LwQ": 2,"GdWo":2, "Y" : 2,"Mod" : 2,"IR2" : 2,
                                      "TA": 3, "Fin":3,"Rec" : 3,"MnPrv":3, "Gtl" : 3, "IR1" : 3,
                                      "Gd" : 4,"BLQ" : 4,"GdPrv":4 , "Reg" : 4,
                                      "Ex" : 5 ,"ALQ" : 5, 
                                      "GLQ" : 6 })

for c in cols_2:
    all_data[c] = all_data[c].replace( {"No" : 0,"N" : 0,"None" : 0,
                                       "Sal" : 1,"Mn" : 1, "Y" : 1,
                                      "Sev" : 2, "Av": 2,
                                      "Maj2" : 3,"Gd" : 3, 
                                       "Maj1" : 4,
                                       "Mod": 5,
                                      "Min2" : 6,
                                      "Min1" : 7,
                                      "Typ" : 8})
    
#then do the map manually also 
cols_map1 = ('OverallQual','OverallCond')
for c in cols_map1:
    all_data[c] = all_data[c].replace({1 : 1, 2 : 1, 3 : 1, # bad
                                       4 : 2, 5 : 2, 6 : 2, # average
                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                       })

cols_map2 = ('GarageFinish','PavedDrive')
for c in cols_map2:
    all_data[c] = all_data[c].replace({0 : 1, 1 : 1, # bad
                                       2 : 2, 3 : 2 # good
                                       })
cols_map3 = ('GarageCond','GarageQual','FireplaceQu','KitchenQual','HeatingQC',
             'BsmtCond','BsmtQual','ExterCond','ExterQual')
for c in cols_map3:
    all_data[c] = all_data[c].replace({1 : 1, # bad
                                       2 : 1, 3 : 1, # average
                                       4 : 2, 5 : 2 # good
                                       })
cols_map4 = ('BsmtFinType1','BsmtFinType2')
for c in cols_map4:
    all_data[c] = all_data[c].replace({1 : 1, # unfinished
                                       2 : 1, 3 : 1, # rec room
                                       4 : 2, 5 : 2, 6 : 2 # living quarters
                                        })
cols_map5 = ('Fence','BsmtExposure')
for c in cols_map5:
    all_data[c] = all_data[c].replace({0 : 1,# bad
                                       1 : 1, 2:1, # avarge
                                       3:2 ,4: 2 # good
                                       })
all_data['PoolQC'] = all_data['PoolQC'].replace({1 : 1, 2 : 1, # average
                                                 3 : 2, 4 : 2 # good
                                                 })    
all_data['Functional'] = all_data['Functional'].replace({1 : 1, 2 : 1, # bad
                                       3 : 2, 4 : 2, # major
                                       5 : 3, 6 : 3, 7 : 3, # minor
                                       8 : 4 # typical
                                       })
all_data['LotShape'] = all_data['LotShape'].replace({1 : 1, 2 : 1, 3:1, # bad
                                       4: 2 # good
                                       })
    
# 2* Combinations of existing features
# Total SF for house (incl. basement)
all_data["AllSF"] = all_data["GrLivArea"] + all_data["TotalBsmtSF"]
# Total SF for 1st + 2nd floors
all_data["AllFlrsSF"] = all_data["1stFlrSF"] + all_data["2ndFlrSF"]
# Total number of bathrooms
all_data["TotalBath"] = all_data["BsmtFullBath"] + (0.5 * all_data["BsmtHalfBath"]) + all_data["FullBath"] + (0.5 * all_data["HalfBath"])
# Total SF for porch
all_data["AllPorchSF"] = all_data["OpenPorchSF"] + all_data["EnclosedPorch"] + all_data["3SsnPorch"] + all_data["ScreenPorch"]
#commted for now
'''# Has masonry veneer or not
train["HasMasVnr"] = train.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, 
                                               "Stone" : 1, "None" : 0})
# House completed before sale or not
train["BoughtOffPlan"] = train.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, 
                                                      "Family" : 0, "Normal" : 0, "Partial" : 1}'''
 
# 3- Polynomials on the top 10 existing features
# nothing for now 


# 5- skewe numerical continouce numbers
#================================================
# Log transform of the skewed numerical features to lessen impact of outliers
# a skewness with an absolute value > 0.75 is considered at least moderately skewed
 
   
#get numeric features   
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
dataset_train = all_data[:ntrain]
#transform the numeric features using log(x + 1)
skewed = dataset_train[numeric_feats].apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[skewed > 0.75]
skewed = skewed.index
all_data[skewed]=np.log1p(all_data[skewed])
y_train= np.log(y_train)
  
# or 
'''
from scipy.special import boxcox1p
num_features = all_data.dtypes[all_data.dtypes != "object"].index
skew_features = all_data[num_features].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)
skew_amount = pd.DataFrame({'Skew Amount' :skew_features})
high_skew = skew_amount[abs(skew_amount) > 1]
high_skew_features = high_skew.index
lam = 0.20 #Lambda 
for feature in high_skew_features:
    all_data[feature] = boxcox1p(all_data[feature], lam)
print("Number of features transformed: ", high_skew_features.shape[0])
y_train= np.log1p(y_train)  '''
    

# Finally    
all_data = pd.get_dummies(all_data)
print(all_data.shape)

# 6- train data and test data after preprocessing
#================================================
dataset_train = all_data[:ntrain]
dataset_test = all_data[ntrain:]


#============================================================================================
# Modeling (prediction)
#============================================================================================

# 1- XGBoost
#================================================
# version 1 :
xgboost_model = xgb.XGBRegressor(
        colsample_bytree=0.2, #Subsample ratio of columns when constructing each tree
        gamma=0.0, #Minimum loss reduction required to make a further partition on a leaf node of the tree.
        learning_rate=0.05, #Boosting learning rate (xgb’s “eta”)
        max_depth=6, #Maximum tree depth for base learners
        min_child_weight=1.5, #Minimum sum of instance weight(hessian) needed in a child
        n_estimators=7200, #Number of boosted trees to fit
        reg_alpha=0.9, #L1 regularization term on weights
        reg_lambda=0.6, #L2 regularization term on weights
        subsample=0.2, #Subsample ratio of the training instance.
        seed=42, #Random number seed. (Deprecated, please use random_state)
        silent=1 #Whether to print messages while running boosting
        )

# version 2 : 
# Parameter tuning (Grid Search)
#=================
# i- Tune learning rate and number of estimators
param_test = {
 'learning_rate': [0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4 , 0.05 , 0.08],
  'n_estimators': range( 400, 1000,7200)
}
# ii- Tune max_depth and min_child_weight
param_test = {
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2)
}
# iii- Tune gamma
param_test = {
 'gamma':[i/10.0 for i in range(0,5)]
}
# iv - Tune subsample and colsample_bytree
param_test = {
 'subsample': [i/100.0 for i in range(10,110,10)],
 'colsample_bytree':[i/100.0 for i in range(10,110,10)]
}
# ivv- Tuning Regularization Parameters
param_test = {
  'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05,0.9]
}

Grid_Search_model = GridSearchCV(estimator=xgboost_model, scoring="neg_mean_squared_error", n_jobs=-1,
                        param_grid=param_test, verbose=0)
Grid_Search_model.fit(dataset_train, y_train)
# summarize results
print("Best: %f using %s" % (Grid_Search_model.best_score_, Grid_Search_model.best_params_))

# model after edit paramter
xgboost_model = xgb.XGBRegressor(
        colsample_bytree=0.4, #Subsample ratio of columns when constructing each tree
        gamma=0.0, #Minimum loss reduction required to make a further partition on a leaf node of the tree.
        learning_rate=0.05, #Boosting learning rate (xgb’s “eta”)
        max_depth=5, #Maximum tree depth for base learners
        min_child_weight=1, #Minimum sum of instance weight(hessian) needed in a child
        n_estimators=400, #Number of boosted trees to fit
        reg_alpha=0.001, #L1 regularization term on weights
        reg_lambda=0.6, #L2 regularization term on weights
        subsample=0.7, #Subsample ratio of the training instance.
        seed=42, #Random number seed. (Deprecated, please use random_state)
        silent=1 #Whether to print messages while running boosting
        )

# fit a model
#=================
xgboost_model.fit(dataset_train, y_train)

# test the model on train set 
#============================
# run prediction on training set to get an idea of how well it does
y_pred_train_xgboost = xgboost_model.predict(dataset_train)
print("Model score on training set: ", rmse( y_train, y_pred_train_xgboost))

# test the model on test set 
#============================
# make prediction on test set
y_pred_xgboost = xgboost_model.predict(dataset_test)


#Get Execl Sheet 
#===============
#submit this prediction and get the score
pred1 = pd.DataFrame({'Id': test_ID, 'SalePrice': np.exp(y_pred_xgboost)})
pred1.to_csv('HousePrices_1_1.csv', header=True, index=False)


# 2- Lasso
#================================================
# Lasso
#====== 
#found this best alpha through cross-validation
best_alpha = 0.00099
lasso_model = Lasso(alpha=best_alpha, max_iter=50000)


# LassoCV
#========
lasso_model = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], 
                max_iter = 50000, cv = 10)
lasso_model.fit(dataset_train, y_train)
alpha = lasso_model.alpha_
print("Best alpha :", alpha)
print("Try again for more precision with alphas centered around " + str(alpha))
lasso_model = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                          alpha * 1.4], 
                max_iter = 50000, cv = 10)

# Fit model 
#==========
lasso_model.fit(dataset_train, y_train)

# test the model on train set 
#============================
# run prediction on training set to get an idea of how well it does
y_pred_train_lasso = lasso_model.predict(dataset_train)
print("Model score on training set: ", rmse( y_train, y_pred_train_lasso))

# test the model on test set 
#============================
# make prediction on test set
y_pred_lasso = lasso_model.predict(dataset_test)


#Get Execl Sheet 
#===============
#submit this prediction and get the score
pred1 = pd.DataFrame({'Id': test_ID, 'SalePrice': np.exp(y_pred_lasso)})
pred1.to_csv('HousePrices_2_1.csv', header=True, index=False)



# 3- neural network model
#================================================
# Feature Scaling
sc = StandardScaler()
dataset_train_ann = sc.fit_transform(dataset_train)
dataset_test_ann = sc.transform(dataset_test)


# Initialising the ANN as seq of layers
ann_model = Sequential()
# Adding the input layer and the first hidden layer
# number of nodes in hidden layer = (number of input +number of output layer ) /2 (not offical role) 
# random initilize for wights with (uniform or normal )fun
# activation fun is rectifier (because it represent prob for output)
ann_model.add(Dense(output_dim = 16 ,init = 'normal', activation = 'relu' , input_dim = 271)) 
# Adding the second hidden layer
ann_model.add(Dense(output_dim = 8 ,init = 'normal', activation = 'relu')) 
# Adding the output layer
ann_model.add(Dense(output_dim = 1 ,init = 'normal')) 
# Compiling the ANN
#adam : alog for finding best wight
ann_model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# Fit model 
#==========
#batch_size  update after how many rows
ann_model.fit(dataset_train_ann, y_train, batch_size = 5, nb_epoch = 1000) # nb_epoch


# test the model on test set 
#============================
# make prediction on test set
y_pred_aan = ann_model.predict(dataset_test_ann)
y_pred_aan = np.exp(y_pred_aan)
y_pred_aan.resize((1459))

#Get Execl Sheet 
#===============
pred1 = pd.DataFrame(y_pred_aan, index=test_ID, columns=["SalePrice"]) 
pred1.to_csv('HousePrices_3_1.csv', header=True, index_label='Id') 


# 4- ElasticNetl
#================================================

# version 1 :
elasticNet_model = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 
                                    0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                          max_iter = 50000, cv = 10)
elasticNet_model .fit(dataset_train, y_train)
alpha = elasticNet_model .alpha_
ratio = elasticNet_model .l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )

# version 2 :
print("Try again for more precision with l1_ratio centered around " + str(ratio))
elasticNet_model  = ElasticNetCV(l1_ratio = [ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],
                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                          max_iter = 50000, cv = 10)
elasticNet_model.fit(dataset_train, y_train)
if (elasticNet_model.l1_ratio_ > 1):
    elasticNet_model.l1_ratio_ = 1    
alpha = elasticNet_model.alpha_
ratio = elasticNet_model.l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )

# version 3 :
print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) + 
      " and alpha centered around " + str(alpha))
elasticNet_model  = ElasticNetCV(l1_ratio = ratio,
                          alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, alpha * .9, 
                                    alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, 
                                    alpha * 1.35, alpha * 1.4], 
                          max_iter = 50000, cv = 10)

# Fit model 
#==========
elasticNet_model.fit(dataset_train, y_train)
if (elasticNet_model.l1_ratio_ > 1):
    elasticNet_model.l1_ratio_ = 1    
alpha = elasticNet_model.alpha_
ratio = elasticNet_model.l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )


# test the model on train set 
#============================
# run prediction on training set to get an idea of how well it does
y_pred_train_elasticNet = elasticNet_model.predict(dataset_train)
print("Model score on training set: ", rmse( y_train, y_pred_train_elasticNet))

# test the model on test set 
#============================
# make prediction on test set
y_pred_elasticNet = elasticNet_model.predict(dataset_test)


#Get Execl Sheet 
#===============
pred1 = pd.DataFrame({'Id': test_ID, 'SalePrice': np.exp(y_pred_elasticNet)})
pred1.to_csv('HousePrices_4_1.csv', header=True, index=False)



# 6- LightGBM :
#================================================

LightGBM_model= lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

# Fit model 
#==========
LightGBM_model.fit(dataset_train, y_train)


# test the model on train set 
#============================
# run prediction on training set to get an idea of how well it does
y_pred_train_LightGBM = LightGBM_model.predict(dataset_train)
print("Model score on training set: ", rmse( y_train, y_pred_train_LightGBM))

# test the model on test set 
#============================
# make prediction on test set
y_pred_LightGBM = LightGBM_model.predict(dataset_test)



#Get Execl Sheet 
#===============
pred1 = pd.DataFrame({'Id': test_ID, 'SalePrice': np.exp(y_pred_LightGBM)})
pred1.to_csv('HousePrices_6.csv', header=True, index=False)


# 7 - Ridge
#================================================

Ridge_model = Ridge(alpha=0.01)

# Fit model 
#==========
Ridge_model.fit(dataset_train, y_train)

# test the model on test set 
#============================
# make prediction on test set
y_pred_Ridge= Ridge_model.predict(dataset_test)


#Get Execl Sheet 
#===============
pred1 = pd.DataFrame({'Id': test_ID, 'SalePrice': np.exp(y_pred_Ridge)})
pred1.to_csv('HousePrices_7_1.csv', header=True, index=False)


# 5 - Ensembling Ridge, XGBoost , LightGBM and Lasso
#================================================
#Ensemble prediction:
ensemble=pd.DataFrame()
#ensemble =(y_pred_xgboost + y_pred_lasso + y_pred_LightGBM)/3
#ensemble =(y_pred_xgboost + y_pred_lasso + y_pred_LightGBM +y_pred_aan)/4
#ensemble = 0.15*y_pred_xgboost + 0.3*y_pred_lasso + 0.55*y_pred_LightGBM
ensemble = 0.25* y_pred_Ridge + 0.15*y_pred_xgboost + 0.15*y_pred_lasso + 0.45*y_pred_LightGBM
#ensemble = [np.median(i) for i in zip(y_pred_xgboost,y_pred_lasso,y_pred_aan)] 


#Get Execl Sheet 
#===============
pred1 = pd.DataFrame()
pred1['Id'] = test_ID
pred1['SalePrice'] = np.exp(ensemble)
pred1.to_csv('HousePrices_5_20.csv',index=False)

