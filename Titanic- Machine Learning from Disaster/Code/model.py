#==============================================
#Titanic -Classification problem
#==============================================


#References
#==============================================
# https://www.kaggle.com/ydalat/titanic-a-step-by-step-intro-to-machine-learning/notebook
# https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling

#================================================================
#Preprocessing Data 
#================================================================

# data analysis , wrangling and visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm 
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression , LogisticRegressionCV ,Perceptron ,SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC ,LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE , RFECV




# Outlier detection 
def detect_outliers(df,n,features):
    outlier_indices = [] 
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col],25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR
        # Determine a list of indices of outliers for feature col
        #Any data points outside 1.5 time the IQR (1.5 time IQR below Q1, or 1.5 time IQR above Q3), is considered an outlier.
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index       
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices) # if the same row comes twice or more in the list that means it contains more than one feature has outlier       
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return multiple_outliers

#==============================================
#Preprocessing Data 
#==============================================
  
# train data
dataset_train = pd.read_csv('train.csv')
# sumbit data 
dataset_test = pd.read_csv('test.csv')

#important variables
train_ID = dataset_train['PassengerId']
test_ID = dataset_test['PassengerId']
dataset_train.drop("PassengerId", axis = 1, inplace = True)
dataset_test.drop("PassengerId", axis = 1, inplace = True)



#============================================================================================
#General study (train data)
#============================================================================================
print (dataset_train.columns)
print (dataset_train.columns.values)
dataset_train.isnull().sum() # to get null coulmns
dataset_train.head() # five top records 
dataset_train.tail() # five button records 
dataset_train.info() # to get data type
dataset_train.describe() # count , mean  , min .... for data type : number (float , int ...)
dataset_train.describe(include=['O']) # count , mean  , min .... for data type : object


#============================================================================================
#univariable study (train data)
#============================================================================================
dataset_train.Survived.describe()
#histogram  plot
plt.hist(dataset_train["Survived"], color='blue')
plt.show()


#============================================================================================
# Multivariate study (train data)
#============================================================================================
# Method one :pick selected features
dataset_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
dataset_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
dataset_train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
dataset_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Survived vs age chart 
g = sns.FacetGrid(dataset_train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
# Survived and pclass vs age chart
grid = sns.FacetGrid(dataset_train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
# Embarked , survived , sex , pclass
grid = sns.FacetGrid(dataset_train, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
# fare , sex , survived , embarked
grid = sns.FacetGrid(dataset_train, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


#Method two : take all the features using
# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
#1- correlation matrix (heat map)
corrmat = dataset_train[["Survived","SibSp","Parch","Age","Fare","Pclass"]].corr()
sns.heatmap(corrmat,annot=True, fmt = ".2f", cmap = "coolwarm");


#Method three : scatterplot for  features 
'''sns.set()
cols = [ 'Age','Fare']
sns.pairplot(dataset_train[cols], size = 2.5) 
plt.show();


plt.figure()
sns.pairplot(data=dataset_train[cols], hue="Survived", dropna=True) 
'''


#============================================================================================
# Basic cleaning (train data , test data)
#============================================================================================

# 1- outliers (train data)
#================================================
#i - Univariate analysis (no need)

#ii- bivariate analysis for numerical (continuous , discreate)
#("Age","SibSp","Parch","Fare")

# method one :  manually using scatter plot
'''#scatter plot Survived/Age
data = pd.concat([dataset_train['Survived'],dataset_train['Age']],axis=1)
data.plot.scatter(x='Age' , y='Survived',ylim=(0,1));
# we can consder above > 80 is outliar so there is no outliar here

#bivariate analysis  Survived/Fare
data = pd.concat([dataset_train['Survived'],dataset_train['Fare']],axis=1)
data.plot.scatter(x='Fare' , y='Survived',ylim=(0,1));
#deleting points (outliars values)
dataset_train = dataset_train.drop(dataset_train[dataset_train['Fare'] >500].index)

# should do it for "Parch","Fare" as well '''


# method two : using another tech
Outliers_to_drop = detect_outliers(dataset_train,2,["Age","SibSp","Parch","Fare"])
dataset_train.loc[Outliers_to_drop] # Show the outliers rows (found 10 outliers)

#PassengerID 28, 89 and 342 passenger have an high Ticket Fare
#The seven others have very high values of SibSP.
# Drop outliers
dataset_train = dataset_train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)


# 2- missing data (train data , test data)
#================================================
#to apply process once at test and train 

y_train = dataset_train.Survived.values
all_data = [dataset_train, dataset_test]



# how many features are missing values 
# train datat set
dataset_train.columns[dataset_train.isnull().any()]
total = dataset_train.isnull().sum().sort_values(ascending=False)
percent = (dataset_train.isnull().sum()/dataset_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data

#test data set
dataset_test.columns[dataset_test.isnull().any()]
total = dataset_test.isnull().sum().sort_values(ascending=False)
percent = (dataset_test.isnull().sum()/dataset_test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data



#method one: drop columns with high null values from the train set only
'''
dataset_train = dataset_train.drop((missing_data[missing_data['Total'] > 0]).index,1)
#drop the row with null value 
dataset_train.isnull().sum().max() #just checking that there's no missing data missing...
'''

#method two : fill with values

# 'Name'
# create new features 
# 1- Title :(get title from name as a new coulmn)
for dataset in all_data:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False).fillna('None')
    #dataset['NameLength']= dataset['Name'].apply(len)


#replace many titles with a more common name or classify them as Rare
pd.crosstab(dataset_train['Title'], dataset_train['Sex'])
for dataset in all_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')  

# new groups
dataset_train['Title'].value_counts()   
dataset_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False)


#convert the categorical titles 
dataset_train = pd.get_dummies(dataset_train, columns = ["Title"])
dataset_test = pd.get_dummies(dataset_test, columns = ["Title"])
all_data = [dataset_train, dataset_test]

'''title_mapping = {"None":0 ,"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in all_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)'''
'''labelencoder_X = LabelEncoder()
for dataset in all_data:
    dataset['Title'] = labelencoder_X.fit_transform(dataset['Title'])'''

'''# 2 - Name length 
#create name length bands and determine correlations with Survived.
dataset_train['NameLengthBand'] = pd.cut(dataset_train['NameLength'], 5)
dataset_train[['NameLengthBand', 'Survived']].groupby(['NameLengthBand'], as_index=False).mean().sort_values(by='NameLengthBand', ascending=True)

#replace Name length with ordinals based on these bands.
for dataset in all_data:    
    dataset.loc[ dataset['NameLength'] <= 26, 'NameLength'] = 0
    dataset.loc[(dataset['NameLength'] > 26) & (dataset['NameLength'] <= 40), 'NameLength'] = 1
    dataset.loc[(dataset['NameLength'] > 40) & (dataset['NameLength'] <= 54), 'NameLength'] = 2
    dataset.loc[(dataset['NameLength'] > 54) & (dataset['NameLength'] <= 68), 'NameLength'] = 3
    dataset.loc[ dataset['NameLength'] > 68, 'NameLength'] =5


dataset_train[['NameLength', 'Survived']].groupby(['NameLength'], as_index=False).mean().sort_values(by='NameLength', ascending=True)
'''

#remove the NameLengthBand feature.
#dataset_train = dataset_train.drop(['NameLengthBand'], axis=1)
dataset_train = dataset_train.drop(['Name'], axis=1)
dataset_test = dataset_test.drop(['Name'], axis=1)
all_data = [dataset_train, dataset_test]

# 'sex'
#Converting a categorical feature
for dataset in all_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    #dataset['Sex'] = labelencoder_X.fit_transform(dataset['Sex'])

#After being a numerical
corrmat = dataset_train[["Survived","SibSp","Parch","Age","Fare","Pclass","Sex"]].corr()
sns.heatmap(corrmat,annot=True, fmt = ".2f", cmap = "coolwarm");

# 'Age'
#plot
grid = sns.FacetGrid(dataset_train, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

#skewness for continuous features
dataset_train['Age'].skew() #(no need)

'''
#iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age depond on sex and pclass.
avg_ages = np.zeros((2,3))
avg_ages
for dataset in all_data:
    for i in range(0, 2):
        for j in range(0, 3):
            #get vector of age that satisfy condition below (same class , same gender , not null age ) 
            guess_df = dataset[(dataset['Sex'] == i)&(dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            curr_avg = guess_df.median()

            # Convert random age float to nearest .5 age
            avg_ages[i,j] = int( curr_avg/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = avg_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)
'''
#using correlation map
# Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows
for dataset in all_data:
    index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)
    
    for i in index_NaN_age :
        age_med = dataset["Age"].median()
        age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
        if not np.isnan(age_pred) :
            dataset['Age'].iloc[i] = age_pred
        else :
            dataset['Age'].iloc[i] = age_med
            
     
'''#create Age bands and determine correlations with Survived.
dataset_train['AgeBand'] = pd.cut(dataset_train['Age'], 5)
dataset_train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

#replace Age with ordinals based on these bands.
for dataset in all_data:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] =5


dataset_train[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Age', ascending=True)

dataset_train = pd.get_dummies(dataset_train, columns = ["Age"])
dataset_test = pd.get_dummies(dataset_test, columns = ["Age"])
all_data = [dataset_train, dataset_test]

#remove the AgeBand feature.
dataset_train = dataset_train.drop(['AgeBand'], axis=1)
all_data = [dataset_train, dataset_test]'''
        


# 'SibSp' , 'Parch'
#create a new feature for FamilySize which combines Parch and SibSp
for dataset in all_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['Single'] = dataset['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    dataset['SmallF'] = dataset['FamilySize'].map(lambda s: 1 if  s == 2  else 0)
    dataset['MedF'] = dataset['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
    dataset['LargeF'] = dataset['FamilySize'].map(lambda s: 1 if s >= 5 else 0)

dataset_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
dataset_train[['Single', 'Survived']].groupby(['Single'], as_index=False).mean().sort_values(by='Survived', ascending=False)
dataset_train[['MedF', 'Survived']].groupby(['MedF'], as_index=False).mean().sort_values(by='Survived', ascending=False)
dataset_train[['SmallF', 'Survived']].groupby(['SmallF'], as_index=False).mean().sort_values(by='Survived', ascending=False)
dataset_train[['LargeF', 'Survived']].groupby(['LargeF'], as_index=False).mean().sort_values(by='Survived', ascending=False)

    
'''for dataset in all_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
dataset_train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by='Survived', ascending=False)'''


dataset_train = dataset_train.drop(['Parch', 'SibSp','FamilySize'], axis=1)
dataset_test = dataset_test.drop(['Parch', 'SibSp','FamilySize'], axis=1)
all_data = [dataset_train, dataset_test]

# 'Fare'
#fill  nulls with the most common occurance
for dataset in all_data:
    dataset['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)

#skewness for continuous features
dataset_train['Fare'].skew()

#histogram  plot
plt.hist(dataset_train['Fare'], color='blue')
plt.show()


# Apply log to Fare to reduce skewness distribution
for dataset in all_data:
    dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0) # we dont have nulls now

'''#create Fare bands and determine correlations with Survived.
dataset_train['FareBand'] = pd.qcut(dataset_train['Fare'], 2)
dataset_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

#replace fare with ordinals based on these bands.
for dataset in all_data:
    dataset.loc[ dataset['Fare'] <= 2.671, 'Fare'] = 0 #2.671
    dataset.loc[ dataset['Fare'] > 2.671, 'Fare'] = 1
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    
dataset_train = dataset_train.drop(['FareBand'], axis=1)
all_data = [dataset_train, dataset_test] '''


#'Cabin'
for dataset in all_data:
    dataset['cabin_letter']=dataset.Cabin.str.extract('^([A-Z]+)',expand=False).fillna('None')
    #dataset['cabin_letter']=dataset['cabin_letter'].map({'None':0,'D':1,'E':2,'B':3,'F':4,'C':5,'G':6,'A':7,'T':8}).astype(int) # we can use labelencoder
    #dataset['Has_Cabin'] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)


dataset_train[["cabin_letter","Survived"]].groupby(['cabin_letter'],as_index=False).mean().sort_values(by='Survived',ascending=False)
#dataset_train[["Has_Cabin","Survived"]].groupby(['Has_Cabin'],as_index=False).mean().sort_values(by='Survived',ascending=False)


#Converting a categorical feature
dataset_train = pd.get_dummies(dataset_train, columns = ['cabin_letter'])
dataset_test = pd.get_dummies(dataset_test, columns = ['cabin_letter'])
all_data = [dataset_train, dataset_test]

   

dataset_train = dataset_train.drop(['Cabin'], axis=1)
dataset_test = dataset_test.drop(['Cabin'], axis=1)
all_data = [dataset_train, dataset_test]


# 'Embarked'
#get most common (Embarked)
#fill  nulls with the most common occurance
for dataset in all_data:
    dataset['Embarked'].fillna(dataset['Embarked'].dropna().mode()[0], inplace=True)
  
dataset_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#Converting a categorical feature
dataset_train = pd.get_dummies(dataset_train, columns = ["Embarked"], prefix="Em")
dataset_test = pd.get_dummies(dataset_test, columns = ["Embarked"], prefix="Em")
all_data = [dataset_train, dataset_test]

'''for dataset in all_data:
    #dataset['Embarked']=dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    #dataset['Embarked']= labelencoder_X.fit_transform(dataset['Embarked'])'''

#'Pclass'
#Converting a categorical feature
dataset_train = pd.get_dummies(dataset_train, columns = ["Pclass"], prefix="PC")
dataset_test = pd.get_dummies(dataset_test, columns = ["Pclass"], prefix="PC")
all_data = [dataset_train, dataset_test]

'''
#create an artificial feature combining Pclass and Age.
for dataset in all_data:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass'''

#'Ticket'
dataset_train = dataset_train.drop(['Ticket'], axis=1)
dataset_test = dataset_test.drop(['Ticket'], axis=1)
all_data = [dataset_train, dataset_test]



#  train data and test data after preprocessing
#================================================
y_train = dataset_train["Survived"]
dataset_train = dataset_train.drop("Survived", axis=1)
#not exist in test set
dataset_train = dataset_train.drop("cabin_letter_T", axis=1)


#============================================================================================
# Modeling (prediction)
#============================================================================================

#  Cross-validation scores
#================================================

kfold = StratifiedKFold(n_splits=10)

random_state = 2
classifiers = []
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(SVC(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(GaussianNB())
classifiers.append(Perceptron(random_state=random_state))
classifiers.append(LinearSVC(random_state=random_state))
classifiers.append(SGDClassifier(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state = random_state))
classifiers.append(RandomForestClassifier(random_state = random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, dataset_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":['Logistic Regression', 'SVC', 'KNN', 'Gaussian',
    'Perceptron', 'linear SVC', 'SGD', 'Decision Tree', 'Random Forest','AdaBoost','ExtraTrees', 
    'GradientBoosting','MultipleLayerPerceptron','LinearDiscriminantAnalysis']})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


#   Hyperparameter tunning for some selected models 
#    (SVC, AdaBoost, RandomForest , ExtraTrees and the GradientBoosting )
#================================================

# Adaboost
#============
DTC_model = DecisionTreeClassifier()
adaDTC_model = AdaBoostClassifier(DTC_model, random_state=7)
ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}
gs_adaDTC_model = GridSearchCV(adaDTC_model,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gs_adaDTC_model.fit(dataset_train,y_train)

adaDTC_best= gs_adaDTC_model.best_estimator_
gs_adaDTC_model.best_score_


# ExtraTrees
#============ 
ExtC_model = ExtraTreesClassifier()
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

gs_ExtC_model = GridSearchCV(ExtC_model,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gs_ExtC_model.fit(dataset_train,y_train)

ExtC_best= gs_ExtC_model.best_estimator_
gs_ExtC_model.best_score_


# RFC
#============ 
RFC_model = RandomForestClassifier()
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

gs_RFC_model = GridSearchCV(RFC_model,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gs_RFC_model.fit(dataset_train,y_train)

random_forest_best=gs_RFC_model.best_estimator_
gs_RFC_model.best_score_


# Gradient boosting
#===================
GBC_model = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gs_GBC_model = GridSearchCV(GBC_model,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gs_GBC_model.fit(dataset_train,y_train)

GBC_best=gs_GBC_model.best_estimator_
gs_GBC_model.best_score_


# SVC classifier
#===================
SVMC_model = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gs_SVMC_model = GridSearchCV(SVMC_model,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gs_SVMC_model.fit(dataset_train,y_train)

SVMC_best= gs_SVMC_model.best_estimator_
gs_SVMC_model.best_score_




# Get the best trade-off between bias and variance:
#================================================

# 1- compare classifiers results between themselves and applied to the same test data
test_Survived_AdaDTC = pd.Series(adaDTC_best.predict(dataset_test), name="AdaDTC")
test_Survived_ExtC = pd.Series(ExtC_best.predict(dataset_test), name="ExtC")
test_Survived_GBC = pd.Series(GBC_best.predict(dataset_test), name="GBC")
test_Survived_SVMC = pd.Series(SVMC_best.predict(dataset_test), name="SVMC")
test_Survived_random_forest = pd.Series(random_forest_best.predict(dataset_test), name="random_forest")


# Concatenate all classifier results
ensemble_results = pd.concat([test_Survived_AdaDTC, test_Survived_ExtC, test_Survived_GBC,test_Survived_SVMC,test_Survived_random_forest],axis=1)
g= sns.heatmap(ensemble_results.corr(),annot=True)



# 2- ensemble classifiers together 

#As we can see in heat map, Adaboost has the lowest correlation to other predictors
#This indicates that it does not predict differently than the others when it comes to the test data.
#We will therefore 'ensemble' the remaining four predictors.

VotingPredictor = VotingClassifier(estimators=[ ('ExtC', ExtC_best), ('GBC',GBC_best), ('SVMC', SVMC_best), ('random_forest', random_forest_best)],voting='hard', n_jobs=4)
VotingPredictor = VotingPredictor.fit(dataset_train, y_train)
y_pred_voting = VotingPredictor.predict(dataset_test)


#Get Execl Sheet 
#===============
#submit this prediction and get the score
pred = pd.DataFrame({'PassengerId': test_ID, 'Survived': y_pred_voting})
pred.to_csv('Titanic_v.csv', header=True, index=False)



# Logistic Regression classifier
#===================

# feature selection 
rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')
rfecv.fit(dataset_train,y_train)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(dataset_train.columns[rfecv.support_]))

Selected_features = ['Sex', 'Title_Master', 'Title_Mr', 'Title_Rare', 'LargeF', 'cabin_letter_D', 'cabin_letter_E', 'cabin_letter_G', 'cabin_letter_None', 'Em_S', 'PC_3', 'T_A4', 'T_C',  'T_PP', 'T_SOC', 'T_SOPP', 'T_STONO', 'T_WC', 'T_WEP']
Selected_features = ['Sex', 'Title_Master', 'LargeF', 'cabin_letter_None']
X = dataset_train[Selected_features]

Logistic_model = LogisticRegression()
logistic_param_grid = {'C': [10**-i for i in range(-5, 5)], 'class_weight': [None, 'balanced'],
                        'penalty': ['l1','l2'],
                        'tol': [1e-10],
                        'solver': ['liblinear']}

gs_Logistic_model = GridSearchCV(Logistic_model,param_grid = logistic_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gs_Logistic_model.fit(X,y_train)

logistic_best= gs_Logistic_model.best_estimator_
gs_Logistic_model.best_score_

y_pred_logisic = pd.Series(logistic_best.predict(dataset_test[Selected_features]), name="logistic")



#Get Execl Sheet 
#===============
#submit this prediction and get the score
pred = pd.DataFrame({'PassengerId': test_ID, 'Survived': y_pred_logisic})
pred.to_csv('Titanic_l_10.csv', header=True, index=False)