import numpy as np
import scipy
import pandas as pd
import sklearn
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_validate 
from numpy import random, arange
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold







# load data
mdata= pd.read_csv('train_imperson_without4n7_balanced_data.csv') 
mdata.head()

# Investigating the presence of missing values
mdata.info()

# data is complete
m1= mdata.dropna()
m1.info()

# Investigating the presence of duplicate cases
# We should assume data are iid, hence keep duplicates
m2 = mdata.drop_duplicates() 
m2.info()
#duplicate rows for reference
mdata[mdata.duplicated() == True].head()


# Descriptive statistics on the training set
mdata.describe().T.describe()

# Separate X and Y
X, Y = mdata.loc[:, mdata.columns != '155'], mdata['155'] 
X.head()

# Statistical analysis of the features matrix
X.describe().T.describe() 
X.std().value_counts()




#Dataset 1: features with variance greater than zero
#Xselected with 0 var taken out
selector = VarianceThreshold()
selector.fit(X)
col = X.columns[selector.get_support()] 
Xselected= X.loc[:, col ]
Xselected.head()
Xselected.shape
Xselected.skew().median()



# Dataset 2: variance greater than zero and data scaled to a range [0, 1]
#normalised range 0 1
scaler1 = MinMaxScaler().fit(Xselected) 
scaled0_1=scaler1.transform(Xselected)

# dataframe normalised range 0_1
NX3df = pd.DataFrame(scaled0_1, columns= Xselected.columns) 
NX3df.head()





#Dataset 3: variance greater than zero and power transform
#Power transform
pt = PowerTransformer(method = 'yeo-johnson').fit(Xselected) 
poweredX2 = pt.transform(Xselected)



#dataframe power gaussian
PX2df = pd.DataFrame(poweredX2, columns= Xselected.columns) 
PX2df.head()

PX2df.skew().median()


#Normalizer transforms to norm 1
scaler = Normalizer().fit(Xselected) 
normalizedX2 = scaler.transform(Xselected)

#dataframe with norm 1
NX2df = pd.DataFrame(normalizedX2, columns = Xselected.columns) 
NX2df.head()


# Feature selection Part 1: Tree based feature importance

%matplotlib inline

mdata= pd.read_csv('train_imperson_without4n7_balanced_data.csv')
X, Y = mdata.loc[:, mdata.columns != '155'], mdata['155']

#eliminate features with 0 standard deviation
X1_sigma = X.loc[:, (X.std()>0)]

%%time

forest = RandomForestClassifier()
forest.fit(X1_sigma, Y)
rf_importance = forest.feature_importances_ 
rf_importance

# Rank the features by importance

rf_ranked = sorted((rating, idx) for (idx, rating) in enumerate(r f_importance))

# look at the 20 most important features • rf_ranked[-20:]
importances = [] 
for _ in range(10):
    forest.fit(X1_sigma, Y)

importances.append(forest.feature_importances_) 
df = pd.DataFrame(importances)

df.plot(legend=False)
fig, ax = plt.subplots(figsize=(20,10))
ax.set_title("Variation of RF column importance over 10 runs")
ax.set_xlabel("Column Number")
ax.set_ylabel("Importance")
df.boxplot(grid=False, whis=1000., ax=ax)


# Using xgboost to get the feature importance

%%time

xgb = XGBClassifier()
xgb.fit(X1_sigma, Y)
xg_importance = xgb.feature_importances_
xg_importance

xg_ranked = sorted((rating, idx) for (idx, rating) in enumerate(i mportance))

# look at the 20 most important features • xg_ranked[-20:]

np.array([x[0] for x in reversed(xg_ranked)]).cumsum()


# Feature Importance with Catboost
from catboost import CatBoostClassifier
cb = CatBoostClassifier()

%%time
cb.fit(X1_sigma, Y, verbose=100)
 

cb_importance = cb.get_feature_importance() • cb_importance


cb_ranked = sorted((rating, idx) for (idx, rating) in enumerate(c b_importance))

# look at the 20 most important features • cb_ranked[-20:]

importances = [] 
for _ in range(10):
    cb = CatBoostClassifier()
    cb.fit(X1_sigma, Y, verbose=500) 
    importances.append(cb.get_feature_importance())

df = pd.DataFrame(importances) 
df.plot(legend=False)


np.array([x[0] for x in reversed(cb_ranked)]).cumsum()

def top20(ranked):
    return [x[1] for x in ranked[-20:]]

print("RF:", top20(rf_ranked))
print("xgboost:", top20(xg_ranked))
print("catboost:", top20(cb_ranked))
print("Intersection:", set(top20(rf_ranked)).intersection(top20(xg_ranked)).intersection(top20(cb_ranked)))


# Evaluating the models on the subset of features

X_top_features = X1_sigma.iloc[:, list(selected_features)]

# shuffle the rows

indexes = arange(Y.shape[0])
numpy.random.shuffle(indexes)
X_cv = X_top_features.iloc[indexes,:]
Y_cv = Y.iloc[indexes]
rf = RandomForestClassifier()
rf_cv = cross_validate(rf, X_cv, Y_cv, cv=5, scoring="accuracy")
rf_cv

gb = XGBClassifier(verbosity=0)
gb_cv = cross_validate(gb, X_cv, Y_cv, cv=5, scoring="accuracy")
gb_cv




# Feature Selection Part 2: Recursive Feature Elimination

%matplotlib inline
import numpy as np
import pandas as pd
mdata= pd.read_csv('train_imperson_without4n7_balanced_data.csv')
X, Y = mdata.loc[:, mdata.columns != '155'], mdata['155']

#eliminate features with 0 standard deviation
X1_sigma = X.loc[:, (X.std()>0)]
initial_selection = {0, 32, 2, 34, 72, 11, 48, 49, 18, 20, 22, 57, 29, 31}
X_top_features = X1_sigma.iloc[:, list(initial_selection)]

# shuffle the rows
indexes = arange(Y.shape[0])
random.shuffle(indexes)
X_cv = X_top_features.iloc[indexes,:]
Y_cv = Y.iloc[indexes]




%%time
cb = CatBoostClassifier(verbose=False, early_stopping_rounds=50)
rfecv = RFECV(cb, scoring="accuracy", n_jobs=-1, verbose=1)
rfecv.fit(X_cv, Y_cv)
rfecv.n_features_
rfecv.grid_scores_
rfecv.support_

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores)


# Validating the 6 selected features using 5-fold CV

X_top_6 = X_cv[X_cv.columns[rfecv.support_]]
cb = CatBoostClassifier(verbose=500, early_stopping_rounds=50)


%time 
cb_cv = cross_validate(cb, X_top_6, Y_cv, cv=5, scoring="ac curacy")
cb_cv

# get the top column names/numbers from the original dataframe:

list(X_top_6.columns)






# Feature selection using Chi-squared

mdata_test= pd.read_csv('test_imperson_without4n7_balanced_data.c sv')
mdata_train= pd.read_csv('train_imperson_without4n7_balanced_data.csv')
Xtrain, Ytrain = mdata_train.loc[:, mdata_train.columns != '155'], mdata_train['155']
Xtest, Ytest = mdata_test.loc[:, mdata_test.columns != '155'], mdata_test['155']
X1_sigma = Xtrain.loc[:, (Xtrain.std()>0)]
X2_sigma = Xtest.loc[:, (Xtest.std()>0)]

params = {'loss_function':'Logloss','eval_metric':'AUC','verbose':54}
cb = CatBoostClassifier(**params)
cb.fit(Xtrain, Ytrain,eval_set=(Xtest,Ytest),use_best_model = True,plot=True)
feat_important = [t for t in zip(features,cb.get_feature_importance())]
feat_important_df = pd.DataFrame(feat_important,columns=['feature ', 'VarImp'])
feat_important_df = feat_important_df.sort_values('VarImp', ascending=False)
feat_important_df[feat_important_df['VarImp']>0]
 


len(feat_important_df[feat_important_df['VarImp']>0])

selected_features=[]
for i in feat_important_df[feat_important_df['VarImp']>0]['feature']:
    selected_features.append(int(i))
    
selected_features = set(selected_features)
print(selected_features)


X_top_features = Xtrain.iloc[:,list(selected_features)]
indexes = arange(Ytrain.shape[0])
np.random.shuffle(indexes)



# chi-squared (chi2) statistical test for non-negative features to select 10 of the best features

data = pd.read_csv('test_imperson_without4n7_balanced_data.csv')
Xtrain, Ytrain = mdata_train.loc[:, mdata_train.columns != '155'], mdata_train['155']

# apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(Xtrain,Ytrain)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Xtrain.columns)

# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['feature','Score'] #naming the dataframe columns

print(featureScores.nlargest(10,'Score')) #print 10 best features



selected_features=[]
for i in featureScores.nlargest(10,'Score')['feature']:
    selected_features.append(int(i)-3)

selected_features = set(selected_features)
X_top_features = Xtrain.iloc[:, list(selected_features)]
indexes = arange(Ytrain.shape[0])
numpy.random.shuffle(indexes)
X_cv = X_top_features.iloc[indexes,:]




Y_cv = Ytrain.iloc[indexes]
cb = CatBoostClassifier(verbose=False, early_stopping_rounds=50)
cb_cv = cross_validate(cb, X_cv, Y_cv, cv=5, scoring="accuracy")
cb_cv

gb = XGBClassifier(verbosity=0)
gb_cv = cross_validate(gb, X_cv, Y_cv, cv=5, scoring="accuracy")
gb_cv

Xtrain, Ytrain = Xselected, Y

# apply SelectKBest class to extract top 10 best features bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(Xtrain,Ytrain)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Xtrain.columns)

#concat two dataframes for better visualization featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#naming the dataframe columns
featureScores.columns = ['feature','Score']

#print 10 best features
print(featureScores.nlargest(10,'Score')) 

selected_features = set((25,16,22,13,77,15,2,3,23,34))


X_top_features = Xselected.iloc[:, list(selected_features)] 
indexes = numpy.arange(Ytrain.shape[0]) 
numpy.random.shuffle(indexes)
X_cv = X_top_features.iloc[indexes,:]
Y_cv = Ytrain.iloc[indexes]
indexes = arange(Ytrain.shape[0]) 
numpy.random.shuffle(indexes)
X_cv = X_top_features.iloc[indexes,:] 
Y_cv = Ytrain.iloc[indexes]
cb = CatBoostClassifier(verbose=False, early_stopping_rounds=50) 
cb_cv = cross_validate(cb, X_cv, Y_cv, cv=5, scoring="accuracy") 
cb_cv

gb = XGBClassifier(verbosity=0)
gb_cv = cross_validate(gb, X_cv, Y_cv, cv=5, scoring="accuracy")
gb_cv

Xtrain, Ytrain = NX2df, Y

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10) 
fit = bestfeatures.fit(Xtrain,Ytrain)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Xtrain.columns)


#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1) #naming the dataframe columns
featureScores.columns = ['feature','Score'] #print 10 best features
print(featureScores.nlargest(10,'Score'))
selected_features = set((25,16,22,13,77,15,2,3,23,27)) 
X_top_features = NX2df.iloc[:, list(selected_features)] 
indexes = numpy.arange(Ytrain.shape[0]) 
numpy.random.shuffle(indexes)
X_cv = X_top_features.iloc[indexes,:] 
Y_cv = Ytrain.iloc[indexes]
cb = CatBoostClassifier(verbose=False, early_stopping_rounds=50) 
cb_cv = cross_validate(cb, X_cv, Y_cv, cv=5, scoring="accuracy") 
cb_cv

gb = XGBClassifier(verbosity=0)
gb_cv = cross_validate(gb, X_cv, Y_cv, cv=5, scoring="accuracy")
gb_cv

Xtrain, Ytrain = NX3df, Y


#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10) 
fit = bestfeatures.fit(Xtrain,Ytrain)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Xtrain.columns)


#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1) #naming the dataframe columns
featureScores.columns = ['feature','Score'] #print 10 best features
print(featureScores.nlargest(10,'Score'))
selected_features = set((25,16,22,13,77,15,2,3,23,34)) 
X_top_features = NX3df.iloc[:, list(selected_features)] 
indexes = numpy.arange(Ytrain.shape[0]) 
numpy.random.shuffle(indexes)
X_cv = X_top_features.iloc[indexes,:] 
Y_cv = Ytrain.iloc[indexes]
cb = CatBoostClassifier(verbose=False, early_stopping_rounds=50) 
cb_cv = cross_validate(cb, X_cv, Y_cv, cv=5, scoring="accuracy") 
cb_cv

gb = XGBClassifier(verbosity=0)
gb_cv = cross_validate(gb, X_cv, Y_cv, cv=5, scoring="accuracy")
gb_cv

from sklearn import feature_selection
Xtrain, Ytrain = PX2df, Y


#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(k=10)
fit = bestfeatures.fit(Xtrain,Ytrain)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Xtrain.columns)


#concat two dataframes for better visualization featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#naming the dataframe columns
featureScores.columns = ['feature','Score'] #print 10 best features
print(featureScores.nlargest(10,'Score')) 
selected_features = set((22,32,25,13,16,15,29,28,23,0))
X_top_features = PX2df.iloc[:, list(selected_features)] 
indexes = numpy.arange(Ytrain.shape[0]) 
numpy.random.shuffle(indexes)
X_cv = X_top_features.iloc[indexes,:]
Y_cv = Ytrain.iloc[indexes]
cb = CatBoostClassifier(verbose=False, early_stopping_rounds=50) 
cb_cv = cross_validate(cb, X_cv, Y_cv, cv=5, scoring="accuracy") 
cb_cv
gb = XGBClassifier(verbosity=0)
gb_cv = cross_validate(gb, X_cv, Y_cv, cv=5, scoring="accuracy") 
gb_cv

from sklearn.ensemble import RandomForestClassifier
from sklearn import feature_selection


#apply SelectKBest class to extract top 10 best features bestfeatures = SelectKBest(k=10)
fit = bestfeatures.fit(Xtest,Ytest)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Xtest.columns)


#concat two dataframes for better visualization featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#naming the dataframe columns
featureScores.columns = ['feature','Score'] #print 10 best features
print(featureScores.nlargest(10,'Score'))
f = msb / msw
selected_features = set((66,72,64,76,35,47,48,44,78,5)) 
X_top_features = Xtest.iloc[:, list(selected_features)] 
indexes = numpy.arange(Ytest.shape[0]) 
numpy.random.shuffle(indexes)
Xtest_cv = X_top_features.iloc[indexes,:] 
Ytest_cv = Ytest.iloc[indexes]
Xtrain, Ytrain = Xselected, Y


#apply SelectKBest class to extract top 10 best features bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(Xtrain,Ytrain)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Xtrain.columns)


#concat two dataframes for better visualization featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#naming the dataframe columns
featureScores.columns = ['feature','Score'] #print 10 best features
print(featureScores.nlargest(10,'Score'))

selected_features = set((25,16,22,13,77,15,2,3,23,34)) 

import numpy
X_top_features = Xselected.iloc[:, list(selected_features)] 
indexes = numpy.arange(Ytrain.shape[0]) 
numpy.random.shuffle(indexes)
X_cv = X_top_features.iloc[indexes,:] 
Y_cv = Ytrain.iloc[indexes]

from sklearn.ensemble import RandomForestClassifier #Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)


#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_cv,Y_cv)


# prediction on test set
y_pred=clf.predict(Xtest_cv)


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics


# Model Accuracy, how often is the classifier correct? print("Accuracy:",metrics.accuracy_score(Ytest_cv, y_pred))
Xtrain, Ytrain = NX2df, Y


#apply SelectKBest class to extract top 10 best features bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(Xtrain,Ytrain)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Xtrain.columns)


#concat two dataframes for better visualization featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#naming the dataframe columns
featureScores.columns = ['feature','Score'] #print 10 best features
print(featureScores.nlargest(10,'Score')) 
selected_features = set((25,16,22,13,77,15,2,3,23,27))
X_top_features = NX2df.iloc[:, list(selected_features)]

indexes = numpy.arange(Ytrain.shape[0]) 
numpy.random.shuffle(indexes)
X_cv = X_top_features.iloc[indexes,:] 
Y_cv = Ytrain.iloc[indexes]


#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)


#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_cv,Y_cv)


# prediction on test set
y_pred=clf.predict(Xtest_cv)


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics


# Model Accuracy, how often is the classifier correct? print("Accuracy:",metrics.accuracy_score(Ytest_cv, y_pred))
Xtrain, Ytrain = NX3df, Y


#apply SelectKBest class to extract top 10 best features bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(Xtrain,Ytrain)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Xtrain.columns)


#concat two dataframes for better visualization featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#naming the dataframe columns
featureScores.columns = ['feature','Score'] #print 10 best features
print(featureScores.nlargest(10,'Score'))
selected_features = set((25,16,22,13,77,15,2,3,23,34)) 
X_top_features = NX3df.iloc[:, list(selected_features)] 
indexes = numpy.arange(Ytrain.shape[0]) 
numpy.random.shuffle(indexes)
X_cv = X_top_features.iloc[indexes,:] 
Y_cv = Ytrain.iloc[indexes]


#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)


#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_cv,Y_cv)

# prediction on test set
y_pred=clf.predict(Xtest_cv)


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics


# Model Accuracy, how often is the classifier correct? print("Accuracy:",metrics.accuracy_score(Ytest_cv, y_pred))
Xtrain, Ytrain = PX2df, Y


#apply SelectKBest class to extract top 10 best features bestfeatures = SelectKBest(k=10)
fit = bestfeatures.fit(Xtrain,Ytrain)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Xtrain.columns)


#concat two dataframes for better visualization featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#naming the dataframe columns
featureScores.columns = ['feature','Score'] #print 10 best features
print(featureScores.nlargest(10,'Score'))

import numpy
selected_features = set((22,32,25,13,16,15,29,28,23,0)) 
X_top_features = PX2df.iloc[:, list(selected_features)] 
indexes = numpy.arange(Ytrain.shape[0]) 
numpy.random.shuffle(indexes)
X_cv = X_top_features.iloc[indexes,:]
Y_cv = Ytrain.iloc[indexes]


#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)


#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_cv,Y_cv)


# prediction on test set
y_pred=clf.predict(Xtest_cv)


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics


# Model Accuracy, how often is the classifier correct? print("Accuracy:",metrics.accuracy_score(Ytest_cv, y_pred))
from sklearn import feature_selection


#apply SelectKBest class to extract top 10 best features bestfeatures = SelectKBest(k=10)
fit = bestfeatures.fit(Xtest,Ytest)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Xtest.columns)


#concat two dataframes for better visualization featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#naming the dataframe columns
featureScores.columns = ['feature','Score'] #print 10 best features
print(featureScores.nlargest(10,'Score'))
f = msb / msw
selected_features = set((66,72,64,76,35,47,48,44,78,5)) 
X_top_features = Xtest.iloc[:, list(selected_features)] 
indexes = numpy.arange(Ytest.shape[0]) 
numpy.random.shuffle(indexes)
Xtest_cv = X_top_features.iloc[indexes,:] 
Ytest_cv = Ytest.iloc[indexes]

from sklearn.naive_bayes import GaussianNB
Xtrain, Ytrain = Xselected, Y


#apply SelectKBest class to extract top 10 best features bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(Xtrain,Ytrain)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Xtrain.columns)


#concat two dataframes for better visualization featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#naming the dataframe columns
featureScores.columns = ['feature','Score'] #print 10 best features
print(featureScores.nlargest(10,'Score'))
selected_features = set((25,16,22,13,77,15,2,3,23,34)) 

import numpy
X_top_features = Xselected.iloc[:, list(selected_features)] 
indexes = numpy.arange(Ytrain.shape[0]) 
numpy.random.shuffle(indexes)
X_cv = X_top_features.iloc[indexes,:] 
Y_cv = Ytrain.iloc[indexes]

import time

%time
model = GaussianNB() 
model.fit(X_cv, Y_cv)
GaussianNB(priors=None, var_smoothing=1e-09)
model.score(X_cv,Y_cv)
model.score(Xtest_cv,Ytest_cv)
%time

Xtrain, Ytrain = NX2df, Y


#apply SelectKBest class to extract top 10 best features bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(Xtrain,Ytrain)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Xtrain.columns)


#concat two dataframes for better visualization featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#naming the dataframe columns
featureScores.columns = ['feature','Score'] #print 10 best features

print(featureScores.nlargest(10,'Score'))
selected_features = set((25,16,22,13,77,15,2,3,23,27)) 
X_top_features = NX2df.iloc[:, list(selected_features)] 
indexes = numpy.arange(Ytrain.shape[0]) 
numpy.random.shuffle(indexes)
X_cv = X_top_features.iloc[indexes,:] 
Y_cv = Ytrain.iloc[indexes]

%time
model = GaussianNB() 
model.fit(X_cv, Y_cv)
model.predict(X_cv) 
model.score(X_cv,Y_cv) 
model.score(Xtest_cv,Ytest_cv)
%time
Xtrain, Ytrain = NX3df, Y


#apply SelectKBest class to extract top 10 best features bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(Xtrain,Ytrain)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Xtrain.columns)


#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1) #naming the dataframe columns
featureScores.columns = ['feature','Score'] #print 10 best features
print(featureScores.nlargest(10,'Score'))
selected_features = set((25,16,22,13,77,15,2,3,23,34)) 
X_top_features = NX3df.iloc[:, list(selected_features)] 
indexes = numpy.arange(Ytrain.shape[0]) 
numpy.random.shuffle(indexes)
X_cv = X_top_features.iloc[indexes,:] 
Y_cv = Ytrain.iloc[indexes]
model = GaussianNB()
model.fit(X_cv, Y_cv)
model.predict(X_cv) 
model.score(X_cv,Y_cv) 
model.score(Xtest_cv,Ytest_cv)
Xtrain, Ytrain = PX2df, Y


#apply SelectKBest class to extract top 10 best features bestfeatures = SelectKBest(k=10)
fit = bestfeatures.fit(Xtrain,Ytrain)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Xtrain.columns)


#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1) #naming the dataframe columns
featureScores.columns = ['feature','Score'] #print 10 best features
print(featureScores.nlargest(10,'Score'))

import numpy
selected_features = set((22,32,25,13,16,15,29,28,23,0)) 
X_top_features = PX2df.iloc[:, list(selected_features)] 
indexes = numpy.arange(Ytrain.shape[0]) 
numpy.random.shuffle(indexes)
X_cv = X_top_features.iloc[indexes,:]
Y_cv = Ytrain.iloc[indexes]
model = GaussianNB() 
model.fit(X_cv, Y_cv)
model.predict(X_cv) 
model.score(X_cv,Y_cv) 
model.score(Xtest_cv,Ytest_cv)



#load data
traindata= pd.read_csv(r'C:\Users\Student\Git\AML_project\datasets\train_im person_without4n7_balanced_data.csv')
testdata= pd.read_csv(r'C:\Users\Student\Git\AML_project\datasets\test_impe rson_without4n7_balanced_data.csv')


# Separate X and Y
X_train, Y_train = traindata.loc[:, traindata.columns != '155'], traindata[ '155']
X_test, Y_test = testdata.loc[:, testdata.columns != '155'], testdata['155' ]

from sklearn.model_selection import GridSearchCV
pipeline1 = Pipeline([('zero variance', VarianceThreshold()), ('top features', SelectKBest(f_classif)), ('ada',AdaBoostClassifier())])

grid=GridSearchCV(cv=10, estimator=pipeline1,param_grid={'ada__n_estimators': [10,50,100,150], 'ada__learning_rate':[0.001,0.1,0.01],
'top features__k':[15,20,30,35]}, scoring = 'accuracy',
n_jobs=-1)

grid.fit(X_train, Y_train)
print(grid.best_params_)
print(grid.best_score_)

from sklearn.model_selection import GridSearchCV

pipeline2= Pipeline([('zero variance', VarianceThreshold()), ('norm 1', Normalizer()),
('top features', SelectKBest(chi2)), ('model',LogisticRegression())])
grid=GridSearchCV(cv=10, estimator=pipeline2,param_grid={'top features__k':[15,20,30,35], 'model__C': [0.01, 0.1, 1, 10, 100, 1000]},
scoring = 'accuracy', n_jobs=-1)
grid.fit(X_train, Y_train)


# sorted(pipeline2.get_params().keys()) #list of the parameters you can tun e
print(grid.best_params_)
print(grid.best_score_)

from sklearn.model_selection import GridSearchCV

pipeline2a= Pipeline([('zero variance', VarianceThreshold()), ('minmax', MinMaxScaler()),
('top features', SelectKBest(chi2)), ('model',LogisticRegression())])
grid=GridSearchCV(cv=10, estimator=pipeline2a,param_grid={'top features__k':[15,20,30,35],'model__C': [0.01, 0.1, 1, 10, 100, 1000]},
scoring = 'accuracy', n_jobs=-1)
grid.fit(X_train, Y_train)


#sorted(pipeline1.get_params().keys()) #list of the parameters you can tune print(grid.best_params_)
print(grid.best_score_)
pipeline3 = Pipeline([('zero variance', VarianceThreshold()), ('scale 0_1', MinMaxScaler()),
('top features', SelectKBest(f_classif)), ('ada',AdaBoostClassifier())])
grid=GridSearchCV(cv=10, estimator=pipeline3,param_grid={'ada__n_estimators': [10,50,100,150], 'ada__learning_rate':[0.001,0.1,0.01],
'top features__k':[15,20,30,35]}, scoring = 'accuracy',
n_jobs=-1)
grid.fit(X_train, Y_train)


#sorted(pipeline1.get_params().keys()) #list of the parameters you can tune print(grid.best_params_)
print(grid.best_score_)

pipeline1 = Pipeline([
('zero variance', VarianceThreshold()),
('top20 features', SelectKBest(f_classif, k=35)), ('ada',AdaBoostClassifier(n_estimators=100,learning_rate=0.01))])

pipeline1.fit(X_train,Y_train)
Y_predict=pipeline1.predict(X_test) 
accuracy_score(Y_test, Y_predict)*100 
pipeline2= Pipeline([
('zero variance', VarianceThreshold()),
('norm 1', Normalizer()),
('top20 features', SelectKBest(chi2, k=20)), ('model',LogisticRegression(C=1000,solver='newton-cg'))])

pipeline2.fit(X_train,Y_train)

Y_predict=pipeline2.predict(X_test) 
accuracy_score(Y_test, Y_predict)*100 
pipeline3 = Pipeline([
('zero variance', VarianceThreshold()),
('scale 0_1', MinMaxScaler()),
('top20 features', SelectKBest(f_classif, k=35)), ('ada',AdaBoostClassifier(n_estimators=100,learning_rate=0.01))])

pipeline3.fit(X_train,Y_train)

Y_predict=pipeline3.predict(X_test) 
accuracy_score(Y_test, Y_predict)*100 
pipeline5 = Pipeline([('zero variance', VarianceThreshold()), ('scale 0_1', MinMaxScaler()),
('top20 features', SelectKBest(chi2, k=20)), ('ada',AdaBoostClassifier())])

grid=GridSearchCV(cv=10,estimator=pipeline5, param_grid={'ada__n_estimators': [10,50,100,150],
'ada__learning_rate':[0.001,0.1,0.01]},
scoring = 'accuracy', n_jobs=-1)

grid.fit(X_train, Y_train)

print(grid.best_params_) 
print(grid.best_score_) 
pipeline5 = Pipeline([
('zero variance', VarianceThreshold()), ('scale 0_1', MinMaxScaler()),
('top20 features', SelectKBest(chi2, k=20)), ('ada',AdaBoostClassifier())])

grid=GridSearchCV(cv=10, estimator=pipeline5,param_grid={'ada__n_estimators': [10,50,100,150], 'ada__learning_rate':[0.001,0.1,0.01]}, scoring = 'accuracy',
n_jobs=-1)

grid.fit(X_train, Y_train)

print(grid.best_params_) 
print(grid.best_score_) 
pipeline1 = Pipeline([('zero variance', VarianceThreshold()),
('top20 features', SelectKBest(f_classif, k=20)), ('ada',AdaBoostClassifier(n_estimators=150,learning_rate=0.1))])

pipeline1.fit(X_train,Y_train)

Y_predict=pipeline1.predict(X_test) 
accuracy_score(Y_test, Y_predict)*100 
pipeline2 = Pipeline([
('zero variance', VarianceThreshold()),
('scale 0_1', MinMaxScaler()),
('top20 features', SelectKBest(f_classif, k=20)), ('ada',AdaBoostClassifier(n_estimators=150,learning_rate=0.1))])

pipeline2.fit(X_train,Y_train)

Y_predict=pipeline2.predict(X_test) 
accuracy_score(Y_test, Y_predict)*100 
pipeline3= Pipeline([
('zero variance', VarianceThreshold()),
('scale 0_1', MinMaxScaler()),
('top20 features', SelectKBest(chi2, k=20)), ('model',LogisticRegression(C=1000,solver='newton-cg'))])

pipeline3.fit(X_train,Y_train) 
Y_predict=pipeline3.predict(X_test)
accuracy_score(Y_test, Y_predict)*100

# Evaluation
# Pipeline 1 - Accuracy
pipeline1 = Pipeline([
('zero variance', VarianceThreshold()),
('top20 features', SelectKBest(f_classif, k=20)), ('ada',AdaBoostClassifier(n_estimators=150,learning_rate=0.1))])
pipeline1.fit(X_train,Y_train) 
Y_predict1=pipeline1.predict(X_test)
accuracy_score(Y_test, Y_predict1)*100 
matrix = confusion_matrix(Y_test, Y_predict1)
print(matrix)
p1_TP = matrix[0][0]
p1_TN = matrix[1][1]
p1_FP = matrix[1][0]
p1_FN = matrix[0][1]
print('')
print('True Positive = %.0f' % p1_TP) 
print('True Negative = %.0f' % p1_TN)
print('')
print('False Positive = %.0f' % p1_FP)
print('False Negative = %.0f' % p1_FN)


#Pipeline 1 - Evaluation Metrics
p1_Acc = (p1_TP + p1_TN) / (p1_TP + p1_TN + p1_FP + p1_FN) 
print('Acc = %.4f' % p1_Acc)
p1_DR = p1_TP / (p1_TP + p1_FN) 
print('DR(Recall) = %.4f' % p1_DR)
p1_Prec = p1_TP / (p1_TP + p1_FP) 
print('Precision = %.4f' % p1_Prec)
p1_FAR = p1_FP / (p1_TN + p1_FP) 
print('FAR = %.4f' % p1_FAR)
p1_FNR = p1_FN / (p1_FN + p1_TP) 
print('FNR = %.4f' % p1_FNR)
p1_F1 = (2 * p1_TP) / (2*p1_TP + p1_FP + p1_FN) 
print('F1 = %.4f' % p1_F1)

a = ((p1_TP * p1_TN) - (p1_FP * p1_FN))
b = (p1_TP + p1_FP)*(p1_TP + p1_FN)*(p1_TN + p1_FP)*(p1_TN + p1_FN)
p1_MCC = a/b 
print('MCC = %.4f' % p1_MCC)



#Pipeline 1 - ROC Curve
probs = pipeline1.predict_proba(X_test) 
probs = probs[:, 1]
auc = roc_auc_score(Y_test, probs) 
print('AUC: %.3f' % auc)
fpr, tpr, thresholds = roc_curve(Y_test, probs) 
pyplot.plot([0,1],[0,1], linestyle = '--') 
pyplot.plot(fpr,tpr,marker='.')
pyplot.show()


#Pipeline 2
pipeline2 = Pipeline([
('zero variance', VarianceThreshold()),
('scale 0_1', MinMaxScaler()),
('top20 features', SelectKBest(f_classif, k=20)), ('ada',AdaBoostClassifier(n_estimators=150,learning_rate=0.1))])
pipeline2.fit(X_train,Y_train) 
Y_predict2=pipeline2.predict(X_test)
accuracy_score(Y_test, Y_predict2)*100 


#Pipeline 2 - Confusion Matrix
matrix = confusion_matrix(Y_test, Y_predict2)
print(matrix)
p2_TP = matrix[0][0]
p2_TN = matrix[1][1]
p2_FP = matrix[1][0]
p2_FN = matrix[0][1]
print('') 
print('True Positive= %.0f' % p2_TP)
print('True Negative = %.0f' % p2_TN) 
print('')
print('False Positive = %.0f' % p2_FP) 
print('False Negative = %.0f' % p2_FN)


#Pipeline 2 - Evaluation Metrics
p2_Acc = (p2_TP + p2_TN) / (p2_TP + p2_TN + p2_FP + p2_FN) 
print('Acc = %.4f' % p2_Acc)
p2_DR = p2_TP / (p2_TP + p2_FN) 
print('DR(recall) = %.4f' % p2_DR)
p2_Prec = p2_TP / (p2_TP + p2_FP) 
print('Precision = %.4f' % p2_Prec)
p2_FAR = p2_FP / (p2_TN + p2_FP) 
print('FAR = %.4f' % p2_FAR)
p2_FNR = p2_FN / (p2_FN + p2_TP) 
print('FNR = %.4f' % p2_FNR)
p2_F1 = (2 * p2_TP) / (2*p2_TP + p2_FP + p2_FN)
print('F1 = %.4f' % p2_F1)
p2_MCC = ((p2_TP * p2_TN) - (p2_FP * p2_FN)) / math.sqrt((p2_TP + p2_FP)*(p2_TP + p2_FN)*(p2_TN + p2_FP)*(p2_TN + p2_FN))
print('MCC = %.4f' % p2_MCC)



#Pipeline 2 - ROC Curve
probs = pipeline2.predict_proba(X_test) 
probs = probs[:, 1]
auc = roc_auc_score(Y_test, probs) 
print('AUC: %.3f' % auc)
fpr, tpr, thresholds = roc_curve(Y_test, probs) 
pyplot.plot([0,1],[0,1], linestyle = '--') 
pyplot.plot(fpr,tpr,marker='.')
pyplot.show()


#Pipeline 3
pipeline3= Pipeline([
('zero variance', VarianceThreshold()),
('scale 0_1', MinMaxScaler()),
('top20 features', SelectKBest(chi2, k=20)), ('model',LogisticRegression(C=1000,solver='newton-cg'))])
pipeline3.fit(X_train,Y_train) 
Y_predict3=pipeline3.predict(X_test)
accuracy_score(Y_test, Y_predict3)*100 
#Pipleline 3 - Classification Report
report = classification_report(Y_test, Y_predict3) 
print(report)


matrix = confusion_matrix(Y_test, Y_predict3) 
print(matrix)
p3_TP = matrix[0][0]
p3_TN = matrix[1][1]
p3_FP = matrix[1][0]
p3_FN = matrix[0][1]

print('') 
print('True Positive = %.0f' % p3_TP)
print('True Negative = %.0f' % p3_TN)
print('')

print('False Positive = %.0f' % p3_FP) 
print('False Negative = %.0f' % p3_FN)



#Pipeline 3 - Evaluation Metrics

print('Acc = %.4f' % p3_Acc) 
p3_DR = p3_TP / (p3_TP + p3_FN)
print('DR(recall) = %.4f' % p3_DR) 
p3_Prec = p3_TP / (p3_TP + p3_FP)
print('Precision = %.4f' % p3_Prec) 
p3_FAR = p3_FP / (p3_TN + p3_FP)
print('FAR = %.4f' % p3_FAR)
print('FNR = %.4f' % p3_FNR)
print('F1 = %.4f' % p3_F1)
print('MCC = %.4f' % p3_MCC)



#Pipeline 3 - ROC Curve
probs = pipeline3.predict_proba(X_test) 
probs = probs[:, 1]
auc = roc_auc_score(Y_test, probs) 
print('AUC: %.3f' % auc)
fpr, tpr, thresholds = roc_curve(Y_test, probs) 
pyplot.plot([0,1],[0,1], linestyle = '--') 
pyplot.plot(fpr,tpr,marker='.')

pyplot.show()