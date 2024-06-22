import numpy as np 
import pandas as pd
import os
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib import pyplot
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
raw=pd.read_csv('file:///Users/vasudhagarg/Downloads/Titanic-Dataset.csv')
# print(raw.sample(10))
raw.info()
raw['age']=raw['Age'].fillna(raw['Age'].mean())
# print(raw.head())
df=raw.drop(['Age','Cabin'],axis=1)
df.head()
df.info()
new_df=df.dropna()
new_df.head()
new_df.info()
new_df['family']=new_df['SibSp']+new_df['Parch']
new_data=new_df.drop(['SibSp','Parch','Name','Ticket'],axis=1)
# print(new_data.head())
new_data['Embarked'].value_counts().plot.bar()
# print(plt.show())
transform=ColumnTransformer(transformers=[
    ('t1',OneHotEncoder(sparse=False,drop='first'),['Sex']),
    ('t2',OrdinalEncoder(categories=[['Q','C','S']]),['Embarked'])
     ],
     remainder='passthrough')
new_data=transform.fit_transform(new_data)
# print(new_data.shape)
data1=pd.DataFrame(new_data,columns=['Sex','Embarked','PassengerId','Survived','Pclass','Fare','age','family'])
data1.head()
data1.columns.to_list
oreder=['PassengerId','Sex','Embarked','Pclass','Fare','age','family','Survived']
df1=data1[oreder]
df1.sample(10)
plt.subplot(1,2,1)
sns.distplot(df1['age'])
plt.subplot(1,2,2)
sns.boxplot(df1['Fare'])

# print(plt.show())

# remove(cap) outlier from age and drop fare
upper_limit=df['age'].mean()+(3*df['age'].std())
lower_limit=df['age'].mean()-(3*df['age'].std())
# print(upper_limit,lower_limit)

# capping outliers
df1['age_']=np.where(df1['age']>upper_limit,upper_limit,
            np.where(df1['age']<lower_limit,lower_limit,df1['age']))
df1.head()
plt.subplot(2,2,1)
sns.barplot(x=df1['Sex'],y=df1['Survived'])
plt.subplot(2,2,2)
sns.barplot(x=df1['family'],y=df1['Survived'])
plt.subplot(2,2,3)
sns.barplot(x=df1['Pclass'],y=df1['Survived'])
plt.subplot(2,2,4)
sns.barplot(x=df1['Embarked'],y=df1['Survived'])
# print(plt.show())
def myfunc(mem):
    if 0<mem<4:
        
        return 1
    elif mem>4:
        return 2
    else:
        return 0

df1['family']=df1['family'].apply(myfunc)
sns.barplot(x=df1['family'],y=df1['Survived'])
# print(plt.show())

# check point

# small family> alone>big family
df_final=df1.drop(['Fare'],axis=1)
df_final.head()

X=df_final.drop(columns=['Survived'])
y=df_final['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)
print(X_train.shape,X_test.shape)
gbc=GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, min_samples_split=2)
gbc.fit(X_train,y_train)

lr=LogisticRegression()
lr.fit(X_train,y_train)

rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
y_pred1=gbc.predict(X_test)
y_pred2=lr.predict(X_test)
y_pred3=rfc.predict(X_test)
print('gbc: ',accuracy_score(y_test,y_pred1))
print('lr: ',accuracy_score(y_test,y_pred2))
print('rfc: ',accuracy_score(y_test,y_pred3))

param={
    'n_estimators':[100,200,250,300,500],
    'learning_rate':[0.1,0.5,0.7,0.01],
    'max_depth':[10,12,15,20]
#     'max_features':['sqrt', 'log2']
}
gs=GridSearchCV(estimator=GradientBoostingClassifier(),param_grid=param,n_jobs=1, cv=5,verbose=1)
# gs.fit(X_train,y_train)
# print(gs.best_params_,gs.best_score_)

gbc=GradientBoostingClassifier(learning_rate=0.01, n_estimators=250)
gbc.fit(X_train,y_train)
ypred=gbc.predict(X_test)
accuracy_score(y_test,ypred)
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
output = pd.DataFrame({'PassengerId': X_test['PassengerId'], 'Survived': ypred})
output.to_csv('submission_titanic.csv', index=False)
