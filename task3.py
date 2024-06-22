# IRIS FLOWER CLASSIFICATION
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings("ignore")
dataset = pd.read_csv('https://github.com/vasudha830/Encryptix/blob/main/IRIS.csv')
dataset
species = dataset['species'].value_counts().reset_index()
# print(species)

dataset_numeric = dataset.drop(columns=['species'])
correlation_matrix = dataset_numeric.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Feature Correlations for Iris Dataset')
plt.show()


plt.figure(figsize=(10, 6))
#The highest positive correlation
plt.subplot(2, 2, 1)
sns.scatterplot(data=dataset, x='petal_length', y='petal_width', hue='species')
plt.title('petal_length vs petal_width')
#The lowest negative correlation
plt.subplot(2, 2, 2)
sns.scatterplot(data=dataset, x='sepal_width', y='petal_length', hue='species')
plt.title('sepal_width vs petal_length')
plt.subplot(2, 2, 3)
sns.scatterplot(data=dataset, x='petal_length', y='sepal_length', hue='species')
plt.title('petal_length vs sepal_length')
plt.subplot(2, 2, 4)
sns.scatterplot(data=dataset, x='sepal_length', y='sepal_width', hue='species')
plt.title('sepal_length vs sepal_width')
plt.tight_layout()
plt.show()

fig = px.scatter_3d(dataset, x='sepal_length', y='petal_width', z='petal_length', color="species",title='3D Scatter Plot of Iris Dataset')
# fig.show()

# PREPROCESSING
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
def evaluate_model(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return cm, accuracy

#MODELS
# 1. LOGISTIC REGRESSION
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
evaluate_model(y_test, y_pred)

# 2. K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
y_pred = lr.predict(X_test)
evaluate_model(y_test, y_pred)

# 3. DECISION TREE
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = lr.predict(X_test)
evaluate_model(y_test, y_pred)

# 4. RANDOM FOREST
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train, y_train)
y_pred = lr.predict(X_test)
evaluate_model(y_test, y_pred)

# 5. SUPPORT VETCOR MACHINE 
svc = SVC()
svc.fit(X_train, y_train)
y_pred = lr.predict(X_test)
evaluate_model(y_test, y_pred)

# k-Fold Cross Validation
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC()
}
results = {}
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=10)
    results[model_name] = cv_scores

results_df = pd.DataFrame(results)
plt.figure(figsize=(10, 6))
results_df.boxplot()
plt.title('Comparison of model results (10-Fold Cross Validation)')
plt.ylabel('Accuracy')
plt.show()
print(results_df.mean())

