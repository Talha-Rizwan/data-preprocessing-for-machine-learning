# Talha Rizwan Malik 19I-0652
# Syed Muhammad Ibtisam 19i-0422
# Muhammad Anser Qureshi 19I-0680
import csv
import pandas as pd
from sklearn import metrics
from sklearn.impute import KNNImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# data cleaning through simple imputation(mean imputation)
mean_imputation_df = pd.read_csv('derm.csv')
age_mean = int(mean_imputation_df['age'].mean())
mean_imputation_df['age'].fillna(age_mean, inplace=True)
mean_imputation_df.to_csv('derm.0.1.csv')

# data cleaning through KNN imputation
df = pd.read_csv(r'derm.csv')
imputer = KNNImputer()
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df['age']=df['age'].astype('int')
df.to_csv('derm.0.2.csv')

# KNN Classifier for mean imputer
knn_classifier_df = pd.read_csv('derm.0.1.csv')
y = knn_classifier_df[['class']]
X = knn_classifier_df.iloc[:, :-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
knn_imputation_2_test1 = KNeighborsClassifier(n_neighbors=5)
knn_imputation_2_test1.fit(X_train, y_train)
r = knn_imputation_2_test1.score(X_test,y_test)
print('Accuracy of Simple Imputation : ', r)

# KNN Classifier for KNN imputer
knn_classifier_df_1 = pd.read_csv('derm.0.2.csv')
y = knn_classifier_df_1[['class']]
X = knn_classifier_df_1.iloc[:, :-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=1)
knn_imputation_2_test1 = KNeighborsClassifier()
knn_imputation_2_test1.fit(X_train, y_train)
r2 = knn_imputation_2_test1.score(X_test,y_test)
print('Accuracy of KNN Imputation : ',r2)

