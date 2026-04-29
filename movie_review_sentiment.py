import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


df=pd.read_csv('movies_sentiment_data.csv')

# print(df.head())
# print(df.shape)

df['category']=df['sentiment'].apply(lambda x: 1 if x=='positive' else 0)

# print(df.head())

X_train,X_test,y_train,y_test=train_test_split(df.review,df.category,test_size=0.2, random_state=42)

#MultinomiqlNB Model
model_1=Pipeline(
    [('cv',CountVectorizer()),('nb',MultinomialNB())]
)

model_1.fit(X_train,y_train)
y_pred_1=model_1.predict(X_test)
report_1=classification_report(y_test,y_pred_1)
print('Classification report of MultinomiqlNB')
print(report_1)


#KNN Model
model_2=Pipeline(
    [('cv',CountVectorizer()),
     ('knn',KNeighborsClassifier(n_neighbors=10,metric='euclidean'))]
)

model_2.fit(X_train,y_train)
y_pred_2=model_2.predict(X_test)
report_2=classification_report(y_test,y_pred_2)
print('Classification report of KNN')
print(report_2)


#RandomForest Model
model_3=Pipeline(
    [('cv',CountVectorizer()),
     ('rfc',RandomForestClassifier(n_estimators=50,criterion='entropy',random_state=42))]
)

model_3.fit(X_train,y_train)
y_pred_3=model_3.predict(X_test)
report_3=classification_report(y_test,y_pred_3)
print('Classification report of RandomForest')
print(report_3)