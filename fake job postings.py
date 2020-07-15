# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 15:57:57 2020

@author: HP
"""

import pandas as pd
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("fake_job_postings.csv")

data.info()

data.isnull().sum()
data = data.fillna('No info ')

df = data.apply(lambda x: x.astype(str).str.lower())

df["fraudulent"].value_counts()

plt.figure(1,figsize=(10,8))
sns.countplot(hue=df.fraudulent,x=df.employment_type);
plt.title('Which type of jobs have more fraudulent postings');


plt.figure(figsize = (15, 8))
sns.countplot(y=data.function,hue=data.fraudulent);


labels=data.location.value_counts().index[:10]
values=data.location.value_counts().values[:10]
plt.figure(figsize = (15, 8))

ax = sns.barplot(x=labels, y=values)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
for i, p in enumerate(ax.patches):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height + 0.9, values[i],ha="center")
    
text = data[data.columns[1:-1]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
target = data['fraudulent']    
    
df1 = pd.concat([text,target],axis=1)    
    
df1=df1.add_prefix('X_')

df1['X_0'] = df1['X_0'].str.replace('\d+', '')

stop = set(stopwords.words('english'))
def remove_stopword(word):
    return word not in words

df1['X_0'] = df1['X_0'].str.lower().str.split()
d= df1['X_0'].apply(lambda x : [item for item in x if item not in stop])

d = pd.concat([d, target], axis=1)
d1=d.sample(frac=0.4, replace=True, random_state=1)

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

vectorizer = CountVectorizer()
bag_of_words = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False).fit_transform(d1['X_0'])
print(bag_of_words.toarray())
s=bag_of_words.toarray()

tfidf = TfidfTransformer()
X = tfidf.fit_transform(s)
print(X)
print(X.shape)
print(X.toarray())
s1=X.toarray()

x = s1
y = d1.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

from sklearn.naive_bayes import MultinomialNB
from collections import Counter
classifier = MultinomialNB()

from sklearn.neighbors import KNeighborsClassifier
nb = KNeighborsClassifier(n_neighbors=4)
nb.fit(x_train,y_train)

classifier.fit(x_train,y_train)

expected = y_test
predicted = classifier.predict(x_test)

collections=Counter(y_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predicted))
print(confusion_matrix(y_test,predicted))

from sklearn.metrics import accuracy_score

accuracy_score(expected,predicted)
   
    
    
    