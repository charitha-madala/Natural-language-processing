# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 15:31:24 2020

@author: DELL
"""


import numpy as np
import pandas as pd
import pickle#used to save models
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter= '\t',quoting=3)
#ignores tab and quotes
import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
c=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])#replacing special characters with spaces
    review=review.lower()#turning into lower case
    review=review.split()#turning into a list
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    c.append(review)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(c).toarray()
y=dataset.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
pickle.dump(cv.vocabulary_,open("feature.pkl","wb"))

from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(input_dim=1500,init="random_uniform",activation="sigmoid",output_dim=1000))
model.add(Dense(init="random_uniform",activation="sigmoid",output_dim=100))
model.add(Dense (output_dim=1,init="random_uniform",activation="sigmoid"))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=50,batch_size=10)
y_pred= model.predict(x_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
loaded_vec=CountVectorizer(decode_error='replace',vocabulary=pickle.load(open('feature.pkl','rb')))
da="It was a pleasure"#driver code
da=da.split("delimiter")
result=model.predict(loaded_vec.transform(da))
print(result)
prediction=result>0.5
print(prediction)