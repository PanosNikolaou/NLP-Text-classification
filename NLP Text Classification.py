# -*- coding: utf-8 -*-
"""
Description:
    NLP Classification with CountVectorizer, tf-idf and SGDClassifier

Author : Nikolaou Panagiotis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
import pickle

# read the excel file and use the columns B,C,D
df = pd.read_excel('Use Case_Classification Python Test.xlsx', sheet_name="Training Data", usecols="B,C,D")

# create a hashtable from the target values
counter = Counter(df['news_type'].tolist())

# map the news categories
news_types = {i[0]: idx for idx, i in enumerate(counter.most_common(5))}

df = df[df['news_type'].map(lambda x: x in news_types)]

# get the description from news title
description_list = df['news_title'].tolist()

# map the news types
news_types_list = [news_types[i] for i in df['news_type'].tolist()]

# convert to np array
news_types_list = np.array(news_types_list)

# create a new count vectorizer
count_vect = CountVectorizer()

# convert text into vector 
x_train_counts = count_vect.fit_transform(description_list)

# save the vocabulary for future use
pickle.dump(count_vect.vocabulary_, open("vocab.pickle", 'wb'))

# create a TfidfTransformer
tfidf_transformer = TfidfTransformer()

# compute the IDF values
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

# split data to test and train for x and y
train_x, test_x, train_y, test_y = train_test_split(x_train_tfidf, news_types_list, test_size=0.3)

# create a new support vector machine and fir the train data
clf = svm.SVC(kernel='linear').fit(train_x, train_y)

# create a new SGD classifier and use the train data
sgdc = SGDClassifier(max_iter=1000, tol=0.01)
sgdc.fit(train_x, train_y)

# make a prediction to test data
y_score = clf.predict(test_x)

# get the score for the predicted data
n_right = 0
for i in range(len(y_score)):
    if y_score[i] == test_y[i]:
        n_right += 1

# compute the accurancy of the model
print("Accuracy: %.2f%%" % ((n_right/float(len(test_y)) * 100)))

# save the model for future use
pickle.dump(clf,open("model.pickle", 'wb'))

# read the sheet name data to be classified
df_classify = pd.read_excel('Use Case_Classification Python Test.xlsx', sheet_name="Data to be Classified", usecols="A,B,C")

# createa new dataframe from the news title column
description_list_new = df_classify['news_title'].tolist()

# convert text into vector 
x_train_counts_new = count_vect.transform(description_list_new)

# compute the IDF values
x_train_tfidf_new = tfidf_transformer.transform(x_train_counts_new)

# make the prediction with the trained model
df_classify['sys_news_type'] = clf.predict(x_train_tfidf_new)

# create the tags for the news types
w = ["leadership", "partnership/alliance", "m&a", "finance", "product/solution"]
l = [0, 1, 2, 3, 4]
trans = {l1:w1 for w1,l1 in zip(w,l)}

# assign the text tags to the numeric values
for index, row in df_classify.iterrows():
    df_classify['sys_news_type'][index] = trans[row['sys_news_type']]

# save the column to the excel file
with pd.ExcelWriter("Use Case_Classification Python Test.xlsx",engine="openpyxl",mode="a",on_sheet_exists="replace") as writer:
    df_classify.to_excel(writer, sheet_name='Data to be Classified')
