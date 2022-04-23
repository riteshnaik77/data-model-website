import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import pickle


train_df=pd.read_csv("BBC_News_Train.csv", engine="python")

test_df=pd.read_csv("BBC_News_Test.csv", engine="python")

print(train_df.head())
print(test_df.head())

# Numerical Encoding or Categorization for Category Column
train_df["Label_Encoding"] = train_df["Category"].factorize()[0]

# Frequency Distribution for Each Class
print (train_df["Category"].value_counts())
print (train_df["Label_Encoding"].value_counts())
# Based on frequency distribution  we can say that data is balanced, not suffering from class imbalance.

# Preserving the Category Coding
category_labels_to_id = {"business":0,"tech":1,"politics":2,"sport":3,"entertainment":4}
id_to_category = {0:"business",1:"tech",2:"politics",3:"sport",4:"entertainment"}

# Check the number of Null in our Data Set
train_df.isnull().sum()

"""
Setting TF-IDF
--------------
Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
min_df = Ignore all the words that have a document frequency less than min_df
"""

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=7, norm='l2', encoding='utf-8', ngram_range=(1, 3),lowercase = True,stop_words='english')

# Training the tfidf feature
tfidf_feature = tfidf.fit_transform(train_df.Text).toarray()

with open('news_classification_tfidf_vectorizer.pkl', 'wb') as output:
    pickle.dump(tfidf, output)

#Chi Square is a statistical technique, used for calculating correlation of features with outcome

N = 5  # We are going to look for top 3 categories
labels = train_df.Label_Encoding

#For each category, find words that are highly corelated to it
for category, category_id in sorted(category_labels_to_id.items()):
  features_chi2 = chi2(tfidf_feature, labels == category_id)              # Do chi2 analyses of all items in this category
  indices = np.argsort(features_chi2[0])                                  # Sorts the indices of features_chi2[0] - the chi-squared stats of each feature
  feature_names = np.array(tfidf.get_feature_names())[indices]            # Converts indices to feature names ( in increasing order of chi-squared stat values)
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]         # List of single word features ( in increasing order of chi-squared stat values)
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]          # List for two-word features ( in increasing order of chi-squared stat values)
  trigrams = [v for v in feature_names if len(v.split(" "))==3]
  print("# '{}':".format(category))
  print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:]))) # Print 3 unigrams with highest Chi squared stat
  print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:]))) # Print 3 bigrams with highest Chi squared stat
  print("  . Most correlated Trigrams:\n       . {}".format('\n       . '.join(trigrams[-N:]))) # Print 3 bigrams with highest Chi squared stat

# Train Test Split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier()

#Split Data
X_train, X_test, y_train, y_test= train_test_split(tfidf_feature, labels, test_size=0.25, random_state=0)

model.fit(X_train,y_train)

pickle.dump(model, open("news_classification.pkl", "wb"))


predicted_train = model.predict(X_train)
predicted_test = model.predict(X_test)

from sklearn.metrics import classification_report

print (classification_report(y_test,predicted_test))

print (classification_report(y_train,predicted_train))

""" Module for prediction"""
id_to_category = {0:"business",1:"tech",2:"politics",3:"sport",4:"entertainment"}


test_article = "Iron man actor robert junior came for promotion. The film is getting lot of attention from movie lovers across the globe. Its gonna be interesting to see how this movie performs on box-office."

test_article = input("Enter the text of your article\n")

test_article = test_article.lower()

test_frame = pd.DataFrame({"Text":[test_article]})
print (test_frame)

test_feature = tfidf.transform(test_frame.Text).toarray()
print("Checking this", test_feature)

prediction = model.predict(test_feature)

print (f"Model predicts this excerpt belong to  {str.upper(id_to_category[prediction[0]])} category!")
