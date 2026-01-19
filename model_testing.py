import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from torch.optim import Adam
# def tfidf(train_dataset,validation_dataset,test_dataset):
#     # TF-IDF Vectorization
#     tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2))

#     # Fit on training data only
#     X_train_text = tfidf.fit_transform(train_dataset['text_str'])
#     X_val_text = tfidf.transform(validation_dataset['text_str'])
#     X_test_text = tfidf.transform(test_dataset['text_str'])

#     # Get labels
#     y_train = train_dataset['label'].values
#     y_val = validation_dataset['label'].values
#     y_test = test_dataset['label'].values

#     # print(f"Training text features shape: {X_train_text.shape}")
input_file = "cleaned_fake_reviews.csv"
x=pd.read_csv(input_file)
x = x.dropna()
# x['label']=[0.0 if i=='OR' else 1.0 for i in x['label']]
# x.to_csv('fake review/cleaned_fake_reviews.csv', index=False)
labels = x['label'].values
reviews = x['text_'].values
print(len(x))
x.rename(columns={'text_':'text'}, inplace=True)
x1=pd.read_csv("Cell_Phones_and_Accessories/output1.csv")
x1 = x1.dropna().reset_index(drop=True)
x1=x1.drop(['_id','reviewerID','asin','reviewerName','helpful','summary','unixReviewTime','reviewTime'],axis=1)
x1['class']=[0 if i==0.0 else 2.0 for i in x1['class']]
print(len(x1))
print(x1.columns)
# print (x1.head(10))
# text_set = set(x['text_'])
# matches = 0
# for review in x1['reviewText']:
#     if review in text_set:
#         matches += 1
#         print("Found match")

# print(f"Total matches: {matches}")
x1.rename(columns={'class':'label'}, inplace=True)
x1.rename(columns={'reviewText':'text'}, inplace=True)
x1.rename(columns={'overall':'rating'}, inplace=True)

x1 = x1[['category', 'rating', 'text', 'label']]
x1.to_csv('spam_reviews.csv', index=False)
# merged_df = pd.concat([x, x1], ignore_index=True)
# merged_df.to_csv('merged_fake_spam_reviews.csv', index=False)
# print(x1['label'].value_counts())
# print(merged_df['label'].value_counts())
print(x['label'].isnull().sum())
print(x1['label'].isnull().sum())

# Drop duplicates keeping first occurrence

# x = x.drop_duplicates(subset=['text_str'], keep='first').reset_index(drop=True)
# x1 = x1.drop_duplicates(subset=['text_str'], keep='first').reset_index(drop=True)

#preprocessing part


x['text']=x['text'].str.lower()
x1['text']=x1['text'].str.lower()

# train_dataset = merged_df.sample(frac=0.8, random_state=42)
# #remove html /url tags
# x['text'] = x['text'].str.replace(r'http\S+|www\S+|https\S+', '', case=False, regex=True)
# x1['text'] = x1['text'].str.replace(r'http\S+|www\S+|https\S+', '', case=False, regex=True)
# #remove punctuations
# x['text'] = x['text'].str.replace(r'<[^>]+>', '', regex=True)
# x1['text'] = x1['text'].str.replace(r'<[^>]+>', '', regex=True)

#tokenizitaion

x['text'] = x['text'].apply(word_tokenize)
x1['text'] = x1['text'].apply(word_tokenize)

#lemmatization
# lemmatizer = WordNetLemmatizer()
# x['text'] = x['text'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])
# x1['text'] = x1['text'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])
# #null count
#lemmatization
lemmatizer = WordNetLemmatizer()
x['text'] = x['text'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])
x1['text'] = x1['text'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])

# Create text_str column from tokenized text
x['text_str'] = x['text'].apply(lambda tokens: ' '.join(tokens))
x1['text_str'] = x1['text'].apply(lambda tokens: ' '.join(tokens))

# Drop duplicates keeping first occurrence
x = x.drop_duplicates(subset=['text_str'], keep='first').reset_index(drop=True)
x1 = x1.drop_duplicates(subset=['text_str'], keep='first').reset_index(drop=True)

# train_val_x1, test_dataset_x1 = train_test_split(x1, test_size=0.1, stratify=x['label'], random_state=42)
# train_dataset_x1, validation_dataset_x1 = train_test_split(train_val_x1, test_size=0.125, stratify=train_val['label'], random_state=42)

# print(len(train_dataset_x1))
# print(len(validation_dataset_x1))
# print(len(test_dataset_x1))

#feature integration 
#for spam part 
# Review text

# ✔ Always include
# → TF-IDF / embeddings

def apply_tfidf(train_dataset, validation_dataset, test_dataset):
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1,2))

    # Fit on training data only
    X_train_text = vectorizer.fit_transform(train_dataset['text_str'])
    X_val_text = vectorizer.transform(validation_dataset['text_str'])
    X_test_text = vectorizer.transform(test_dataset['text_str'])

    # Get labels
    y_train = train_dataset['label'].values
    y_val = validation_dataset['label'].values
    y_test = test_dataset['label'].values

    return X_train_text, X_val_text, X_test_text, y_train, y_val, y_test, vectorizer

# Split both datasets
train_val, test_dataset = train_test_split(x, test_size=0.1, stratify=x['label'], random_state=42)
train_dataset, validation_dataset = train_test_split(train_val, test_size=0.125, stratify=train_val['label'], random_state=42)

train_val_x1, test_dataset_x1 = train_test_split(x1, test_size=0.1, stratify=x1['label'], random_state=42)
train_dataset_x1, validation_dataset_x1 = train_test_split(train_val_x1, test_size=0.125, stratify=train_val_x1['label'], random_state=42)

# Apply TF-IDF
X_train, X_val, X_test, y_train, y_val, y_test, tfidf_vectorizer = apply_tfidf(train_dataset_x1, validation_dataset_x1, test_dataset_x1)
# tfidf(train_dataset_x1,validation_dataset_x1,test_dataset_x1)
# text length
x1['text_len'] = x1['text_str'].apply(lambda x: len(x.split()))

# exclamation count
x1['excl_count'] = x1['text_str'].apply(lambda x: x.count('!'))

# url presence
x1['has_url'] = x1['text_str'].apply(lambda x: 1 if 'http' in x or 'www' in x else 0)




# 3. Category

# ✔ Include

# One-hot encoding (ML)

# Embedding (DL)

# This helps detect category–text mismatch
# One-hot encoding for category
category_dummies = pd.get_dummies(x1['category'], prefix='cat')
