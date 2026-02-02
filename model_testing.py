import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import time

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from torch.optim import Adam
#x=fake , x1=spam
input_file = "/home/mezba/Downloads/CSE 330/fake review/Cell_Phones_and_Accessories/output_fake.csv"
x=pd.read_csv(input_file)
x = x.dropna()
x['label']=[0.0 if i=='OR' else 1.0 for i in x['label']]
x.to_csv('cleaned_fake_reviews.csv', index=False)
labels = x['label'].values
reviews = x['text_'].values
print(len(x))
x.rename(columns={'text_':'text'}, inplace=True)
x1=pd.read_csv("Cell_Phones_and_Accessories/output_home.csv")
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


# x = x.drop_duplicates(subset=['text_str'], keep='first').reset_index(drop=True)
# x1 = x1.drop_duplicates(subset=['text_str'], keep='first').reset_index(drop=True)

x1['has_url'] = x1['text'].apply(lambda x: 1 if 'http' in x or 'www' in x or 'https' in x else 0)
#preprocessing part


x['text']=x['text'].str.lower()
x1['text']=x1['text'].str.lower()

# train_dataset = merged_df.sample(frac=0.8, random_state=42)
# #remove html /url tags
x['text'] = x['text'].str.replace(r'http\S+|www\S+|https\S+', '', case=False, regex=True)
x1['text'] = x1['text'].str.replace(r'http\S+|www\S+|https\S+', '', case=False, regex=True)
#remove punctuations
x['text'] = x['text'].str.replace(r'<[^>]+>', '', regex=True)
x1['text'] = x1['text'].str.replace(r'<[^>]+>', '', regex=True)

#tokenizitaion

x['text'] = x['text'].apply(word_tokenize)
x1['text'] = x1['text'].apply(word_tokenize)

#lemmatization
lemmatizer = WordNetLemmatizer()
x['text'] = x['text'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])
x1['text'] = x1['text'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])
#lemmatization
# lemmatizer = WordNetLemmatizer()
# x['text'] = x['text'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])
# x1['text'] = x1['text'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])

# Create text_str column from tokenized text
x['text_str'] = x['text'].apply(lambda tokens: ' '.join(tokens))
x1['text_str'] = x1['text'].apply(lambda tokens: ' '.join(tokens))
x1['label'] = x1['label'].map({0.0: 0, 2.0: 1})

# Drop duplicates keeping first occurrence
x = x.drop_duplicates(subset=['text_str'], keep='first').reset_index(drop=True)
x1 = x1.drop_duplicates(subset=['text_str'], keep='first').reset_index(drop=True)

x1 = x1.sample(frac=1, random_state=42).reset_index(drop=True)

# train_dataset_x1, validation_dataset_x1 = train_test_split(train_val_x1, test_size=0.125, stratify=train_val['label'], random_state=42)

# print(len(train_dataset_x1))
# print(len(validation_dataset_x1))
# print(len(test_dataset_x1))

#feature integration 
#for spam part 
def apply_tfidf(train_dataset, validation_dataset, test_dataset):
    vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    min_df=3,
    max_df=0.9
)
    X_train_text = vectorizer.fit_transform(train_dataset['text_str'])
    X_val_text = vectorizer.transform(validation_dataset['text_str'])

    # Get labels
    y_train = train_dataset['label'].values
    y_val = validation_dataset['label'].values

    return X_train_text, X_val_text, y_train, y_val, vectorizer

# train_val, test_dataset = train_test_split(x, test_size=0.1, stratify=x['label'], random_state=42)
# train_dataset, validation_dataset = train_test_split(train_val, test_size=0.125, stratify=train_val['label'], random_state=42)

train_val_x1, test_dataset_x1 = train_test_split(x1, test_size=0.1, stratify=x1['label'], random_state=42)
train_dataset_x1, validation_dataset_x1 = train_test_split(train_val_x1, test_size=0.125, stratify=train_val_x1['label'], random_state=42)


def add_features(df):
    df = df.copy()
    
    df['text_len'] = df['text_str'].apply(lambda x: len(x.split()))
    df['excl_count'] = df['text'].apply(lambda x: x.count('!') or x.count('#'))
    # df['has_url'] = df['text'].apply(lambda x: 1 if 'http' in x or 'www' in x else 0)
    df['text_str']=df['category']+' '+df['text_str']
    # category_dummies = pd.get_dummies(df['category'])
    # category_dummies = category_dummies.reindex(columns=all_categories, fill_value=0)

    return df  # pd.concat([df, category_dummies], axis=1)

train_dataset_x1 = add_features(train_dataset_x1)
validation_dataset_x1 = add_features(validation_dataset_x1)
print("has 1 ",x1['has_url'].sum())
# Apply TF-IDF
X_train, X_val, y_train, y_val, tfidf_vectorizer = apply_tfidf(train_dataset_x1, validation_dataset_x1, test_dataset_x1)

def combine_features(tfidf_features, df):
    text_len = df['text_len'].values.reshape(-1, 1)*0.2
    excl = df['excl_count'].values.reshape(-1, 1)*0.2
    url = df['has_url'].values.reshape(-1, 1)*3
    # rating = df['rating'].values.reshape(-1, 1)
    return hstack([tfidf_features, text_len, excl, url])

# category_cols = [col for col in train_dataset_x1.columns if col.startswith('cat_')]

X_train_combined = combine_features(X_train, train_dataset_x1)
X_val_combined = combine_features(X_val, validation_dataset_x1)

print(x1.columns)
print(x1['text_str'].head(2))

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,          
        min_samples_split=10,   
        min_samples_leaf=5,     
        random_state=42, 
        class_weight='balanced', 
        n_jobs=-1
),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1),
    'Naive Bayes': MultinomialNB(),
}

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

results = []

for name, model in models.items():
    print(f"\n{'='*80}")
    print(f"Training {name}...")
    print(f"{'='*80}")
    
    start_time = time.time()
    model.fit(X_train_combined, y_train)
    training_time = time.time() - start_time
    
    # Evaluate
    train_accuracy = model.score(X_train_combined, y_train)*100
    val_accuracy = model.score(X_val_combined, y_val)*100
    
    print(f"Training Time: {training_time:.2f}s")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    
    y_pred = model.predict(X_val_combined)
    print("\nTest Classification Report:")
    print(classification_report(y_val, y_pred))
    
    results.append({
        'Model': name,
        'Train_Acc': train_accuracy,
        'Val_Acc': val_accuracy,
        'Training_Time': training_time
    })

# Print summary comparison
print("\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

best_model_name = results_df.loc[results_df['Val_Acc'].idxmax(), 'Model']
print(f"\n Best Model: {best_model_name}")
print(f"   Validation Accuracy: {results_df.loc[results_df['Val_Acc'].idxmax(), 'Val_Acc']:.2f}%")

best_model = models[best_model_name]
y_pred_best = best_model.predict(X_val_combined)
  
# def plot_class_distribution(y, title):
#     sns.countplot(x=y)
#     plt.title(title)
#     plt.xlabel('Class')
#     plt.ylabel('Count')
#     plt.show()  
# plot_class_distribution(y_train, 'Training Set Class Distribution')
# plot_class_distribution(y_val, 'Validation Set Class Distribution')
# cm = confusion_matrix(y_val, y_pred)
# sns.heatmap(cm, annot=True, fmt='d')
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()
# cv_scores = cross_val_score(model, X_train_combined, y_train, cv=5)
# print(f"Cross-validation scores: {cv_scores}")
# print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
# Check rating distribution
# print("Real reviews (0) ratings:")
# print(train_dataset_x1[train_dataset_x1['label']==0]['rating'].value_counts())
# print("\nSpam reviews (1) ratings:")
# print(train_dataset_x1[train_dataset_x1['label']==1]['rating'].value_counts())
#for roberta part
