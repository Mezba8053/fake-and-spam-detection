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
# #null count
#lemmatization
lemmatizer = WordNetLemmatizer()
x['text'] = x['text'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])
x1['text'] = x1['text'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])

# Create text_str column from tokenized text
x['text_str'] = x['text'].apply(lambda tokens: ' '.join(tokens))
x1['text_str'] = x1['text'].apply(lambda tokens: ' '.join(tokens))
x1['label'] = x1['label'].map({0.0: 0, 2.0: 1})

# Drop duplicates keeping first occurrence
x = x.drop_duplicates(subset=['text_str'], keep='first').reset_index(drop=True)
x1 = x1.drop_duplicates(subset=['text_str'], keep='first').reset_index(drop=True)

# Shuffle the data to avoid any ordering bias
x1 = x1.sample(frac=1, random_state=42).reset_index(drop=True)

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
    vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,1),
    min_df=3,
    max_df=0.9
)


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

# Get all unique categories from training data for consistent encoding
all_categories = train_dataset_x1['category'].unique()

def add_features(df):
    df = df.copy()
    df['text_len'] = df['text_str'].apply(lambda x: len(x.split()))
    df['excl_count'] = df['text_str'].apply(lambda x: x.count('!'))
    df['has_url'] = df['text_str'].apply(lambda x: 1 if 'http' in x or 'www' in x else 0)

    category_dummies = pd.get_dummies(df['category'])
    category_dummies = category_dummies.reindex(columns=all_categories, fill_value=0)

    return pd.concat([df, category_dummies], axis=1)

train_dataset_x1 = add_features(train_dataset_x1)
validation_dataset_x1 = add_features(validation_dataset_x1)
test_dataset_x1 = add_features(test_dataset_x1)

# Apply TF-IDF
X_train, X_val, X_test, y_train, y_val, y_test, tfidf_vectorizer = apply_tfidf(train_dataset_x1, validation_dataset_x1, test_dataset_x1)

# Combine TF-IDF with other features (removed rating due to data leakage)
def combine_features(tfidf_features, df):
    text_len = df['text_len'].values.reshape(-1, 1)
    excl = df['excl_count'].values.reshape(-1, 1)
    url = df['has_url'].values.reshape(-1, 1)

    from scipy.sparse import hstack
    return hstack([tfidf_features, text_len, excl, url])

# Get category columns
category_cols = [col for col in train_dataset_x1.columns if col.startswith('cat_')]

X_train_combined = combine_features(X_train, train_dataset_x1)
X_val_combined = combine_features(X_val, validation_dataset_x1)
X_test_combined = combine_features(X_test, test_dataset_x1)

print(x1.columns)
print(x1['text_str'].head(2))

# Define multiple models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1),
    'Naive Bayes': MultinomialNB(),
    # 'SVM': SVC(kernel='rbf', random_state=42, class_weight='balanced')  # Uncomment if you want to try SVM (slower)
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
    test_accuracy = model.score(X_test_combined, y_test)*100
    
    print(f"Training Time: {training_time:.2f}s")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    y_pred = model.predict(X_test_combined)
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_pred))
    
    results.append({
        'Model': name,
        'Train_Acc': train_accuracy,
        'Val_Acc': val_accuracy,
        'Test_Acc': test_accuracy,
        'Training_Time': training_time
    })

# Print summary comparison
print("\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Select best model based on validation accuracy
best_model_name = results_df.loc[results_df['Val_Acc'].idxmax(), 'Model']
print(f"\n Best Model: {best_model_name}")
print(f"   Validation Accuracy: {results_df.loc[results_df['Val_Acc'].idxmax(), 'Val_Acc']:.2f}%")
print(f"   Test Accuracy: {results_df.loc[results_df['Val_Acc'].idxmax(), 'Test_Acc']:.2f}%")

# Use the best model for final evaluation
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_combined)

# Uncomment below for visualization
# cm = confusion_matrix(y_test, y_pred_best)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title(f"Confusion Matrix - {best_model_name}")
# plt.show()
# #plotting part
  
def plot_class_distribution(y, title):
    sns.countplot(x=y)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()  
plot_class_distribution(y_train, 'Training Set Class Distribution')
# plot_class_distribution(y_val, 'Validation Set Class Distribution')
plot_class_distribution(y_test, 'Test Set Class Distribution')
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
cv_scores = cross_val_score(model, X_train_combined, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
# Check rating distribution
print("Real reviews (0) ratings:")
print(train_dataset_x1[train_dataset_x1['label']==0]['rating'].value_counts())
print("\nSpam reviews (1) ratings:")
print(train_dataset_x1[train_dataset_x1['label']==1]['rating'].value_counts())
#now for next model
#for roberta part
