"""
Title: Sentiment Analysis
Abdelhalim Mokhtar
"""
# Import necessary libraries
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Read dataset
path = 'movie_data.tsv'
data = pd.read_table(path, header=None, skiprows=1, names=['Review', 'Sentiment'])
# Get reviews
X = data.Review
# Get sentiments
y = data.Sentiment

# Using CountVectorizer to convert text to tokens
vect = CountVectorizer(stop_words='english', ngram_range=(1, 1), max_df=.80, min_df=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)
# Using training data to transform text into feature counts for each post
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
X_test_dtm = vect.transform(X_test)

# Accuracy calculation with Naive Bayes
NB = MultinomialNB()
NB.fit(X_train_dtm, y_train)
y_pred = NB.predict(X_test_dtm)
print('\nNaive Bayes')
print('Accuracy Score: ', metrics.accuracy_score(y_test, y_pred) * 100, '%', sep='')
print('Confusion Matrix: ', metrics.confusion_matrix(y_test, y_pred), sep='\n')

# Accuracy calculation with Logistic Regression
LR = LogisticRegression(solver='liblinear', dual=False)
LR.fit(X_train_dtm, y_train)
y_pred = LR.predict(X_test_dtm)
print('\nLogistic Regression')
print('Accuracy Score: ', metrics.accuracy_score(y_test, y_pred) * 100, '%', sep='')
print('Confusion Matrix: ', metrics.confusion_matrix(y_test, y_pred), sep='\n')

# Accuracy calculation with SVM
SVM = LinearSVC(dual=False)
SVM.fit(X_train_dtm, y_train)
y_pred = SVM.predict(X_test_dtm)
print('\nSupport Vector Machine')
print('Accuracy Score: ', metrics.accuracy_score(y_test, y_pred) * 100, '%', sep='')
print('Confusion Matrix: ', metrics.confusion_matrix(y_test, y_pred), sep='\n')

# Accuracy calculation with KNN
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train_dtm, y_train)
y_pred = KNN.predict(X_test_dtm)
print('\nK Nearest Neighbors (NN = 3)')
print('Accuracy Score: ', metrics.accuracy_score(y_test, y_pred) * 100, '%', sep='')
print('Confusion Matrix: ', metrics.confusion_matrix(y_test, y_pred), sep='\n')

# Naive Bayes analysis
tokens_words = vect.get_feature_names_out()
print('\nAnalysis')
print('No. of tokens: ', len(tokens_words))
counts = NB.feature_count_
df_table = {'Token': tokens_words, 'Negative': counts[0, :], 'Positive': counts[1, :]}
tokens = pd.DataFrame(df_table, columns=['Token', 'Positive', 'Negative'])
positives = len(tokens[tokens['Positive'] > tokens['Negative']])
print('No. of positive tokens: ', positives)
print('No. of negative tokens: ', len(tokens_words) - positives)
# Check the positivity/negativity of a specific token, for example the word awesome
token_search = ['awesome']
print('\nSearch Results for token/s:', token_search)
print(tokens.loc[tokens['Token'].isin(token_search)])
# Analyze false negatives (actual:1; predicted:0) (i.e. a negative review predicted for a positive review)
print(X_test[y_pred < y_test])
# Analysis False Positives (Actual: 0; Predicted: 1)(Predicted positive review for a negative review)
# Analyze false positives (actual:0; predicted:1) (i.e. a positive review predicted for a negative review)
print(X_test[y_pred > y_test])

# Test an opinion on the best performing model (LR logistic regression)
trainingVector = CountVectorizer(stop_words='english', ngram_range=(1, 1), max_df=.80, min_df=5)
trainingVector.fit(X)
X_dtm = trainingVector.transform(X)
LR_complete = LogisticRegression(solver='liblinear', dual=False)
LR_complete.fit(X_dtm, y)

tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

joblib.dump(en_stopwords, 'stopwords.pkl')
joblib.dump(LR_complete, 'model.pkl')
joblib.dump(trainingVector, 'vectorizer.pkl')
