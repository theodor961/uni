import pandas as pd 
import re

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# Loading a dataset
data_source_url = "https://raw.githubusercontent.com/satyajeetkrjha/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
airline_tweets = pd.read_csv(data_source_url)


# Data cleaning
features = airline_tweets.iloc[:, 10].values
labels = airline_tweets.iloc[:, 1].values

processed_features = []

for sentence in range(0, len(features)):
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))
    processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    processed_feature = processed_feature.lower()
    processed_features.append(processed_feature)


# Converting textual data to numerical vectors
vectorizer = TfidfVectorizer(max_features=2500, stop_words='english')
processed_features = vectorizer.fit_transform(processed_features).toarray()


# Dividing Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)


# Training the model
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)
print("Model is successfully trained")


# Evaluating the model
predictions = text_classifier.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
print("Accuracy Score:", accuracy_score(y_test, predictions))


# Testing the model with a new examples
while True:
    new_example = input("Enter the example you want to classify (or type 'exit' to end): ")

    if new_example.lower() == 'exit':
        print("Exiting the program.")
        break

    processed_new_example = re.sub(r'\W', ' ', str(new_example))
    processed_new_example = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_new_example)
    processed_new_example = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_new_example) 
    processed_new_example = re.sub(r'\s+', ' ', processed_new_example, flags=re.I)
    processed_new_example = re.sub(r'^b\s+', '', processed_new_example)
    processed_new_example = processed_new_example.lower()

    new_example_vectorized = vectorizer.transform([processed_new_example]).toarray()

    new_example_prediction = text_classifier.predict(new_example_vectorized)

    print("Predicted Sentiment for the new example:", new_example_prediction[0])
    print("\n")

