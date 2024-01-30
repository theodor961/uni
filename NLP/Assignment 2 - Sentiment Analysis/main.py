import pandas as pd 
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB



# Function for data cleaning
def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'^b\s+', '', text)
    text = text.lower()
    return text


# Function for text vectorization (using tf-idf)
def vectorize_text(features):
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=2500, stop_words='english')
    # N-Gram Count Vectorization
    # vectorizer = CountVectorizer(ngram_range=(1,2), stop_words='english')
    return vectorizer.fit_transform(features).toarray(), vectorizer


# Function for training a model
def train_model(X_train, y_train, model_type='RandomForest'):
    if model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=200, random_state=0)
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(random_state=0)
    elif model_type == 'SVM':
        model = SVC(random_state=0)
    elif model_type == 'NaiveBayes':
        model = MultinomialNB()
    else:
        raise ValueError("Invalid model type")

    model.fit(X_train, y_train)
    return model


# Function for evaluating a model
def evaluate_model(model, X_test, y_test, algorithm, feature_engineering):
    predictions = model.predict(X_test)
    confusion_mat = confusion_matrix(y_test, predictions)
    classification_rep = classification_report(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)

    print("Confusion Matrix:\n", confusion_mat)
    print("Classification Report:\n", classification_rep)
    print("Accuracy Score:", accuracy)

    # Create a DataFrame with evaluation results
    results_df = pd.DataFrame({
        'Algorithm': [algorithm],
        'FeatureEngineering': [feature_engineering],
        # 'ConfusionMatrix': [confusion_mat],
        'ClassificationReport': [classification_rep],
        'Accuracy': [accuracy]
    })

    # Save the results to a CSV file
    results_filename = f"{algorithm}_{feature_engineering}_results.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"Evaluation results saved as {results_filename}")


# Loading a dataset
data_source_url = "https://raw.githubusercontent.com/satyajeetkrjha/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
airline_tweets = pd.read_csv(data_source_url)


# Data cleaning
features = airline_tweets.iloc[:, 10].apply(clean_text)
labels = airline_tweets.iloc[:, 1].values


# Converting textual data to numerical vectors
processed_features, vectorizer = vectorize_text(features)


# Testing the model with a new example
while True:
    # Get user input for the model type and feature engineering method
    model_type = input("Enter the model type you want to train (RandomForest, LogisticRegression, SVM, NaiveBayes), or type 'exit' to end: ")

    # Check if the user wants to exit
    if model_type.lower() == 'exit':
        print("Exiting the program.")
        break
    
    # Dividing Data into Training and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

    # Train the selected model
    text_classifier = train_model(X_train, y_train, model_type)
    print(f"{model_type} model is successfully trained")

    # Evaluate the model
    evaluate_model(text_classifier, X_test, y_test, model_type, 'tfidf')

    # Testing the model with a new example
    while True:
        new_example = input("Enter the example you want to classify (or type 'exit' to end): ")

        if new_example.lower() == 'exit':
            print("Exiting the model testing.")
            break

        processed_new_example = clean_text(new_example)
        new_example_vectorized = vectorizer.transform([processed_new_example]).toarray()

        new_example_prediction = text_classifier.predict(new_example_vectorized)

        print("Predicted Sentiment for the new example:", new_example_prediction[0])
        print("\n")