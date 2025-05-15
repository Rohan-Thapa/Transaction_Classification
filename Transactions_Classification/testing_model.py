import pickle
import nltk
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download the below resources for the NLTK if not already, and using it for the first time.
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')

def nltk_preprocess(texts):
    """
    Tokenize, remove stopwords and lemmatize are done in this function.
    And its name should be similar to the name used in the pipeline of the pickle
    """

    lemmatizer = nltk.WordNetLemmatizer()
    stop_words = set(nltk.corpus.stopwords.words('english'))
    cleaned = []
    for doc in texts:
        tokens = nltk.word_tokenize(doc.lower())
        lemmas = [
            lemmatizer.lemmatize(tok, pos='v')
            for tok in tokens
            if tok.isalpha() and tok not in stop_words
        ]
        cleaned.append(" ".join(lemmas))
    return cleaned

model_filename = 'transaction_classifier.pkl'
with open(model_filename, 'rb') as f:
    loaded_model: Pipeline = pickle.load(f)

print(f"Loaded model from '{model_filename}' successfully.")

if __name__ == "__main__":
    # (description, actual_label) is the format for the test_data below.
    test_data = [
        ("Internet bill payment", "Utilities"),
        ("Movie theater tickets", "Entertainment"),
        ("Coffee at local cafe", "Food"),
        ("Book shopping online", "Shopping"),
        ("Bus ticket to city", "Travelling"),
        ("Mobile topup recharge", "TopUp"),
        ("Charity donation for cause", "Bill Split"),
        ("Random expense", "Others")
    ]

    descriptions = [d for d, _ in test_data]
    actual_labels = [lbl for _, lbl in test_data]

    # Prediction for the provided description.
    predicted_labels = loaded_model.predict(descriptions)

    # Compare and print each of the description details
    correct = 0
    print("\nDescription".ljust(30), "Actual".ljust(15), "Predicted".ljust(15), "Match")
    print("-" * 75)
    for desc, actual, pred in zip(descriptions, actual_labels, predicted_labels):
        match = (actual == pred)
        if match:
            correct += 1
        print(f"{desc.ljust(30)} {actual.ljust(15)} {pred.ljust(15)} {match}")

    # Accuracy calculation for the model accuracy.
    total = len(test_data)
    accuracy = correct / total * 100
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2f}%")
