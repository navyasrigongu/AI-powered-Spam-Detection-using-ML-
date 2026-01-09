import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords (only first time)
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("SpamN.csv", encoding="latin-1")[['v1', 'v2']]
data.columns = ['label', 'message']

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

data['clean_message'] = data['message'].apply(clean_text)

# Feature extraction
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(data['clean_message']).toarray()

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(data['label'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
# model = MultinomialNB()
# model.fit(X_train, y_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# Improve Spam Recall (Advanced)

# Try Logistic Regression:

# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Prediction function
def predict_spam(text):
    text_lower = text.lower()
    spam_keywords = [
    'free', 'win', 'winner', 'prize', 'lottery', 'jackpot',
    'cash', 'reward', 'bonus', 'giveaway', 'claim', 'money',

    'urgent', 'immediately', 'limited time', 'hurry',
    'last chance', 'today only', 'expires',

    'click', 'tap', 'open', 'link', 'subscribe',
    'download', 'verify', 'confirm',

    'account', 'bank', 'debit', 'credit', 'card', 'pin', 'otp',
    'suspended', 'blocked', 'security alert',

    'offer', 'discount', 'deal', 'promotion', 'special offer',
    'exclusive', 'trial', 'free trial',

    'guaranteed', 'risk-free', 'congratulations', 'selected'
]


    count = sum(1 for word in spam_keywords if word in text_lower)
    if count >= 2:
        return "Spam 🚨"


    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned]).toarray()
    result = model.predict(vector)
    return "Spam 🚨" if result[0] == 1 else "Not Spam ✅"
if __name__ == "__main__":
    # Test manually
    #print(predict_spam("Congratulations! You won a free ticket"))
    print(predict_spam("Hey, are we meeting today?"))
