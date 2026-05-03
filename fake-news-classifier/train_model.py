"""
Train the Fake News Classifier Model
This script trains the model and saves it for deployment
"""

import pandas as pd
import re
import joblib
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TRAINING FAKE NEWS CLASSIFIER")
print("="*60)

# Download stopwords
print("\nDownloading NLTK data...")
try:
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    print("✓ NLTK stopwords downloaded")
except:
    print("✗ Error downloading stopwords")
    stop_words = []

# Load dataset
print("\nLoading dataset...")
df = pd.read_csv('data/fake_and_real_news.csv')
print(f"✓ Dataset loaded: {len(df):,} articles")
print(f"  - Fake: {len(df[df['label']=='Fake']):,}")
print(f"  - Real: {len(df[df['label']=='Real']):,}")

# Preprocessing function
def preprocess_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

# Preprocess text
print("\nPreprocessing text...")
df['cleaned_text'] = df['Text'].apply(preprocess_text)
print("✓ Text preprocessing complete")

# Prepare features and target
X = df['cleaned_text']
y = df['label']

# Train/test split
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

# Build and train Logistic Regression pipeline
print("\nTraining Logistic Regression model...")
print("  - TF-IDF vectorization (max_features=10000)")
print("  - Stopword removal")
print("  - N-gram range: (1, 2)")

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        stop_words=stop_words,
        ngram_range=(1, 2),
        min_df=2
    )),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

pipeline.fit(X_train, y_train)
print("✓ Training complete")

# Evaluate model
print("\nEvaluating model...")
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label='Fake', average='weighted')

print(f"\n{'='*60}")
print("MODEL PERFORMANCE")
print(f"{'='*60}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"F1-Score: {f1:.4f}")
print(f"{'='*60}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
model_path = 'models/best_model.pkl'
print(f"\nSaving model to: {model_path}")
joblib.dump(pipeline, model_path)
print("✓ Model saved successfully!")

# Test the saved model
print("\nTesting saved model...")
loaded_model = joblib.load(model_path)
test_text = "Breaking news: Scientists discover new planet in our solar system!"
prediction = loaded_model.predict([test_text])[0]
probabilities = loaded_model.predict_proba([test_text])[0]

print(f"\nTest Prediction:")
print(f"  Text: '{test_text}'")
print(f"  Prediction: {prediction}")
print(f"  Probabilities: Fake={probabilities[0]:.2%}, Real={probabilities[1]:.2%}")

print("\n" + "="*60)
print("✓ MODEL TRAINING COMPLETE!")
print("="*60)
print(f"\nYou can now run the Gradio app:")
print(f"  python app.py")
print("="*60)
