"""
Fake News Classifier - Gradio Web Application
Deployed ML model for classifying news articles as Fake or Real
"""

import gradio as gr
import joblib
import re
import warnings
warnings.filterwarnings('ignore')

# Load the trained model
MODEL_PATH = 'models/best_model.pkl'

print("Loading model...")
try:
    model = joblib.load(MODEL_PATH)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("Please ensure you've run the training notebook first!")
    exit(1)

def preprocess_text(text):
    """
    Clean and preprocess text (same as training)
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def predict_news(article_text):
    """
    Predict if news article is Fake or Real
    
    Args:
        article_text (str): News article text
        
    Returns:
        tuple: (prediction_label, confidence_dict)
    """
    if not article_text or article_text.strip() == "":
        return "Please enter a news article to classify.", {"Fake": 0.0, "Real": 0.0}
    
    try:
        # Preprocess text
        cleaned_text = preprocess_text(article_text)
        
        # Make prediction
        prediction = model.predict([cleaned_text])[0]
        probabilities = model.predict_proba([cleaned_text])[0]
        
        # Get confidence scores
        # Model classes: ['Fake', 'Real']
        classes = model.classes_
        confidence_dict = {
            classes[0]: float(probabilities[0]),
            classes[1]: float(probabilities[1])
        }
        
        # Format output
        confidence_pct = max(probabilities) * 100
        
        if prediction == "Fake":
            result = f"🚨 **FAKE NEWS** (Confidence: {confidence_pct:.1f}%)"
        else:
            result = f"✅ **REAL NEWS** (Confidence: {confidence_pct:.1f}%)"
        
        return result, confidence_dict
        
    except Exception as e:
        return f"Error during prediction: {str(e)}", {"Fake": 0.0, "Real": 0.0}

# Example news articles for testing
examples = [
    # Fake news example
    """Breaking: Scientists Discover That Earth is Actually Flat! 
    In a shocking revelation that contradicts centuries of scientific consensus, 
    researchers at the Institute of Alternative Facts have conclusively proven 
    that the Earth is flat. The team used highly questionable methodology and 
    cherry-picked data to reach this groundbreaking conclusion. Government 
    officials are covering up this truth to maintain control over the population. 
    Wake up sheeple! Share this before it gets deleted!""",
    
    # Real news example
    """WASHINGTON (Reuters) - The U.S. Federal Reserve kept interest rates unchanged 
    on Wednesday and signaled it still plans three rate cuts this year, despite 
    recent data showing inflation remains elevated. The central bank's policy-setting 
    committee voted unanimously to hold its benchmark overnight interest rate in the 
    current 5.25%-5.50% range, where it has been since July. Fed Chair Jerome Powell 
    said in a press conference that officials still expect to lower rates later this 
    year, though the timing will depend on incoming economic data.""",
    
    # Fake news example
    """SHOCKING: Celebrity Endorses Miracle Weight Loss Pill That Doctors HATE! 
    You won't believe what happened when this famous actress took these amazing 
    pills! She lost 50 pounds in just 2 weeks without any exercise! Doctors are 
    furious because this one weird trick is putting them out of business. Big Pharma 
    is trying to hide this from you! Click here now before this offer disappears! 
    Limited time only! Act fast!"""
]

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Fake News Classifier") as demo:
    
    gr.Markdown(
        """
        # 📰 Fake News Classifier
        ### Classify news articles as **Fake** or **Real** using Machine Learning
        
        This application uses a trained **Logistic Regression** model with **TF-IDF vectorization** 
        to analyze news articles and predict their authenticity.
        
        **How to use:**
        1. Paste a news article in the text box below
        2. Click "Classify Article" 
        3. View the prediction and confidence scores
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            # Input
            article_input = gr.Textbox(
                label="Enter News Article",
                placeholder="Paste your news article here...",
                lines=10,
                max_lines=20
            )
            
            # Classify button
            classify_btn = gr.Button("🔍 Classify Article", variant="primary", size="lg")
            
            # Clear button
            clear_btn = gr.ClearButton([article_input], value="Clear", size="sm")
            
        with gr.Column(scale=1):
            # Output - Prediction
            prediction_output = gr.Markdown(
                label="Prediction",
                value="*Prediction will appear here*"
            )
            
            # Output - Confidence scores
            confidence_output = gr.Label(
                label="Confidence Scores",
                num_top_classes=2
            )
    
    # Examples section
    gr.Markdown("### 📝 Example Articles (Click to try)")
    gr.Examples(
        examples=examples,
        inputs=article_input,
        label="Sample News Articles"
    )
    
    # Model info
    gr.Markdown(
        """
        ---
        **Model Information:**
        - **Algorithm:** Logistic Regression
        - **Vectorization:** TF-IDF (10,000 features)
        - **Accuracy:** ~99.6%
        - **Dataset:** Fake and Real News Dataset
        
        **Note:** This classifier is trained on a specific dataset and may not generalize 
        perfectly to all types of news articles. Use it as a tool to assist in identifying 
        potentially fake news, but always verify information from multiple reliable sources.
        """
    )
    
    # Connect the button to the prediction function
    classify_btn.click(
        fn=predict_news,
        inputs=article_input,
        outputs=[prediction_output, confidence_output]
    )

# Launch the app
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Gradio Web Application...")
    print("="*60)
    
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        show_error=True
    )
