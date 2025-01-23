from textblob import TextBlob

def analyze_text(text):
    if not text.strip():
        return "Neutral"  # Prevent crash on empty input
    
    blob = TextBlob(text)
    sentiment = "Positive" if blob.sentiment.polarity > 0 else "Negative"
    return sentiment
