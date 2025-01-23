import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Define stop words (to filter out greetings & irrelevant words)
STOP_WORDS = {"hi", "hello", "hey", "this", "is", "my", "name", "i", "am", "how", "do", "you"}

def extract_key_points(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)

    # Extract only relevant nouns and adjectives, removing stop words
    key_points = [word.lower() for word, tag in tagged_words 
                  if tag in ["NN", "NNP", "JJ"] and word.lower() not in STOP_WORDS]

    return key_points if key_points else ["general"]  # Prevent empty key points

def generate_summary(text, sentiment, key_points):
    summary = f"{text}.\n\nSummary:\nUser is interested in {', '.join(key_points)}.\n"
    
    if sentiment == "Positive":
        summary += "The sentiment is positive, indicating enthusiasm."
    else:
        summary += "The sentiment is negative, indicating some concerns."

    return summary
