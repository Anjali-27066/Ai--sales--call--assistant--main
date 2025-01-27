import os
import json
import time
import pyaudio
import speech_recognition as sr
import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login
from product_recommender import ProductRecommender
from objection_handler import ObjectionHandler
from sentence_transformers import SentenceTransformer
from vosk import Model, KaldiRecognizer
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt
import pandas as pd

# Load environment variables
load_dotenv()

# Paths to models and data
vosk_model_path = r"C:\Users\anjali\Desktop\infosys project\vosk-model-small-en-us-0.15"
beauty_products_path = r"C:\Users\anjali\Desktop\infosys project\beauty_products_no_relevance.csv"
objections_path = r"C:\Users\anjali\Desktop\infosys project\objections.csv"

# Google Sheets authentication
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
credentials_path = r"C:\Users\anjali\Desktop\infosys project\google_service_account.json"
credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
gc = gspread.authorize(credentials)

# Open or create the Google Sheet
try:
    spreadsheet = gc.open("AI Dashboard History")
    sheet = spreadsheet.sheet1
except gspread.exceptions.SpreadsheetNotFound:
    print("Spreadsheet not found. Creating a new one...")
    spreadsheet = gc.create("AI Dashboard History")
    sheet = spreadsheet.sheet1
    
    # Share the new spreadsheet with the service account email
    service_account_email = "your-service-account@your-project-id.iam.gserviceaccount.com"
    spreadsheet.share(service_account_email, perm_type="user", role="writer")
    print("New spreadsheet created and shared successfully.")

# Initialize product recommender and objection handler
product_recommender = ProductRecommender(beauty_products_path)
objection_handler = ObjectionHandler(objections_path)

# Hugging Face API key
huggingface_api_key = "huggingface_api_key"
login(token=huggingface_api_key)

# Load sentiment analysis model from Hugging Face
model_name = "tabularisai/multilingual-sentiment-analysis"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Initialize Vosk speech recognition model
if os.path.exists(vosk_model_path):
    vosk_model = Model(vosk_model_path)
    recognizer = KaldiRecognizer(vosk_model, 16000)
    print("Vosk model loaded successfully.")
else:
    raise ValueError("Invalid Vosk model path.")

# Initialize audio stream
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=4000)
stream.start_stream()

# Initialize transcribed_chunks in session_state if it doesn't exist
if "transcribed_chunks" not in st.session_state:
    st.session_state.transcribed_chunks = []

# Helper functions
def preprocess_text(text):
    return text.strip().lower()

def analyze_sentiment(text):
    try:
        if not text.strip():
            return "NEUTRAL", 0.0

        processed_text = preprocess_text(text)
        result = sentiment_analyzer(processed_text)[0]

        sentiment_map = {
            'Very Negative': "NEGATIVE",
            'Negative': "NEGATIVE",
            'Neutral': "NEUTRAL",
            'Positive': "POSITIVE",
            'Very Positive': "POSITIVE"
        }

        sentiment = sentiment_map.get(result['label'], "NEUTRAL")
        return sentiment, result['score']

    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return "NEUTRAL", 0.5

def save_sentiment_analysis_to_google_sheets(speech_text, sentiment, score):
    try:
        sheet.append_row([speech_text, sentiment, score, "Sentiment Analysis", time.ctime()])
        print("Sentiment analysis saved to Google Sheets successfully.")
    except Exception as e:
        print(f"Error saving sentiment analysis to Google Sheets: {e}")

def save_call_details_to_google_sheets(full_text, sentiment, score, summary_text):
    try:
        sheet.append_row([full_text, sentiment, score, "Call Details", time.ctime(), summary_text])
        print("Call details saved to Google Sheets successfully.")
    except Exception as e:
        print(f"Error saving call details to Google Sheets: {e}")

def save_product_recommendations_to_google_sheets(user_input, recommendations):
    try:
        sheet.append_row([user_input, ", ".join(recommendations), "Product Recommendations", time.ctime()])
        print("Product recommendations saved to Google Sheets successfully.")
    except Exception as e:
        print(f"Error saving product recommendations to Google Sheets: {e}")

def plot_sentiment_pie_chart(sentiment_counts):
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
    st.pyplot(fig)

# Streamlit interface with sidebar
st.title("Dashboard")
st.markdown("Ready to assist! What can I do for you today?")
st.write("Real-time--Analysis")

# Create a sidebar for tabs
with st.sidebar:
    st.subheader("Navigation")
    selected_tab = st.radio("Select a page:", ("Sentiment Analysis", "Call Details and Summary", "Product Recommendations", "Dashboard History"))

# Display the selected tab content
if selected_tab == "Sentiment Analysis":
    st.header("Sentiment Analysis")

    if st.button("Start Sentiment Analysis"):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        st.info("Listening...")

        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)

                st.info("Analyzing...")

                # Convert speech to text
                speech_text = recognizer.recognize_google(audio)
                st.write("Transcription:", speech_text)

                # Append transcribed text to session_state for future processing
                st.session_state.transcribed_chunks.append(speech_text)

                # Sentiment analysis
                sentiment, score = analyze_sentiment(speech_text)
                st.write(f"Sentiment: {sentiment} (Score: {score})")

                # Save sentiment analysis data to Google Sheets
                save_sentiment_analysis_to_google_sheets(speech_text, sentiment, score)

        except Exception as e:
            st.error(f"Error: {e}")

elif selected_tab == "Call Details and Summary":
    st.header("Call Details and Summary")

    if st.button("Generate Call Summary"):
        if st.session_state.transcribed_chunks:  # Ensure there are chunks to summarize
            # Concatenate all the transcribed text
            full_text = " ".join(st.session_state.transcribed_chunks)

            # Sentiment analysis for the entire conversation
            sentiment, score = analyze_sentiment(full_text)
            st.write(f"Sentiment for the entire call: {sentiment} (Score: {score})")

            # Initialize sentiment counts (default to 0 for each sentiment type)
            sentiment_counts = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}

            # Ensure sentiment is one of the expected values
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
            else:
                sentiment_counts["NEUTRAL"] += 1  # Default to "NEUTRAL" if invalid sentiment

            # Plot pie chart of sentiments for the entire call
            plot_sentiment_pie_chart(sentiment_counts)

            # Summarization (example using Hugging Face summarizer)
            summarizer = pipeline("summarization")
            summary = summarizer(full_text, max_length=150, min_length=50, do_sample=False)
            summary_text = summary[0]['summary_text']
            st.write("Summary:", summary_text)

            # Highlight if product details are mentioned (e.g., 'hair serum details')
            if "hair serum" in full_text.lower():
                st.warning("Mentioned: Hair serum details")

            # Save to Google Sheets
            save_call_details_to_google_sheets(full_text, sentiment, score, summary_text)

        else:
            st.warning("No transcription data available to summarize.")

elif selected_tab == "Product Recommendations":
    st.header("Product Recommendations")

    user_input = st.text_input("Enter a keyword (e.g., moisturizer, serum):", "")

    if st.button("Get Recommendations"):
        try:
            if user_input.strip():
                # Get product recommendations
                recommendations = product_recommender.get_recommendations(user_input.strip())
                
                if recommendations:
                    st.write("*Recommended Products:*")
                    sentiment_counts = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}  # Initialize sentiment counts
                    
                    # Perform sentiment analysis on each recommendation
                    for product in recommendations:
                        sentiment, score = analyze_sentiment(product)
                        st.write(f"- {product} | Sentiment: {sentiment} (Score: {score})")
                        
                        # Update sentiment counts
                        if sentiment in sentiment_counts:
                            sentiment_counts[sentiment] += 1
                        else:
                            sentiment_counts["NEUTRAL"] += 1  # Default to "NEUTRAL" if invalid sentiment

                    # Plot sentiment pie chart for the recommendations
                    plot_sentiment_pie_chart(sentiment_counts)

                    # Save recommendations and sentiment to Google Sheets
                    save_product_recommendations_to_google_sheets(user_input, recommendations)
                else:
                    st.warning("No products found for the given keyword.")
            else:
                st.warning("Please enter a keyword to get recommendations.")
        except Exception as e:
            st.error(f"Error in fetching recommendations: {e}")


elif selected_tab == "Dashboard History":
    st.header("History of Analysis and Recommendations")

    if sheet:
        # Fetch all data from the sheet
        data = sheet.get_all_records()
        
        if data:
            # Convert the data to a pandas DataFrame for better display
            df = pd.DataFrame(data)
            
            # Display the data in a table
            st.dataframe(df)
        else:
            st.warning("No data available in Google Sheets.")
    else:
        st.warning("Google Sheets not connected.")
