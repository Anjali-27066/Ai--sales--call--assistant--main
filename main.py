from sentiment_analysis import transcribe_with_chunks
from google_sheets import store_data_in_sheet
from env_setup import config
from product_recommender import ProductRecommender
from objection_handler import ObjectionHandler, load_objections
from sentence_transformers import SentenceTransformer

def main():
    objections_file_path = r"C:\Users\anjali\Downloads\AI-Sales-Call-Assistant--main (1)\AI-Sales-Call-Assistant--main\data\objections.csv"
    recommendations_file_path = r"C:\Users\anjali\Downloads\AI-Sales-Call-Assistant--main (1)\AI-Sales-Call-Assistant--main\data\recommendations.csv"
    
    objections_dict = load_objections(objections_file_path)

    product_recommender = ProductRecommender(recommendations_file_path)
    objection_handler = ObjectionHandler(objections_file_path)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    transcribed_chunks = transcribe_with_chunks(objections_dict)

    total_text = ""
    sentiment_scores = []

    for chunk, sentiment, score in transcribed_chunks:
        if chunk.strip():  
            total_text += chunk + " "
            if sentiment == "POSITIVE" or sentiment == "VERY POSITIVE":
                sentiment_scores.append(score)
            elif sentiment == "NEGATIVE" or sentiment == "VERY NEGATIVE":
                sentiment_scores.append(-score)
            else:
                sentiment_scores.append(0)  

            query_embedding = model.encode([chunk])
            
            product_distances, _ = product_recommender.index.search(query_embedding, 1)
            if product_distances[0][0] < 1.5: 
                recommendations = product_recommender.get_recommendations(chunk)
                if recommendations:
                    print(f"Recommendations for chunk: '{chunk}'")
                    for idx, rec in enumerate(recommendations, 1):
                        print(f"{idx}. {rec}")
            
            objection_distances, _ = objection_handler.index.search(query_embedding, 1)
            if objection_distances[0][0] < 1.5:  # Same threshold as real-time
                objection_responses = objection_handler.handle_objection(chunk)
                if objection_responses:
                    for response in objection_responses:
                        print(f"Objection Response: {response}")

    if sentiment_scores: 
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        overall_sentiment = "POSITIVE" if average_sentiment > 0 else "NEGATIVE" if average_sentiment < 0 else "NEUTRAL"
    else:
        overall_sentiment = "NEUTRAL"

    print(f"Overall Sentiment: {overall_sentiment}")
    print(f"Conversation Summary: {total_text.strip()}")
    
    store_data_in_sheet(config["google_sheet_id"], transcribed_chunks, total_text.strip(), overall_sentiment)

if __name__ == "__main__":
    main()
