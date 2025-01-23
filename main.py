from speech_recognition_handler import recognize_speech
from sentiment_analysis import analyze_text
from text_processing import extract_key_points, generate_summary
from product_recommend import recommend_product

def main():
    while True:
        text = recognize_speech()
        if text and text.lower() == "stop":
            print("Stopping real-time analysis...")
            break

        if text:
            sentiment = analyze_text(text)
            key_points = extract_key_points(text)

            summary = generate_summary(text, sentiment, key_points)
            recommendations = recommend_product(key_points)

            # ✅ Print output in correct order
            print(f"\nSentiment: {sentiment}")
            print(f"\nKey Points: {', '.join(key_points)}")
            print(f"\n{summary}")

            print("\nProduct Recommendations:")
            for rec in recommendations:
                print(f"- {rec['name']}, Price: ${rec['price']}, Category: {rec['category']}")

            print("\nObjection Response:")
            print("Hello! I'm happy to help you find a product that fits your needs. "
                  "Expensive products are often sought after due to their quality, "
                  "but it's important to consider your personal preferences and budget.")

if __name__ == "__main__":
    main()
