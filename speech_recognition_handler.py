import speech_recognition as sr  # ✅ Correct package name

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Reduce background noise impact

        try:
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=10)  # Extended listening time
            text = recognizer.recognize_google(audio)
            print(f"Recognized Text: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, could not understand the audio.")
        except sr.RequestError:
            print("Could not request results, check your internet connection.")
        except sr.WaitTimeoutError:
            print("No speech detected, try again.")

    return None
