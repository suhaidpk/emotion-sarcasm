import cv2
from deepface import DeepFace
import pyttsx3
import time

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 165)
engine.setProperty('volume', 1.0)

# 🗣️ Voice gender/mood selection
voices = engine.getProperty('voices')
# 0 = male (usually), 1 = female (depends on OS)
engine.setProperty('voice', voices[1].id)  # change 1 → 0 for male voice

# Proverbs or relief quotes for each emotion
emotion_quotes = {
    'angry': "Anger is one letter short of danger. Take a deep breath.",
    'disgust': "Disgust passes quickly, but kindness lasts longer.",
    'fear': "Do not fear failure. Fear never trying.",
    'happy': "Don’t be too happy, life is balance. Enjoy this beautiful moment.",
    'neutral': "Show some emotion you idiot.",
    'sad': "Even the darkest night will end, and the sun will rise again.",
    'surprise': "Life is full of surprises. Smile and welcome them."
}

# Initialize webcam
cap = cv2.VideoCapture(0)
print("Starting camera... Press 'q' to quit.")

# Emotion tracking variables
last_emotion = None
emotion_start_time = 0
spoke_quote = False
subtitle_text = ""
subtitle_start_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze emotions in real-time
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion'].lower()
        emotions = result[0]['emotion']

        # Display detected emotion
        cv2.putText(frame, f'Emotion: {dominant_emotion}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show all emotions
        y0 = 90
        for emo, val in emotions.items():
            cv2.putText(frame, f"{emo}: {val:.2f}%", (50, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y0 += 25

        # Emotion stability check
        current_time = time.time()
        if dominant_emotion != last_emotion:
            last_emotion = dominant_emotion
            emotion_start_time = current_time
            spoke_quote = False  # reset
        else:
            # Emotion stayed same for 5 seconds
            if (current_time - emotion_start_time >= 5):
                quote = emotion_quotes.get(dominant_emotion, "")
                if quote:
                    if not spoke_quote:
                        print(f"🧠 Emotion stable ({dominant_emotion}). Saying: {quote}")
                        engine.say(quote)
                        engine.runAndWait()
                        subtitle_text = quote
                        subtitle_start_time = current_time
                        spoke_quote = True
                    else:
                        # Only print after the first time
                        print(f"🙂 Still {dominant_emotion}: {quote}")
                emotion_start_time = current_time  # reset timer for next 5s cycle

        # Show subtitle for 5 seconds
        if subtitle_text and (time.time() - subtitle_start_time < 5):
            cv2.putText(frame, subtitle_text, (30, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            subtitle_text = ""

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Emotion Detector (DeepFace + Subtitles + Relief Quotes)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
