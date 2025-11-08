import cv2
from deepface import DeepFace
import pyttsx3
import time
import threading
from PIL import ImageFont, ImageDraw, Image
import numpy as np


emotion_quotes = {
    'angry': "Whoa, chill out buddy. Want a Snickers?",
    'disgust': "did u fart",
    'fear': "scaredy cat",
    'happy': "Someone’s too happy today. u r going to regret it",
    'neutral': "Are u dead",
    'sad': "haaa...haaa...haaa",
    'surprise': "what u never seen this"
}


last_spoken_time = {}
COOLDOWN = 10  


def speak_quote(quote):
    """Run TTS safely in background without blocking or overlap."""
    def _speak():
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            engine.setProperty('voice', voices[1].id)  
            engine.setProperty('rate', 195)          
            engine.setProperty('volume', 1.0)
            engine.say(quote)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print("Speech error:", e)

    threading.Thread(target=_speak, daemon=True).start()



cap = cv2.VideoCapture(0)
print("Starting camera... Press 'q' to quit.")


last_emotion = None
emotion_start_time = 0
subtitle_text = ""
subtitle_start_time = 0


try:
    font_path = "C:\\Windows\\Fonts\\times.ttf"  
    subtitle_font = ImageFont.truetype(font_path, 18)
except:
    subtitle_font = ImageFont.load_default()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion'].lower()
        emotions = result[0]['emotion']

      
        cv2.putText(frame, f'Emotion: {dominant_emotion}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

      
        y0 = 90
        for emo, val in emotions.items():
            cv2.putText(frame, f"{emo}: {val:.2f}%", (50, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y0 += 22

        
        current_time = time.time()
        if dominant_emotion != last_emotion:
            last_emotion = dominant_emotion
            emotion_start_time = current_time
        else:
            if current_time - emotion_start_time >= 5:
                quote = emotion_quotes.get(dominant_emotion, "")
                last_time = last_spoken_time.get(dominant_emotion, 0)

                if quote and (current_time - last_time >= COOLDOWN):
                    print(f"😏 Sarcasm for {dominant_emotion}: {quote}")
                    speak_quote(quote)
                    subtitle_text = quote
                    subtitle_start_time = current_time
                    last_spoken_time[dominant_emotion] = current_time

                emotion_start_time = current_time

        
        if subtitle_text and (time.time() - subtitle_start_time < 5):
           
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            text = subtitle_text
            text_width = subtitle_font.getlength(text)
            text_x = (frame.shape[1] - text_width) // 2
            text_y = frame.shape[0] - 35
            draw.text((text_x, text_y), text, font=subtitle_font, fill=(0, 255, 255))
            frame = np.array(img_pil)
        else:
            subtitle_text = ""

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Emotion Detector (DeepFace + Sarcastic Replies)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
