# 😏 Emotion-Sarcasm Detector

Not just an analyzer, but helps you control the emotion with sarcastic replies!

## 📋 Overview

**Emotion-Sarcasm** is a Python-based real-time facial emotion detection system that analyzes your emotions through your webcam and responds with hilarious sarcastic quotes. It's fun, interactive, and a great way to lighten the mood!

### Features

- 🎥 **Real-time Facial Emotion Detection** - Uses DeepFace to detect 7 different emotions
- 🎤 **Text-to-Speech (TTS)** - Speaks out sarcastic quotes based on your detected emotion
- 💬 **Sarcastic Replies** - Witty responses tailored to each emotion:
  - 😠 **Angry**: "Whoa, chill out buddy. Want a Snickers?"
  - 🤮 **Disgust**: "did u fart"
  - 😨 **Fear**: "scaredy cat"
  - 😊 **Happy**: "Someone's too happy today. u r going to regret it"
  - 😐 **Neutral**: "Are u dead"
  - 😢 **Sad**: "haaa...haaa...haaa"
  - 😲 **Surprise**: "what u never seen this"
- 📊 **Emotion Confidence Display** - Shows confidence percentages for all detected emotions
- ⏱️ **Smart Cooldown** - Avoids repeated sarcastic remarks (10-second cooldown per emotion)
- 📝 **On-screen Subtitles** - Displays sarcastic quotes as subtitles on the video feed

## 🛠️ Requirements

- Python 3.7+
- Webcam/Camera
- Windows (for system fonts) or adjustable for other OS

### Dependencies

```
tensorflow==2.15.0
numpy
opencv-python
deepface
pyttsx3
```

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/suhaidpk/emotion-sarcasm.git
   cd emotion-sarcasm
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

## 🚀 Usage

1. **Run the emotion detector**
   ```bash
   python "face emotion.py"
   ```

2. **What happens next:**
   - Your webcam will activate
   - The system detects your facial emotion in real-time
   - After holding the same emotion for 5+ seconds, a sarcastic quote will be spoken out loud
   - The quote appears as a subtitle on the video feed
   - Press `q` to quit the application

## 📝 How It Works

1. **Emotion Detection**: Uses DeepFace to analyze the frame and detect the dominant emotion and confidence levels for all 7 emotions
2. **Emotion Tracking**: Tracks emotion changes and waits 5 seconds of consistent emotion before triggering a response
3. **Sarcasm Generation**: Selects a pre-defined sarcastic quote based on the detected emotion
4. **Text-to-Speech**: Uses pyttsx3 to speak the quote naturally
5. **Display**: Shows the emotion, confidence scores, and subtitle on the video feed

## 🎮 Controls

- **`q` key**: Quit the application

## ⚙️ Configuration

You can customize the following in `face emotion.py`:

- **Sarcastic Quotes**: Modify the `emotion_quotes` dictionary (lines 10-18)
- **Cooldown Period**: Change `COOLDOWN = 10` (line 22) to adjust seconds between quotes
- **Emotion Threshold**: Change the `5` on line 87 to adjust how long an emotion must be held
- **Voice Settings**: Modify lines 31-33 for voice, rate, and volume
- **Font**: Adjust font path on line 55 for different systems

## 🖥️ System Requirements

- **OS**: Windows (primary), Linux/Mac (with font path adjustments)
- **RAM**: 4GB+ recommended
- **GPU**: Optional (speeds up emotion detection)
- **Webcam**: Working USB camera or built-in webcam

## ⚠️ Notes

- The first run may take time to download DeepFace models
- Ensure good lighting for better emotion detection accuracy
- The system works best when your face is clearly visible
- Font path is set for Windows; modify for Linux/Mac if needed

## 🤝 Contributing

Feel free to fork, modify, and improve! Some ideas:
- Add more sarcastic quotes
- Support different languages
- Add emotion history tracking
- Create a GUI interface
- Add emotion statistics

## 📄 License

This project is open source and available for personal and educational use.

## 👨‍💻 Author

Created by [suhaidpk](https://github.com/suhaidpk)

---

**Have fun detecting emotions and enjoy the sarcasm!** 😏
