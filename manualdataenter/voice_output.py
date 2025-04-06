from gtts import gTTS
import os

def speak_prediction(text):
    tts = gTTS(text)
    tts.save("prediction.mp3")