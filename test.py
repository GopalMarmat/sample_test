import os
import threading
import sounddevice as sd
import numpy as np
import requests
import speech_recognition as sr
from scipy.io.wavfile import write

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle

# ---------------- CONFIG ----------------
SAMPLE_RATE = 44100
AUDIO_FILE = "answer.wav"
Window.size = (480, 700)

recognizer = sr.Recognizer()

# ----------------------------------------

class InterviewApp(App):

    def build(self):
        root = BoxLayout(orientation="vertical", padding=15, spacing=10)

        # Background
        with root.canvas.before:
            Color(0.1, 0.1, 0.1, 1)
            self.rect = Rectangle(pos=root.pos, size=root.size)

        root.bind(pos=self.update_rect, size=self.update_rect)

        # Response label
        self.response_label = Label(
            text="Press Start to record your answer",
            size_hint_y=None,
            halign="left",
            valign="top",
            markup=True,
            color=(1, 1, 1, 1)
        )
        self.response_label.bind(width=self.update_text_width)
        self.response_label.bind(texture_size=self.update_text_height)

        scroll = ScrollView()
        scroll.add_widget(self.response_label)
        root.add_widget(scroll)

        # Buttons
        bar = BoxLayout(size_hint_y=None, height=70, spacing=10)

        self.start_btn = Button(text="Start", background_color=(0.3, 0.6, 0.3, 1))
        self.stop_btn = Button(text="Stop", background_color=(0.7, 0.2, 0.2, 1), disabled=True)
        self.gen_btn = Button(text="Generate", background_color=(0.2, 0.4, 0.7, 1), disabled=True)

        self.start_btn.bind(on_press=self.start_recording)
        self.stop_btn.bind(on_press=self.stop_recording)
        self.gen_btn.bind(on_press=self.generate_response)

        bar.add_widget(self.start_btn)
        bar.add_widget(self.gen_btn)
        bar.add_widget(self.stop_btn)

        root.add_widget(bar)

        self.recording = None
        return root

    # ---------- UI helpers ----------
    def update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def update_text_width(self, instance, width):
        instance.text_size = (width, None)

    def update_text_height(self, instance, size):
        instance.height = size[1]

    # ---------- Audio Recording ----------
    def start_recording(self, _):
        self.response_label.text = "[b]🎙 Recording...[/b]"
        self.start_btn.disabled = True
        self.stop_btn.disabled = False
        self.gen_btn.disabled = True

        self.recording = sd.rec(
            int(60 * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )

    def stop_recording(self, _):
        sd.stop()

        # Convert to int16 for compatibility
        audio_int16 = np.int16(
            self.recording / np.max(np.abs(self.recording)) * 32767
        )

        write(AUDIO_FILE, SAMPLE_RATE, audio_int16)

        self.response_label.text = "[b]✅ Recording saved.[/b]\nTap Generate."
        self.stop_btn.disabled = True
        self.gen_btn.disabled = False
        self.start_btn.disabled = False

    # ---------- AI Pipeline ----------
    def generate_response(self, _):
        self.response_label.text = "[b]⏳ Transcribing & generating feedback...[/b]"
        self.gen_btn.disabled = True
        threading.Thread(target=self.ai_pipeline, daemon=True).start()

    def ai_pipeline(self):
        try:
            transcript = self.transcribe()
            print("Model testing",transcript)
            feedback = self.call_llama(transcript)

            self.response_label.text = (
                "[b] Interviewer Say:[/b]\n"
                f"{transcript}\n\n"
                "[b]🤖 Generated Answer:[/b]\n"
                f"{feedback}"
            )

        except Exception as e:
            self.response_label.text = f"[b]❌ Error[/b]\n{str(e)}"
        finally:
            self.gen_btn.disabled = False

    # ---------- Google Speech Recognition ----------
    def transcribe(self):
        with sr.AudioFile(AUDIO_FILE) as source:
            audio = recognizer.record(source)
            try:
                return recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                return "❌ Could not understand the audio."
            except sr.RequestError:
                return "⚠️ Google API error."

    # ---------- LLaMa Feedback ----------
    import requests

    def call_llama(self,transcript):
        url = "http://localhost:11434/api/chat"

        payload = {
            "model": "phi3:mini",
            "messages": [
                {
                    "role": "system",
                    "content": """You are helpfull assistant, You task is to generate response as per question,
                    Generate reponse in proper structure like below:
                    1. point1.....
                    2. point2.....
                    .
                    .
                    """
                },
                {
                    "role": "user",
                    "content": transcript
                }
            ],
            "stream": False,
            "options": {
                "num_predict": 150
            }
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        return response.json()["message"]["content"]


if __name__ == "__main__":
    InterviewApp().run()
