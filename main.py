import os
import sys
import pyaudio
from google.cloud import speech
from google.api_core.exceptions import ServiceUnavailable
from datetime import datetime
import psutil
import pyttsx3
import platform
import random
import requests
import threading

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
client = speech.SpeechClient()
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "bUVaaEEK4hfvE2ekIRTdM0eYA3Schtbk")

RATE = 16000
CHUNK = int(RATE / 20)  

tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 170)

tts_engine.say("")
tts_engine.runAndWait()

def speak(text):
    """Threaded speech so text prints immediately while speaking."""
    def _speak():
        tts_engine.say(text)
        tts_engine.runAndWait()
    print(f"Assistant: {text}")
    sys.stdout.flush()
    threading.Thread(target=_speak, daemon=True).start()

def mic_stream():
    audio_interface = pyaudio.PyAudio()
    stream = audio_interface.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    try:
        while True:
            yield speech.StreamingRecognizeRequest(audio_content=stream.read(CHUNK))
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()
        raise StopIteration

def get_greeting():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning!"
    elif 12 <= hour < 17:
        return "Good afternoon!"
    elif 17 <= hour < 21:
        return "Good evening!"
    else:
        return "Hello!"

def ask_mistral(query):
    try:
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "mistral-small-latest",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are KindStep, a helpful voice assistant for visually impaired users. Give SHORT, CLEAR answers in 1-2 sentences maximum. Be conversational, friendly, and concise. Speak naturally as if talking to someone who is listening, not reading. Keep responses under 30 words when possible."
                },
                {"role": "user", "content": query}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        response = requests.post(url, headers=headers, json=data, timeout=15)
        response.raise_for_status()
        response_json = response.json()
        return response_json['choices'][0]['message']['content'].strip()
    except requests.exceptions.Timeout:
        return "Sorry, the model is taking too long to respond. Please try again."
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return "Sorry, I'm having trouble connecting to my AI service right now."
    except KeyError:
        return "Sorry, I received an unexpected response."
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "Sorry, something went wrong. Please try again."

def handle_query(query):
    query_lower = query.lower()

    if "exit" in query_lower or "quit" in query_lower:
        speak("Exiting the assistant. Goodbye!")
        sys.exit(0)

    elif any(word in query_lower for word in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]):
        speak(get_greeting())

    elif any(phrase in query_lower for phrase in ["time", "what time", "current time", "tell me the time", "time is it"]):
        current_time = datetime.now().strftime('%I:%M %p')
        speak(f"The current time is {current_time}")

    elif any(phrase in query_lower for phrase in ["date", "day is it", "what day", "today", "what's the date", "today's date"]):
        today = datetime.now().strftime('%A, %B %d, %Y')
        speak(f"Today is {today}")

    elif "battery" in query_lower:
        try:
            battery = psutil.sensors_battery()
            if battery:
                percent = battery.percent
                plugged = "charging" if battery.power_plugged else "not charging"
                speak(f"Battery is at {percent:.0f} percent and {plugged}")
            else:
                speak("Battery information not available.")
        except:
            speak("Battery information not available on this system.")

    elif any(phrase in query_lower for phrase in ["your name", "who are you", "what's your purpose", "purpose", "what do you do"]):
        responses = [
            "My name is KindStep. I'm your AI-powered voice assistant.",
            "I am KindStep, your personal assistant powered by artificial intelligence.",
            "I am here to help you with daily tasks and answer your questions.",
            "I'm KindStep. I assist with time, date, battery status, and can answer any questions you have."
        ]
        speak(random.choice(responses))

    elif any(phrase in query_lower for phrase in ["how are you", "how's it going", "how do you do", "what's up"]):
        responses = [
            "I am doing great, thank you!",
            "I'm fine, how about you?",
            "All good here. Hope you're having a nice day!",
            "I'm doing well! How can I help you today?"
        ]
        speak(random.choice(responses))

    elif "help" in query_lower:
        speak("You can ask me about time, date, battery, or any question you have. I'm powered by AI so feel free to ask me anything. Say exit to quit.")

    else:
        answer = ask_mistral(query)
        speak(answer)

try:
    print("\n" + "═" * 70)
    print("KINDSTEP ASSISTANT")
    print("═" * 70)
    sys.stdout.flush()

    ask_mistral("Hello, please initialize the assistant.")

    responses = client.streaming_recognize(
        speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=RATE,
                language_code="en-US"
            ),
            interim_results=True
        ),
        requests=mic_stream()
    )

    transcript_count = 0

    for response in responses:
        for result in response.results:
            transcript = result.alternatives[0].transcript
            confidence = result.alternatives[0].confidence if result.is_final else 0

            if not result.is_final:
                sys.stdout.write(f"\r💬 {transcript}...")
                sys.stdout.flush()
            else:
                sys.stdout.write("\r" + " " * 100 + "\r")
                sys.stdout.flush()

                transcript_count += 1
                timestamp = datetime.now().strftime('%H:%M:%S')
                confidence_str = f"{confidence * 100:.0f}%" if confidence > 0 else "N/A"

                print(f"\n[{timestamp}] You: {transcript}")
                print(f"           └─ Confidence: {confidence_str}")
                sys.stdout.flush()

                handle_query(transcript)

                print("─" * 70)
                sys.stdout.flush()

except ServiceUnavailable:
    print("\n" + "═" * 70)
    print("CONNECTION ERROR")
    print("═" * 70)
    print("The connection to Google Speech API was lost.")
    print("Please check your internet connection and try again.")
    print("═" * 70 + "\n")
    sys.stdout.flush()
    speak("Connection error. Please check your internet.")

except KeyboardInterrupt:
    print("\n\n" + "═" * 70)
    print("KINDSTEP SHUTTING DOWN")
    print("═" * 70)
    print(f"  Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if 'transcript_count' in locals():
        print(f"  Total interactions: {transcript_count}")
    print("═" * 70 + "\n")
    sys.stdout.flush()
    speak("Goodbye!")

except Exception as e:
    print("\n" + "═" * 70)
    print("UNEXPECTED ERROR")
    print("═" * 70)
    print(f"Error: {str(e)}")
    print("═" * 70 + "\n")
    sys.stdout.flush()
    speak("An unexpected error occurred. Please restart the assistant.")
