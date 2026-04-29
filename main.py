import sys
import os
import tempfile
import time
import logging
import json

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import keyboard
import pyautogui
import pyperclip
import winsound

from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s      %(message)s', datefmt='%H:%M:%S')

# Load configuration
try:
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
except FileNotFoundError:
    logging.error("File config.json not found!")
    sys.exit(1)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logging.error("Missing GROQ_API_KEY in .env file!")
    sys.exit(1)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

HOTKEY = config.get("HOTKEY", "f8")
AI_HOTKEY = config.get("AI_HOTKEY", "alt+f8")
SAMPLE_RATE = config.get("SAMPLE_RATE", 16000)
CHANNELS = config.get("CHANNELS", 1)
LANGUAGE = config.get("LANGUAGE", "sk")

# Global variables for audio
audio_data = []
is_recording = False

def beep_start():
    winsound.Beep(1000, 150)

def beep_stop():
    winsound.Beep(500, 150)

def audio_callback(indata, frames, time_info, status):
    if is_recording:
        audio_data.append(indata.copy())

def type_text(text: str):
    if not text:
        return
    time.sleep(0.2)
    pyperclip.copy(text)
    pyautogui.hotkey('ctrl', 'v')

def main():
    global is_recording, audio_data
    
    logging.info("Initializing API client...")
    client = Groq(api_key=GROQ_API_KEY)
    logging.info("Client initialized successfully.")

    openrouter_client = None
    if OPENROUTER_API_KEY:
        openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )
        logging.info("OpenRouter client initialized (backup for SK proofreading).")
    else:
        logging.warning("OPENROUTER_API_KEY not found — OpenRouter backup disabled.")
    
    print(f"\n  ✅  WhisperDictation is running (Single-Cloud version).")
    print(f"  Hold [{HOTKEY.upper()}] for standard transcription (Groq) — release to type text.")
    print(f"  Hold [{AI_HOTKEY.upper()}] for transcription + AI grammar (Groq -> Llama3) — release to type text.")
    print(f"  Press [Ctrl+C] in this console to exit.\n")
    
    system_instruction = "Si profesionálny korektor VÝLUČNE pre slovenský jazyk. NIKDY nepoužiješ češtinu. Slovenčina ≠ čeština. Zakázané české znaky v tvojom výstupe: ě, ů. Zakázané české slová: moc, díky, hezky, omlouvám, není, ale, protože, jenom, taky, nějak, vůbec, víte, říkám, řekl, může, musí, věc. Ak sa tieto slová vyskytnú vo vstupe — sú to chyby Whisper-u — VŽDY ich nahraď slovenským ekvivalentom. Oprav gramatiku a štylistiku, zachovaj všetky fakty a význam. Vráť IBA čistý opravený slovenský text. Žiadne vysvetlivky, žiadny úvod, žiadne formátovanie."
    
    # List of models from strongest to fastest (Cascade)
    model_cascade = [
        {"client": client,            "model": "llama-3.3-70b-versatile"},
        {"client": client,            "model": "llama-3.1-8b-instant"},
        {"client": openrouter_client, "model": "google/gemma-3-27b-it"},
    ]

    def is_czech_contaminated(text: str) -> bool:
        czech_chars = ['ě', 'ů']
        czech_words = ['moc ', 'díky', 'není ', 'jenom', 'taky ', 'vůbec', 'říkám']
        t = text.lower()
        return any(c in t for c in czech_chars) or any(w in t for w in czech_words)

    def call_groq_with_fallback(text_input):
        for entry in model_cascade:
            api_client = entry["client"]
            model_name = entry["model"]
            if api_client is None:
                continue
            try:
                response = api_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": text_input}
                    ]
                )
                if response.choices and response.choices[0].message.content:
                    result = response.choices[0].message.content.strip()
                    if is_czech_contaminated(result):
                        logging.warning(f"Model {model_name} returned Czech! Trying another model...")
                        continue
                    return result
            except Exception as e:
                logging.warning(f"Model {model_name} failed: {e}. Trying another...")
                continue
        
        logging.error("All models failed. Critical error. Using original text.")
        for _ in range(3):
            winsound.Beep(400, 100)
        return text_input

    try:
        while True:
            # 1. Waiting for key press in polling loop
            if not keyboard.is_pressed(HOTKEY) and not keyboard.is_pressed(AI_HOTKEY):
                time.sleep(0.01)
                continue
            
            # Determine which hotkey was pressed.
            is_ai_mode = keyboard.is_pressed(AI_HOTKEY)
            
            # 2. Start recording
            is_recording = True
            audio_data = []
            beep_start()
            
            trigger_key = AI_HOTKEY if is_ai_mode else HOTKEY
            mode_name = "Llama3 AI Proofreading" if is_ai_mode else "Basic transcription (Groq)"
            logging.info(f"Recording... (Mode: {mode_name}) - Release for transcription")
            
            stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', callback=audio_callback)
            stream.start()
            
            # 3. While the user holds the key, wait (polling loop arch)
            while keyboard.is_pressed(trigger_key):
                time.sleep(0.05)
            
            # 4. Key released, stop recording
            is_recording = False
            stream.stop()
            stream.close()
            beep_stop()
            logging.info("Recording finished. Sending to cloud...")
            
            # 5. Process audio
            if not audio_data:
                continue
                
            audio_np = np.concatenate(audio_data, axis=0)
            duration = len(audio_np) / SAMPLE_RATE
            if duration < 0.3:
                logging.info("Recording was too short, ignoring.")
                continue
                
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_filename = tmp_file.name
                wav.write(tmp_filename, SAMPLE_RATE, audio_np)
                
            try:
                # Step 1: Transcription via Whisper-large-v3 (Groq)
                with open(tmp_filename, "rb") as file:
                    transcription = client.audio.transcriptions.create(
                      file=(tmp_filename, file.read()),
                      model="whisper-large-v3-turbo",
                      language=LANGUAGE,
                      prompt="Slovenčina. Bežný hovorený text. čšžýáíéúäôľščťžýáíé vedľa, veľakrát, neprídem, prídeme, ďakujem, môžem, vôbec, väčší, prídem, tíšší, nôž, môj, tvoj, stroj"
                    )
                
                text = transcription.text.strip()
                logging.info(f"Recognized text (Groq): {text}")
                
                # Step 2: AI Proofreading (Mode 2) with Groq Llama3
                if text and is_ai_mode:
                    logging.info("Applying AI proofreading (Model cascade)...")
                    text = call_groq_with_fallback(text)
                    logging.info(f"Final corrected text: {text}")

                if text:
                    type_text(text)
            except Exception as e:
                logging.error(f"Error during processing: {e}")
            finally:
                os.remove(tmp_filename)
                
    except KeyboardInterrupt:
        print("\n  Shutting down application...")

if __name__ == "__main__":
    main()