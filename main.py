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
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)s      %(message)s', datefmt='%H:%M:%S')

# Načítanie konfigurácie
try:
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
except FileNotFoundError:
    logging.error("Subor config.json nebol najdeny!")
    sys.exit(1)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logging.error("Chýba GROQ_API_KEY v .env súbore!")
    sys.exit(1)

HOTKEY = config.get("HOTKEY", "f8")
AI_HOTKEY = config.get("AI_HOTKEY", "alt+f8")
SAMPLE_RATE = config.get("SAMPLE_RATE", 16000)
CHANNELS = config.get("CHANNELS", 1)
LANGUAGE = config.get("LANGUAGE", "sk")

# Globálne premenné pre zvuk
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
    
    logging.info("Inicializujem API klienta...")
    client = Groq(api_key=GROQ_API_KEY)
    logging.info("Klient inicializovaný úspešne.")
    
    print(f"\n  ✅  WhisperDictation beží (Single-Cloud verzia).")
    print(f"  Drž [{HOTKEY.upper()}] pre štandardný prepis (Groq) — pusti pre napísanie textu.")
    print(f"  Drž [{AI_HOTKEY.upper()}] pre prepis + AI gramatiku (Groq -> Llama3) — pusti pre napísanie textu.")
    print(f"  Stlač [Ctrl+C] v tejto konzole pre ukončenie.\n")
    
    system_instruction = "Si profesionálny slovenský jazykový korektor. Komunikuješ výlučne v spisovnej slovenčine. Akýkoľvek pokus o použitie češtiny bude považovaný za chybu. Tvoja štylistika musí byť 100% slovenská. Tvojou jedinou úlohou je upraviť hrubý text z diktafónu do spisovnej, formálnej slovenčiny. ZAKAZUJEM ti používať akékoľvek české slová (ako napr. omlouvám se, díky, moc, hezky). Oprav gramatiku a štylistiku, ale prísne zachovaj všetky fakty a význam. Vráť IBA čistý opravený text. Žiadny úvod, žiadne vysvetlivky, žiadne formátovanie."
    
    # Zoznam modelov od najsilnejšieho po najrýchlejší (Kaskáda)
    model_cascade = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

    def call_groq_with_fallback(text_input):
        for model_name in model_cascade:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": text_input}
                    ]
                )
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content.strip()
            except Exception as e:
                logging.warning(f"Model {model_name} zlyhal: {e}. Skúšam ďalší...")
                continue
        
        logging.error("Všetky modely zlyhali. Kritická chyba. Používam pôvodný text.")
        for _ in range(3):
            winsound.Beep(400, 100)
        return text_input

    try:
        while True:
            # 1. Čakáme na stlačenie klávesy v polling loope
            if not keyboard.is_pressed(HOTKEY) and not keyboard.is_pressed(AI_HOTKEY):
                time.sleep(0.01)
                continue
            
            # Zistíme, ktorý hotkey bol stlačený.
            is_ai_mode = keyboard.is_pressed(AI_HOTKEY)
            
            # 2. Začíname nahrávať
            is_recording = True
            audio_data = []
            beep_start()
            
            trigger_key = AI_HOTKEY if is_ai_mode else HOTKEY
            mode_name = "Llama3 AI Korektúra" if is_ai_mode else "Základný prepis (Groq)"
            logging.info(f"Nahrávam... (Režim: {mode_name}) - Pusti pre prepis")
            
            stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', callback=audio_callback)
            stream.start()
            
            # 3. Kým používateľ drží klávesu, čakáme (polling loop arch)
            while keyboard.is_pressed(trigger_key):
                time.sleep(0.05)
            
            # 4. Klávesa pustená, zastavujeme nahrávanie
            is_recording = False
            stream.stop()
            stream.close()
            beep_stop()
            logging.info("Nahrávanie ukončené. Odosielam do cloudu...")
            
            # 5. Spracovanie audia
            if not audio_data:
                continue
                
            audio_np = np.concatenate(audio_data, axis=0)
            duration = len(audio_np) / SAMPLE_RATE
            if duration < 0.3:
                logging.info("Nahrávka bola príliš krátka, ignorujem.")
                continue
                
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_filename = tmp_file.name
                wav.write(tmp_filename, SAMPLE_RATE, audio_np)
                
            try:
                # Krok 1: Prepis cez Whisper-large-v3 (Groq)
                with open(tmp_filename, "rb") as file:
                    transcription = client.audio.transcriptions.create(
                      file=(tmp_filename, file.read()),
                      model="whisper-large-v3",
                      language=LANGUAGE
                    )
                
                text = transcription.text.strip()
                logging.info(f"Rozpoznaný text (Groq): {text}")
                
                # Krok 2: AI Korektúra (Mode 2) s Groq Llama3
                if text and is_ai_mode:
                    logging.info("Aplikujem AI korektúru (Kaskáda modelov)...")
                    text = call_groq_with_fallback(text)
                    logging.info(f"Výsledný skorigovaný text: {text}")

                if text:
                    type_text(text)
            except Exception as e:
                logging.error(f"Chyba pri spracovaní: {e}")
            finally:
                os.remove(tmp_filename)
                
    except KeyboardInterrupt:
        print("\n  Vypínam aplikáciu...")

if __name__ == "__main__":
    main()