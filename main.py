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
import google.generativeai as genai
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GROQ_API_KEY or not GEMINI_API_KEY:
    logging.error("Chýba GROQ_API_KEY alebo GEMINI_API_KEY v .env súbore!")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

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
    
    logging.info("Inicializujem API klientov...")
    client = Groq(api_key=GROQ_API_KEY)
    logging.info("Klienti inicializovaní úspešne.")
    
    print(f"\n  ✅  WhisperDictation beží (Multi-Cloud verzia).")
    print(f"  Drž [{HOTKEY.upper()}] pre štandardný prepis (Groq) — pusti pre napísanie textu.")
    print(f"  Drž [{AI_HOTKEY.upper()}] pre prepis + AI gramatiku (Groq -> Gemini) — pusti pre napísanie textu.")
    print(f"  Stlač [Ctrl+C] v tejto konzole pre ukončenie.\n")
    
    # Príprava Gemini modelov s inštrukciami
    system_instruction = "Si profesionálny asistent. Oprav gramatiku a preštylizuj text do formálnej slovenčiny. Prísne zachovaj všetky fakty (dni, časy, mená). Nevymýšľaj si žiadny text navyše. Nepoužívaj české slová. Vráť IBA čistý, opravený text bez úvodu alebo záveru."
    model_pro = genai.GenerativeModel(model_name="gemini-3.1-pro-preview", system_instruction=system_instruction)
    model_flash = genai.GenerativeModel(model_name="gemini-3-flash-preview", system_instruction=system_instruction)
    
    try:
        while True:
            # 1. Čakáme na stlačenie klávesy v polling loope
            if not keyboard.is_pressed(HOTKEY) and not keyboard.is_pressed(AI_HOTKEY):
                time.sleep(0.01)
                continue
            
            # Zistíme, ktorý hotkey bol stlačený.
            # Skontrolujeme najskôr AI_HOTKEY, lebo môže obsahovať základný HOTKEY (napr. alt+f8 vs f8)
            is_ai_mode = keyboard.is_pressed(AI_HOTKEY)
            
            # 2. Začíname nahrávať
            is_recording = True
            audio_data = []
            beep_start()
            
            trigger_key = AI_HOTKEY if is_ai_mode else HOTKEY
            mode_name = "Gemini AI Korektúra" if is_ai_mode else "Základný prepis (Groq)"
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
                
                # Krok 2: AI Korektúra (Mode 2) s Google Gemini
                if text and is_ai_mode:
                    original_text = text
                    
                    try:
                        # Attempt 1: gemini-3.1-pro-preview
                        logging.info("Aplikujem AI korektúru cez gemini-3.1-pro-preview...")
                        response = model_pro.generate_content(text)
                        
                        if response.text:
                            text = response.text.strip()
                            logging.info(f"Skorigovaný text: {text}")
                        else:
                            raise ValueError("Prázdna odpoveď od Gemini Pro.")
                            
                    except Exception as e:
                        logging.warning(f"Chyba pri gemini-3.1-pro-preview: {e}. Skúšam fallback na gemini-3-flash-preview...")
                        
                        try:
                            # Attempt 2 (Fallback): gemini-3-flash-preview
                            fallback_response = model_flash.generate_content(text)
                            
                            if fallback_response.text:
                                text = fallback_response.text.strip()
                                logging.info(f"Skorigovaný text (Fallback): {text}")
                            else:
                                raise ValueError("Prázdna odpoveď od Gemini Flash.")
                                
                        except Exception as fallback_error:
                            # Attempt 3 (Final Rescue)
                            logging.error(f"Fallback zlyhal: {fallback_error}. Kritická chyba. Používam pôvodný text.")
                            for _ in range(3):
                                winsound.Beep(400, 100)
                            text = original_text

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