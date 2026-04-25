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

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

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

    openrouter_client = None
    if OPENROUTER_API_KEY:
        openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )
        logging.info("OpenRouter klient inicializovaný (záloha pre SK korektúru).")
    else:
        logging.warning("OPENROUTER_API_KEY nenájdený — OpenRouter záloha vypnutá.")
    
    print(f"\n  ✅  WhisperDictation beží (Single-Cloud verzia).")
    print(f"  Drž [{HOTKEY.upper()}] pre štandardný prepis (Groq) — pusti pre napísanie textu.")
    print(f"  Drž [{AI_HOTKEY.upper()}] pre prepis + AI gramatiku (Groq -> Llama3) — pusti pre napísanie textu.")
    print(f"  Stlač [Ctrl+C] v tejto konzole pre ukončenie.\n")
    
    system_instruction = "Si profesionálny korektor VÝLUČNE pre slovenský jazyk. NIKDY nepoužiješ češtinu. Slovenčina ≠ čeština. Zakázané české znaky v tvojom výstupe: ě, ů. Zakázané české slová: moc, díky, hezky, omlouvám, není, ale, protože, jenom, taky, nějak, vůbec, víte, říkám, řekl, může, musí, věc. Ak sa tieto slová vyskytnú vo vstupe — sú to chyby Whisper-u — VŽDY ich nahraď slovenským ekvivalentom. Oprav gramatiku a štylistiku, zachovaj všetky fakty a význam. Vráť IBA čistý opravený slovenský text. Žiadne vysvetlivky, žiadny úvod, žiadne formátovanie."
    
    # Zoznam modelov od najsilnejšieho po najrýchlejší (Kaskáda)
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
                        logging.warning(f"Model {model_name} vrátil češtinu! Skúšam ďalší model...")
                        continue
                    return result
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
                      model="whisper-large-v3-turbo",
                      language=LANGUAGE,
                      prompt="Slovenčina. Bežný hovorený text. čšžýáíéúäôľščťžýáíé vedľa, veľakrát, neprídem, prídeme, ďakujem, môžem, vôbec, väčší, prídem, tíšší, nôž, môj, tvoj, stroj"
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