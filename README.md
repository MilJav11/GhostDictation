# 🎙️ Ghost Dictation — AI Voice Dictation for Windows

A lightweight Python dictation tool for Windows with cloud-powered 
transcription and intelligent Slovak grammar correction.

## 🏗️ Architecture: Dual-Cloud (Groq LPU + OpenRouter)

- **Transcription:** `whisper-large-v3-turbo` via Groq LPU — ultra-fast SK speech-to-text
- **AI Correction:** Model cascade with automatic fallback:
  1. `llama-3.3-70b-versatile` (Groq) — primary
  2. `llama-3.1-8b-instant` (Groq) — fast fallback
  3. `google/gemma-3-27b-it` (OpenRouter) — SK language fallback
- **Czech contamination guard:** Auto-detects and retries if AI returns Czech

## ⌨️ Hotkeys
- **[F8]** — Transcription only (Whisper → paste)
- **[ALT+F8]** — Transcription + Slovak grammar correction (Whisper → Llama3 → paste)

## 🛠️ Installation
1. `python -m venv venv`
2. `venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. Create `.env` with your keys:
```
GROQ_API_KEY=sk-gsk-...
OPENROUTER_API_KEY=sk-or-...  # optional, enables SK fallback model
```
5. `python main.py`

## 🛡️ Security
- API keys managed via `.env` (excluded from Git via `.gitignore`)
- Zero hallucination policy via strict system prompting
- Clipboard-based paste — full Unicode/diacritics support
