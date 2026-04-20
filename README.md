# 🎙️ GhostDictation: Multi-Cloud Voice Assistant

Professional Edge-to-Cloud dictation tool for Windows, built with Python. 

## 🚀 Architectural Highlights
* **Multi-Cloud Routing:** Uses **Groq API (Whisper-large-v3)** for ultra-low latency transcription (< 1s) and **Google Gemini API** for semantic correction and professional styling.
* **Resilience Pattern:** Implements a **Cascade Fallback / Circuit Breaker**. If the primary AI model fails or hits a rate limit, the system gracefully degrades to a faster backup model, and ultimately to raw text output.
* **Security:** Strict separation of concerns. All API keys are securely loaded via `.env` file, isolating them from the application logic.
* **Native Integration:** Bypasses OS keyboard input issues by utilizing direct clipboard pasting (`pyperclip`), ensuring 100% accuracy for Unicode characters (Slovak diacritics).

## ⌨️ Usage
* `Hold F8` - Raw dictation (Lightning fast, ideal for prompt engineering).
* `Hold ALT + F8` - AI correction (Records audio, transcribes, and reformats into professional corporate language).
