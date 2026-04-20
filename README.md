# 🎙️ Ghost Dictation: High-Performance AI Voice Assistant

A specialized Python-based dictation tool designed for ultra-low latency transcription and intelligent grammar correction.

## 🏗️ Architecture: Single-Cloud (Groq LPU)
This project utilizes a **Single-Cloud architecture** via **Groq API** to achieve near-instant results by leveraging LPU (Language Processing Units).

* **Transcription:** Powered by `whisper-large-v3-turbo` for high-speed speech-to-text.
* **Intelligence:** Powered by `llama-3.3-70b-versatile` for semantic Slovak grammar correction and formal styling.
* **Resilience:** Implemented a **Model Cascade Fallback** system. If the primary model fails or reaches a limit, the system automatically attempts recovery via a secondary model (`llama-3.1-8b-instant`) before falling back to the raw transcript.

## ⌨️ Global Hotkeys
* **[F8] (Standard Mode):** Immediate transcription of speech to the current cursor position.
* **[ALT + F8] (AI Mode):** Transcription followed by an AI-driven grammar and style check, specifically tuned for formal Slovak business communication.

## 🛡️ Security & Quality
* **Zero-Environment Leaks:** Sensitive API keys are managed exclusively via `.env` files (excluded from Git).
* **Zero Hallucination Policy:** Strict system prompting ensures the AI doesn't invent facts or mix languages (Slovak/Czech).
* **Clipboard Integration:** Uses `pyperclip` for direct OS-level pasting to ensure 100% compatibility with Unicode characters and diacritics.

## 🛠️ Installation
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`.
3. Install dependencies: `pip install -r requirements.txt`.
4. Add your `GROQ_API_KEY` to the `.env` file.
