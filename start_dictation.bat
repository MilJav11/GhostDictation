@echo off
title Ghost Dictation
echo Zapinam virtualne prostredie a startujem diktafon...
cd /d "C:\Users\a\Desktop\GhostDictation"
call venv\Scripts\activate
python main.py
pause