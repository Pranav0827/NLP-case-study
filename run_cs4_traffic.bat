@echo off
title Pune Traffic Grievance AI Prototype (CS4)
cd /d "%~dp0"
echo ========================================================
echo  Starting Case Study 4: Pune Traffic Management Prototype
echo  (6 Categories, Location NER, Informal Text Robustness)
echo ========================================================
python -m streamlit run cs4_traffic/app.py
pause
