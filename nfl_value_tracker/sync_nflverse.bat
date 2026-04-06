@echo off
set START_DIR=%~dp0
cd /d "%START_DIR%"

echo [NFLVerse Sync] Activating Virtual Environment...
call venv_312\Scripts\activate.bat

echo [NFLVerse Sync] Running Pipeline...
python pipeline.py

echo [NFLVerse Sync] Exporting Power BI CSVs...
python export_powerbi.py

echo [NFLVerse Sync] Complete!
