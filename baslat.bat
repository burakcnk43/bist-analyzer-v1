@echo off
chcp 65001 >nul
title BIST AI ANALYZER PRO

echo ============================================
echo   BIST AI ANALYZER PRO - BASLATILIYOR
echo ============================================
echo.

REM Sanal ortamı aktifleştir
echo [1/3] Sanal ortam aktifleştiriliyor...
call venv\Scripts\activate.bat
echo     Python hazir.

REM Dashboard başlat
echo [2/3] Dashboard baslatiliyor...
start "BIST Dashboard" cmd /c "venv\Scripts\streamlit run src\presentation\dashboard\app.py --server.port 8501"
echo     Dashboard: http://localhost:8501

REM API başlat
echo [3/3] API baslatiliyor...
start "BIST API" cmd /c "venv\Scripts\uvicorn src.presentation.api.fastapi_routes:app --host 0.0.0.0 --port 8000 --reload"
echo     API: http://localhost:8000/docs

echo.
echo ============================================
echo   SISTEM CALISIYOR!
echo   Dashboard: http://localhost:8501
echo   API Docs: http://localhost:8000/api/docs
echo ============================================
echo.
echo Kapatmak icin bu pencereyi kapatabilirsiniz.
pause