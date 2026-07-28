@echo off
chcp 65001 >nul
title BCBIST AI V2

echo ============================================
echo   BCBIST AI V2 baslatiliyor
echo ============================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo Sanal ortam olusturuluyor...
    python -m venv venv
    if errorlevel 1 (
        echo Python bulunamadi. Python'u yukleyip bu dosyayi tekrar calistirin.
        pause
        exit /b 1
    )
)

if not exist "venv\Scripts\streamlit.exe" (
    echo Gerekli kutuphaneler yukleniyor. Bu islem ilk seferde birkac dakika surebilir...
    venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Kutuphaneler yuklenemedi. Internet baglantisini kontrol edip tekrar deneyin.
        pause
        exit /b 1
    )
)

echo.
echo V2 adresi: http://localhost:8502/bcbistv2
venv\Scripts\python.exe -m streamlit run src\presentation\dashboard\v2_app.py --server.port 8502 --server.baseUrlPath bcbistv2
