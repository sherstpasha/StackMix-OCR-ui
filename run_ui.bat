@echo off
echo ========================================
echo  StackMix OCR Web Interface
echo ========================================
echo.

REM Активация виртуального окружения
if exist "env\Scripts\activate.bat" (
    echo [*] Активация виртуального окружения...
    call env\Scripts\activate.bat
) else (
    echo [!] Виртуальное окружение не найдено!
    echo [*] Создание виртуального окружения...
    python -m venv env
    call env\Scripts\activate.bat
    echo [*] Установка зависимостей...
    pip install -r requirements.txt
)

echo.
echo [*] Запуск веб-интерфейса...
echo [*] После запуска откройте браузер по адресу: http://localhost:7860
echo.

python app.py

pause
