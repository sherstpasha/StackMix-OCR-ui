# StackMix OCR Web Interface Launcher
# PowerShell script for easy launch

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " StackMix OCR Web Interface" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (Test-Path "env\Scripts\Activate.ps1") {
    Write-Host "[*] Activating virtual environment..." -ForegroundColor Green
    & "env\Scripts\Activate.ps1"
} else {
    Write-Host "[!] Virtual environment not found!" -ForegroundColor Yellow
    Write-Host "[*] Creating virtual environment..." -ForegroundColor Green
    python -m venv env
    
    & "env\Scripts\Activate.ps1"
    
    Write-Host "[*] Installing dependencies..." -ForegroundColor Green
    pip install --upgrade pip
    pip install -r requirements.txt
}

Write-Host ""
Write-Host "[*] Starting web interface..." -ForegroundColor Green
Write-Host "[*] Open your browser at: http://localhost:7860" -ForegroundColor Yellow
Write-Host ""

python app.py

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
