$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python -m PyInstaller `
  --noconfirm `
  --clean `
  --onedir `
  --windowed `
  --name DataCollector `
  --add-data "config\cameras.json;config" `
  --add-data "models\yolov8n.pt;models" `
  --hidden-import ultralytics `
  --hidden-import cv2 `
  --hidden-import PIL.Image `
  --hidden-import PIL.ImageTk `
  src\data_collector_app.py

New-Item -ItemType Directory -Force "dist\DataCollector\config" | Out-Null
New-Item -ItemType Directory -Force "dist\DataCollector\models" | Out-Null
Copy-Item -Force "config\cameras.json" "dist\DataCollector\config\cameras.json"
Copy-Item -Force "models\yolov8n.pt" "dist\DataCollector\models\yolov8n.pt"

Write-Host ""
Write-Host "Built: dist\DataCollector\DataCollector.exe"
Write-Host "Config: dist\DataCollector\config\cameras.json"
Write-Host "Model: dist\DataCollector\models\yolov8n.pt"
