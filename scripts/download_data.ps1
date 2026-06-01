# Download NASA SMAP/MSL telemetry dataset from Kaggle (Telemanom layout).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    Write-Error "Kaggle CLI not found. Install: pip install kaggle"
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host "Downloading nasa-anomaly-detection-dataset-smap-msl..."
& $Python -m kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl -p $Root --unzip

# Kaggle unpacks to data/data/{train,test}; normalize to data/{train,test}
$NestedTrain = Join-Path $Root "data\data\train"
$NestedTest = Join-Path $Root "data\data\test"
$TrainDir = Join-Path $Root "data\train"
$TestDir = Join-Path $Root "data\test"
if (Test-Path $NestedTrain) {
    New-Item -ItemType Directory -Force -Path $TrainDir, $TestDir | Out-Null
    Copy-Item (Join-Path $NestedTrain "*.npy") -Destination $TrainDir -Force
    Copy-Item (Join-Path $NestedTest "*.npy") -Destination $TestDir -Force
    Write-Host "Copied NPY channels into data\train and data\test"
}

$labels = Join-Path $Root "data\labeled_anomalies.csv"
Write-Host "Fetching labeled_anomalies.csv from Telemanom..."
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv" -OutFile $labels

Write-Host "Dataset ready under $Root\data\"
