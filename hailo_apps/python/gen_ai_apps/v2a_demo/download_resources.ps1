$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

# The repo root makes hailo_apps importable in a standalone (not pip-installed) setup.
$env:PYTHONPATH = "$((Resolve-Path "$PSScriptRoot\..\..\..\..").Path);$env:PYTHONPATH"
python -m hailo_apps.installation.download_resources --group v2a_demo --arch hailo10h
if ($LASTEXITCODE -ne 0) {
    throw "Failed to download the demo HEF resources."
}

# mkdir resources
New-Item -ItemType Directory -Force -Path resources | Out-Null

# cd resources
Set-Location resources

$baseUrl = "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/v2a_demo"
$files = @(
    "en_US-joe-medium.onnx",
    "en_US-joe-medium.onnx.json",
    "go_hailo.onnx",
    "hey_hailo.onnx",
    "hey_hailo_v3.onnx",
    "word_embeddings_weight.npy"
)

foreach ($file in $files) {
    Write-Host "Downloading $file..."
    Invoke-WebRequest -Uri "$baseUrl/$file" -OutFile $file
}

# cd ..
Set-Location ..
