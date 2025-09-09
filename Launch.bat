@echo off
cd /d "%~dp0"

:: Ustawienie zmiennej środowiskowej dla PyTorch w celu użycia nowoczesnego alokatora pamięci CUDA
set PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

echo Starting Stable Diffusion GUI...
python engine\\text_to_image_gui.py

if errorlevel 1 (
    echo.
    echo Error: Failed to start application
    echo Make sure Python is installed and requirements are met
    pause
)