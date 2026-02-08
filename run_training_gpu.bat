@echo off
setlocal

:: Set path to venv310
set VENV_DIR=%~dp0venv310
set SITE_PACKAGES=%VENV_DIR%\Lib\site-packages

:: Add NVIDIA CUDA libraries to PATH
set PATH=%SITE_PACKAGES%\nvidia\cuda_runtime\bin;%PATH%
set PATH=%SITE_PACKAGES%\nvidia\cudnn\bin;%PATH%
set PATH=%SITE_PACKAGES%\nvidia\cublas\bin;%PATH%
set PATH=%SITE_PACKAGES%\nvidia\cufft\bin;%PATH%
set PATH=%SITE_PACKAGES%\nvidia\curand\bin;%PATH%
set PATH=%SITE_PACKAGES%\nvidia\cusolver\bin;%PATH%
set PATH=%SITE_PACKAGES%\nvidia\cusparse\bin;%PATH%

echo ==========================================================
echo 🚀 STARTING GPU TRAINING
echo ==========================================================
echo Python: %VENV_DIR%\Scripts\python.exe
echo TensorFlow Version:
"%VENV_DIR%\Scripts\python.exe" -c "import tensorflow as tf; print(tf.__version__)"
echo GPU Available:
"%VENV_DIR%\Scripts\python.exe" -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))"
echo ==========================================================

:: Run command
"%VENV_DIR%\Scripts\python.exe" %*

endlocal
exit /b %ERRORLEVEL%
