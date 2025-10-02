@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ENV_NAME=DIPLOM_PO_bat"
set "DEFAULT_ARCHIVE=%~dp0..\DIPLOM_PO_win-64.tar.gz"

if "%~1"=="" (
  set "ARCHIVE=%DEFAULT_ARCHIVE%"
) else (
  set "ARCHIVE=%~1"
)

if "%HOST%"=="" set "HOST=0.0.0.0"
if "%PORT%"=="" set "PORT=8000"

set "RUN_CONDA_UNPACK="

set "ROOT=%~dp0"
set "ENV_DIR=%ROOT%\.conda_env\%ENV_NAME%"

echo [INFO] Project root: %ROOT%
echo [INFO] Archive path: %ARCHIVE%
echo [INFO] Target env:   %ENV_DIR%

if not exist "%ARCHIVE%" (
  echo [ERROR] Archive not found: "%ARCHIVE%"
  echo         Передай путь первым аргументом или положи архив на уровень выше.
  echo         Пример: run.bat "C:\path\to\DIPLOM_PO_win-64.tar.gz"
  exit /b 1
)

if not exist "%ENV_DIR%\python.exe" (
  echo [INFO] Extracting environment...
  mkdir "%ENV_DIR%" 2>nul

  tar -xzf "%ARCHIVE%" -C "%ENV_DIR%"
  if errorlevel 1 (
    echo [ERROR] tar extraction failed. Убедись, что доступна команда "tar".
    echo         Вариант: установить Git for Windows ^(добавит tar в PATH^).
    exit /b 1
  )

  if exist "%ENV_DIR%\Scripts\conda-unpack.exe" (
    if "%RUN_CONDA_UNPACK%"=="1" (
      echo [INFO] Running conda-unpack...
      "%ENV_DIR%\Scripts\conda-unpack.exe" 1>nul 2>nul
      if errorlevel 1 echo [WARN] conda-unpack returned non-zero; continuing...
    ) else (
      echo [INFO] Skipping conda-unpack on Windows to avoid long-path issues.
    )
  ) else (
    echo [INFO] conda-unpack not found ^(skipping^).
  )
) else (
  echo [INFO] Environment already present. Skipping extraction.
)

pushd "%ROOT%"
echo [INFO] Starting Uvicorn on %HOST%:%PORT% ...
"%ENV_DIR%\python.exe" -m uvicorn gateway:app --host %HOST% --port %PORT% --reload
set "EXITCODE=%ERRORLEVEL%"
popd

endlocal & exit /b %EXITCODE%
