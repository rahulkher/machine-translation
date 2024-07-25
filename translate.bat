@echo off

setlocal
set "ollama_path="

:: Check if Ollama is installed
for /F "tokens=*" %%i in ('where ollama 2^>nul') do (
    set "ollama_path=%%i"
)

if not defined ollama_path (
    echo Ollama is not installed or not in the system PATH.
    echo Make sure you have ollama installed and added to system PATH
    goto end
)

:: Check if Ollama is running
set process_name=ollama.exe
tasklist /FI "IMAGENAME eq %process_name%" | find /I "%process_name%" >nul 2>&1
if "%ERRORLEVEL%"=="0" (
    echo Ollama is already running.
) else (
    echo Ollama is not running. Starting Ollama server
    start "Ollama" cmd /c "ollama serve
)

endlocal

setlocal
set audiofile=%1


echo Starting Application....
call D:
call cd translation-whisper
echo.
echo Activating translation environment...
call translation\Scripts\activate

echo.
echo Running translation Scripts
echo.
python main.py --audiofile %audiofile%

pause
:end	
endlocal 


