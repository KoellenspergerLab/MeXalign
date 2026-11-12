@echo off
setlocal enableextensions
REM MeXalign GUI launcher (Windows) — double-click to run.
REM
REM On first run it auto-creates the conda env "mexalign" from mexalign.yml (Miniconda
REM must be installed once), then launches. Later runs just launch. To override
REM detection, set PYTHON below, e.g.
REM   set "PYTHON=%USERPROFILE%\miniconda3\envs\mexalign\python.exe"

set "PYTHON="

cd /d "%~dp0"

if defined PYTHON goto verify

REM 1) Use an existing Python that already has the packages: the mexalign env in a
REM    known conda base, the base's own python, or python/py on PATH.
REM    The per-user .conda\envs location is checked first — conda puts the env
REM    there whenever the conda base is read-only / system-wide (e.g. ProgramData).
if exist "%USERPROFILE%\.conda\envs\mexalign\python.exe" (
    "%USERPROFILE%\.conda\envs\mexalign\python.exe" -c "import PySide6, SimpleITK" >nul 2>nul && (
        set "PYTHON=%USERPROFILE%\.conda\envs\mexalign\python.exe"
        goto run
    )
)
for %%E in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniforge3" "%USERPROFILE%\mambaforge" "%PROGRAMDATA%\miniconda3" "%PROGRAMDATA%\Anaconda3") do (
    if exist "%%~E\envs\mexalign\python.exe" (
        "%%~E\envs\mexalign\python.exe" -c "import PySide6, SimpleITK" >nul 2>nul && (
            set "PYTHON=%%~E\envs\mexalign\python.exe"
            goto run
        )
    )
    if exist "%%~E\python.exe" (
        "%%~E\python.exe" -c "import PySide6, SimpleITK" >nul 2>nul && (
            set "PYTHON=%%~E\python.exe"
            goto run
        )
    )
)
for %%P in (python py) do (
    %%P -c "import PySide6, SimpleITK" >nul 2>nul && (
        set "PYTHON=%%P"
        goto run
    )
)

REM 2) Nothing ready — locate conda.
set "CONDA="
set "CONDA_BASE="
for %%E in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniforge3" "%USERPROFILE%\mambaforge" "%PROGRAMDATA%\miniconda3" "%PROGRAMDATA%\Anaconda3") do (
    if not defined CONDA if exist "%%~E\Scripts\conda.exe" (
        set "CONDA=%%~E\Scripts\conda.exe"
        set "CONDA_BASE=%%~E"
    )
)
if not defined CONDA (
    where conda >nul 2>nul && set "CONDA=conda"
)

if not defined CONDA (
    echo Conda/Miniconda was not found on this PC.
    echo Install Miniconda once from:
    echo     https://www.anaconda.com/download/success
    echo then double-click this file again.
    echo.
    pause
    exit /b 1
)

if not defined CONDA_BASE (
    for /f "delims=" %%i in ('"%CONDA%" info --base 2^>nul') do set "CONDA_BASE=%%i"
)

REM Create the 'mexalign' env only if it does not exist yet.
set "ENV_PY="
if defined CONDA_BASE if exist "%CONDA_BASE%\envs\mexalign\python.exe" set "ENV_PY=%CONDA_BASE%\envs\mexalign\python.exe"
if not defined ENV_PY (
    echo First-time setup: creating the "mexalign" environment from mexalign.yml ...
    echo This downloads the packages and may take a few minutes.
    call "%CONDA%" env create -f mexalign.yml
    if errorlevel 1 (
        echo Environment creation failed — see the messages above.
        pause
        exit /b 1
    )
)

REM Resolve the env's Python, then verify before launching. The env may live under
REM the conda base (%CONDA_BASE%\envs), the per-user store (%USERPROFILE%\.conda\envs),
REM or a custom envs_dirs — so try the common spots, then ask conda directly.
if defined CONDA_BASE if exist "%CONDA_BASE%\envs\mexalign\python.exe" (
    set "PYTHON=%CONDA_BASE%\envs\mexalign\python.exe"
    goto verify
)
if exist "%USERPROFILE%\.conda\envs\mexalign\python.exe" (
    set "PYTHON=%USERPROFILE%\.conda\envs\mexalign\python.exe"
    goto verify
)
for %%E in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniforge3" "%USERPROFILE%\mambaforge" "%PROGRAMDATA%\miniconda3" "%PROGRAMDATA%\Anaconda3") do (
    if exist "%%~E\envs\mexalign\python.exe" (
        set "PYTHON=%%~E\envs\mexalign\python.exe"
        goto verify
    )
)
REM Last resort: let conda report the env's interpreter path (any location).
for /f "delims=" %%i in ('"%CONDA%" run -n mexalign python -c "import sys;print(sys.executable)" 2^>nul') do set "PYTHON=%%i"
if defined PYTHON if exist "%PYTHON%" goto verify

echo Could not find the mexalign environment's Python after setup. See messages above.
pause
exit /b 1

:verify
REM Final check that the packages import before launching (mirrors the .command).
"%PYTHON%" -c "import PySide6, SimpleITK" >nul 2>nul
if errorlevel 1 (
    echo Could not import PySide6 / SimpleITK with:
    echo     %PYTHON%
    echo The environment may be incomplete or incompatible. If it mentions
    echo   "DLL load failed while importing QtCore"
    echo this PC's Windows may be too old for this Qt version. Tell the
    echo developer your Windows version ^(press Win+R, type winver^).
    echo.
    pause
    exit /b 1
)

:run
echo Launching with: %PYTHON%
"%PYTHON%" -m mexalign.gui_full
if errorlevel 1 (
    echo.
    echo The app did not start ^(see the error above^). If it mentions
    echo   "DLL load failed while importing QtCore"
    echo this PC's Windows may be too old for this Qt version. Tell the
    echo developer your Windows version ^(press Win+R, type winver^).
    echo.
    pause
)
