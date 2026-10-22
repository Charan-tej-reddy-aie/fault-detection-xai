@echo off
setlocal
echo ========================================================
echo   Push Project to GitHub
echo ========================================================
echo.
set /p REPO_URL="Enter your GitHub Repository URL (e.g. https://github.com/username/repo-name.git): "
if "%REPO_URL%"=="" (
    echo [ERROR] No URL entered. Aborting.
    pause
    exit /b 1
)

echo.
echo [1/3] Checking Git remote...
git remote remove origin 2>nul
git remote add origin %REPO_URL%

echo [2/3] Setting branch to main...
git branch -M main

echo [3/3] Pushing to GitHub...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo   Successfully pushed to GitHub!
    echo ========================================================
) else (
    echo.
    echo [NOTE] If authentication failed, make sure you have logged in
    echo        or configured your GitHub Personal Access Token (PAT).
)
echo.
pause
