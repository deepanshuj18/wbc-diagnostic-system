@echo off
echo Starting WBC Diagnostic System v2.0...
cd "%~dp0infra"
docker-compose up -d
echo.
echo Services starting! Check status with: docker-compose ps
echo.
echo Frontend will be at: http://localhost:3000
echo Backend will be at: http://localhost:4000
echo ML Service will be at: http://localhost:8000
echo.
echo Press any key to check service status...
pause >nul
docker-compose ps
pause



