@echo off
title WBC Diagnostic System - Quick Test
cls
echo.
echo ============================================================
echo          WBC DIAGNOSTIC SYSTEM - QUICK TEST
echo ============================================================
echo.

echo Testing MALIGNANT case (ID from Wisconsin Breast Cancer Dataset)...
echo -----------------------------------------------------------------
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"features\": [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}" 2>nul
echo.
echo.

echo Testing BENIGN case (ID from Wisconsin Breast Cancer Dataset)...
echo ------------------------------------------------------------------
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"features\": [13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]}" 2>nul
echo.
echo.
echo ============================================================
echo Test completed!
echo.
pause

