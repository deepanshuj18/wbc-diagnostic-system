# Quick Test Commands for WBC Diagnostic System

## Run Both Tests at Once

### Windows Batch File
Simply double-click or run:
```cmd
test_predictions.bat
```

## Individual Commands

### Test MALIGNANT Case
```cmd
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"features\": [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}"
```

**Expected Result:**
```json
{"prediction": 0, "result": "Malignant", "probability": ~4e-10, "calibrated_probability": 0.0}
```

### Test BENIGN Case
```cmd
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"features\": [13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]}"
```

**Expected Result:**
```json
{"prediction": 1, "result": "Benign", "probability": ~0.97, "calibrated_probability": 0.6}
```

## Model Status Check
```cmd
curl http://localhost:8000/model_status
```

## Expected Output Summary

| Case Type | Prediction | Result | Confidence | Verification |
|-----------|-----------|---------|------------|--------------|
| Malignant | 0 | Malignant | Very Low (~0%) | ✅ Correct |
| Benign | 1 | Benign | Very High (~97%) | ✅ Correct |

## Notes
- Both cases are from the actual Wisconsin Breast Cancer Dataset
- Malignant case: prediction=0 (correct labeling fixed!)
- Benign case: prediction=1 (correct labeling fixed!)
- Calibration is working (calibrated_probability differs from raw probability)
- All model fixes are in place and verified

