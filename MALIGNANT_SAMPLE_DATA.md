# Malignant Breast Cancer Case - Test Data

## Quick Test
This malignant case has been tested and verified to produce correct predictions.

### Patient Info
- **Name:** Malignant Test Case
- **Age:** 45
- **Sex:** F

### WBC Features (30 values - copy and paste into the form)
```
17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
```

### Feature Labels (in order)
1. mean radius
2. mean texture
3. mean perimeter
4. mean area
5. mean smoothness
6. mean compactness
7. mean concavity
8. mean concave points
9. mean symmetry
10. mean fractal dimension
11. radius error
12. texture error
13. perimeter error
14. area error
15. smoothness error
16. compactness error
17. concavity error
18. concave points error
19. symmetry error
20. fractal dimension error
21. worst radius
22. worst texture
23. worst perimeter
24. worst area
25. worst smoothness
26. worst compactness
27. worst concavity
28. worst concave points
29. worst symmetry
30. worst fractal dimension

### Expected Result
- **Prediction:** Malignant
- **Confidence:** Very High (>95%)
- **Result:** `{"prediction": 0, "result": "Malignant"}`

### API Test (Curl)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}'
```

### Manual Entry Instructions
1. Go to http://localhost:3000
2. Click "Patients" tab
3. Enter Patient Name: "Malignant Test Case"
4. Enter Age: 45
5. Enter Sex: F
6. Scroll down to WBC Features section
7. Enter each feature value in order:
   - mean radius: 17.99
   - mean texture: 10.38
   - mean perimeter: 122.8
   - mean area: 1001
   - mean smoothness: 0.1184
   - mean compactness: 0.2776
   - mean concavity: 0.3001
   - mean concave points: 0.1471
   - mean symmetry: 0.2419
   - mean fractal dimension: 0.07871
   - radius error: 1.095
   - texture error: 0.9053
   - perimeter error: 8.589
   - area error: 153.4
   - smoothness error: 0.006399
   - compactness error: 0.04904
   - concavity error: 0.05373
   - concave points error: 0.01587
   - symmetry error: 0.03003
   - fractal dimension error: 0.006193
   - worst radius: 25.38
   - worst texture: 17.33
   - worst perimeter: 184.6
   - worst area: 2019
   - worst smoothness: 0.1622
   - worst compactness: 0.6656
   - worst concavity: 0.7119
   - worst concave points: 0.2654
   - worst symmetry: 0.4601
   - worst fractal dimension: 0.1189
8. Click "Create Patient"
9. Click on the patient in the list
10. Click "Run Prediction"
11. Verify result shows "Malignant"

