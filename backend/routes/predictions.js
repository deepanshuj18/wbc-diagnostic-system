// backend/routes/predictions.js
const express = require('express');
const router = express.Router();
const auth = require('../middleware/auth');
const predictionController = require('../controllers/predictionController');

router.post('/predict', auth, predictionController.predictForPatient);
router.get('/', auth, predictionController.listPredictions);
router.get('/:id', auth, predictionController.getPrediction);

module.exports = router;
