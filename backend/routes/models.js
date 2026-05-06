// backend/routes/models.js
const express = require('express');
const router = express.Router();
const auth = require('../middleware/auth');
const axios = require('axios');
const Model = require('../models/Model');
const Prediction = require('../models/Prediction');

router.get('/status', auth, async (req, res) => {
  try {
    const modelUrl = (process.env.MODEL_API_URL || 'http://ml:8000') + '/model_status';
    const resp = await axios.get(modelUrl, { timeout: 5000 });
    res.json(resp.data);
  } catch (err) {
    res.status(503).json({ error: 'Model server unavailable' });
  }
});

router.get('/versions', auth, async (req, res) => {
  const models = await Model.find().sort({ training_date: -1 });
  res.json(models);
});

router.get('/evaluation', auth, async (req, res) => {
  try {
    const modelUrl = (process.env.MODEL_API_URL || 'http://ml:8000') + '/model_evaluation';
    const resp = await axios.get(modelUrl, { timeout: 15000 });
    res.json(resp.data);
  } catch (err) {
    res.status(503).json({ error: 'Failed to fetch evaluation metrics: ' + err.message });
  }
});

router.get('/embeddings', auth, async (req, res) => {
  try {
    const modelUrl = (process.env.MODEL_API_URL || 'http://ml:8000') + '/embeddings_visualization';
    const resp = await axios.post(modelUrl, {}, { timeout: 30000 });
    res.json(resp.data);
  } catch (err) {
    res.status(503).json({ error: 'Failed to fetch embeddings: ' + err.message });
  }
});

module.exports = router;




