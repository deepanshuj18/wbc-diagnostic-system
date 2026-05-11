// backend/controllers/predictionController.js
const axios = require('axios');
const Prediction = require('../models/Prediction');
const Patient = require('../models/Patient');

exports.predictForPatient = async (req, res) => {
  const { patientId, method = 'shap' } = req.body;
  try {
    const patient = await Patient.findById(patientId);
    if (!patient) return res.status(404).json({ error: 'Patient not found' });
    
    if (!patient.features || Object.keys(patient.features).length === 0) {
      return res.status(400).json({ error: 'Patient has no features. Please add WBC features first.' });
    }
    
    // FIX #1: Send named features dict to guarantee correct feature order
    // Instead of Object.values() which has no guaranteed order
    const modelUrl = (process.env.MODEL_API_URL || 'http://ml:8000') + '/predict';
    const resp = await axios.post(modelUrl, { 
      named_features: patient.features,
      method,
      age: patient.age,
      gender: patient.gender || patient.sex 
    }, { timeout: 20_000 });
    const data = resp.data;
    
    if (data.error) {
      return res.status(503).json({ error: data.error });
    }
    
    const pred = new Prediction({
      patient: patient._id,
      user: req.user._id,
      model_version: data.model_version,
      prediction: data.prediction,
      result: data.result,
      probability: data.probability,
      calibrated_probability: data.calibrated_probability,
      confidence: data.confidence,
      uncertainty: data.uncertainty,
      explanation_method: data.explanation_method,
      shap_values: data.shap_values,
      embedding: data.embedding_mean,
      feature_names: data.feature_names,
      warnings: data.warnings || []
    });
    await pred.save();
    res.json({ prediction: pred, raw: data });
  } catch (err) {
    console.error('Prediction error:', err.message);
    res.status(500).json({ error: err.message });
  }
};

exports.listPredictions = async (req, res) => {
  const preds = await Prediction.find({ user: req.user._id }).sort({ createdAt: -1 }).populate('patient');
  res.json(preds);
};

exports.getPrediction = async (req, res) => {
  const p = await Prediction.findById(req.params.id).populate('patient');
  if (!p) return res.status(404).json({ error: 'Not found' });
  res.json(p);
};
