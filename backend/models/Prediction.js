// backend/models/Prediction.js
const mongoose = require('mongoose');
const PredictionSchema = new mongoose.Schema({
  patient: { type: mongoose.Schema.Types.ObjectId, ref: 'Patient' },
  user: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  model_version: String,
  prediction: Number,
  result: String, // "Benign" or "Malignant"
  probability: Number,
  calibrated_probability: Number,
  uncertainty: Number,
  explanation_method: String,
  shap_values: [Number],
  embedding: [Number],
  feature_names: [String],
  createdAt: { type: Date, default: Date.now }
});
module.exports = mongoose.model('Prediction', PredictionSchema);
