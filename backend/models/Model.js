// backend/models/Model.js
const mongoose = require('mongoose');

const ModelSchema = new mongoose.Schema({
  version: String,
  metrics: {
    accuracy: Number,
    auc: Number,
    precision: Number,
    recall: Number
  },
  input_dim: Number,
  embed_dim: Number,
  mask_ratio: Number,
  training_date: { type: Date, default: Date.now },
  is_active: { type: Boolean, default: true }
});

module.exports = mongoose.model('Model', ModelSchema);







