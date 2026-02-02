// backend/models/FairnessMetrics.js
const mongoose = require('mongoose');

const FairnessMetricsSchema = new mongoose.Schema({
  model_version: String,
  overall: {
    accuracy: Number,
    precision: Number,
    recall: Number,
    f1: Number
  },
  age_groups: {
    '<40': {
      accuracy: Number,
      precision: Number,
      recall: Number,
      f1: Number,
      count: Number
    },
    '40-60': {
      accuracy: Number,
      precision: Number,
      recall: Number,
      f1: Number,
      count: Number
    },
    '>60': {
      accuracy: Number,
      precision: Number,
      recall: Number,
      f1: Number,
      count: Number
    }
  },
  gender_groups: {
    M: {
      accuracy: Number,
      precision: Number,
      recall: Number,
      f1: Number,
      count: Number
    },
    F: {
      accuracy: Number,
      precision: Number,
      recall: Number,
      f1: Number,
      count: Number
    },
    Other: {
      accuracy: Number,
      precision: Number,
      recall: Number,
      f1: Number,
      count: Number
    }
  },
  demographic_parity_diff: Number,
  test_samples: Number,
  computedAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model('FairnessMetrics', FairnessMetricsSchema);

