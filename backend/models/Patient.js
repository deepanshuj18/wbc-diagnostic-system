// backend/models/Patient.js
const mongoose = require('mongoose');
const PatientSchema = new mongoose.Schema({
  name: String,
  age: Number,
  sex: String,
  gender: { type: String, enum: ['M', 'F', 'Other'], default: 'Other' },
  // We'll store the WBC features in an object keyed by feature name
  features: { type: mongoose.Schema.Types.Mixed },
  verified: { type: Boolean, default: false }, // For continuous learning
  verifiedBy: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  verifiedAt: Date,
  createdBy: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  createdAt: { type: Date, default: Date.now }
});
module.exports = mongoose.model('Patient', PatientSchema);
