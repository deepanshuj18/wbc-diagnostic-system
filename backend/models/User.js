// backend/models/User.js
const mongoose = require('mongoose');
const UserSchema = new mongoose.Schema({
  name: String,
  email: { type: String, unique: true },
  passwordHash: String,
  role: { type: String, enum: ['clinician','admin'], default: 'clinician' },
  createdAt: { type: Date, default: Date.now }
});
module.exports = mongoose.model('User', UserSchema);
