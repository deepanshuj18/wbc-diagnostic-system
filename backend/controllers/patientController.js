// backend/controllers/patientController.js
const Patient = require('../models/Patient');
const Prediction = require('../models/Prediction');

exports.createPatient = async (req, res) => {
  try {
    const p = new Patient({...req.body, createdBy: req.user._id});
    await p.save();
    res.json(p);
  } catch (err) { res.status(500).json({ error: err.message }); }
};

exports.listPatients = async (req, res) => {
  const patients = await Patient.find({ createdBy: req.user._id }).sort({ createdAt: -1 });
  res.json(patients);
};

exports.getPatient = async (req, res) => {
  const p = await Patient.findById(req.params.id);
  if (!p) return res.status(404).json({ error: 'Not found' });
  res.json(p);
};

exports.updatePatient = async (req, res) => {
  const p = await Patient.findByIdAndUpdate(req.params.id, req.body, { new: true });
  res.json(p);
};

exports.deletePatient = async (req, res) => {
  try {
    const patientId = req.params.id;
    // Delete all predictions associated with this patient
    await Prediction.deleteMany({ patient: patientId });
    // Delete the patient
    await Patient.findByIdAndDelete(patientId);
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};
