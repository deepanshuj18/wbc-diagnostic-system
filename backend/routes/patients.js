// backend/routes/patients.js
const express = require('express');
const router = express.Router();
const auth = require('../middleware/auth');
const patientController = require('../controllers/patientController');

router.post('/', auth, patientController.createPatient);
router.get('/', auth, patientController.listPatients);
router.get('/:id', auth, patientController.getPatient);
router.put('/:id', auth, patientController.updatePatient);
router.delete('/:id', auth, patientController.deletePatient);

module.exports = router;
