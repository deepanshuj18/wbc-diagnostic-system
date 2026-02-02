import React, { useEffect, useState } from 'react';
import API from '../api';
import PatientDetail from './PatientDetail';

// WBC Wisconsin Breast Cancer features (from sklearn dataset)
const WBC_FEATURES = [
  'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
  'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
  'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
  'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
  'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
  'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
];

export default function PatientList(){
  const [patients, setPatients] = useState([]);
  const [selected, setSelected] = useState(null);
  const [form, setForm] = useState({name:'', age:'', sex:'', features:{}});
  const [featureInputs, setFeatureInputs] = useState({});
  
  useEffect(()=> { 
    load(); 
    // Initialize feature inputs
    const initFeatures = {};
    WBC_FEATURES.forEach(f => initFeatures[f] = '');
    setFeatureInputs(initFeatures);
  }, []);
  
  const load = async ()=> { 
    try {
      const res = await API.get('/patients'); 
      setPatients(res.data); 
    } catch (err) {
      console.error('Failed to load patients', err);
    }
  };
  
  const create = async ()=> {
    try {
      // Convert string inputs to numbers
      const features = {};
      Object.keys(featureInputs).forEach(key => {
        const val = featureInputs[key];
        features[key] = val === '' ? 0 : parseFloat(val);
      });
      
      const res = await API.post('/patients', {
        ...form, 
        features,
        age: parseInt(form.age) || 0
      });
      setForm({name:'', age:'', sex:'', features:{}});
      setFeatureInputs({...featureInputs});
      WBC_FEATURES.forEach(f => setFeatureInputs(prev => ({...prev, [f]: ''})));
      load();
      alert('Patient created successfully!');
    } catch (err) {
      alert('Failed to create patient: ' + (err.response?.data?.error || err.message));
    }
  };
  
  const deletePatient = async (patientId, patientName, e) => {
    e.stopPropagation(); // Prevent selection when clicking delete
    if (!window.confirm(`Are you sure you want to delete patient "${patientName}"?`)) {
      return;
    }
    
    try {
      await API.delete(`/patients/${patientId}`);
      if (selected && selected._id === patientId) {
        setSelected(null); // Clear selection if deleted patient was selected
      }
      load();
      alert('Patient deleted successfully!');
    } catch (err) {
      alert('Failed to delete patient: ' + (err.response?.data?.error || err.message));
    }
  };
  
  return (
    <div className="card">
      <h3>Patients</h3>
      <div className="patient-grid">
        <div className="patient-list">
          <h4>Patient List</h4>
          <ul>
            {patients.map(p=> (
              <li key={p._id} onClick={()=>setSelected(p)}>
                {p.name} ({p.age})
                <button 
                  className="delete-btn"
                  onClick={(e) => deletePatient(p._id, p.name, e)}
                  title="Delete patient"
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
          {patients.length === 0 && <div>No patients yet</div>}
        </div>
        <div className="patient-form">
          <h4>New Patient</h4>
          <div className="form-group">
            <label>Name <input value={form.name} onChange={e=>setForm({...form, name:e.target.value})} /></label>
            <label>Age <input type="number" value={form.age} onChange={e=>setForm({...form, age:e.target.value})} /></label>
            <label>Sex <input value={form.sex} onChange={e=>setForm({...form, sex:e.target.value})} placeholder="M/F" /></label>
          </div>
          <div className="features-preview" style={{maxHeight: '400px', overflowY: 'auto', border: '1px solid #ddd', padding: '10px', borderRadius: '5px'}}>
            <small><strong>WBC Features ({WBC_FEATURES.length} total):</strong></small>
            {WBC_FEATURES.map(f => (
              <label key={f} style={{display: 'block', margin: '5px 0'}}>
                {f}: <input 
                  type="number" 
                  step="0.0001"
                  value={featureInputs[f] || ''} 
                  onChange={e=>setFeatureInputs({...featureInputs, [f]: e.target.value})}
                  style={{width: '100px', marginLeft: '5px'}}
                />
              </label>
            ))}
          </div>
          <button onClick={create}>Create Patient</button>
        </div>
        <div className="patient-detail">
          {selected ? <PatientDetail patient={selected} onRefresh={load} /> : <div>Select a patient to view details and run predictions</div>}
        </div>
      </div>
    </div>
  );
}
