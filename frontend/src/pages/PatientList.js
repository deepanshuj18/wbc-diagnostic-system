import React, { useEffect, useState } from 'react';
import API from '../api';
import PatientDetail from './PatientDetail';

// WBC Wisconsin Breast Cancer features grouped by category
const WBC_FEATURE_GROUPS = {
  'Mean Values': [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension'
  ],
  'Standard Error': [
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error'
  ],
  'Worst Values': [
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
  ]
};

const WBC_FEATURES = Object.values(WBC_FEATURE_GROUPS).flat();

export default function PatientList() {
  const [patients, setPatients] = useState([]);
  const [selected, setSelected] = useState(null);
  const [form, setForm] = useState({ name: '', age: '', sex: '', features: {} });
  const [featureInputs, setFeatureInputs] = useState({});
  const [expandedGroups, setExpandedGroups] = useState({
    'Mean Values': true,
    'Standard Error': false,
    'Worst Values': false
  });

  useEffect(() => {
    load();
    // Initialize feature inputs
    const initFeatures = {};
    WBC_FEATURES.forEach(f => initFeatures[f] = '');
    setFeatureInputs(initFeatures);
  }, []);

  const load = async () => {
    try {
      const res = await API.get('/patients');
      setPatients(res.data);
    } catch (err) {
      console.error('Failed to load patients', err);
    }
  };

  const create = async () => {
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
      setForm({ name: '', age: '', sex: '', features: {} });
      setFeatureInputs({ ...featureInputs });
      WBC_FEATURES.forEach(f => setFeatureInputs(prev => ({ ...prev, [f]: '' })));
      load();
      alert('Patient created successfully!');
    } catch (err) {
      alert('Failed to create patient: ' + (err.response?.data?.error || err.message));
    }
  };

  const deletePatient = async (patientId, patientName, e) => {
    e.stopPropagation();
    if (!window.confirm(`Are you sure you want to delete patient "${patientName}"?`)) {
      return;
    }

    try {
      await API.delete(`/patients/${patientId}`);
      if (selected && selected._id === patientId) {
        setSelected(null);
      }
      load();
      alert('Patient deleted successfully!');
    } catch (err) {
      alert('Failed to delete patient: ' + (err.response?.data?.error || err.message));
    }
  };

  const toggleGroup = (groupName) => {
    setExpandedGroups(prev => ({
      ...prev,
      [groupName]: !prev[groupName]
    }));
  };

  const fillSampleData = () => {
    // Sample malignant data from WBC dataset
    const sampleData = {
      'mean radius': 17.99, 'mean texture': 10.38, 'mean perimeter': 122.8, 'mean area': 1001, 'mean smoothness': 0.1184,
      'mean compactness': 0.2776, 'mean concavity': 0.3001, 'mean concave points': 0.1471, 'mean symmetry': 0.2419, 'mean fractal dimension': 0.07871,
      'radius error': 1.095, 'texture error': 0.9053, 'perimeter error': 8.589, 'area error': 153.4, 'smoothness error': 0.006399,
      'compactness error': 0.04904, 'concavity error': 0.05373, 'concave points error': 0.01587, 'symmetry error': 0.03003, 'fractal dimension error': 0.006193,
      'worst radius': 25.38, 'worst texture': 17.33, 'worst perimeter': 184.6, 'worst area': 2019, 'worst smoothness': 0.1622,
      'worst compactness': 0.6656, 'worst concavity': 0.7119, 'worst concave points': 0.2654, 'worst symmetry': 0.4601, 'worst fractal dimension': 0.1189
    };
    setFeatureInputs(sampleData);
  };

  return (
    <div className="card">
      <h3 style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <span>👥</span> Patient Management
      </h3>
      <div className="patient-grid">
        <div className="patient-list">
          <h4 style={{ fontSize: '1.1em', marginBottom: '15px' }}>📋 Patient List</h4>
          <ul>
            {patients.map(p => (
              <li
                key={p._id}
                onClick={() => setSelected(p)}
                className={selected?._id === p._id ? 'selected' : ''}
              >
                <span>{p.name} ({p.age})</span>
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
          {patients.length === 0 && (
            <div style={{
              textAlign: 'center',
              padding: '30px 20px',
              color: 'var(--color-text-muted)',
              fontSize: '0.9em'
            }}>
              No patients yet. Create one to get started!
            </div>
          )}
        </div>

        <div className="patient-form">
          <h4 style={{ fontSize: '1.1em', marginBottom: '15px' }}>➕ Create New Patient</h4>

          <div className="form-group">
            <label>👤 Patient Name</label>
            <input
              placeholder="Enter full name"
              value={form.name}
              onChange={e => setForm({ ...form, name: e.target.value })}
            />

            <label>🎂 Age</label>
            <input
              type="number"
              placeholder="Enter age"
              value={form.age}
              onChange={e => setForm({ ...form, age: e.target.value })}
            />

            <label>⚧ Sex</label>
            <select
              value={form.sex}
              onChange={e => setForm({ ...form, sex: e.target.value })}
            >
              <option value="">Select sex</option>
              <option value="M">Male</option>
              <option value="F">Female</option>
            </select>
          </div>

          <div style={{ marginTop: '20px', marginBottom: '15px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h4 style={{ fontSize: '1em', margin: 0 }}>🔬 WBC Features (30 total)</h4>
            <button
              onClick={fillSampleData}
              style={{
                padding: '6px 12px',
                fontSize: '0.85em',
                background: 'var(--gradient-warning)',
                minWidth: 'auto'
              }}
              title="Fill with sample malignant case data"
            >
              📝 Fill Sample
            </button>
          </div>

          <div style={{
            maxHeight: '450px',
            overflowY: 'auto',
            border: '1px solid var(--glass-border)',
            borderRadius: 'var(--radius-md)',
            padding: '10px',
            background: 'var(--glass-bg)'
          }}>
            {Object.entries(WBC_FEATURE_GROUPS).map(([groupName, features]) => (
              <div key={groupName} style={{ marginBottom: '15px' }}>
                <div
                  onClick={() => toggleGroup(groupName)}
                  style={{
                    padding: '12px',
                    background: groupName === 'Mean Values'
                      ? 'var(--gradient-primary)'
                      : groupName === 'Standard Error'
                        ? 'var(--gradient-info)'
                        : 'var(--gradient-warning)',
                    color: 'white',
                    borderRadius: 'var(--radius-sm)',
                    cursor: 'pointer',
                    fontWeight: '600',
                    fontSize: '0.95em',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    userSelect: 'none',
                    marginBottom: '8px',
                    transition: 'all var(--transition-fast)'
                  }}
                >
                  <span>{groupName} ({features.length})</span>
                  <span style={{ fontSize: '1.2em' }}>{expandedGroups[groupName] ? '▼' : '▶'}</span>
                </div>

                {expandedGroups[groupName] && (
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(2, 1fr)',
                    gap: '10px',
                    padding: '10px',
                    background: 'rgba(255, 255, 255, 0.5)',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--glass-border)'
                  }}>
                    {features.map(f => (
                      <div key={f} style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                        <label style={{
                          fontSize: '0.75em',
                          fontWeight: '600',
                          color: 'var(--color-text-secondary)',
                          margin: 0,
                          textTransform: 'capitalize'
                        }}>
                          {f.replace('mean ', '').replace('worst ', '').replace(' error', '')}
                        </label>
                        <input
                          type="number"
                          step="0.0001"
                          placeholder="0.0"
                          value={featureInputs[f] || ''}
                          onChange={e => setFeatureInputs({ ...featureInputs, [f]: e.target.value })}
                          style={{
                            fontSize: '0.9em',
                            padding: '8px',
                            margin: 0
                          }}
                        />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>

          <button
            onClick={create}
            style={{ marginTop: '15px', width: '100%' }}
          >
            ✨ Create Patient with Features
          </button>
        </div>

        <div className="patient-detail">
          {selected ? (
            <PatientDetail patient={selected} onRefresh={load} />
          ) : (
            <div style={{
              textAlign: 'center',
              padding: '60px 20px',
              color: 'var(--color-text-muted)'
            }}>
              <div style={{ fontSize: '3em', marginBottom: '15px' }}>📊</div>
              <p style={{ fontSize: '1.1em', fontWeight: '500' }}>Select a patient</p>
              <p style={{ fontSize: '0.9em' }}>Click on a patient from the list to view details and run predictions</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
