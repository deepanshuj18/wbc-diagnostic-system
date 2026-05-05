import React, { useState, useEffect } from 'react';
import API from '../api';
import SHAPPlot from '../components/SHAPPlot';

export default function PatientDetail({ patient, onRefresh }){
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [explainerMethod, setExplainerMethod] = useState('shap');

  useEffect(() => {
    loadPredictions();
  }, [patient]);

  const loadPredictions = async () => {
    try {
      const res = await API.get('/predictions');
      const patientPreds = res.data.filter(p => p.patient?._id === patient._id);
      setPredictions(patientPreds);
    } catch (err) {
      console.error('Failed to load predictions', err);
    }
  };

  const triggerPredict = async () => {
    if (!patient.features || Object.keys(patient.features).length === 0) {
      alert('Patient has no features. Cannot run prediction.');
      return;
    }
    
    setLoading(true);
    try {
      const res = await API.post('/predictions/predict', { 
        patientId: patient._id, 
        method: explainerMethod 
      });
      setPrediction(res.data);
      await loadPredictions();
      await onRefresh();
      alert('Prediction completed successfully!');
    } catch (err) {
      alert(err.response?.data?.error || err.message || 'Failed to run prediction');
    } finally { 
      setLoading(false); 
    }
  };

  const latestPred = predictions.length > 0 ? predictions[0] : prediction;

  return (
    <div>
      <h4>{patient.name}</h4>
      <div className="patient-info">
        <div>Age: {patient.age}</div>
        <div>Sex: {patient.sex}</div>
        <div>Created: {new Date(patient.createdAt).toLocaleDateString()}</div>
      </div>
      
      <div style={{marginTop: '20px'}}>
        <div style={{marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '10px'}}>
          <label style={{fontWeight: 'bold'}}>Explanation Method:</label>
          <select 
            value={explainerMethod} 
            onChange={(e) => setExplainerMethod(e.target.value)}
            style={{padding: '5px 10px', borderRadius: '5px', border: '1px solid #ddd'}}
          >
            <option value="shap">SHAP</option>
            <option value="lime">LIME</option>
            <option value="ig">Integrated Gradients</option>
          </select>
        </div>
        <button onClick={triggerPredict} disabled={loading}>
          {loading ? 'Running Prediction...' : 'Run Prediction'}
        </button>
      </div>
      
      {latestPred && (
        <div className="card" style={{marginTop: '20px'}}>
          <h4>Latest Prediction Result</h4>
          <div className="prediction-summary">
            <div><strong>Result:</strong> 
              <span style={{color: latestPred.prediction === 1 ? '#4CAF50' : '#f44336', fontWeight: 'bold'}}>
                {' '}{latestPred.result || (latestPred.prediction === 1 ? 'Benign' : 'Malignant')}
              </span>
            </div>
            <div><strong>Calibrated Probability:</strong> 
              {((latestPred.calibrated_probability || latestPred.probability) * 100).toFixed(2)}%
            </div>
            {latestPred.calibrated_probability !== undefined && latestPred.probability !== latestPred.calibrated_probability && (
              <div><strong>Raw Probability:</strong> {(latestPred.probability * 100).toFixed(2)}%</div>
            )}
            <div><strong>Uncertainty:</strong> {latestPred.uncertainty?.toFixed(4)}</div>
            <div><strong>Explanation Method:</strong> {latestPred.explanation_method?.toUpperCase() || 'SHAP'}</div>
            <div><strong>Model Version:</strong> {latestPred.model_version || 'N/A'}</div>
          </div>
          
          {latestPred.warnings && latestPred.warnings.length > 0 && (
            <div style={{
              marginTop: '15px',
              padding: '12px 16px',
              background: 'linear-gradient(135deg, #fff3e0, #ffe0b2)',
              border: '1px solid #ffb74d',
              borderRadius: '8px',
              color: '#e65100'
            }}>
              <strong>⚠️ Warnings:</strong>
              <ul style={{margin: '8px 0 0 0', paddingLeft: '20px'}}>
                {latestPred.warnings.map((w, i) => (
                  <li key={i} style={{marginBottom: '4px'}}>{w}</li>
                ))}
              </ul>
            </div>
          )}
          
          {latestPred.shap_values && latestPred.shap_values.length > 0 && (
            <div style={{marginTop: '20px'}}>
              <h5>{(latestPred.explanation_method || 'SHAP').toUpperCase()} Feature Importance</h5>
              <SHAPPlot 
                shapValues={latestPred.shap_values}
                featureNames={latestPred.feature_names || (Array.isArray(latestPred.shap_values) ? 
                  Array(latestPred.shap_values.length).fill(0).map((_, i) => `Feature ${i+1}`) : 
                  []
                )}
                methodName={(latestPred.explanation_method || 'SHAP').toUpperCase()}
              />
            </div>
          )}
        </div>
      )}
      
      {predictions.length > 0 && (
        <div className="card" style={{marginTop: '20px'}}>
          <h4>Prediction History</h4>
          <ul>
            {predictions.map((p, idx) => (
              <li key={p._id || idx}>
                {new Date(p.createdAt).toLocaleString()}: {p.prediction === 1 ? 'Benign' : 'Malignant'} 
                ({(p.probability * 100).toFixed(1)}%)
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
