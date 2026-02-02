import React, { useEffect, useState } from 'react';
import API from '../api';

export default function Dashboard(){
  const [metrics, setMetrics] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(()=> {
    fetchMetrics();
  }, []);
  
  const fetchMetrics = async () => {
    setLoading(true);
    try {
      const [patientsRes, predsRes, modelRes] = await Promise.all([
        API.get('/patients'),
        API.get('/predictions'),
        API.get('/models/status').catch(() => ({ data: { error: 'Model server unavailable' } }))
      ]);
      
      const predictions = predsRes.data;
      const malignant = predictions.filter(p => p.prediction === 0).length;
      const benign = predictions.filter(p => p.prediction === 1).length;
      
      setMetrics({ 
        patients: patientsRes.data.length, 
        predictions: predictions.length,
        malignant,
        benign
      });
      setModelStatus(modelRes.data);
    } catch (err) { 
      console.error('Failed to fetch metrics', err); 
      setMetrics({ patients: 0, predictions: 0, malignant: 0, benign: 0 });
    } finally {
      setLoading(false);
    }
  };
  
  if (loading) {
    return <div className="card">Loading...</div>;
  }
  
  return (
    <div>
      <h2>Dashboard Overview</h2>
      
      <div className="card">
        <h3>Application Statistics</h3>
        <div className="grid">
          <div className="tile">
            <div style={{fontSize: '2em', fontWeight: 'bold', color: '#2563eb'}}>
              {metrics?.patients ?? 0}
            </div>
            <div>Total Patients</div>
          </div>
          <div className="tile">
            <div style={{fontSize: '2em', fontWeight: 'bold', color: '#2563eb'}}>
              {metrics?.predictions ?? 0}
            </div>
            <div>Total Predictions</div>
          </div>
        </div>
        
        {metrics?.predictions > 0 && (
          <div className="grid" style={{marginTop: '20px'}}>
            <div className="tile" style={{background: '#fee2e2'}}>
              <div style={{fontSize: '1.8em', fontWeight: 'bold', color: '#dc2626'}}>
                {metrics?.malignant ?? 0}
              </div>
              <div>Malignant Cases</div>
            </div>
            <div className="tile" style={{background: '#d1fae5'}}>
              <div style={{fontSize: '1.8em', fontWeight: 'bold', color: '#059669'}}>
                {metrics?.benign ?? 0}
              </div>
              <div>Benign Cases</div>
            </div>
          </div>
        )}
      </div>
      
      <div className="card">
        <h3>Model Status</h3>
        {modelStatus?.error ? (
          <div className="error">
            Model server unavailable. Please ensure the ML service is running.
          </div>
        ) : modelStatus ? (
          <div className="prediction-summary">
            <div><strong>Model Version:</strong> {modelStatus.model_version || 'N/A'}</div>
            <div><strong>Uptime:</strong> {Math.floor((modelStatus.uptime_seconds || 0) / 60)} minutes</div>
            <div><strong>Input Dimension:</strong> {modelStatus.input_dim || 'N/A'}</div>
            <div><strong>Embedding Dimension:</strong> {modelStatus.embed_dim || 'N/A'}</div>
          </div>
        ) : (
          <div>No model status available</div>
        )}
      </div>
    </div>
  );
}
