import React, { useEffect, useState } from 'react';
import API from '../api';

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
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
    return (
      <div>
        <h2>📊 Dashboard Overview</h2>
        <div className="card" style={{ textAlign: 'center', padding: '60px' }}>
          <div className="loading" style={{ margin: '0 auto', width: '40px', height: '40px' }}></div>
          <p style={{ marginTop: '20px', color: 'var(--color-text-muted)' }}>Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <h2 style={{ fontSize: '2.5em', marginBottom: '30px' }}>
        📊 Dashboard Overview
      </h2>

      <div className="card">
        <h3 style={{ marginBottom: '25px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span>📈</span> Application Statistics
        </h3>
        <div className="grid">
          <div className="tile" style={{ background: 'var(--gradient-primary)' }}>
            <div>{metrics?.patients ?? 0}</div>
            <div>👥 Total Patients</div>
          </div>
          <div className="tile" style={{ background: 'var(--gradient-info)' }}>
            <div>{metrics?.predictions ?? 0}</div>
            <div>🔬 Total Predictions</div>
          </div>
        </div>

        {metrics?.predictions > 0 && (
          <div className="grid" style={{ marginTop: '20px' }}>
            <div className="tile danger">
              <div>{metrics?.malignant ?? 0}</div>
              <div>⚠️ Malignant Cases</div>
            </div>
            <div className="tile success">
              <div>{metrics?.benign ?? 0}</div>
              <div>✅ Benign Cases</div>
            </div>
          </div>
        )}
      </div>

      <div className="card">
        <h3 style={{ marginBottom: '25px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span>🧠</span> AI Model Status
        </h3>
        {modelStatus?.error ? (
          <div className="error">
            <strong>⚠️ Model Server Unavailable</strong>
            <p style={{ marginTop: '8px', marginBottom: 0 }}>
              Please ensure the ML service is running on port 8000.
            </p>
          </div>
        ) : modelStatus ? (
          <div style={{
            background: 'var(--gradient-success)',
            color: 'white',
            padding: 'var(--space-xl)',
            borderRadius: 'var(--radius-lg)',
            boxShadow: 'var(--shadow-lg)'
          }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              <div>
                <div style={{ fontSize: '0.9em', opacity: 0.9, marginBottom: '5px' }}>Model Version</div>
                <div style={{ fontSize: '1.3em', fontWeight: 'bold' }}>{modelStatus.model_version || 'N/A'}</div>
              </div>
              <div>
                <div style={{ fontSize: '0.9em', opacity: 0.9, marginBottom: '5px' }}>Uptime</div>
                <div style={{ fontSize: '1.3em', fontWeight: 'bold' }}>
                  {Math.floor((modelStatus.uptime_seconds || 0) / 60)} min
                </div>
              </div>
              <div>
                <div style={{ fontSize: '0.9em', opacity: 0.9, marginBottom: '5px' }}>Input Dimension</div>
                <div style={{ fontSize: '1.3em', fontWeight: 'bold' }}>{modelStatus.input_dim || 'N/A'}</div>
              </div>
              <div>
                <div style={{ fontSize: '0.9em', opacity: 0.9, marginBottom: '5px' }}>Embedding Dimension</div>
                <div style={{ fontSize: '1.3em', fontWeight: 'bold' }}>{modelStatus.embed_dim || 'N/A'}</div>
              </div>
            </div>
            <div style={{
              marginTop: '20px',
              padding: '15px',
              background: 'rgba(255, 255, 255, 0.2)',
              borderRadius: 'var(--radius-md)',
              textAlign: 'center',
              fontWeight: '600'
            }}>
              ✅ System Ready for Predictions
            </div>
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '40px', color: 'var(--color-text-muted)' }}>
            No model status available
          </div>
        )}
      </div>
    </div>
  );
}
