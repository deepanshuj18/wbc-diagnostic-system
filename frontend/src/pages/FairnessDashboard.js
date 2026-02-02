import React, { useState, useEffect } from 'react';
import { getFairnessMetrics } from '../api';

export default function FairnessDashboard() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadMetrics();
  }, []);

  const loadMetrics = async () => {
    try {
      setLoading(true);
      const data = await getFairnessMetrics();
      setMetrics(data);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderBarChart = (label, value, color) => (
    <div key={label} style={{ marginBottom: '15px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
        <span style={{ fontWeight: 'bold' }}>{label}</span>
        <span style={{ fontWeight: 'bold', color: color || '#333' }}>
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div style={{
        width: '100%',
        height: '25px',
        backgroundColor: '#e0e0e0',
        borderRadius: '5px',
        overflow: 'hidden'
      }}>
        <div style={{
          width: `${value * 100}%`,
          height: '100%',
          backgroundColor: color || '#4CAF50',
          transition: 'width 0.5s ease'
        }} />
      </div>
    </div>
  );

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '40px' }}>
        <div style={{ fontSize: '18px' }}>Loading fairness metrics...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ textAlign: 'center', padding: '40px', color: '#d32f2f' }}>
        <div style={{ fontSize: '18px' }}>Error: {error}</div>
        <button onClick={loadMetrics} style={{
          marginTop: '15px',
          padding: '10px 20px',
          backgroundColor: '#1976d2',
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: 'pointer'
        }}>
          Retry
        </button>
      </div>
    );
  }

  if (!metrics || !metrics.fairness_metrics) {
    return <div style={{ padding: '20px' }}>No fairness data available</div>;
  }

  const fm = metrics.fairness_metrics;
  const colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'];

  return (
    <div style={{ padding: '20px' }}>
      <h2 style={{ marginBottom: '30px', color: '#1976d2' }}>Fairness & Bias Audit Dashboard</h2>
      <p style={{ marginBottom: '30px', color: '#666' }}>
        Model fairness metrics across demographic groups (v2.0.0)
      </p>

      {/* Overall Metrics */}
      <div style={{
        backgroundColor: 'white',
        padding: '25px',
        borderRadius: '10px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        marginBottom: '30px'
      }}>
        <h3 style={{ marginBottom: '20px', color: '#333' }}>Overall Performance</h3>
        {renderBarChart('Accuracy', fm.overall?.accuracy || 0, '#4CAF50')}
        {renderBarChart('Precision', fm.overall?.precision || 0, '#2196F3')}
        {renderBarChart('Recall', fm.overall?.recall || 0, '#FF9800')}
        {renderBarChart('F1-Score', fm.overall?.f1 || 0, '#9C27B0')}
      </div>

      {/* Age Groups */}
      <div style={{
        backgroundColor: 'white',
        padding: '25px',
        borderRadius: '10px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        marginBottom: '30px'
      }}>
        <h3 style={{ marginBottom: '20px', color: '#333' }}>Performance by Age Group</h3>
        {['age_<40', 'age_40-60', 'age_>60'].map((key, idx) => {
          const ageGroup = fm[key];
          if (!ageGroup) return null;
          return (
            <div key={key} style={{ marginBottom: '20px', padding: '15px', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
              <h4 style={{ marginBottom: '10px', color: '#555' }}>
                {key.replace('age_', 'Age ')} ({ageGroup.count} samples)
              </h4>
              {renderBarChart('Accuracy', ageGroup.accuracy, '#4CAF50')}
              {renderBarChart('Precision', ageGroup.precision, '#2196F3')}
              {renderBarChart('Recall', ageGroup.recall, '#FF9800')}
            </div>
          );
        })}
      </div>

      {/* Gender Groups */}
      <div style={{
        backgroundColor: 'white',
        padding: '25px',
        borderRadius: '10px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        marginBottom: '30px'
      }}>
        <h3 style={{ marginBottom: '20px', color: '#333' }}>Performance by Gender</h3>
        {['gender_M', 'gender_F', 'gender_Other'].map((key, idx) => {
          const genderGroup = fm[key];
          if (!genderGroup) return null;
          const genderLabel = key.replace('gender_', '').replace('M', 'Male').replace('F', 'Female');
          return (
            <div key={key} style={{ marginBottom: '20px', padding: '15px', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
              <h4 style={{ marginBottom: '10px', color: '#555' }}>
                {genderLabel} ({genderGroup.count} samples)
              </h4>
              {renderBarChart('Accuracy', genderGroup.accuracy, '#4CAF50')}
              {renderBarChart('Precision', genderGroup.precision, '#2196F3')}
              {renderBarChart('Recall', genderGroup.recall, '#FF9800')}
            </div>
          );
        })}
      </div>

      {/* Demographic Parity */}
      {fm.demographic_parity_diff !== undefined && (
        <div style={{
          backgroundColor: fm.demographic_parity_diff > 0.1 ? '#ffebee' : '#e8f5e9',
          padding: '20px',
          borderRadius: '10px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          border: `2px solid ${fm.demographic_parity_diff > 0.1 ? '#f44336' : '#4CAF50'}`
        }}>
          <h3 style={{ marginBottom: '10px', color: '#333' }}>Demographic Parity</h3>
          <p style={{ fontSize: '16px', color: '#555' }}>
            Demographic Parity Difference: <strong>
              {fm.demographic_parity_diff.toFixed(4)}
            </strong>
          </p>
          <p style={{ fontSize: '14px', color: '#666', marginTop: '10px' }}>
            {fm.demographic_parity_diff > 0.1 
              ? '⚠️ Significant bias detected (>0.1 difference)'
              : '✓ Fair model performance'}
          </p>
        </div>
      )}

      <div style={{ marginTop: '30px', fontSize: '14px', color: '#888', textAlign: 'center' }}>
        Test samples: {metrics.test_samples || 'N/A'}
      </div>
    </div>
  );
}

