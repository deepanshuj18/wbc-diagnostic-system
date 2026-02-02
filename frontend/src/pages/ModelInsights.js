import React, { useState, useEffect } from 'react';
import { getEmbeddingsVisualization } from '../api';

export default function ModelInsights() {
  const [embeddings, setEmbeddings] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadEmbeddings();
  }, []);

  const loadEmbeddings = async () => {
    try {
      setLoading(true);
      const data = await getEmbeddingsVisualization();
      setEmbeddings(data);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderTSNEPlot = () => {
    if (!embeddings || !embeddings.embeddings) return null;

    const { embeddings: coords, labels, label_names } = embeddings;
    const benign = coords.filter((_, i) => labels[i] === 0);
    const malignant = coords.filter((_, i) => labels[i] === 1);

    // Scale to 0-100 for display
    const scale = (val, min, max) => ((val - min) / (max - min)) * 100;
    
    const allX = coords.map(p => p[0]);
    const allY = coords.map(p => p[1]);
    const minX = Math.min(...allX);
    const maxX = Math.max(...allX);
    const minY = Math.min(...allY);
    const maxY = Math.max(...allY);

    return (
      <div style={{ position: 'relative', width: '100%', height: '500px', border: '2px solid #e0e0e0', borderRadius: '10px', backgroundColor: 'white' }}>
        {benign.map((point, idx) => (
          <div
            key={`benign-${idx}`}
            style={{
              position: 'absolute',
              left: `${scale(point[0], minX, maxX)}%`,
              bottom: `${scale(point[1], minY, maxY)}%`,
              width: '8px',
              height: '8px',
              backgroundColor: '#4CAF50',
              borderRadius: '50%',
              border: '1px solid white'
            }}
            title={`Benign: [${point[0].toFixed(2)}, ${point[1].toFixed(2)}]`}
          />
        ))}
        {malignant.map((point, idx) => (
          <div
            key={`malignant-${idx}`}
            style={{
              position: 'absolute',
              left: `${scale(point[0], minX, maxX)}%`,
              bottom: `${scale(point[1], minY, maxY)}%`,
              width: '8px',
              height: '8px',
              backgroundColor: '#f44336',
              borderRadius: '50%',
              border: '1px solid white'
            }}
            title={`Malignant: [${point[0].toFixed(2)}, ${point[1].toFixed(2)}]`}
          />
        ))}
        <div style={{ position: 'absolute', top: '10px', left: '10px', backgroundColor: 'rgba(255,255,255,0.9)', padding: '10px', borderRadius: '5px' }}>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '5px' }}>
            <div style={{ width: '12px', height: '12px', backgroundColor: '#4CAF50', borderRadius: '50%', marginRight: '8px' }}></div>
            <span>Benign ({benign.length})</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <div style={{ width: '12px', height: '12px', backgroundColor: '#f44336', borderRadius: '50%', marginRight: '8px' }}></div>
            <span>Malignant ({malignant.length})</span>
          </div>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '40px' }}>
        <div style={{ fontSize: '18px' }}>Generating embedding visualization...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ textAlign: 'center', padding: '40px', color: '#d32f2f' }}>
        <div style={{ fontSize: '18px' }}>Error: {error}</div>
        <button onClick={loadEmbeddings} style={{
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

  return (
    <div style={{ padding: '20px' }}>
      <h2 style={{ marginBottom: '30px', color: '#1976d2' }}>Model Insights & Visualization</h2>
      <p style={{ marginBottom: '30px', color: '#666' }}>
        t-SNE visualization of learned embeddings colored by prediction
      </p>

      <div style={{
        backgroundColor: 'white',
        padding: '25px',
        borderRadius: '10px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        marginBottom: '30px'
      }}>
        <h3 style={{ marginBottom: '20px', color: '#333' }}>Embedding Space</h3>
        {renderTSNEPlot()}
        <p style={{ marginTop: '15px', fontSize: '14px', color: '#666' }}>
          This visualization shows how the model clusters similar patients in embedding space. 
          Well-separated clusters indicate good feature representation.
        </p>
      </div>

      {/* Feature Importance Placeholder */}
      <div style={{
        backgroundColor: 'white',
        padding: '25px',
        borderRadius: '10px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        marginBottom: '30px'
      }}>
        <h3 style={{ marginBottom: '20px', color: '#333' }}>Model Architecture</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px' }}>
          <div style={{ padding: '15px', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
            <div style={{ fontWeight: 'bold', color: '#555', marginBottom: '5px' }}>Input Features</div>
            <div style={{ fontSize: '24px', color: '#1976d2' }}>30</div>
          </div>
          <div style={{ padding: '15px', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
            <div style={{ fontWeight: 'bold', color: '#555', marginBottom: '5px' }}>Embedding Dim</div>
            <div style={{ fontSize: '24px', color: '#1976d2' }}>32</div>
          </div>
          <div style={{ padding: '15px', backgroundColor: '#f5f5f5', borderRadius: '8px' }}>
            <div style={{ fontWeight: 'bold', color: '#555', marginBottom: '5px' }}>Architecture</div>
            <div style={{ fontSize: '18px', color: '#1976d2' }}>MAE → LR/TabNet</div>
          </div>
        </div>
      </div>

      {/* Model Info */}
      <div style={{
        backgroundColor: '#e3f2fd',
        padding: '20px',
        borderRadius: '10px',
        border: '2px solid #1976d2'
      }}>
        <h3 style={{ marginBottom: '15px', color: '#1976d2' }}>Model Version: 2.0.0</h3>
        <ul style={{ lineHeight: '1.8', color: '#555' }}>
          <li><strong>Hybrid Architecture:</strong> Masked Autoencoder + TabNet/Logistic Regression</li>
          <li><strong>Explainability:</strong> SHAP, LIME, Integrated Gradients</li>
          <li><strong>Calibration:</strong> Temperature Scaling / Isotonic Regression</li>
          <li><strong>Fairness:</strong> Demographic parity monitoring</li>
        </ul>
      </div>
    </div>
  );
}

