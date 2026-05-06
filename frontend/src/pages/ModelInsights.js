import React, { useState, useEffect } from 'react';
import { getModelEvaluation, getEmbeddingsVisualization } from '../api';

export default function ModelInsights() {
  const [evaluation, setEvaluation] = useState(null);
  const [embeddings, setEmbeddings] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => { loadData(); }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const [evalData, embData] = await Promise.all([
        getModelEvaluation(),
        getEmbeddingsVisualization()
      ]);
      setEvaluation(evalData);
      setEmbeddings(embData);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div style={{ textAlign: 'center', padding: '60px' }}><div style={{ fontSize: '18px', color: 'var(--color-text-secondary)' }}>Loading model evaluation data...</div></div>;
  if (error) return (
    <div style={{ textAlign: 'center', padding: '60px' }}>
      <div style={{ fontSize: '18px', color: '#ef4444' }}>Error: {error}</div>
      <button onClick={loadData} style={{ marginTop: '15px', padding: '10px 24px' }}>Retry</button>
    </div>
  );

  const s = evaluation?.summary || {};
  const cm = evaluation?.confusion_matrix || [[0,0],[0,0]];
  const tabs = [
    { key: 'overview', label: '📊 Overview' },
    { key: 'confusion', label: '🎯 Confusion Matrix' },
    { key: 'curves', label: '📈 ROC & PR Curves' },
    { key: 'features', label: '🔬 Feature Importance' },
    { key: 'embeddings', label: '🧬 Embeddings' },
  ];

  const cardStyle = {
    background: 'var(--glass-bg)', backdropFilter: 'blur(10px)',
    padding: '25px', borderRadius: '16px',
    border: '1px solid var(--glass-border)', boxShadow: 'var(--shadow-lg)',
    marginBottom: '24px'
  };

  const metricCard = (label, value, color, icon) => (
    <div style={{
      flex: '1', minWidth: '140px', padding: '20px',
      background: `linear-gradient(135deg, ${color}15, ${color}08)`,
      borderRadius: '14px', border: `1px solid ${color}30`, textAlign: 'center'
    }}>
      <div style={{ fontSize: '14px', color: 'var(--color-text-secondary)', marginBottom: '8px' }}>{icon} {label}</div>
      <div style={{ fontSize: '28px', fontWeight: '800', color }}>{(value * 100).toFixed(1)}%</div>
    </div>
  );

  // SVG-based curve renderer
  const renderCurve = (points, width, height, color, label) => {
    if (!points || points.length < 2) return null;
    const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p[0] * width} ${height - p[1] * height}`).join(' ');
    return <path d={path} fill="none" stroke={color} strokeWidth="2.5" />;
  };

  return (
    <div style={{ padding: '10px 0' }}>
      <h2 style={{ marginBottom: '8px', background: 'var(--gradient-primary)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
        Model Insights & Evaluation
      </h2>
      <p style={{ marginBottom: '24px', color: 'var(--color-text-secondary)' }}>
        Comprehensive model performance analysis on test data ({evaluation?.test_samples || 'N/A'} samples)
      </p>

      {/* Sub-tabs */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '28px', flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.key} onClick={() => setActiveTab(t.key)} style={{
            padding: '10px 20px', borderRadius: '10px', fontSize: '14px', fontWeight: '600',
            background: activeTab === t.key ? 'var(--gradient-primary)' : 'var(--glass-bg)',
            color: activeTab === t.key ? 'white' : 'var(--color-text-secondary)',
            border: activeTab === t.key ? 'none' : '1px solid var(--glass-border)',
            cursor: 'pointer', transition: 'all 0.2s', textTransform: 'none', letterSpacing: '0',
            boxShadow: activeTab === t.key ? 'var(--shadow-md)' : 'none'
          }}>{t.label}</button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {activeTab === 'overview' && (
        <div>
          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', marginBottom: '24px' }}>
            {metricCard('Accuracy', s.accuracy || 0, '#10b981', '🎯')}
            {metricCard('Precision', s.precision || 0, '#3b82f6', '🔍')}
            {metricCard('Recall', s.recall || 0, '#f59e0b', '📡')}
            {metricCard('F1-Score', s.f1 || 0, '#8b5cf6', '⚡')}
            {metricCard('ROC AUC', s.roc_auc || 0, '#06b6d4', '📈')}
            {metricCard('Avg Precision', s.avg_precision || 0, '#ec4899', '🏆')}
          </div>

          {/* Classification Report Table */}
          {evaluation?.classification_report && (
            <div style={cardStyle}>
              <h3 style={{ marginBottom: '16px', color: 'var(--color-text-primary)' }}>Classification Report</h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid var(--glass-border)' }}>
                      {['Class', 'Precision', 'Recall', 'F1-Score', 'Support'].map(h => (
                        <th key={h} style={{ padding: '12px 16px', textAlign: 'left', fontWeight: '700', color: 'var(--color-text-secondary)', textTransform: 'uppercase', fontSize: '12px', letterSpacing: '0.5px' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {['Malignant', 'Benign'].map(cls => {
                      const r = evaluation.classification_report[cls];
                      if (!r) return null;
                      return (
                        <tr key={cls} style={{ borderBottom: '1px solid var(--glass-border)' }}>
                          <td style={{ padding: '12px 16px', fontWeight: '600' }}>
                            <span style={{ display: 'inline-block', width: '10px', height: '10px', borderRadius: '50%', backgroundColor: cls === 'Malignant' ? '#ef4444' : '#10b981', marginRight: '8px' }}></span>
                            {cls}
                          </td>
                          <td style={{ padding: '12px 16px' }}>{(r.precision * 100).toFixed(1)}%</td>
                          <td style={{ padding: '12px 16px' }}>{(r.recall * 100).toFixed(1)}%</td>
                          <td style={{ padding: '12px 16px', fontWeight: '600' }}>{(r['f1-score'] * 100).toFixed(1)}%</td>
                          <td style={{ padding: '12px 16px' }}>{r.support}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Model Architecture */}
          <div style={cardStyle}>
            <h3 style={{ marginBottom: '16px', color: 'var(--color-text-primary)' }}>Model Architecture</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
              {[
                { label: 'Input Features', value: '30', color: '#3b82f6' },
                { label: 'Embedding Dim', value: '32', color: '#8b5cf6' },
                { label: 'Architecture', value: 'MAE → LR/TabNet', color: '#06b6d4' },
                { label: 'Calibration', value: 'Temp Scaling', color: '#10b981' },
              ].map(item => (
                <div key={item.label} style={{ padding: '16px', background: `${item.color}10`, borderRadius: '12px', border: `1px solid ${item.color}25` }}>
                  <div style={{ fontSize: '12px', fontWeight: '600', color: 'var(--color-text-secondary)', marginBottom: '6px', textTransform: 'uppercase' }}>{item.label}</div>
                  <div style={{ fontSize: '20px', fontWeight: '700', color: item.color }}>{item.value}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* CONFUSION MATRIX TAB */}
      {activeTab === 'confusion' && (
        <div style={cardStyle}>
          <h3 style={{ marginBottom: '20px' }}>Confusion Matrix</h3>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <div>
              <div style={{ display: 'flex', marginBottom: '8px', paddingLeft: '120px' }}>
                <div style={{ width: '120px', textAlign: 'center', fontWeight: '700', fontSize: '13px', color: 'var(--color-text-secondary)' }}>Pred: Malignant</div>
                <div style={{ width: '120px', textAlign: 'center', fontWeight: '700', fontSize: '13px', color: 'var(--color-text-secondary)' }}>Pred: Benign</div>
              </div>
              {[['True: Malignant', 0], ['True: Benign', 1]].map(([label, row]) => (
                <div key={label} style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
                  <div style={{ width: '120px', fontWeight: '700', fontSize: '13px', color: 'var(--color-text-secondary)', textAlign: 'right', paddingRight: '12px' }}>{label}</div>
                  {[0, 1].map(col => {
                    const val = cm[row]?.[col] || 0;
                    const isCorrect = row === col;
                    return (
                      <div key={col} style={{
                        width: '120px', height: '90px', display: 'flex', flexDirection: 'column',
                        alignItems: 'center', justifyContent: 'center', margin: '2px',
                        borderRadius: '12px', fontWeight: '800', fontSize: '28px',
                        background: isCorrect ? 'linear-gradient(135deg, #10b981, #059669)' : 'linear-gradient(135deg, #ef4444, #dc2626)',
                        color: 'white', boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
                      }}>
                        {val}
                        <div style={{ fontSize: '10px', fontWeight: '500', opacity: 0.85, marginTop: '4px' }}>
                          {isCorrect ? (row === 0 ? 'TN' : 'TP') : (row === 0 ? 'FP' : 'FN')}
                        </div>
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
          <p style={{ textAlign: 'center', marginTop: '20px', fontSize: '14px', color: 'var(--color-text-muted)' }}>
            TN = True Negative, TP = True Positive, FP = False Positive, FN = False Negative
          </p>
        </div>
      )}

      {/* ROC & PR CURVES TAB */}
      {activeTab === 'curves' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '24px' }}>
          {/* ROC Curve */}
          <div style={cardStyle}>
            <h3 style={{ marginBottom: '4px' }}>ROC Curve</h3>
            <p style={{ fontSize: '14px', color: 'var(--color-text-muted)', marginBottom: '16px' }}>AUC = {evaluation?.roc_curve?.auc?.toFixed(4) || 'N/A'}</p>
            <div style={{ position: 'relative', background: 'var(--color-bg-primary)', borderRadius: '12px', padding: '16px' }}>
              <svg viewBox="0 0 320 280" style={{ width: '100%', height: 'auto' }}>
                {/* Grid */}
                {[0, 0.25, 0.5, 0.75, 1].map(v => (
                  <g key={v}>
                    <line x1={40} y1={240 - v * 200} x2={300} y2={240 - v * 200} stroke="var(--glass-border)" strokeWidth="0.5" />
                    <text x={35} y={244 - v * 200} textAnchor="end" fill="var(--color-text-muted)" fontSize="10">{v.toFixed(1)}</text>
                    <line x1={40 + v * 260} y1={40} x2={40 + v * 260} y2={240} stroke="var(--glass-border)" strokeWidth="0.5" />
                    <text x={40 + v * 260} y={256} textAnchor="middle" fill="var(--color-text-muted)" fontSize="10">{v.toFixed(1)}</text>
                  </g>
                ))}
                {/* Diagonal */}
                <line x1={40} y1={240} x2={300} y2={40} stroke="var(--color-text-muted)" strokeWidth="1" strokeDasharray="4,4" opacity="0.5" />
                {/* ROC curve */}
                {evaluation?.roc_curve && (() => {
                  const { fpr, tpr } = evaluation.roc_curve;
                  const path = fpr.map((f, i) => `${i === 0 ? 'M' : 'L'} ${40 + f * 260} ${240 - tpr[i] * 200}`).join(' ');
                  const fillPath = path + ` L ${40 + fpr[fpr.length - 1] * 260} 240 L 40 240 Z`;
                  return (<>
                    <path d={fillPath} fill="rgba(99, 102, 241, 0.1)" />
                    <path d={path} fill="none" stroke="#6366f1" strokeWidth="2.5" />
                  </>);
                })()}
                <text x={170} y={274} textAnchor="middle" fill="var(--color-text-secondary)" fontSize="11" fontWeight="600">False Positive Rate</text>
                <text x={12} y={140} textAnchor="middle" fill="var(--color-text-secondary)" fontSize="11" fontWeight="600" transform="rotate(-90, 12, 140)">True Positive Rate</text>
              </svg>
            </div>
          </div>

          {/* PR Curve */}
          <div style={cardStyle}>
            <h3 style={{ marginBottom: '4px' }}>Precision-Recall Curve</h3>
            <p style={{ fontSize: '14px', color: 'var(--color-text-muted)', marginBottom: '16px' }}>AP = {evaluation?.pr_curve?.average_precision?.toFixed(4) || 'N/A'}</p>
            <div style={{ position: 'relative', background: 'var(--color-bg-primary)', borderRadius: '12px', padding: '16px' }}>
              <svg viewBox="0 0 320 280" style={{ width: '100%', height: 'auto' }}>
                {[0, 0.25, 0.5, 0.75, 1].map(v => (
                  <g key={v}>
                    <line x1={40} y1={240 - v * 200} x2={300} y2={240 - v * 200} stroke="var(--glass-border)" strokeWidth="0.5" />
                    <text x={35} y={244 - v * 200} textAnchor="end" fill="var(--color-text-muted)" fontSize="10">{v.toFixed(1)}</text>
                    <line x1={40 + v * 260} y1={40} x2={40 + v * 260} y2={240} stroke="var(--glass-border)" strokeWidth="0.5" />
                    <text x={40 + v * 260} y={256} textAnchor="middle" fill="var(--color-text-muted)" fontSize="10">{v.toFixed(1)}</text>
                  </g>
                ))}
                {evaluation?.pr_curve && (() => {
                  const { recall, precision } = evaluation.pr_curve;
                  const path = recall.map((r, i) => `${i === 0 ? 'M' : 'L'} ${40 + r * 260} ${240 - precision[i] * 200}`).join(' ');
                  const fillPath = path + ` L ${40 + recall[recall.length - 1] * 260} 240 L 40 240 Z`;
                  return (<>
                    <path d={fillPath} fill="rgba(16, 185, 129, 0.1)" />
                    <path d={path} fill="none" stroke="#10b981" strokeWidth="2.5" />
                  </>);
                })()}
                <text x={170} y={274} textAnchor="middle" fill="var(--color-text-secondary)" fontSize="11" fontWeight="600">Recall</text>
                <text x={12} y={140} textAnchor="middle" fill="var(--color-text-secondary)" fontSize="11" fontWeight="600" transform="rotate(-90, 12, 140)">Precision</text>
              </svg>
            </div>
          </div>

          {/* Calibration Plot */}
          <div style={cardStyle}>
            <h3 style={{ marginBottom: '4px' }}>Calibration Plot</h3>
            <p style={{ fontSize: '14px', color: 'var(--color-text-muted)', marginBottom: '16px' }}>Predicted vs actual probabilities</p>
            <div style={{ position: 'relative', background: 'var(--color-bg-primary)', borderRadius: '12px', padding: '16px' }}>
              <svg viewBox="0 0 320 280" style={{ width: '100%', height: 'auto' }}>
                {[0, 0.25, 0.5, 0.75, 1].map(v => (
                  <g key={v}>
                    <line x1={40} y1={240 - v * 200} x2={300} y2={240 - v * 200} stroke="var(--glass-border)" strokeWidth="0.5" />
                    <text x={35} y={244 - v * 200} textAnchor="end" fill="var(--color-text-muted)" fontSize="10">{v.toFixed(1)}</text>
                    <line x1={40 + v * 260} y1={40} x2={40 + v * 260} y2={240} stroke="var(--glass-border)" strokeWidth="0.5" />
                    <text x={40 + v * 260} y={256} textAnchor="middle" fill="var(--color-text-muted)" fontSize="10">{v.toFixed(1)}</text>
                  </g>
                ))}
                <line x1={40} y1={240} x2={300} y2={40} stroke="var(--color-text-muted)" strokeWidth="1" strokeDasharray="4,4" opacity="0.5" />
                {evaluation?.calibration && evaluation.calibration.predicted.map((pred, i) => (
                  <circle key={i} cx={40 + pred * 260} cy={240 - evaluation.calibration.actual[i] * 200} r="5" fill="#f59e0b" stroke="white" strokeWidth="1.5" />
                ))}
                <text x={170} y={274} textAnchor="middle" fill="var(--color-text-secondary)" fontSize="11" fontWeight="600">Mean Predicted</text>
                <text x={12} y={140} textAnchor="middle" fill="var(--color-text-secondary)" fontSize="11" fontWeight="600" transform="rotate(-90, 12, 140)">Fraction Positive</text>
              </svg>
            </div>
          </div>
        </div>
      )}

      {/* FEATURE IMPORTANCE TAB */}
      {activeTab === 'features' && (
        <div style={cardStyle}>
          <h3 style={{ marginBottom: '20px' }}>Top Feature Importance</h3>
          {evaluation?.feature_importance?.length > 0 ? (
            evaluation.feature_importance.map((feat, idx) => {
              const maxImp = evaluation.feature_importance[0].importance;
              const pct = maxImp > 0 ? (feat.importance / maxImp) * 100 : 0;
              const colors = ['#6366f1', '#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe'];
              const color = colors[Math.min(Math.floor(idx / 3), colors.length - 1)];
              return (
                <div key={feat.name} style={{ marginBottom: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <span style={{ fontWeight: '600', fontSize: '14px', color: 'var(--color-text-primary)' }}>
                      {idx + 1}. {feat.name}
                    </span>
                    <span style={{ fontWeight: '700', fontSize: '14px', color }}>{feat.importance.toFixed(4)}</span>
                  </div>
                  <div style={{ width: '100%', height: '10px', backgroundColor: 'var(--color-bg-primary)', borderRadius: '5px', overflow: 'hidden' }}>
                    <div style={{ width: `${pct}%`, height: '100%', background: `linear-gradient(90deg, ${color}, ${color}aa)`, borderRadius: '5px', transition: 'width 0.8s ease' }} />
                  </div>
                </div>
              );
            })
          ) : (
            <p style={{ color: 'var(--color-text-muted)' }}>Feature importance not available for current model configuration.</p>
          )}
        </div>
      )}

      {/* EMBEDDINGS TAB */}
      {activeTab === 'embeddings' && (
        <div style={cardStyle}>
          <h3 style={{ marginBottom: '4px' }}>t-SNE Embedding Space</h3>
          <p style={{ fontSize: '14px', color: 'var(--color-text-muted)', marginBottom: '16px' }}>
            Visualization of learned embeddings — well-separated clusters indicate good feature representation
          </p>
          {embeddings?.embeddings ? (() => {
            const { embeddings: coords, labels } = embeddings;
            const allX = coords.map(p => p[0]), allY = coords.map(p => p[1]);
            const minX = Math.min(...allX), maxX = Math.max(...allX);
            const minY = Math.min(...allY), maxY = Math.max(...allY);
            const scale = (val, min, max) => ((val - min) / (max - min)) * 100;
            const benign = coords.filter((_, i) => labels[i] === 0);
            const malignant = coords.filter((_, i) => labels[i] === 1);
            return (
              <div style={{ position: 'relative', width: '100%', height: '500px', border: '1px solid var(--glass-border)', borderRadius: '12px', background: 'var(--color-bg-primary)' }}>
                {benign.map((p, i) => (
                  <div key={`b-${i}`} title={`Benign`} style={{
                    position: 'absolute', left: `${scale(p[0], minX, maxX)}%`, bottom: `${scale(p[1], minY, maxY)}%`,
                    width: '8px', height: '8px', backgroundColor: '#10b981', borderRadius: '50%', border: '1px solid white'
                  }} />
                ))}
                {malignant.map((p, i) => (
                  <div key={`m-${i}`} title={`Malignant`} style={{
                    position: 'absolute', left: `${scale(p[0], minX, maxX)}%`, bottom: `${scale(p[1], minY, maxY)}%`,
                    width: '8px', height: '8px', backgroundColor: '#ef4444', borderRadius: '50%', border: '1px solid white'
                  }} />
                ))}
                <div style={{ position: 'absolute', top: '12px', left: '12px', background: 'var(--glass-bg)', padding: '10px 14px', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px', gap: '8px' }}>
                    <div style={{ width: '10px', height: '10px', backgroundColor: '#10b981', borderRadius: '50%' }} />
                    <span style={{ fontSize: '13px' }}>Benign ({benign.length})</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <div style={{ width: '10px', height: '10px', backgroundColor: '#ef4444', borderRadius: '50%' }} />
                    <span style={{ fontSize: '13px' }}>Malignant ({malignant.length})</span>
                  </div>
                </div>
              </div>
            );
          })() : <p style={{ color: 'var(--color-text-muted)' }}>No embedding data available</p>}
        </div>
      )}

      <div style={{ textAlign: 'center', marginTop: '20px', fontSize: '13px', color: 'var(--color-text-muted)' }}>
        Model v{evaluation?.model_version || '2.0.0'} • {evaluation?.test_samples || 'N/A'} test samples
      </div>
    </div>
  );
}
