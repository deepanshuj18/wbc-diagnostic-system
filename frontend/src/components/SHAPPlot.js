import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

export default function SHAPPlot({ featureNames=[], shapValues=[], methodName='SHAP' }){
  const ref = useRef();
  useEffect(()=> {
    if (!shapValues || shapValues.length === 0) return;
    const values = shapValues;
    const absValues = values.map(v => Math.abs(v));

    // Sort by absolute importance (descending)
    const indices = absValues.map((_, i) => i)
      .sort((a, b) => absValues[a] - absValues[b]);

    const sortedNames = indices.map(i => featureNames[i] || `Feature ${i+1}`);
    const sortedValues = indices.map(i => absValues[i]);

    const trace = {
      x: sortedValues,
      y: sortedNames,
      orientation: 'h',
      type: 'bar',
      marker: {
        color: sortedValues.map(v => v > 0 ? 'rgba(99,102,241,0.8)' : 'rgba(239,68,68,0.8)')
      }
    };
    const layout = { title: `Feature importance (${methodName})`, margin: { l: 200 } };
    Plotly.newPlot(ref.current, [trace], layout, {responsive:true});
  }, [featureNames, shapValues, methodName]);
  return <div ref={ref} style={{width: '100%', height: '400px'}} />;
}
