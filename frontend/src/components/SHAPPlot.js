import React, { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

export default function SHAPPlot({ featureNames=[], shapValues=[] }){
  const ref = useRef();
  useEffect(()=> {
    if (!shapValues || shapValues.length === 0) return;
    const values = shapValues;
    const trace = {
      x: values.map(v=>Math.abs(v)),
      y: featureNames,
      orientation: 'h',
      type: 'bar'
    };
    const layout = { title: 'Feature importance (abs SHAP)', margin: { l: 200 } };
    Plotly.newPlot(ref.current, [trace], layout, {responsive:true});
  }, [featureNames, shapValues]);
  return <div ref={ref} style={{width: '100%', height: '400px'}} />;
}
