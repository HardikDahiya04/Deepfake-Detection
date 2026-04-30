import React from 'react';
import RiskGauge from './RiskGauge';

function ResultCard({ result, threshold, onThresholdChange }) {
  // sensitivity is the inverse of threshold: high sensitivity = low threshold = catches more fakes
  const sensitivity = 1 - threshold;
  const prediction = result.fake_probability > threshold ? 'FAKE' : 'REAL';
  const confidence = prediction === 'FAKE' ? result.fake_probability : 1 - result.fake_probability;

  return (
    <div className={`result-card ${prediction === 'FAKE' ? 'result-fake' : 'result-real'}`}>
      <div className="result-header">
        <span className={`result-badge ${prediction === 'FAKE' ? 'badge-fake' : 'badge-real'}`}>
          {prediction}
        </span>
      </div>
      <div className="result-details">
        <div className="detail-row">
          <span className="detail-label">Confidence</span>
          <span className="detail-value">{(confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="confidence-bar">
          <div
            className={`confidence-fill ${prediction === 'FAKE' ? 'fill-fake' : 'fill-real'}`}
            style={{ width: `${confidence * 100}%` }}
          />
        </div>
        {result.fake_probability !== undefined && (
          <div className="detail-row">
            <span className="detail-label">Fake Probability</span>
            <span className="detail-value">{(result.fake_probability * 100).toFixed(1)}%</span>
          </div>
        )}
      </div>

      <RiskGauge fakeProbability={result.fake_probability} threshold={threshold} />

      <div className="threshold-control">
        <div className="threshold-label">
          <span>Detection Sensitivity</span>
          <span className="threshold-value">{(sensitivity * 100).toFixed(0)}%</span>
        </div>
        <input
          type="range"
          className="threshold-slider"
          min="0"
          max="100"
          value={sensitivity * 100}
          onChange={(e) => onThresholdChange(1 - Number(e.target.value) / 100)}
        />
      </div>
    </div>
  );
}

export default ResultCard;
