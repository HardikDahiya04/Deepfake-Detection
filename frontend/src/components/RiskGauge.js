import React from 'react';

function getVerdict(probability) {
  if (probability <= 0.25) return 'Likely Authentic';
  if (probability <= 0.45) return 'Probably Real';
  if (probability <= 0.55) return 'Uncertain';
  if (probability <= 0.75) return 'Suspicious';
  return 'Likely Manipulated';
}

function RiskGauge({ fakeProbability, threshold }) {
  const percent = fakeProbability * 100;
  const thresholdPercent = threshold * 100;

  return (
    <div className="risk-gauge">
      <div className="risk-gauge-bar">
        <div
          className="risk-gauge-indicator"
          style={{ left: `${Math.min(Math.max(percent, 2), 98)}%` }}
        />
        <div
          className="risk-gauge-threshold"
          style={{ left: `${thresholdPercent}%` }}
        />
      </div>
      <div className="risk-gauge-labels">
        <span>Low Risk</span>
        <span>High Risk</span>
      </div>
      <div className="risk-gauge-verdict">{getVerdict(fakeProbability)}</div>
    </div>
  );
}

export default RiskGauge;
