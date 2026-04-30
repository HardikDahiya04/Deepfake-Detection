import React from 'react';

const STEPS = [
  'Uploading',
  'Detecting faces',
  'Analyzing',
  'Complete',
];

function ProgressIndicator({ step }) {
  if (!step) return null;

  return (
    <div className="progress-indicator">
      {STEPS.map((label, i) => {
        const stepNum = i + 1;
        const isCompleted = step > stepNum;
        const isActive = step === stepNum;

        return (
          <React.Fragment key={i}>
            {i > 0 && (
              <div className={`progress-line ${step > stepNum ? 'completed' : ''}`} />
            )}
            <div className={`progress-step ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}>
              <div className="progress-circle">
                {isCompleted ? (
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                ) : (
                  stepNum
                )}
              </div>
              <span className="progress-step-label">{label}</span>
            </div>
          </React.Fragment>
        );
      })}
    </div>
  );
}

export default ProgressIndicator;
