import React, { useState } from 'react';

function formatTimeAgo(timestamp) {
  const seconds = Math.floor((Date.now() - new Date(timestamp).getTime()) / 1000);
  if (seconds < 60) return 'Just now';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
}

function AnalysisHistory({ history, onClear }) {
  const [expanded, setExpanded] = useState(false);

  if (history.length === 0) return null;

  return (
    <div className="analysis-history">
      <div className="history-header" onClick={() => setExpanded(!expanded)}>
        <div className="history-header-left">
          <span>Analysis History</span>
          <span className="history-count">{history.length}</span>
        </div>
        <span className={`history-toggle-icon ${expanded ? 'expanded' : ''}`}>
          &#9660;
        </span>
      </div>

      {expanded && (
        <div className="history-body">
          {history.map((entry) => (
            <div key={entry.id} className="history-entry">
              <span className={`history-entry-badge ${entry.prediction === 'FAKE' ? 'badge-fake' : 'badge-real'}`}>
                {entry.prediction}
              </span>
              <span className="history-entry-name" title={entry.fileName}>
                {entry.fileName.length > 30 ? entry.fileName.slice(0, 27) + '...' : entry.fileName}
              </span>
              <span className="history-entry-confidence">
                {(entry.confidence * 100).toFixed(1)}%
              </span>
              <span className="history-entry-time">
                {formatTimeAgo(entry.timestamp)}
              </span>
            </div>
          ))}
          <div className="history-clear">
            <button className="history-clear-btn" onClick={onClear}>
              Clear History
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default AnalysisHistory;
