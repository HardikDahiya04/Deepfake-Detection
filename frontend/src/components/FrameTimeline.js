import React, { useState } from 'react';

function scoreToColor(score) {
  // Normalize score (attention intensity) to a green-to-red color
  const clamped = Math.min(Math.max(score, 0), 1);
  const r = Math.round(clamped * 239 + (1 - clamped) * 34);
  const g = Math.round((1 - clamped) * 197 + clamped * 68);
  const b = Math.round(68);
  return `rgb(${r}, ${g}, ${b})`;
}

function FrameTimeline({ frameAnalysis }) {
  const [selectedFrame, setSelectedFrame] = useState(0);

  if (!frameAnalysis) return null;

  const { per_frame_scores, attention_frames } = frameAnalysis;
  if (!per_frame_scores || per_frame_scores.length === 0) return null;

  // Normalize scores to 0-1 range for coloring
  const maxScore = Math.max(...per_frame_scores);
  const minScore = Math.min(...per_frame_scores);
  const range = maxScore - minScore || 1;
  const normalizedScores = per_frame_scores.map(s => (s - minScore) / range);

  return (
    <div className="frame-timeline">
      <div className="frame-timeline-header">
        <span className="frame-timeline-title">Frame-by-Frame Analysis</span>
        <span style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>
          {per_frame_scores.length} frames
        </span>
      </div>

      <div className="frame-bar">
        {normalizedScores.map((score, i) => (
          <div
            key={i}
            className={`frame-segment ${selectedFrame === i ? 'selected' : ''}`}
            style={{ backgroundColor: scoreToColor(score) }}
            onClick={() => setSelectedFrame(i)}
          >
            <span className="frame-segment-tooltip">
              Frame {i + 1}: {(per_frame_scores[i]).toFixed(3)}
            </span>
          </div>
        ))}
      </div>

      <div className="frame-labels">
        <span>Frame 1</span>
        <span>Frame {per_frame_scores.length}</span>
      </div>

      {attention_frames && attention_frames[selectedFrame] && (
        <div className="frame-preview">
          <img
            src={`data:image/jpeg;base64,${attention_frames[selectedFrame]}`}
            alt={`Attention heatmap for frame ${selectedFrame + 1}`}
          />
          <p className="frame-preview-label">
            Attention Heatmap - Frame {selectedFrame + 1} (Score: {per_frame_scores[selectedFrame].toFixed(3)})
          </p>
        </div>
      )}
    </div>
  );
}

export default FrameTimeline;
