import React, { useRef, useState, useEffect } from 'react';

function UploadZone({ file, preview, dragActive, fileInputRef, onFile, onDrop, setDragActive, includeFrames, onIncludeFramesChange }) {
  const videoRef = useRef(null);
  const [poster, setPoster] = useState(null);

  // Reset poster whenever a new file is selected
  useEffect(() => {
    setPoster(null);
  }, [preview]);

  const handleLoadedMetadata = () => {
    const video = videoRef.current;
    if (!video) return;
    // Seek to 15% of duration (capped at 2s) to get a representative face frame
    const seekTo = isFinite(video.duration) ? Math.min(2, video.duration * 0.15) : 0.5;
    video.currentTime = seekTo;
  };

  const handleSeeked = () => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) return;
    try {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      setPoster(canvas.toDataURL('image/jpeg', 0.85));
    } catch {
      // Ignore canvas errors (e.g. cross-origin)
    }
  };

  return (
    <div
      className={`upload-zone ${dragActive ? 'drag-active' : ''} ${file ? 'has-file' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
      onDragLeave={() => setDragActive(false)}
      onDrop={onDrop}
      onClick={() => !file && fileInputRef.current?.click()}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        onChange={(e) => e.target.files[0] && onFile(e.target.files[0])}
        hidden
      />

      {!file ? (
        <div className="upload-content">
          <div className="upload-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>
          <p className="upload-text">Drop a video here or <span className="browse-link">browse</span></p>
          <p className="upload-hint">Supports MP4, AVI, MOV, MKV, WebM</p>
        </div>
      ) : (
        <div className="preview-section">
          <video
            ref={videoRef}
            src={preview}
            preload="auto"
            controls
            poster={poster || undefined}
            className="video-preview"
            onLoadedMetadata={handleLoadedMetadata}
            onSeeked={handleSeeked}
          />
          <p className="file-name">{file.name}</p>
          <p className="file-size">{(file.size / (1024 * 1024)).toFixed(1)} MB</p>
          <label className="include-frames-toggle" onClick={(e) => e.stopPropagation()}>
            <input
              type="checkbox"
              checked={includeFrames}
              onChange={(e) => onIncludeFramesChange(e.target.checked)}
            />
            Include frame-by-frame analysis
          </label>
        </div>
      )}
    </div>
  );
}

export default UploadZone;
