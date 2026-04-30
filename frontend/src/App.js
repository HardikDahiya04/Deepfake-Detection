import React, { useState, useRef, useCallback, useEffect } from 'react';
import axios from 'axios';
import './App.css';

import Header from './components/Header';
import UploadZone from './components/UploadZone';
import ProgressIndicator from './components/ProgressIndicator';
import ResultCard from './components/ResultCard';
import ProcessingDetails from './components/ProcessingDetails';
import FrameTimeline from './components/FrameTimeline';
import AnalysisHistory from './components/AnalysisHistory';
import Footer from './components/Footer';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  // File state
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  // Analysis state
  const [result, setResult] = useState(null);
  const [loadingStep, setLoadingStep] = useState(null);
  const [error, setError] = useState(null);
  const [includeFrames, setIncludeFrames] = useState(false);

  // User controls
  const [threshold, setThreshold] = useState(0.5);
  const [history, setHistory] = useState([]);

  // Theme
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('deepfake-theme') || 'light';
  });

  // Timeout refs for progress steps
  const stepTimers = useRef([]);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('deepfake-theme', theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  }, []);

  const clearStepTimers = useCallback(() => {
    stepTimers.current.forEach(t => clearTimeout(t));
    stepTimers.current = [];
  }, []);

  const handleFile = useCallback((selectedFile) => {
    const allowed = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm'];
    if (!allowed.includes(selectedFile.type) && !selectedFile.name.match(/\.(mp4|avi|mov|mkv|webm)$/i)) {
      setError('Please upload a valid video file (MP4, AVI, MOV, MKV, WebM)');
      return;
    }
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError(null);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, [handleFile]);

  const handleSubmit = async () => {
    if (!file) return;
    setLoadingStep(1);
    setError(null);
    setResult(null);
    clearStepTimers();

    // Simulate progress steps while waiting for the API
    stepTimers.current.push(setTimeout(() => setLoadingStep(2), 1500));
    stepTimers.current.push(setTimeout(() => setLoadingStep(3), 3500));

    const formData = new FormData();
    formData.append('video', file);

    const url = includeFrames ? `${API_URL}/predict?include_frames=true` : `${API_URL}/predict`;

    try {
      const response = await axios.post(url, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000,
      });

      clearStepTimers();
      setLoadingStep(4);

      const data = response.data;
      setResult(data);

      // Add to history
      const prediction = data.fake_probability > threshold ? 'FAKE' : 'REAL';
      const confidence = prediction === 'FAKE' ? data.fake_probability : 1 - data.fake_probability;
      setHistory(prev => [{
        id: Date.now(),
        fileName: file.name,
        fileSize: file.size,
        prediction,
        confidence,
        fakeProbability: data.fake_probability,
        timestamp: new Date().toISOString(),
      }, ...prev]);

      // Clear "Complete" step after a short delay
      setTimeout(() => setLoadingStep(null), 800);

    } catch (err) {
      clearStepTimers();
      setLoadingStep(null);
      const msg = err.response?.data?.detail || err.message || 'Prediction failed';
      setError(msg);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setLoadingStep(null);
    clearStepTimers();
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const isLoading = loadingStep !== null && loadingStep < 4;

  return (
    <div className="app">
      <Header theme={theme} onToggleTheme={toggleTheme} />

      <main className="main">
        <UploadZone
          file={file}
          preview={preview}
          dragActive={dragActive}
          fileInputRef={fileInputRef}
          onFile={handleFile}
          onDrop={handleDrop}
          setDragActive={setDragActive}
          includeFrames={includeFrames}
          onIncludeFramesChange={setIncludeFrames}
        />

        {/* Actions */}
        <div className="actions">
          {file && (
            <>
              <button
                className="btn btn-primary"
                onClick={handleSubmit}
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <span className="spinner" />
                    Analyzing...
                  </>
                ) : (
                  'Analyze Video'
                )}
              </button>
              <button className="btn btn-secondary" onClick={handleReset} disabled={isLoading}>
                Reset
              </button>
            </>
          )}
        </div>

        {/* Progress */}
        <ProgressIndicator step={loadingStep} />

        {/* Error */}
        {error && (
          <div className="error-card">
            <p>{error}</p>
          </div>
        )}

        {/* Results */}
        {result && (
          <>
            <ResultCard
              result={result}
              threshold={threshold}
              onThresholdChange={setThreshold}
            />
            <ProcessingDetails details={result.processing_details} />
            <FrameTimeline frameAnalysis={result.frame_analysis} />
          </>
        )}

        {/* History */}
        <AnalysisHistory
          history={history}
          onClear={() => setHistory([])}
        />

      </main>

      <Footer />
    </div>
  );
}

export default App;
