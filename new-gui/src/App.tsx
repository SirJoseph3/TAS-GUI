import React, { useEffect } from 'react';
import { TitleBar } from './components/TitleBar';
import { Sidebar } from './components/Sidebar';
import { MainContent } from './components/MainContent';
import { StatusBar } from './components/StatusBar';
import { useProcessingStore } from './stores/processingStore';

function parseProgressLine(line: string) {
  const clean = (line || '').trim();
  if (!clean) return null;

  const out: {
    progress?: number;
    currentFrame?: number;
    totalFrames?: number;
    fps?: number;
    elapsed?: string;
    eta?: string;
    isInternal?: boolean;
    downloadProgress?: number;
    downloadPackage?: string;
    installStart?: boolean;
    installComplete?: boolean;
  } = {};

  if (clean.includes('[PROGRESS]')) {
    out.isInternal = true;
  }

  // Parse dependency download progress markers
  const downloadProgressMatch = clean.match(/\[DOWNLOAD_PROGRESS:(\d+)\]/);
  if (downloadProgressMatch) {
    out.downloadProgress = Number(downloadProgressMatch[1]);
    out.isInternal = true;
  }

  const downloadStartMatch = clean.match(/\[DOWNLOAD_START:([^\]]+)\]/);
  if (downloadStartMatch) {
    out.downloadPackage = downloadStartMatch[1];
    out.isInternal = true;
  }

  if (clean.includes('[INSTALL_START]')) {
    out.installStart = true;
    out.isInternal = true;
  }

  if (clean.includes('[INSTALL_COMPLETE]')) {
    out.installComplete = true;
    out.isInternal = true;
  }

  const pctMatch = clean.match(/(\d{1,3})%/);
  if (pctMatch) {
    const pct = Number(pctMatch[1]);
    if (!Number.isNaN(pct) && pct >= 0 && pct <= 100) out.progress = pct;
  }

  const framesMatch = clean.match(/Frames?:\s*(\d+)\s*\/\s*(\d+)/i);
  if (framesMatch) {
    out.currentFrame = Number(framesMatch[1]);
    out.totalFrames = Number(framesMatch[2]);
  }

  const fpsMatch = clean.match(/FPS:\s*(\d+(?:\.\d+)?)/i);
  if (fpsMatch) out.fps = Number(fpsMatch[1]);

  const elapsedMatch = clean.match(/Elapsed:\s*([0-9:]+)/i);
  if (elapsedMatch) out.elapsed = elapsedMatch[1];

  const etaMatch = clean.match(/ETA:\s*([0-9:]+)/i);
  if (etaMatch) out.eta = etaMatch[1];

  if (Object.keys(out).length === 0) return null;
  return out;
}

function App() {
  const {
    addLog,
    setError,
    completeProcessing,
    failProcessing,
    updateProgress,
    setFrameInfo,
    setPerformance,
  } = useProcessingStore();

  useEffect(() => {
    // Setup Electron IPC listeners
    if (window.electronAPI) {
      window.electronAPI.onProcessOutput((data) => {
        const line = String(data.data ?? '');

        const parsed = parseProgressLine(line);
        if (parsed) {
          // Handle dependency download progress
          if (typeof parsed.downloadProgress === 'number') {
            updateProgress(parsed.downloadProgress);
          }
          
          // Show download package info in logs
          if (parsed.downloadPackage) {
            addLog({
              type: 'info',
              message: `Downloading: ${parsed.downloadPackage}`,
              timestamp: Date.now(),
            });
          }
          
          // Show install status
          if (parsed.installStart) {
            addLog({
              type: 'info',
              message: 'Installing packages...',
              timestamp: Date.now(),
            });
          }
          
          if (parsed.installComplete) {
            addLog({
              type: 'success',
              message: 'Dependencies installed successfully!',
              timestamp: Date.now(),
            });
          }
          
          if (typeof parsed.currentFrame === 'number' && typeof parsed.totalFrames === 'number') {
            setFrameInfo(parsed.currentFrame, parsed.totalFrames);
            if (parsed.totalFrames > 0) {
              updateProgress(Math.min(100, Math.round((parsed.currentFrame / parsed.totalFrames) * 100)));
            }
          }
          if (typeof parsed.progress === 'number') {
            updateProgress(parsed.progress);
          }
          if (
            typeof parsed.fps === 'number'
            || typeof parsed.elapsed === 'string'
            || typeof parsed.eta === 'string'
          ) {
            const s = useProcessingStore.getState();
            setPerformance(
              typeof parsed.fps === 'number' ? parsed.fps : s.fps,
              typeof parsed.elapsed === 'string' ? parsed.elapsed : s.elapsed,
              typeof parsed.eta === 'string' ? parsed.eta : s.eta
            );
          }
        }

        // Don't spam logs with internal progress lines.
        if (!parsed?.isInternal) {
          addLog({
            type: data.type === 'stderr' ? 'stderr' : 'stdout',
            message: line,
            timestamp: Date.now(),
          });
        }
      });

      window.electronAPI.onProcessFinished((data) => {
        if (data.code === 0) {
          completeProcessing();
          addLog({ type: 'success', message: 'Processing completed successfully!', timestamp: Date.now() });
        } else {
          const msg = `Process exited with code ${data.code}`;
          setError(msg);
          failProcessing(msg);
        }
      });

      window.electronAPI.onProcessError((data) => {
        const msg = data.error || 'Unknown process error';
        setError(msg);
        failProcessing(msg);
        addLog({ type: 'error', message: `Error: ${msg}`, timestamp: Date.now() });
      });

      return () => {
        window.electronAPI.removeAllListeners('process-output');
        window.electronAPI.removeAllListeners('process-finished');
        window.electronAPI.removeAllListeners('process-error');
      };
    }
  }, [addLog, setError, completeProcessing, failProcessing, updateProgress, setFrameInfo, setPerformance]);

  return (
    <div className="flex flex-col h-screen bg-background text-text-primary overflow-hidden">
      {/* Custom Title Bar */}
      <TitleBar />

      {/* Main Layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar Navigation */}
        <Sidebar />

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col min-w-0">
          <MainContent />
          <StatusBar />
        </div>
      </div>
    </div>
  );
}

export default App;