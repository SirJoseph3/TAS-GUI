import React, { useEffect, useState, useRef } from 'react';
import { Play, Square, Trash2, Monitor, Maximize2, X, Copy, Check } from 'lucide-react';
import { useProcessingStore } from '../stores/processingStore';
import { useSettingsStore } from '../stores/settingsStore';
import { useNavigationStore } from '../stores/navigationStore';
import { buildProcessingArgs } from '../utils/buildProcessingArgs';

export const ProcessingPanel: React.FC = () => {
  const {
    status,
    logs,
    startProcessing,
    stopProcessing,
    addLog,
    clearLogs,
    setError,
    error,
    progress,
    fps,
    eta,
    currentFrame,
    totalFrames,
  } = useProcessingStore();
  const settings = useSettingsStore();
  const { activeTab } = useNavigationStore();

  const isImageUpscaleTab = activeTab === 'imageupscale';

  const [previewSrc, setPreviewSrc] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [copied, setCopied] = useState(false);
  const logContainerRef = useRef<HTMLDivElement>(null);

  const isProcessing = status === 'processing' || status === 'stopping';

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  const handleCopyLogs = async () => {
    const logText = logs
      .map((log) => `[${new Date(log.timestamp).toLocaleTimeString()}] ${log.message}`)
      .join('\n');
    
    try {
      await navigator.clipboard.writeText(logText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy logs:', err);
    }
  };

  useEffect(() => {
    if (!settings.livePreview || status !== 'processing') {
      setPreviewSrc(null);
      return;
    }

    let cancelled = false;

    const tick = async () => {
      try {
        const src = await window.electronAPI?.preview.getImage();
        if (!cancelled) {
          setPreviewSrc(src || null);
        }
      } catch {
        if (!cancelled) setPreviewSrc(null);
      }
    };

    tick();
    const id = window.setInterval(tick, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [settings.livePreview, status]);

  const handleStart = async () => {
    if (!settings.inputPath && !settings.youtubeUrl) {
      setError('Please select an input file or enter a YouTube URL');
      return;
    }

    if (settings.inputPath && settings.youtubeUrl) {
      setError('Please use either Input Video OR YouTube URL (clear the other)');
      return;
    }

    startProcessing();
    addLog({ type: 'info', message: 'Starting processing...', timestamp: Date.now() });

    try {
      const args = buildProcessingArgs(settings);

      const result = await window.electronAPI?.processing.start(args);

      if (!result?.success) {
        setError(result?.error || 'Failed to start processing');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  };

  const handleStop = async () => {
    stopProcessing();
    await window.electronAPI?.processing.stop();
  };

  // Console-only view for Image Upscale tab
  if (isImageUpscaleTab) {
    return (
      <div className="w-96 bg-surface border-l border-border flex flex-col">
        {/* Console Output Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-border">
          <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
            <Monitor className="w-4 h-4" />
            Console Output
          </h3>
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopyLogs}
              className="text-text-muted hover:text-text-primary transition-colors"
              title={copied ? 'Copied!' : 'Copy logs'}
            >
              {copied ? <Check className="w-4 h-4 text-success" /> : <Copy className="w-4 h-4" />}
            </button>
            <button
              onClick={clearLogs}
              className="text-text-muted hover:text-text-primary transition-colors"
              title="Clear logs"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Log Output */}
        <div
          ref={logContainerRef}
          className="flex-1 overflow-auto p-4 font-mono text-xs space-y-1"
        >
          {logs.length === 0 ? (
            <span className="text-text-muted">No output yet...</span>
          ) : (
            logs.map((log, index) => (
              <div
                key={index}
                className={`log-line log-${log.type}`}
              >
                <span className="text-text-muted">[{new Date(log.timestamp).toLocaleTimeString()}]</span>{' '}
                {log.message}
              </div>
            ))
          )}
        </div>

        {/* Progress Bar (only when processing) */}
        {status === 'processing' && (
          <div className="px-4 py-3 border-t border-border">
            <div className="flex justify-between text-xs text-text-muted mb-1">
              <span>
                {currentFrame > 0 ? `${currentFrame}/${totalFrames} frames` : 'Processing...'}
              </span>
              <span>{progress}%</span>
            </div>
            <div className="w-full bg-background-secondary rounded-full h-2">
              <div
                className="bg-primary h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            {(fps > 0 || eta) && (
              <div className="flex justify-between text-xs text-text-muted mt-1">
                {fps > 0 && <span>{fps.toFixed(1)} FPS</span>}
                {eta && <span>ETA: {eta}</span>}
              </div>
            )}
          </div>
        )}

        {/* Error Message */}
        {status === 'error' && error && (
          <div className="px-4 py-2 border-t border-border">
            <div className="text-xs text-error bg-error/10 border border-error/20 rounded-lg p-2">
              {error}
            </div>
          </div>
        )}
      </div>
    );
  }

  // Full Processing Panel for other tabs
  return (
    <>
      {/* Fullscreen Modal */}
      {isFullscreen && previewSrc && (
        <div
          className="fixed inset-0 z-50 bg-black flex items-center justify-center"
          onClick={() => setIsFullscreen(false)}
        >
          <img
            src={previewSrc}
            alt="Live preview fullscreen"
            className="max-w-full max-h-full object-contain"
          />
          <button
            onClick={() => setIsFullscreen(false)}
            className="absolute top-4 right-4 text-white hover:text-gray-300 transition-colors"
          >
            <X className="w-8 h-8" />
          </button>
        </div>
      )}

      <div className="w-96 bg-surface border-l border-border flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-border">
          <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
            <Monitor className="w-4 h-4" />
            Processing
          </h3>

          {/* Live Preview Frame */}
          {settings.livePreview && (
            <div className="mb-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-text-muted uppercase">Live Preview</span>
                <span className="text-xs text-text-muted">{status === 'processing' ? 'running' : 'idle'}</span>
              </div>
              <div
                className="w-full aspect-video rounded-lg overflow-hidden bg-background-secondary border border-border flex items-center justify-center relative group cursor-pointer"
                onClick={() => previewSrc && setIsFullscreen(true)}
              >
                {previewSrc ? (
                  <>
                    <img src={previewSrc} alt="Live preview" className="w-full h-full object-contain" />
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-all flex items-center justify-center">
                      <Maximize2 className="w-8 h-8 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                    </div>
                  </>
                ) : (
                  <span className="text-xs text-text-muted">
                    {status === 'processing' ? 'Waiting for preview frame...' : 'Enable + Start to see frames'}
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Progress Bar */}
          {status === 'processing' && (
            <div className="mb-3">
              <div className="flex justify-between text-xs text-text-muted mb-1">
                <span>
                  {currentFrame > 0 ? `${currentFrame}/${totalFrames} frames` : 'Processing...'}
                </span>
                <span>{progress}%</span>
              </div>
              <div className="w-full bg-background-secondary rounded-full h-2">
                <div
                  className="bg-primary h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
              {(fps > 0 || eta) && (
                <div className="flex justify-between text-xs text-text-muted mt-1">
                  {fps > 0 && <span>{fps.toFixed(1)} FPS</span>}
                  {eta && <span>ETA: {eta}</span>}
                </div>
              )}
            </div>
          )}
          
          <div className="flex gap-2">
            <button
              onClick={handleStart}
              disabled={isProcessing}
              className="flex-1 btn-primary flex items-center justify-center gap-2 disabled:opacity-50"
            >
              <Play className="w-4 h-4" />
              Start
            </button>
            <button
              onClick={handleStop}
              disabled={!isProcessing}
              className="flex-1 btn-danger flex items-center justify-center gap-2 disabled:opacity-50"
            >
              <Square className="w-4 h-4" />
              Stop
            </button>
          </div>

          {status === 'error' && error && (
            <div className="mt-3 text-xs text-error bg-error/10 border border-error/20 rounded-lg p-2">
              {error}
            </div>
          )}
        </div>

        {/* Log Output */}
        <div className="flex-1 flex flex-col min-h-0">
          <div className="flex items-center justify-between px-4 py-2 border-b border-border">
            <span className="text-xs font-medium text-text-muted uppercase">Console Output</span>
            <div className="flex items-center gap-2">
              <button
                onClick={handleCopyLogs}
                className="text-text-muted hover:text-text-primary transition-colors"
                title={copied ? 'Copied!' : 'Copy logs'}
              >
                {copied ? <Check className="w-4 h-4 text-success" /> : <Copy className="w-4 h-4" />}
              </button>
              <button
                onClick={clearLogs}
                className="text-text-muted hover:text-text-primary transition-colors"
                title="Clear logs"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          </div>
          <div
            ref={logContainerRef}
            className="flex-1 overflow-auto p-4 font-mono text-xs space-y-1"
          >
            {logs.length === 0 ? (
              <span className="text-text-muted">No output yet...</span>
            ) : (
              logs.map((log, index) => (
                <div
                  key={index}
                  className={`log-line log-${log.type}`}
                >
                  <span className="text-text-muted">[{new Date(log.timestamp).toLocaleTimeString()}]</span>{' '}
                  {log.message}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </>
  );
};
