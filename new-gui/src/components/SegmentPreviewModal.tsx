import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { Play, X, Scissors, Video, RefreshCw } from 'lucide-react';
import { useSettingsStore } from '../stores/settingsStore';
import { useProcessingStore } from '../stores/processingStore';
import { buildProcessingArgs, buildPreviewSegmentOutputPath } from '../utils/buildProcessingArgs';

export interface SegmentPreviewModalProps {
  isOpen: boolean;
  onClose: () => void;
}

function clamp(n: number, min: number, max: number) {
  return Math.min(max, Math.max(min, n));
}

function clampPlayback(pos: number, inS: number, outS: number) {
  const duration = outS - inS;
  const offset = Math.max(0.1, duration * 0.02);
  const minPos = inS + offset;
  const maxPos = outS - offset;
  if (maxPos <= minPos) {
    return (inS + outS) / 2;
  }
  return clamp(pos, minPos, maxPos);
}

function formatTime(totalSeconds: number) {
  const s = Math.max(0, totalSeconds);
  const m = Math.floor(s / 60);
  const r = s % 60;
  const sec = r.toFixed(2).padStart(5, '0');
  return `${m}:${sec}`;
}

function toFileUrl(p: string): string {
  if (!p) return '';
  
  let path = String(p);
  
  const pathParts = path.split(/[\\/]/);
  const uniqueParts: string[] = [];
  let lastPart = '';
  
  for (const part of pathParts) {
    if (part && part !== lastPart) {
      uniqueParts.push(part);
      lastPart = part;
    }
  }
  
  let cleanPath = uniqueParts.join('/');
  
  if (/^[A-Za-z]:/.test(cleanPath)) {
    cleanPath = '/' + cleanPath;
  }
  
  const encodedPath = cleanPath.split('/').map(part => {
    if (!part || part.match(/^[A-Za-z]:$/)) return part;
    return encodeURIComponent(part);
  }).join('/');
  
  return `file:///${encodedPath}`;
}

export const SegmentPreviewModal: React.FC<SegmentPreviewModalProps> = ({ isOpen, onClose }) => {
  const settings = useSettingsStore();
  const { status, startProcessing, addLog, setError, progress } = useProcessingStore();

  const isProcessing = status === 'processing' || status === 'stopping';

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const timelineRef = useRef<HTMLDivElement | null>(null);

  const [durationS, setDurationS] = useState<number>(0);
  const [rangeTouched, setRangeTouched] = useState(false);
  const [initialRangeSet, setInitialRangeSet] = useState(false);
  const [inS, setInS] = useState<number>(0);
  const [outS, setOutS] = useState<number>(0);
  const [dragging, setDragging] = useState<'in' | 'out' | 'timeline' | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [playbackPos, setPlaybackPos] = useState<number>(0);

  const defaultDurS = Number(settings.previewDuration) || 5;

  const stateRef = useRef({ inS, outS, durationS, defaultDurS, playbackPos });
  useEffect(() => {
    stateRef.current = { inS, outS, durationS, defaultDurS, playbackPos };
  }, [inS, outS, durationS, defaultDurS, playbackPos]);

  const canShowInputVideo = Boolean(settings.inputPath) && !settings.youtubeUrl;
  const [inputVideoUrl, setInputVideoUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!canShowInputVideo || !settings.inputPath) {
      setInputVideoUrl(null);
      return;
    }

    const loadVideo = async () => {
      const isElectron = typeof window !== 'undefined' && window.electronAPI;
      
      if (isElectron && window.electronAPI.video) {
        try {
          const result = await window.electronAPI.video.readFile(settings.inputPath);
          if (result.success && result.dataUrl) {
            console.log('Loaded video via IPC as data URL');
            setInputVideoUrl(result.dataUrl);
          } else {
            console.error('Failed to load video via IPC:', result.error);
            const fallbackUrl = toFileUrl(settings.inputPath);
            console.log('Trying fallback file:// URL:', fallbackUrl);
            setInputVideoUrl(fallbackUrl);
          }
        } catch (error) {
          console.error('Error loading video:', error);
          const fallbackUrl = toFileUrl(settings.inputPath);
          setInputVideoUrl(fallbackUrl);
        }
      } else {
        const url = toFileUrl(settings.inputPath);
        console.log('Dev mode - using file:// URL:', url);
        setInputVideoUrl(url);
      }
    };

    loadVideo();
  }, [canShowInputVideo, settings.inputPath]);

  const effectiveInS = rangeTouched ? inS : 0;
  const effectiveOutS = rangeTouched ? outS : defaultDurS;
  const effectiveDurS = Math.max(0, effectiveOutS - effectiveInS);

  useEffect(() => {
    if (!isOpen) return;
    setRangeTouched(false);
    setInitialRangeSet(false);
    setInS(0);
    setOutS(defaultDurS);
    setVideoError(null);
    const initialPos = clampPlayback(0, 0, defaultDurS);
    setPlaybackPos(initialPos);
  }, [isOpen, defaultDurS]);

  // Timeline helper: get time from mouse/touch position
  const getTimeFromClientX = useCallback((clientX: number): number => {
    const el = timelineRef.current;
    if (!el) return 0;
    const rect = el.getBoundingClientRect();
    const x = clamp(clientX - rect.left, 0, rect.width);
    const ratio = rect.width > 0 ? x / rect.width : 0;
    const total = Number.isFinite(durationS) && durationS > 0 ? durationS : defaultDurS;
    return ratio * total;
  }, [durationS, defaultDurS]);

  // Seek video
  const seekTo = useCallback((s: number) => {
    const v = videoRef.current;
    if (!v) return;
    try {
      v.currentTime = clamp(s, 0, v.duration || s);
    } catch {
      // ignore
    }
  }, []);

  // Normalize range (ensure minimum gap)
  const normalizeRange = useCallback((nextIn: number, nextOut: number) => {
    const minGap = 0.5;
    const maxD = Number.isFinite(durationS) && durationS > 0 ? durationS : defaultDurS;
    let a = clamp(nextIn, 0, maxD);
    let b = clamp(nextOut, 0, maxD);
    if (b - a < minGap) {
      b = clamp(a + minGap, 0, maxD);
    }
    return { a, b };
  }, [durationS, defaultDurS]);

  // Convert seconds to percentage
  const percent = useCallback((s: number) => {
    const total = Number.isFinite(durationS) && durationS > 0 ? durationS : defaultDurS;
    return (clamp(s, 0, total) / total) * 100;
  }, [durationS, defaultDurS]);

  // Video metadata loaded
  const handleLoadedMetadata = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    const d = v.duration;
    setVideoError(null);
    if (Number.isFinite(d) && d > 0) {
      setDurationS(d);
      if (!initialRangeSet) {
        const outVal = Math.min(d, defaultDurS);
        setInS(0);
        setOutS(outVal);
        setPlaybackPos(clampPlayback(0, 0, outVal));
        setRangeTouched(true);
        setInitialRangeSet(true);
      }
    }
  }, [initialRangeSet, defaultDurS]);

  // Video time update
  const handleTimeUpdate = useCallback(() => {
    if (dragging) return;
    const v = videoRef.current;
    if (!v) return;
    const currentTime = v.currentTime || 0;
    
    const duration = effectiveOutS - effectiveInS;
    const threshold = Math.max(0.05, duration * 0.01);
    
    const clampedTime = clampPlayback(currentTime, effectiveInS, effectiveOutS);
    setPlaybackPos(clampedTime);
    
    if (currentTime >= effectiveOutS - threshold) {
      v.pause();
      v.currentTime = clampedTime;
    } else if (currentTime <= effectiveInS + threshold) {
      v.currentTime = clampedTime;
    }
  }, [dragging, effectiveInS, effectiveOutS]);

  const handleVideoError = useCallback((e: React.SyntheticEvent<HTMLVideoElement, Event>) => {
    const video = e.currentTarget;
    const errorMsg = video.error 
      ? `Video error (code ${video.error.code}): ${video.error.message}` 
      : 'Failed to load input video';
    console.error('Video load error:', errorMsg, 'URL:', video.src);
    setVideoError(errorMsg);
  }, []);

  // DRAG HANDLERS - Using native document events for smooth dragging
  useEffect(() => {
    if (!dragging || !timelineRef.current) return;

    const handleMove = (e: PointerEvent) => {
      e.preventDefault();
      
      const { inS: currIn, outS: currOut, durationS: currDur, defaultDurS: currDef, playbackPos: currPlayback } = stateRef.current;
      const el = timelineRef.current;
      if (!el) return;

      const rect = el.getBoundingClientRect();
      const x = clamp(e.clientX - rect.left, 0, rect.width);
      const ratio = rect.width > 0 ? x / rect.width : 0;
      const total = Number.isFinite(currDur) && currDur > 0 ? currDur : currDef;
      const t = ratio * total;

      if (dragging === 'in') {
        const minGap = 0.5;
        const maxD = Number.isFinite(currDur) && currDur > 0 ? currDur : currDef;
        
        let a = clamp(t, 0, maxD);
        let b = clamp(currOut, 0, maxD);
        
        if (b - a < minGap) {
          b = clamp(a + minGap, 0, maxD);
        }
        
        setInS(a);
        setOutS(b);
        
        const clampedPlayback = clampPlayback(currPlayback, a, b);
        setPlaybackPos(clampedPlayback);
        seekTo(clampedPlayback);
      } else if (dragging === 'out') {
        const minGap = 0.5;
        const maxD = Number.isFinite(currDur) && currDur > 0 ? currDur : currDef;
        
        let a = clamp(currIn, 0, maxD);
        let b = clamp(t, 0, maxD);
        
        if (b - a < minGap) {
          b = clamp(a + minGap, 0, maxD);
        }
        
        setInS(a);
        setOutS(b);
        
        const clampedPlayback = clampPlayback(currPlayback, a, b);
        setPlaybackPos(clampedPlayback);
        seekTo(clampedPlayback);
      } else if (dragging === 'timeline') {
        const clampedT = clampPlayback(t, currIn, currOut);
        seekTo(clampedT);
        setPlaybackPos(clampedT);
      }
    };

    const handleUp = () => {
      setDragging(null);
    };

    document.addEventListener('pointermove', handleMove, { passive: false });
    document.addEventListener('pointerup', handleUp, { once: true });

    return () => {
      document.removeEventListener('pointermove', handleMove);
      document.removeEventListener('pointerup', handleUp);
    };
  }, [dragging, seekTo]);

  // Handle pointer down on timeline
  const handleTimelinePointerDown = useCallback((e: React.PointerEvent) => {
    if (e.button !== 0) return;

    const target = e.target as HTMLElement;

    // Check if clicking on in-handle
    if (target.closest('[data-handle="in"]')) {
      setDragging('in');
      setRangeTouched(true);
      return;
    }

    // Check if clicking on out-handle
    if (target.closest('[data-handle="out"]')) {
      setDragging('out');
      setRangeTouched(true);
      return;
    }

    // Timeline click - start playback drag
    setDragging('timeline');
    const t = getTimeFromClientX(e.clientX);
    const clampedT = clampPlayback(t, effectiveInS, effectiveOutS);
    seekTo(clampedT);
    setPlaybackPos(clampedT);
  }, [getTimeFromClientX, seekTo, effectiveInS, effectiveOutS]);

  const resetRange = useCallback(() => {
    setRangeTouched(false);
    const outVal = Math.min(defaultDurS, durationS || defaultDurS);
    setInS(0);
    setOutS(outVal);
    const initialPos = clampPlayback(0, 0, outVal);
    setPlaybackPos(initialPos);
    seekTo(initialPos);
  }, [defaultDurS, durationS, seekTo]);

  const runPreview = async () => {
    if (!settings.inputPath && !settings.youtubeUrl) {
      setError('Please select an input file or enter a YouTube URL');
      return;
    }
    if (settings.inputPath && settings.youtubeUrl) {
      setError('Please use either Input Video OR YouTube URL');
      return;
    }
    if (settings.benchmark) {
      setError('Preview Segment requires Benchmark Mode to be OFF');
      return;
    }
    if (!settings.outputPath) {
      setError('Please choose an Output Video path first');
      return;
    }

    const startS = Math.max(0, Math.floor(effectiveInS));
    const durS = Math.max(1, Math.floor(effectiveDurS || defaultDurS));
    const outS2 = startS + durS;

    const previewOut = buildPreviewSegmentOutputPath(settings.outputPath, startS, durS);

    startProcessing();
    addLog({ type: 'info', message: `Starting preview segment: ${startS}s → ${outS2}s`, timestamp: Date.now() });
    addLog({ type: 'info', message: `Preview output: ${previewOut}`, timestamp: Date.now() });

    try {
      const args = buildProcessingArgs(settings, {
        inpoint: startS,
        outpoint: outS2,
        outputOverride: previewOut,
      });

      const result = await window.electronAPI?.processing.start(args);
      if (!result?.success) {
        setError(result?.error || 'Failed to start preview segment');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  };

  if (!isOpen) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6">
      <div className="w-full max-w-6xl h-[82vh] bg-surface border border-border rounded-xl shadow-xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="px-5 py-4 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Scissors className="w-5 h-5 text-primary" />
            <h3 className="text-base font-semibold text-text-primary">Segment Preview</h3>
          </div>
          <button onClick={onClose} className="btn-secondary px-3 py-2 flex items-center gap-2">
            <X className="w-4 h-4" />
            Close
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-3 gap-4 p-5">
          {/* Player + timeline */}
          <div className="lg:col-span-2 flex flex-col min-h-0">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-text-muted uppercase flex items-center gap-2">
                <Video className="w-4 h-4" />
                Input video
              </span>
              <span className="text-xs text-text-muted">
                {Number.isFinite(durationS) && durationS > 0 ? `Duration: ${formatTime(durationS)}` : 'Duration: unknown'}
              </span>
            </div>

            <div className="w-full flex-1 min-h-[240px] rounded-lg overflow-hidden bg-background-secondary border border-border flex items-center justify-center relative">
              {canShowInputVideo && inputVideoUrl ? (
                <>
                  <video
                    key={inputVideoUrl}
                    ref={videoRef}
                    src={inputVideoUrl}
                    controls
                    className="w-full h-full object-contain"
                    onLoadedMetadata={handleLoadedMetadata}
                    onError={handleVideoError}
                    onTimeUpdate={handleTimeUpdate}
                  />
                  {videoError && (
                    <div className="absolute top-2 left-2 right-2 bg-error/90 text-white p-2 rounded text-xs">
                      {videoError}
                    </div>
                  )}
                </>
              ) : (
                <div className="p-4 text-sm text-text-muted text-center max-w-md">
                  Advanced preview player supports <span className="text-text-primary font-medium">local input files</span>.
                </div>
              )}
            </div>

            {/* Timeline */}
            <div className="mt-4">
              <div className="flex items-center justify-between mb-2">
                <div className="text-xs text-text-muted">
                  <span className="font-medium text-text-primary">In</span>: {formatTime(effectiveInS)}
                  <span className="mx-2">|</span>
                  <span className="font-medium text-text-primary">Out</span>: {formatTime(effectiveOutS)}
                  <span className="mx-2">|</span>
                  <span className="font-medium text-text-primary">Len</span>: {formatTime(effectiveDurS || defaultDurS)}
                </div>
                <button onClick={resetRange} className="text-xs text-text-muted hover:text-text-primary flex items-center gap-2">
                  <RefreshCw className="w-3.5 h-3.5" />
                  Reset
                </button>
              </div>

              <div
                ref={timelineRef}
                onPointerDown={handleTimelinePointerDown}
                className={`relative h-12 rounded-lg bg-background-secondary border border-border select-none ${dragging ? 'cursor-grabbing' : 'cursor-pointer'}`}
              >
                {/* Selected range background */}
                <div
                  className="absolute top-0 bottom-0 bg-primary/20"
                  style={{
                    left: `${percent(effectiveInS)}%`,
                    width: `${Math.max(0, percent(effectiveOutS) - percent(effectiveInS))}%`,
                  }}
                />

                {/* Playback position */}
                <div
                  className="absolute top-0 bottom-0 w-1 bg-white/90 z-40 shadow-lg"
                  style={{ left: `${percent(playbackPos)}%`, marginLeft: '-2px' }}
                />

                {/* In handle */}
                <div
                  className="absolute top-0 bottom-0 w-6 -ml-3 cursor-ew-resize z-30 flex items-center justify-center group"
                  style={{ left: `${percent(effectiveInS)}%` }}
                  data-handle="in"
                  title="Drag to set in-point"
                >
                  <div className="w-2 h-full bg-success rounded-full shadow-lg group-hover:w-3 transition-all" />
                </div>

                {/* Out handle */}
                <div
                  className="absolute top-0 bottom-0 w-6 -ml-3 cursor-ew-resize z-30 flex items-center justify-center group"
                  style={{ left: `${percent(effectiveOutS)}%` }}
                  data-handle="out"
                  title="Drag to set out-point"
                >
                  <div className="w-2 h-full bg-warning rounded-full shadow-lg group-hover:w-3 transition-all" />
                </div>
              </div>

              <p className="mt-2 text-xs text-text-muted">
                Tip: Click timeline to seek. Drag green (in) and yellow (out) handles to set range.
              </p>
            </div>

          </div>

          {/* Controls */}
          <div className="lg:col-span-1 flex flex-col gap-4">
            <div className="card p-4">
              <h4 className="text-sm font-semibold text-text-primary mb-3">Selection</h4>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs font-medium text-text-secondary">In (s)</label>
                  <input
                    type="number"
                    min={0}
                    step={0.1}
                    value={Number(effectiveInS.toFixed(2))}
                    onChange={(e) => {
                      const v = Math.max(0, Number(e.target.value) || 0);
                      setRangeTouched(true);
                      const norm = normalizeRange(v, effectiveOutS);
                      setInS(norm.a);
                      setOutS(norm.b);
                      const clampedPlayback = clampPlayback(playbackPos, norm.a, norm.b);
                      setPlaybackPos(clampedPlayback);
                      seekTo(clampedPlayback);
                    }}
                    className="input-field"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-medium text-text-secondary">Out (s)</label>
                  <input
                    type="number"
                    min={0}
                    step={0.1}
                    value={Number(effectiveOutS.toFixed(2))}
                    onChange={(e) => {
                      const v = Math.max(0, Number(e.target.value) || 0);
                      setRangeTouched(true);
                      const norm = normalizeRange(effectiveInS, v);
                      setInS(norm.a);
                      setOutS(norm.b);
                      const clampedPlayback = clampPlayback(playbackPos, norm.a, norm.b);
                      setPlaybackPos(clampedPlayback);
                      seekTo(clampedPlayback);
                    }}
                    className="input-field"
                  />
                </div>
              </div>
            </div>

            <div className="card p-4">
              <h4 className="text-sm font-semibold text-text-primary mb-3">Run</h4>
              <button
                onClick={runPreview}
                disabled={isProcessing}
                className="w-full btn-primary flex items-center justify-center gap-2 disabled:opacity-50"
              >
                <Play className="w-4 h-4" />
                Create Preview
              </button>
              {isProcessing && (
                <div className="mt-3">
                  <div className="flex justify-between text-xs text-text-muted mb-1">
                    <span>Processing...</span>
                    <span>{progress}%</span>
                  </div>
                  <div className="w-full bg-background-secondary rounded-full h-2">
                    <div className="bg-primary h-2 rounded-full" style={{ width: `${progress}%` }} />
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
