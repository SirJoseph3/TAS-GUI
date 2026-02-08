import React from 'react';
import { FolderOpen, Save, Youtube, FileVideo, Monitor, Play, Clock } from 'lucide-react';
import { useSettingsStore } from '../../stores/settingsStore';
import { useProcessingStore } from '../../stores/processingStore';
import { buildProcessingArgs, buildPreviewSegmentOutputPath } from '../../utils/buildProcessingArgs';
import { SegmentPreviewModal } from '../SegmentPreviewModal';

export const GeneralTab: React.FC = () => {
  const settings = useSettingsStore();
  const { status, startProcessing, addLog, setError } = useProcessingStore();

  const [isSegmentPreviewModalOpen, setIsSegmentPreviewModalOpen] = React.useState(false);

  const isProcessing = status === 'processing' || status === 'stopping';

  const handleSelectInput = async () => {
    const result = await window.electronAPI?.dialog.selectFile({
      filters: [
        { name: 'Video Files', extensions: ['mp4', 'mkv', 'avi', 'mov', 'webm', 'm4v', 'gif'] },
        { name: 'All Files', extensions: ['*'] }
      ]
    });
    
    if (result && !result.canceled && result.filePaths) {
      settings.setSetting('inputPath', result.filePaths[0]);
      
      // Auto-generate output path if not set
      if (!settings.outputPath) {
        const inputPath = result.filePaths[0];
        const ext = inputPath.substring(inputPath.lastIndexOf('.'));
        const baseName = inputPath.substring(0, inputPath.lastIndexOf('.'));
        settings.setSetting('outputPath', `${baseName}_enhanced${ext}`);
      }
    }
  };

  const handleRunPreviewSegment = async () => {
    if (!settings.inputPath && !settings.youtubeUrl) {
      setError('Please select an input file or enter a YouTube URL');
      return;
    }

    if (settings.inputPath && settings.youtubeUrl) {
      setError('Please use either Input Video OR YouTube URL (clear the other)');
      return;
    }

    if (settings.benchmark) {
      setError('Preview Segment requires Benchmark Mode to be OFF (it needs an output file).');
      return;
    }

    if (!settings.outputPath) {
      setError('Please choose an Output Video path first (used as the base name for the preview clip).');
      return;
    }

    const startS = Number(settings.previewStart) || 0;
    const durS = Number(settings.previewDuration) || 0;
    const outS = startS + durS;
    if (durS <= 0) {
      setError('Preview Duration must be > 0 seconds.');
      return;
    }

    const previewOut = buildPreviewSegmentOutputPath(settings.outputPath, startS, durS);

    startProcessing();
    addLog({
      type: 'info',
      message: `Starting preview segment: ${startS}s → ${outS}s`,
      timestamp: Date.now(),
    });
    addLog({
      type: 'info',
      message: `Preview output: ${previewOut}`,
      timestamp: Date.now(),
    });

    try {
      const args = buildProcessingArgs(settings, {
        inpoint: startS,
        outpoint: outS,
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

  const handleSelectOutput = async () => {
    const result = await window.electronAPI?.dialog.selectOutput({
      filters: [
        { name: 'MP4 Video', extensions: ['mp4'] },
        { name: 'MKV Video', extensions: ['mkv'] },
        { name: 'GIF', extensions: ['gif'] }
      ]
    });
    
    if (result && !result.canceled && result.filePath) {
      settings.setSetting('outputPath', result.filePath);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="card">
        <h2 className="section-title">
          <FileVideo className="w-5 h-5 text-primary" />
          Input / Output
        </h2>

        {/* Input File */}
        <div className="space-y-2 mb-4">
          <label className="text-sm font-medium text-text-secondary">Input Video</label>
          <div className="flex gap-2">
            <input
              type="text"
              value={settings.inputPath}
              onChange={(e) => settings.setSetting('inputPath', e.target.value)}
              placeholder="Select input video file..."
              className="input-field flex-1"
            />
            <button
              onClick={handleSelectInput}
              className="btn-secondary flex items-center gap-2"
            >
              <FolderOpen className="w-4 h-4" />
              Browse
            </button>
          </div>
        </div>

        {/* YouTube URL */}
        <div className="space-y-2 mb-4">
          <label className="text-sm font-medium text-text-secondary flex items-center gap-2">
            <Youtube className="w-4 h-4 text-error" />
            YouTube URL (Alternative)
          </label>
          <input
            type="text"
            value={settings.youtubeUrl}
            onChange={(e) => settings.setSetting('youtubeUrl', e.target.value)}
            placeholder="https://youtube.com/watch?v=..."
            className="input-field"
          />
        </div>

        {/* Output File */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-text-secondary">Output Video</label>
          <div className="flex gap-2">
            <input
              type="text"
              value={settings.outputPath}
              onChange={(e) => settings.setSetting('outputPath', e.target.value)}
              placeholder="Leave empty for auto-generated name..."
              className="input-field flex-1"
            />
            <button
              onClick={handleSelectOutput}
              className="btn-secondary flex items-center gap-2"
            >
              <Save className="w-4 h-4" />
              Browse
            </button>
          </div>
        </div>

        <div className="mt-4 flex items-center gap-2">
          <button
            onClick={handleRunPreviewSegment}
            disabled={isProcessing}
            className="btn-secondary flex items-center gap-2 disabled:opacity-50"
            title={settings.benchmark ? 'Disable Benchmark Mode to run Preview Segment' : 'Run Preview Segment'}
          >
            <Play className="w-4 h-4" />
            Run Preview Segment
          </button>
          <span className="text-xs text-text-muted">
            Creates a short clip next to your selected output file.
          </span>
        </div>
      </div>

      {/* Live Preview */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="section-title">
            <Monitor className="w-5 h-5 text-primary" />
            Live Preview
          </h2>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.livePreview}
              onChange={(e) => settings.setSetting('livePreview', e.target.checked)}
              className="checkbox-field"
            />
            <span className="text-sm font-medium text-text-primary">Enable</span>
          </label>
        </div>
        <p className="text-xs text-text-muted">
          Shows live preview during processing. May reduce FPS slightly.
        </p>
      </div>

      {/* Preview Segment */}
      <div className="card">
        <h2 className="section-title">
          <Play className="w-5 h-5 text-primary" />
          Preview Segment
        </h2>
        <p className="text-sm text-text-muted mb-4">
          Process only a short segment for preview before full processing.
        </p>

        <div className="flex items-center justify-between mb-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={isSegmentPreviewModalOpen}
              onChange={(e) => setIsSegmentPreviewModalOpen(e.target.checked)}
              className="checkbox-field"
            />
            <span className="text-sm font-medium text-text-primary">Advanced preview (timeline)</span>
          </label>
          <span className="text-xs text-text-muted">Topaz-style segment picker</span>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-text-secondary flex items-center gap-2">
              <Clock className="w-4 h-4" />
              Start Time (seconds)
            </label>
            <input
              type="number"
              min={0}
              value={settings.previewStart}
              onChange={(e) => settings.setSetting('previewStart', parseInt(e.target.value) || 0)}
              className="input-field"
              placeholder="0"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-text-secondary">Duration (seconds)</label>
            <select
              value={settings.previewDuration}
              onChange={(e) => settings.setSetting('previewDuration', parseInt(e.target.value))}
              className="select-field"
            >
              <option value={1}>1 second</option>
              <option value={2}>2 seconds</option>
              <option value={3}>3 seconds</option>
              <option value={5}>5 seconds</option>
              <option value={10}>10 seconds</option>
            </select>
          </div>
        </div>
      </div>

      <SegmentPreviewModal
        isOpen={isSegmentPreviewModalOpen}
        onClose={() => setIsSegmentPreviewModalOpen(false)}
      />

      {/* Options */}
      <div className="card">
        <h2 className="section-title">Options</h2>
        
        <label className="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={settings.benchmark}
            onChange={(e) => settings.setSetting('benchmark', e.target.checked)}
            className="checkbox-field"
          />
          <span className="text-text-primary">Benchmark Mode (No Output)</span>
        </label>
        <p className="text-xs text-text-muted mt-1 ml-8">
          Process without saving - useful for testing speed
        </p>
      </div>

      {/* Info Card */}
      <div className="bg-primary/10 border border-primary/20 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-primary mb-2">Quick Start Guide</h3>
        <ul className="text-sm text-text-secondary space-y-1 list-disc list-inside">
          <li>Select your input video or paste a YouTube URL</li>
          <li>Choose enhancement options from the sidebar (Upscaling, Interpolation, etc.)</li>
          <li>Enable Live Preview to see results in real-time</li>
          <li>Use Preview Segment to test settings on a short clip first</li>
          <li>Click Start to begin processing</li>
        </ul>
      </div>
    </div>
  );
};
