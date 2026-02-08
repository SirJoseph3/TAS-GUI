import React from 'react';
import { useProcessingStore } from '../stores/processingStore';
import { Play, Pause, CheckCircle, XCircle, Clock } from 'lucide-react';

export const StatusBar: React.FC = () => {
  const { status, progress, currentFrame, totalFrames, fps, elapsed, eta } = useProcessingStore();

  const getStatusIcon = () => {
    switch (status) {
      case 'processing':
        return <Play className="w-4 h-4 text-primary animate-pulse" />;
      case 'stopping':
        return <Pause className="w-4 h-4 text-warning" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-success" />;
      case 'error':
        return <XCircle className="w-4 h-4 text-error" />;
      default:
        return <Clock className="w-4 h-4 text-text-muted" />;
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'processing':
        return 'Processing...';
      case 'stopping':
        return 'Stopping...';
      case 'completed':
        return 'Completed';
      case 'error':
        return 'Error';
      default:
        return 'Ready';
    }
  };

  return (
    <div className="h-10 bg-background-secondary border-t border-border flex items-center justify-between px-4 text-sm">
      {/* Left: Status */}
      <div className="flex items-center gap-2">
        {getStatusIcon()}
        <span className={`
          ${status === 'processing' ? 'text-primary' : ''}
          ${status === 'completed' ? 'text-success' : ''}
          ${status === 'error' ? 'text-error' : ''}
          ${status === 'idle' ? 'text-text-muted' : ''}
        `}>
          {getStatusText()}
        </span>
      </div>

      {/* Center: Progress Info */}
      {status === 'processing' && (
        <div className="flex items-center gap-4 text-text-secondary">
          <span>Frame: {currentFrame.toLocaleString()} / {totalFrames.toLocaleString()}</span>
          <span>FPS: {fps.toFixed(1)}</span>
          <span>Elapsed: {elapsed}</span>
          <span>ETA: {eta}</span>
        </div>
      )}

      {/* Right: Progress Bar */}
      <div className="flex items-center gap-3 w-64">
        <div className="flex-1 h-2 bg-surface rounded-full overflow-hidden">
          <div 
            className={`h-full bg-primary transition-all duration-300 ${
              status === 'processing' ? 'progress-bar-striped' : ''
            }`}
            style={{ width: `${progress}%` }}
          />
        </div>
        <span className="text-text-muted w-10 text-right">{progress}%</span>
      </div>
    </div>
  );
};