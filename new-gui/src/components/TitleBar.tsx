import React, { useState, useEffect } from 'react';
import { Minus, Square, X } from 'lucide-react';
import { TASLogo } from './TASLogo';

export const TitleBar: React.FC = () => {
  const [isMaximized, setIsMaximized] = useState(false);

  useEffect(() => {
    const checkMaximized = async () => {
      if (window.electronAPI) {
        const maximized = await window.electronAPI.window.isMaximized();
        setIsMaximized(maximized);
      }
    };
    checkMaximized();
  }, []);

  const handleMinimize = () => {
    window.electronAPI?.window.minimize();
  };

  const handleMaximize = async () => {
    await window.electronAPI?.window.maximize();
    const maximized = await window.electronAPI?.window.isMaximized();
    setIsMaximized(maximized || false);
  };

  const handleClose = () => {
    window.electronAPI?.window.close();
  };

  return (
    <div className="h-10 bg-background-secondary border-b border-border flex items-center justify-between drag-region select-none">
      {/* App Icon and Title */}
      <div className="flex items-center gap-2 px-4">
        <TASLogo size={20} className="text-primary" />
        <span className="text-sm font-semibold text-text-primary">
          The Anime Scripter
        </span>
      </div>

      {/* Window Controls */}
      <div className="flex items-center no-drag">
        <button
          onClick={handleMinimize}
          className="w-12 h-10 flex items-center justify-center text-text-secondary hover:bg-surface-hover transition-colors"
          title="Minimize"
        >
          <Minus className="w-4 h-4" />
        </button>
        <button
          onClick={handleMaximize}
          className="w-12 h-10 flex items-center justify-center text-text-secondary hover:bg-surface-hover transition-colors"
          title={isMaximized ? 'Restore' : 'Maximize'}
        >
          <Square className="w-3.5 h-3.5" />
        </button>
        <button
          onClick={handleClose}
          className="w-12 h-10 flex items-center justify-center text-text-secondary hover:bg-error hover:text-white transition-colors"
          title="Close"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};