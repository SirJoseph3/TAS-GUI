import React, { useState } from 'react';
import { 
  Home, 
  Target,
  Maximize2, 
  Image,
  Activity, 
  Wrench, 
  Film,
  RefreshCw,
  Scissors, 
  Layers,
  Settings,
  Zap,
  Github,
  FolderOpen
} from 'lucide-react';
import { useNavigationStore, TabId } from '../stores/navigationStore';
import { PresetManager } from './PresetManager';

interface NavItem {
  id: TabId;
  label: string;
  icon: React.ElementType;
}

const navItems: NavItem[] = [
  { id: 'general', label: 'General', icon: Home },
  { id: 'objdetect', label: 'Object Detection', icon: Target },
  { id: 'upscale', label: 'Upscaling', icon: Maximize2 },
  { id: 'imageupscale', label: 'Image Upscale', icon: Image },
  { id: 'interpolate', label: 'Interpolation', icon: Activity },
  { id: 'restore', label: 'Restoration', icon: Wrench },
  { id: 'scene', label: 'Scene Detection', icon: Film },
  { id: 'dedup', label: 'Deduplication', icon: RefreshCw },
  { id: 'segment', label: 'Segmentation', icon: Scissors },
  { id: 'depth', label: 'Depth', icon: Layers },
  { id: 'encoding', label: 'Encoding', icon: Settings },
  { id: 'performance', label: 'Performance', icon: Zap },
];

export const Sidebar: React.FC = () => {
  const { activeTab, setActiveTab } = useNavigationStore();
  const [isPresetManagerOpen, setIsPresetManagerOpen] = useState(false);

  const handleOpenGithub = () => {
    window.electronAPI?.openExternal('https://github.com/NevermindNilas');
  };

  return (
    <>
    <PresetManager isOpen={isPresetManagerOpen} onClose={() => setIsPresetManagerOpen(false)} />
    <div className="w-64 bg-surface border-r border-border flex flex-col">
      <div className="p-4 flex-1">
        {/* Preset Button */}
        <button
          onClick={() => setIsPresetManagerOpen(true)}
          className="w-full flex items-center gap-2 px-3 py-2 mb-4 rounded-lg text-sm font-medium bg-primary/10 text-primary hover:bg-primary/20 transition-colors border border-primary/30"
        >
          <FolderOpen className="w-4 h-4" />
          Presets
        </button>
        
        <h2 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-3">
          Settings
        </h2>
        <nav className="space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? 'bg-primary text-white shadow-lg shadow-primary/25'
                    : 'text-text-secondary hover:bg-surface-light/50 hover:text-text-primary'
                }`}
              >
                <Icon className="w-4 h-4" />
                {item.label}
              </button>
            );
          })}
        </nav>
      </div>
      <div className="p-4 border-t border-border">
        <button
          onClick={handleOpenGithub}
          className="w-full flex items-center gap-2 px-3 py-2 text-sm text-text-secondary hover:text-text-primary transition-colors"
        >
          <Github className="w-4 h-4" />
          <span>Creator: NevermindNilas</span>
        </button>
      </div>
    </div>
    </>
  );
};
