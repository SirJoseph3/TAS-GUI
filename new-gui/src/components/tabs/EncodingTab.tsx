import React from 'react';
import { Video, Info } from 'lucide-react';
import { useSettingsStore } from '../../stores/settingsStore';
import { ENCODE_METHODS } from '../../utils/builtinModelLists';

const bitDepths = ['8bit', '16bit'];

export const EncodingTab: React.FC = () => {
  const settings = useSettingsStore();

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="card">
        <h2 className="section-title">
          <Video className="w-5 h-5 text-primary" />
          Video Encoding
        </h2>

        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-text-secondary">Encoding Method</label>
            <select
              value={settings.encodeMethod}
              onChange={(e) => settings.setSetting('encodeMethod', e.target.value)}
              className="select-field"
            >
              {ENCODE_METHODS.map((method) => (
                <option key={method} value={method}>{method}</option>
              ))}
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-text-secondary">Bit Depth</label>
            <select
              value={settings.bitDepth}
              onChange={(e) => settings.setSetting('bitDepth', e.target.value)}
              className="select-field"
            >
              {bitDepths.map((depth) => (
                <option key={depth} value={depth}>{depth}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Resize Section */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="section-title">Output Resize</h2>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.resizeEnabled}
              onChange={(e) => settings.setSetting('resizeEnabled', e.target.checked)}
              className="checkbox-field"
            />
            <span className="text-sm font-medium text-text-primary">Enable Resize</span>
          </label>
        </div>

        {settings.resizeEnabled && (
          <div className="space-y-2">
            <label className="text-sm font-medium text-text-secondary">
              Resize Factor: {settings.resizeFactor}x
            </label>
            <input
              type="range"
              min={0.1}
              max={10}
              step={0.1}
              value={settings.resizeFactor}
              onChange={(e) => settings.setSetting('resizeFactor', parseFloat(e.target.value))}
              className="w-full accent-primary"
            />
            <p className="text-xs text-text-muted">Example: 0.5 = half size, 2.0 = double size</p>
          </div>
        )}
      </div>

      <div className="bg-info/10 border border-info/20 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-info mt-0.5" />
          <div>
            <h3 className="text-sm font-semibold text-info mb-1">About Encoding</h3>
            <p className="text-sm text-text-secondary">
              x264 is widely compatible, x265 gives smaller files but slower.
              10bit keeps more color information. NVENC uses NVIDIA GPU for fast encoding.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
