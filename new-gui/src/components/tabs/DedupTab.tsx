import React from 'react';
import { RefreshCw, Info } from 'lucide-react';
import { useSettingsStore } from '../../stores/settingsStore';
import { DEDUP_METHODS } from '../../utils/builtinModelLists';

export const DedupTab: React.FC = () => {
  const settings = useSettingsStore();

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="section-title">
            <RefreshCw className="w-5 h-5 text-primary" />
            Frame Deduplication
          </h2>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.dedupEnabled}
              onChange={(e) => settings.setSetting('dedupEnabled', e.target.checked)}
              className="checkbox-field"
            />
            <span className="text-sm font-medium text-text-primary">Enable</span>
          </label>
        </div>

        {settings.dedupEnabled && (
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-text-secondary">Deduplication Method</label>
              <select
                value={settings.dedupMethod}
                onChange={(e) => settings.setSetting('dedupMethod', e.target.value)}
                className="select-field"
              >
                {DEDUP_METHODS.map((method) => (
                  <option key={method} value={method}>{method}</option>
                ))}
              </select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-text-secondary">
                Sensitivity: {settings.dedupSens}
              </label>
              <input
                type="range"
                min={0}
                max={100}
                value={settings.dedupSens}
                onChange={(e) => settings.setSetting('dedupSens', parseInt(e.target.value))}
                className="w-full accent-primary"
              />
              <p className="text-xs text-text-muted">Lower = more aggressive removal</p>
            </div>
          </div>
        )}
      </div>

      <div className="bg-info/10 border border-info/20 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-info mt-0.5" />
          <div>
            <h3 className="text-sm font-semibold text-info mb-1">About Deduplication</h3>
            <p className="text-sm text-text-secondary">
              Removes duplicate frames (useful for anime with still shots). SSIM-CUDA is fastest with GPU.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
