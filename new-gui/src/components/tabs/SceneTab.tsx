import React from 'react';
import { Film, Info } from 'lucide-react';
import { useSettingsStore } from '../../stores/settingsStore';
import { SCENECHANGE_METHODS } from '../../utils/builtinModelLists';

export const SceneTab: React.FC = () => {
  const settings = useSettingsStore();

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="section-title">
            <Film className="w-5 h-5 text-primary" />
            Scene Change Detection
          </h2>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.scenechangeEnabled}
              onChange={(e) => settings.setSetting('scenechangeEnabled', e.target.checked)}
              className="checkbox-field"
            />
            <span className="text-sm font-medium text-text-primary">Enable</span>
          </label>
        </div>

        {settings.scenechangeEnabled && (
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-text-secondary">Detection Method</label>
              <select
                value={settings.scenechangeMethod}
                onChange={(e) => settings.setSetting('scenechangeMethod', e.target.value)}
                className="select-field"
              >
                {SCENECHANGE_METHODS.map((method) => (
                  <option key={method} value={method}>{method}</option>
                ))}
              </select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-text-secondary">
                Scene Change Sensitivity: {settings.scenechangeSens}
              </label>
              <input
                type="range"
                min={0}
                max={100}
                value={settings.scenechangeSens}
                onChange={(e) => settings.setSetting('scenechangeSens', parseInt(e.target.value))}
                className="w-full accent-primary"
              />
              <p className="text-xs text-text-muted">
                Lower = only obvious scene changes detected
              </p>
            </div>

            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.autoclip}
                onChange={(e) => settings.setSetting('autoclip', e.target.checked)}
                className="checkbox-field"
              />
              <span className="text-text-primary">Auto Clip Scenes</span>
            </label>
          </div>
        )}
      </div>

      <div className="bg-info/10 border border-info/20 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-info mt-0.5" />
          <div>
            <h3 className="text-sm font-semibold text-info mb-1">About Scene Detection</h3>
            <p className="text-sm text-text-secondary">
              Finds where scenes change in your video to prevent interpolation from blending different scenes.
              MaxxVIT is most accurate, Differential is fastest. Auto-Clip splits videos at scene changes.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
