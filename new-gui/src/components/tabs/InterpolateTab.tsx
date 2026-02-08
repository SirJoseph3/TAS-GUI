import React from 'react';
import { Activity, Info, Upload, Sparkles } from 'lucide-react';
import { useSettingsStore } from '../../stores/settingsStore';
import { INTERPOLATE_METHODS, CUSTOM_BACKENDS, CUSTOM_BACKEND_LABELS } from '../../utils/builtinModelLists';

export const InterpolateTab: React.FC = () => {
  const settings = useSettingsStore();

  const methodOptions = INTERPOLATE_METHODS.filter((m) => m !== 'custom');

  const handleCustomToggle = (checked: boolean) => {
    settings.setSetting('useCustomInterpolateModel', checked);
    if (checked) {
      settings.setSetting('interpolateMethod', 'custom');
    } else if (settings.interpolateMethod === 'custom') {
      settings.setSetting('interpolateMethod', methodOptions[0] || 'rife4.22');
    }
  };

  const handleSelectCustomModel = async () => {
    const result = await window.electronAPI?.dialog.selectFile({
      filters: [
        { name: 'Model Files', extensions: ['pth', 'onnx', 'engine', 'pt'] },
        { name: 'All Files', extensions: ['*'] }
      ]
    });
    
    if (result && !result.canceled && result.filePaths) {
      settings.setSetting('interpolateCustomModel', result.filePaths[0]);
      settings.setSetting('interpolateMethod', 'custom');
      settings.setSetting('useCustomInterpolateModel', true);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="section-title">
            <Activity className="w-5 h-5 text-primary" />
            Frame Interpolation
          </h2>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.interpolateEnabled}
              onChange={(e) => settings.setSetting('interpolateEnabled', e.target.checked)}
              className="checkbox-field"
            />
            <span className="text-sm font-medium text-text-primary">Enable</span>
          </label>
        </div>

        {settings.interpolateEnabled && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-text-secondary">Interpolation Factor</label>
                <select
                  value={settings.interpolateFactor}
                  onChange={(e) => settings.setSetting('interpolateFactor', Number(e.target.value))}
                  className="select-field"
                >
                  <option value={2}>2x (30→60fps)</option>
                  <option value={3}>3x (30→90fps)</option>
                  <option value={4}>4x (30→120fps)</option>
                </select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-text-secondary">Method</label>
                <div className="flex items-center gap-3">
                  <select
                    value={settings.interpolateMethod === 'custom' ? methodOptions[0] || '' : settings.interpolateMethod}
                    onChange={(e) => settings.setSetting('interpolateMethod', e.target.value)}
                    className={`select-field flex-1 ${settings.useCustomInterpolateModel ? 'opacity-50 cursor-not-allowed' : ''}`}
                    disabled={settings.useCustomInterpolateModel}
                  >
                    {methodOptions.map((method) => (
                      <option key={method} value={method}>{method}</option>
                    ))}
                  </select>
                  <label className="flex items-center gap-2 text-sm font-medium text-text-primary">
                    <input
                      type="checkbox"
                      checked={settings.useCustomInterpolateModel}
                      onChange={(e) => handleCustomToggle(e.target.checked)}
                      className="checkbox-field"
                    />
                    Custom model
                  </label>
                </div>
              </div>
            </div>

            {/* Custom Model */}
            {settings.interpolateMethod === 'custom' && (
              <div className="pt-2 border-t border-white/10">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-text-secondary flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-primary" />
                    Custom Model
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={settings.interpolateCustomModel}
                      onChange={(e) => settings.setSetting('interpolateCustomModel', e.target.value)}
                      placeholder="Select custom RIFE model..."
                      className="input-field flex-1"
                    />
                    <button
                      onClick={handleSelectCustomModel}
                      className="btn-secondary flex items-center gap-2"
                    >
                      <Upload className="w-4 h-4" />
                      Browse
                    </button>
                  </div>
                </div>
                
                <div className="mt-3 space-y-2">
                  <label className="text-sm font-medium text-text-secondary">Custom Backend</label>
                  <select
                    value={settings.interpolateCustomBackend}
                    onChange={(e) => settings.setSetting('interpolateCustomBackend', e.target.value)}
                    className="select-field"
                  >
                    {CUSTOM_BACKENDS.map((b) => (
                      <option key={b} value={b}>
                        {CUSTOM_BACKEND_LABELS[b]}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            )}

            <div className="flex gap-4 pt-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.ensemble}
                  onChange={(e) => settings.setSetting('ensemble', e.target.checked)}
                  className="checkbox-field"
                />
                <span className="text-text-primary">Ensemble (Better Quality)</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.slowmo}
                  onChange={(e) => settings.setSetting('slowmo', e.target.checked)}
                  className="checkbox-field"
                />
                <span className="text-text-primary">Slow Motion</span>
              </label>
            </div>
          </div>
        )}
      </div>

      <div className="bg-info/10 border border-info/20 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-info mt-0.5" />
          <div>
            <h3 className="text-sm font-semibold text-info mb-1">About Interpolation</h3>
            <p className="text-sm text-text-secondary">
              Adds frames between existing ones for smoother motion. 2x doubles the frame rate.
              RIFE models are popular - rife4.22 is reliable, rife4.25-heavy gives best quality.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
