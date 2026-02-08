import React from 'react';
import { Layers, Info, Upload, Sparkles } from 'lucide-react';
import { useSettingsStore } from '../../stores/settingsStore';
import { DEPTH_METHODS } from '../../utils/builtinModelLists';

const depthQualities = ['low', 'medium', 'high'];

export const DepthTab: React.FC = () => {
  const settings = useSettingsStore();

  const methodOptions = DEPTH_METHODS.filter((m) => m !== 'custom');

  const handleCustomToggle = (checked: boolean) => {
    settings.setSetting('useCustomDepthModel', checked);
    if (checked) {
      settings.setSetting('depthMethod', 'custom');
    } else if (settings.depthMethod === 'custom') {
      settings.setSetting('depthMethod', methodOptions[0] || 'small_v2');
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
      settings.setSetting('depthCustomModel', result.filePaths[0]);
      settings.setSetting('depthMethod', 'custom');
      settings.setSetting('useCustomDepthModel', true);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="section-title">
            <Layers className="w-5 h-5 text-primary" />
            Depth Map Generation
          </h2>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.depthEnabled}
              onChange={(e) => settings.setSetting('depthEnabled', e.target.checked)}
              className="checkbox-field"
            />
            <span className="text-sm font-medium text-text-primary">Enable</span>
          </label>
        </div>

        {settings.depthEnabled && (
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-text-secondary">Depth Method</label>
              <div className="flex items-center gap-3">
                <select
                  value={settings.depthMethod === 'custom' ? methodOptions[0] || '' : settings.depthMethod}
                  onChange={(e) => settings.setSetting('depthMethod', e.target.value)}
                  className={`select-field flex-1 ${settings.useCustomDepthModel ? 'opacity-50 cursor-not-allowed' : ''}`}
                  disabled={settings.useCustomDepthModel}
                >
                  {methodOptions.map((method) => (
                    <option key={method} value={method}>{method}</option>
                  ))}
                </select>
                <label className="flex items-center gap-2 text-sm font-medium text-text-primary">
                  <input
                    type="checkbox"
                    checked={settings.useCustomDepthModel}
                    onChange={(e) => handleCustomToggle(e.target.checked)}
                    className="checkbox-field"
                  />
                  Custom model
                </label>
              </div>
            </div>

            {settings.useCustomDepthModel && (
              <div className="pt-2 border-t border-white/10">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-text-secondary flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-primary" />
                    Custom Depth Model
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={settings.depthCustomModel}
                      onChange={(e) => settings.setSetting('depthCustomModel', e.target.value)}
                      placeholder="Select custom depth model..."
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
              </div>
            )}
            <div className="space-y-2">
              <label className="text-sm font-medium text-text-secondary">Quality</label>
              <select
                value={settings.depthQuality}
                onChange={(e) => settings.setSetting('depthQuality', e.target.value)}
                className="select-field"
              >
                {depthQualities.map((q) => (
                  <option key={q} value={q}>{q}</option>
                ))}
              </select>
            </div>
          </div>
        )}
      </div>

      <div className="bg-info/10 border border-info/20 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-info mt-0.5" />
          <div>
            <h3 className="text-sm font-semibold text-info mb-1">About Depth Generation</h3>
            <p className="text-sm text-text-secondary">
              Creates grayscale depth maps for 3D effects.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
