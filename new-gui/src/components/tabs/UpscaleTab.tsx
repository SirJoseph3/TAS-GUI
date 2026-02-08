import React from 'react';
import { Maximize2, Info, Upload, Sparkles } from 'lucide-react';
import { useSettingsStore } from '../../stores/settingsStore';
import { UPSCALE_METHODS, CUSTOM_BACKENDS, CUSTOM_BACKEND_LABELS } from '../../utils/builtinModelLists';

export const UpscaleTab: React.FC = () => {
  const settings = useSettingsStore();

  const methodOptions = UPSCALE_METHODS.filter((m) => m !== 'custom');

  const handleCustomToggle = (checked: boolean) => {
    settings.setSetting('useCustomUpscaleModel', checked);
    if (checked) {
      settings.setSetting('upscaleMethod', 'custom');
    } else if (settings.upscaleMethod === 'custom') {
      settings.setSetting('upscaleMethod', methodOptions[0] || 'shufflecugan');
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
      settings.setSetting('upscaleCustomModel', result.filePaths[0]);
      settings.setSetting('upscaleMethod', 'custom');
      settings.setSetting('useCustomUpscaleModel', true);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="section-title">
            <Maximize2 className="w-5 h-5 text-primary" />
            AI Upscaling
          </h2>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.upscaleEnabled}
              onChange={(e) => settings.setSetting('upscaleEnabled', e.target.checked)}
              className="checkbox-field"
            />
            <span className="text-sm font-medium text-text-primary">Enable</span>
          </label>
        </div>

        {settings.upscaleEnabled && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-text-secondary">Upscale Factor</label>
                <select
                  value={settings.upscaleFactor}
                  onChange={(e) => settings.setSetting('upscaleFactor', Number(e.target.value))}
                  className="select-field"
                >
                  <option value={1}>1x</option>
                  <option value={2}>2x</option>
                  <option value={3}>3x</option>
                  <option value={4}>4x</option>
                </select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-text-secondary">Method</label>
                <div className="flex items-center gap-3">
                  <select
                    value={settings.upscaleMethod === 'custom' ? methodOptions[0] || '' : settings.upscaleMethod}
                    onChange={(e) => settings.setSetting('upscaleMethod', e.target.value)}
                    className={`select-field flex-1 ${settings.useCustomUpscaleModel ? 'opacity-50 cursor-not-allowed' : ''}`}
                    disabled={settings.useCustomUpscaleModel}
                  >
                    {methodOptions.map((method) => (
                      <option key={method} value={method}>
                        {method}
                      </option>
                    ))}
                  </select>
                  <label className="flex items-center gap-2 text-sm font-medium text-text-primary">
                    <input
                      type="checkbox"
                      checked={settings.useCustomUpscaleModel}
                      onChange={(e) => handleCustomToggle(e.target.checked)}
                      className="checkbox-field"
                    />
                    Custom model
                  </label>
                </div>
              </div>
            </div>

            {/* Custom Model */}
            {settings.useCustomUpscaleModel && (
              <div className="pt-2 border-t border-white/10">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-text-secondary flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-primary" />
                    Custom Model
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={settings.upscaleCustomModel}
                      onChange={(e) => settings.setSetting('upscaleCustomModel', e.target.value)}
                      placeholder="Select custom .pth or .onnx model..."
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
                    value={settings.upscaleCustomBackend}
                    onChange={(e) => settings.setSetting('upscaleCustomBackend', e.target.value)}
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
          </div>
        )}
      </div>

      <div className="bg-info/10 border border-info/20 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-info mt-0.5" />
          <div>
            <h3 className="text-sm font-semibold text-info mb-1">About Upscaling</h3>
            <p className="text-sm text-text-secondary">
              Upscaling increases video resolution using AI. 2x doubles the resolution, 
              4x quadruples it. TensorRT models are fastest but require NVIDIA GPU.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
