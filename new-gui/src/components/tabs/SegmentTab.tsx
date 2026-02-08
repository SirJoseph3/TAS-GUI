import React from 'react';
import { Scissors, Info, Upload, Sparkles } from 'lucide-react';
import { useSettingsStore } from '../../stores/settingsStore';
import { SEGMENT_METHODS } from '../../utils/builtinModelLists';

export const SegmentTab: React.FC = () => {
  const settings = useSettingsStore();

  const methodOptions = SEGMENT_METHODS.filter((m) => m !== 'custom');

  const handleCustomToggle = (checked: boolean) => {
    settings.setSetting('useCustomSegmentModel', checked);
    if (checked) {
      settings.setSetting('segmentMethod', 'custom');
    } else if (settings.segmentMethod === 'custom') {
      settings.setSetting('segmentMethod', methodOptions[0] || 'anime');
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
      settings.setSetting('segmentCustomModel', result.filePaths[0]);
      settings.setSetting('segmentMethod', 'custom');
      settings.setSetting('useCustomSegmentModel', true);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="section-title">
            <Scissors className="w-5 h-5 text-primary" />
            Video Segmentation
          </h2>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.segmentEnabled}
              onChange={(e) => settings.setSetting('segmentEnabled', e.target.checked)}
              className="checkbox-field"
            />
            <span className="text-sm font-medium text-text-primary">Enable</span>
          </label>
        </div>

        {settings.segmentEnabled && (
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-text-secondary">Segmentation Method</label>
              <div className="flex items-center gap-3">
                <select
                  value={settings.segmentMethod === 'custom' ? methodOptions[0] || '' : settings.segmentMethod}
                  onChange={(e) => settings.setSetting('segmentMethod', e.target.value)}
                  className={`select-field flex-1 ${settings.useCustomSegmentModel ? 'opacity-50 cursor-not-allowed' : ''}`}
                  disabled={settings.useCustomSegmentModel}
                >
                  {methodOptions.map((method) => (
                    <option key={method} value={method}>{method}</option>
                  ))}
                </select>
                <label className="flex items-center gap-2 text-sm font-medium text-text-primary">
                  <input
                    type="checkbox"
                    checked={settings.useCustomSegmentModel}
                    onChange={(e) => handleCustomToggle(e.target.checked)}
                    className="checkbox-field"
                  />
                  Custom model
                </label>
              </div>
            </div>

            {settings.useCustomSegmentModel && (
              <div className="pt-2 border-t border-white/10">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-text-secondary flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-primary" />
                    Custom Segment Model
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={settings.segmentCustomModel}
                      onChange={(e) => settings.setSetting('segmentCustomModel', e.target.value)}
                      placeholder="Select custom segment model..."
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
          </div>
        )}
      </div>

      <div className="bg-info/10 border border-info/20 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-info mt-0.5" />
          <div>
            <h3 className="text-sm font-semibold text-info mb-1">About Segmentation</h3>
            <p className="text-sm text-text-secondary">
              Separates characters from background for advanced editing.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
