import React from 'react';
import { Wrench, Trash2, ArrowUp, ArrowDown, Plus, Info, Upload, Sparkles } from 'lucide-react';
import { useSettingsStore } from '../../stores/settingsStore';
import { RESTORE_METHODS, CUSTOM_BACKENDS, CUSTOM_BACKEND_LABELS } from '../../utils/builtinModelLists';

export const RestoreTab: React.FC = () => {
  const settings = useSettingsStore();
  const [selectedRestoreIndex, setSelectedRestoreIndex] = React.useState<number | null>(null);

  const addRestoreMethod = () => {
    settings.addRestoreChainItem(settings.restoreMethod);
  };

  const methodOptions = RESTORE_METHODS.filter((m) => m !== 'custom');

  const handleCustomToggle = (checked: boolean) => {
    settings.setSetting('useCustomRestoreModel', checked);
    if (checked) {
      settings.setSetting('restoreMethod', 'custom');
    } else if (settings.restoreMethod === 'custom') {
      settings.setSetting('restoreMethod', methodOptions[0] || 'scunet');
    }
  };

  const removeRestoreMethod = () => {
    if (selectedRestoreIndex !== null) {
      settings.removeRestoreChainItem(selectedRestoreIndex);
      setSelectedRestoreIndex(null);
    }
  };

  const moveUp = () => {
    if (selectedRestoreIndex !== null && selectedRestoreIndex > 0) {
      settings.moveRestoreChainItem(selectedRestoreIndex, selectedRestoreIndex - 1);
      setSelectedRestoreIndex(selectedRestoreIndex - 1);
    }
  };

  const moveDown = () => {
    if (selectedRestoreIndex !== null && selectedRestoreIndex < settings.restoreChain.length - 1) {
      settings.moveRestoreChainItem(selectedRestoreIndex, selectedRestoreIndex + 1);
      setSelectedRestoreIndex(selectedRestoreIndex + 1);
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
      settings.setSetting('restoreCustomModel', result.filePaths[0]);
      settings.setSetting('restoreMethod', 'custom');
      settings.setSetting('useCustomRestoreModel', true);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="section-title">
            <Wrench className="w-5 h-5 text-primary" />
            Video Restoration
          </h2>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.restoreEnabled}
              onChange={(e) => settings.setSetting('restoreEnabled', e.target.checked)}
              className="checkbox-field"
            />
            <span className="text-sm font-medium text-text-primary">Enable</span>
          </label>
        </div>

        {settings.restoreEnabled && (
          <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-text-secondary">Add Restoration Method</label>
                <div className="flex flex-col gap-2">
                  <div className="flex gap-2 items-center">
                    <select
                      value={settings.restoreMethod === 'custom' ? methodOptions[0] || '' : settings.restoreMethod}
                      onChange={(e) => settings.setSetting('restoreMethod', e.target.value)}
                      className={`select-field flex-1 ${settings.useCustomRestoreModel ? 'opacity-50 cursor-not-allowed' : ''}`}
                      disabled={settings.useCustomRestoreModel}
                    >
                      {methodOptions.map((method) => (
                        <option key={method} value={method}>{method}</option>
                      ))}
                    </select>
                    <button onClick={addRestoreMethod} className="btn-secondary">
                      <Plus className="w-4 h-4" />
                    </button>
                  </div>
                  <label className="flex items-center gap-2 text-sm font-medium text-text-primary">
                    <input
                      type="checkbox"
                      checked={settings.useCustomRestoreModel}
                      onChange={(e) => handleCustomToggle(e.target.checked)}
                      className="checkbox-field"
                    />
                    Custom model
                  </label>
                </div>
              
              {/* Custom Model */}
              {settings.restoreMethod === 'custom' && (
                <div className="mt-3 pt-3 border-t border-white/10 space-y-3">
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={settings.restoreCustomModel}
                      onChange={(e) => settings.setSetting('restoreCustomModel', e.target.value)}
                      placeholder="Select custom restoration model..."
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
                  
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-text-secondary">Custom Backend</label>
                    <select
                      value={settings.restoreCustomBackend}
                      onChange={(e) => settings.setSetting('restoreCustomBackend', e.target.value)}
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

            {/* Restore Chain List */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-text-secondary">
                Restoration Chain (Order Matters)
              </label>
              <div className="bg-surface border border-border rounded-lg p-2 min-h-[120px]">
                {settings.restoreChain.length === 0 ? (
                  <div className="text-text-muted text-sm text-center py-4">
                    No restoration methods added. Select a method and click + to add.
                  </div>
                ) : (
                  <ul className="space-y-1">
                    {settings.restoreChain.map((method, index) => (
                      <li
                        key={index}
                        onClick={() => setSelectedRestoreIndex(index)}
                        className={`px-3 py-2 rounded cursor-pointer text-sm ${
                          selectedRestoreIndex === index
                            ? 'bg-primary/20 text-primary'
                            : 'hover:bg-surface-light text-text-secondary'
                        }`}
                      >
                        {index + 1}. {method}
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              {/* Chain Controls */}
              {settings.restoreChain.length > 0 && (
                <div className="flex gap-2">
                  <button
                    onClick={removeRestoreMethod}
                    disabled={selectedRestoreIndex === null}
                    className="btn-secondary flex items-center gap-1 disabled:opacity-50"
                  >
                    <Trash2 className="w-4 h-4" />
                    Remove
                  </button>
                  <button
                    onClick={moveUp}
                    disabled={selectedRestoreIndex === null || selectedRestoreIndex === 0}
                    className="btn-secondary flex items-center gap-1 disabled:opacity-50"
                  >
                    <ArrowUp className="w-4 h-4" />
                    Up
                  </button>
                  <button
                    onClick={moveDown}
                    disabled={selectedRestoreIndex === null || selectedRestoreIndex >= settings.restoreChain.length - 1}
                    className="btn-secondary flex items-center gap-1 disabled:opacity-50"
                  >
                    <ArrowDown className="w-4 h-4" />
                    Down
                  </button>
                  <button
                    onClick={() => settings.clearRestoreChain()}
                    className="btn-secondary"
                  >
                    Clear All
                  </button>
                </div>
              )}
            </div>

            {/* Sharpening */}
            <div className="pt-4 border-t border-border">
              <label className="flex items-center gap-2 cursor-pointer mb-3">
                <input
                  type="checkbox"
                  checked={settings.sharpenEnabled}
                  onChange={(e) => settings.setSetting('sharpenEnabled', e.target.checked)}
                  className="checkbox-field"
                />
                <span className="text-text-primary font-medium">Enable Sharpening</span>
              </label>

              {settings.sharpenEnabled && (
                <div className="space-y-2 pl-6">
                  <label className="text-sm font-medium text-text-secondary">
                    Sharpening Sensitivity: {settings.sharpenSens}%
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={settings.sharpenSens}
                    onChange={(e) => settings.setSetting('sharpenSens', parseInt(e.target.value))}
                    className="w-full accent-primary"
                  />
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="bg-info/10 border border-info/20 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-info mt-0.5" />
          <div>
            <h3 className="text-sm font-semibold text-info mb-1">About Restoration</h3>
            <p className="text-sm text-text-secondary">
              Removes noise, compression artifacts, and improves overall quality.
              SCUNet is good for general cleanup, NAFNet for heavy noise.
              Do not overdo sharpening (&gt;70) to avoid halos around objects.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
