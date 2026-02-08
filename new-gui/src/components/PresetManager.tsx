import React, { useState, useEffect } from 'react';
import { FolderOpen, Trash2, Plus, X } from 'lucide-react';
import { useSettingsStore, SettingsState } from '../stores/settingsStore';

interface PresetManagerProps {
  isOpen: boolean;
  onClose: () => void;
}

// Keys to exclude from presets (input/output paths should not be saved)
const EXCLUDED_KEYS: (keyof SettingsState)[] = [
  'inputPath',
  'outputPath',
  'youtubeUrl',
  'imageInputPath',
  'imageOutputPath',
];

export const PresetManager: React.FC<PresetManagerProps> = ({ isOpen, onClose }) => {
  const settings = useSettingsStore();
  const [presets, setPresets] = useState<string[]>([]);
  const [newPresetName, setNewPresetName] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  useEffect(() => {
    if (isOpen) {
      loadPresetList();
    }
  }, [isOpen]);

  const loadPresetList = async () => {
    try {
      const list = await window.electronAPI?.presets.list();
      setPresets(list || []);
    } catch (err) {
      console.error('Failed to load presets:', err);
    }
  };

  const handleSavePreset = async () => {
    if (!newPresetName.trim()) {
      setMessage({ type: 'error', text: 'Please enter a preset name' });
      return;
    }

    setLoading(true);
    try {
      // Get current settings without excluded keys
      const settingsData: Record<string, unknown> = {};
      const state = useSettingsStore.getState();
      
      for (const key of Object.keys(state) as (keyof SettingsState)[]) {
        if (!EXCLUDED_KEYS.includes(key) && typeof state[key] !== 'function') {
          settingsData[key] = state[key];
        }
      }

      const result = await window.electronAPI?.presets.save(newPresetName, settingsData);
      if (result?.success) {
        setMessage({ type: 'success', text: `Preset "${newPresetName}" saved!` });
        setNewPresetName('');
        await loadPresetList();
      } else {
        setMessage({ type: 'error', text: result?.error || 'Failed to save preset' });
      }
    } catch (err) {
      setMessage({ type: 'error', text: 'Failed to save preset' });
    }
    setLoading(false);
    setTimeout(() => setMessage(null), 3000);
  };

  const handleLoadPreset = async (name: string) => {
    setLoading(true);
    try {
      const result = await window.electronAPI?.presets.load(name);
      if (result?.success && result.data) {
        // Apply settings without overwriting paths
        const newSettings: Partial<SettingsState> = {};
        for (const key of Object.keys(result.data)) {
          if (!EXCLUDED_KEYS.includes(key as keyof SettingsState)) {
            newSettings[key as keyof SettingsState] = result.data[key] as never;
          }
        }
        settings.setSettings(newSettings);
        setMessage({ type: 'success', text: `Preset "${name}" loaded!` });
      } else {
        setMessage({ type: 'error', text: result?.error || 'Failed to load preset' });
      }
    } catch (err) {
      setMessage({ type: 'error', text: 'Failed to load preset' });
    }
    setLoading(false);
    setTimeout(() => setMessage(null), 3000);
  };

  const handleDeletePreset = async (name: string) => {
    if (!confirm(`Delete preset "${name}"?`)) return;
    
    setLoading(true);
    try {
      const result = await window.electronAPI?.presets.delete(name);
      if (result?.success) {
        setMessage({ type: 'success', text: `Preset "${name}" deleted` });
        await loadPresetList();
      } else {
        setMessage({ type: 'error', text: result?.error || 'Failed to delete preset' });
      }
    } catch (err) {
      setMessage({ type: 'error', text: 'Failed to delete preset' });
    }
    setLoading(false);
    setTimeout(() => setMessage(null), 3000);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-surface border border-border rounded-xl shadow-2xl w-full max-w-md mx-4">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <h2 className="text-lg font-semibold text-text-primary flex items-center gap-2">
            <FolderOpen className="w-5 h-5 text-primary" />
            Preset Manager
          </h2>
          <button
            onClick={onClose}
            className="text-text-muted hover:text-text-primary transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Message */}
        {message && (
          <div className={`mx-4 mt-4 p-3 rounded-lg text-sm ${
            message.type === 'success' 
              ? 'bg-success/20 text-success border border-success/30' 
              : 'bg-error/20 text-error border border-error/30'
          }`}>
            {message.text}
          </div>
        )}

        {/* Save New Preset */}
        <div className="p-4 border-b border-border">
          <label className="text-sm font-medium text-text-secondary mb-2 block">
            Save Current Settings
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={newPresetName}
              onChange={(e) => setNewPresetName(e.target.value)}
              placeholder="Enter preset name..."
              className="input-field flex-1"
              disabled={loading}
            />
            <button
              onClick={handleSavePreset}
              disabled={loading || !newPresetName.trim()}
              className="btn-primary flex items-center gap-2 disabled:opacity-50"
            >
              <Plus className="w-4 h-4" />
              Save
            </button>
          </div>
        </div>

        {/* Preset List */}
        <div className="p-4 max-h-64 overflow-y-auto">
          <label className="text-sm font-medium text-text-secondary mb-2 block">
            Saved Presets
          </label>
          {presets.length === 0 ? (
            <p className="text-sm text-text-muted py-4 text-center">
              No presets saved yet
            </p>
          ) : (
            <div className="space-y-2">
              {presets.map((name) => (
                <div
                  key={name}
                  className="flex items-center justify-between p-3 bg-background rounded-lg border border-border hover:border-primary/50 transition-colors"
                >
                  <span className="text-sm text-text-primary font-medium">{name}</span>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleLoadPreset(name)}
                      disabled={loading}
                      className="text-primary hover:text-primary/80 transition-colors disabled:opacity-50"
                      title="Load preset"
                    >
                      <FolderOpen className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleDeletePreset(name)}
                      disabled={loading}
                      className="text-error hover:text-error/80 transition-colors disabled:opacity-50"
                      title="Delete preset"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-border">
          <button
            onClick={onClose}
            className="btn-secondary w-full"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
