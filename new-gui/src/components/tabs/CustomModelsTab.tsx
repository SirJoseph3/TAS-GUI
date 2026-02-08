import React from 'react';
import {
  Sparkles,
  Upload,
  Info,
  Maximize2,
  Activity,
  Wrench,
  Scissors,
  Layers,
} from 'lucide-react';
import { useSettingsStore } from '../../stores/settingsStore';
import { CUSTOM_BACKENDS, CUSTOM_BACKEND_LABELS } from '../../utils/builtinModelLists';

type CustomModelKind = 'upscale' | 'interpolate' | 'restore' | 'segment' | 'depth';

export const CustomModelsTab: React.FC = () => {
  const settings = useSettingsStore();

  const selectModelFile = async (kind: CustomModelKind) => {
    const result = await window.electronAPI?.dialog.selectFile({
      filters: [
        { name: 'Model Files', extensions: ['pth', 'onnx', 'engine', 'pt'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });

    if (!result || result.canceled || !result.filePaths?.length) return;
    const path = result.filePaths[0];

    switch (kind) {
      case 'upscale':
        settings.setSetting('upscaleCustomModel', path);
        settings.setSetting('upscaleMethod', 'custom');
        break;
      case 'interpolate':
        settings.setSetting('interpolateCustomModel', path);
        settings.setSetting('interpolateMethod', 'custom');
        break;
      case 'restore':
        settings.setSetting('restoreCustomModel', path);
        settings.setSetting('restoreMethod', 'custom');
        break;
      case 'segment':
        settings.setSetting('segmentCustomModel', path);
        settings.setSetting('segmentMethod', 'custom');
        break;
      case 'depth':
        settings.setSetting('depthCustomModel', path);
        settings.setSetting('depthMethod', 'custom');
        break;
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="card">
        <h2 className="section-title">
          <Sparkles className="w-5 h-5 text-primary" />
          Custom Models
        </h2>
        <p className="text-sm text-text-secondary">
          Put your custom model paths in one place. Choosing a file here will also switch the related
          Method dropdown to <b>custom</b>.
        </p>
        <div className="mt-3 bg-info/10 border border-info/20 rounded-lg p-3">
          <div className="flex items-start gap-3">
            <Info className="w-5 h-5 text-info mt-0.5" />
            <div className="text-sm text-text-secondary">
              <div className="font-semibold text-info">Backends</div>
              <div>
                <b>Auto</b> = try to detect from file extension; <b>CUDA/TensorRT/DirectML/NCNN</b> forces a specific runtime.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Upscale */}
      <div className="card">
        <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
          <Maximize2 className="w-4 h-4" /> Upscale
        </h3>
        <div className="flex gap-2">
          <input
            type="text"
            value={settings.upscaleCustomModel}
            onChange={(e) => settings.setSetting('upscaleCustomModel', e.target.value)}
            placeholder="Select custom upscale model..."
            className="input-field flex-1"
          />
          <button
            onClick={() => selectModelFile('upscale')}
            className="btn-secondary flex items-center gap-2"
          >
            <Upload className="w-4 h-4" /> Browse
          </button>
        </div>
        <div className="mt-3">
          <label className="text-sm font-medium text-text-secondary">Backend</label>
          <select
            value={settings.upscaleCustomBackend}
            onChange={(e) => settings.setSetting('upscaleCustomBackend', e.target.value)}
            className="select-field mt-1"
          >
            {CUSTOM_BACKENDS.map((b) => (
              <option key={b} value={b}>
                {CUSTOM_BACKEND_LABELS[b]}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Interpolate */}
      <div className="card">
        <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
          <Activity className="w-4 h-4" /> Interpolate
        </h3>
        <div className="flex gap-2">
          <input
            type="text"
            value={settings.interpolateCustomModel}
            onChange={(e) => settings.setSetting('interpolateCustomModel', e.target.value)}
            placeholder="Select custom interpolate model..."
            className="input-field flex-1"
          />
          <button
            onClick={() => selectModelFile('interpolate')}
            className="btn-secondary flex items-center gap-2"
          >
            <Upload className="w-4 h-4" /> Browse
          </button>
        </div>
        <div className="mt-3">
          <label className="text-sm font-medium text-text-secondary">Backend</label>
          <select
            value={settings.interpolateCustomBackend}
            onChange={(e) => settings.setSetting('interpolateCustomBackend', e.target.value)}
            className="select-field mt-1"
          >
            {CUSTOM_BACKENDS.map((b) => (
              <option key={b} value={b}>
                {CUSTOM_BACKEND_LABELS[b]}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Restore */}
      <div className="card">
        <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
          <Wrench className="w-4 h-4" /> Restore
        </h3>
        <div className="flex gap-2">
          <input
            type="text"
            value={settings.restoreCustomModel}
            onChange={(e) => settings.setSetting('restoreCustomModel', e.target.value)}
            placeholder="Select custom restore model..."
            className="input-field flex-1"
          />
          <button
            onClick={() => selectModelFile('restore')}
            className="btn-secondary flex items-center gap-2"
          >
            <Upload className="w-4 h-4" /> Browse
          </button>
        </div>
        <div className="mt-3">
          <label className="text-sm font-medium text-text-secondary">Backend</label>
          <select
            value={settings.restoreCustomBackend}
            onChange={(e) => settings.setSetting('restoreCustomBackend', e.target.value)}
            className="select-field mt-1"
          >
            {CUSTOM_BACKENDS.map((b) => (
              <option key={b} value={b}>
                {CUSTOM_BACKEND_LABELS[b]}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Segment */}
      <div className="card">
        <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
          <Scissors className="w-4 h-4" /> Segment
        </h3>
        <div className="flex gap-2">
          <input
            type="text"
            value={settings.segmentCustomModel}
            onChange={(e) => settings.setSetting('segmentCustomModel', e.target.value)}
            placeholder="Select custom segment model..."
            className="input-field flex-1"
          />
          <button
            onClick={() => selectModelFile('segment')}
            className="btn-secondary flex items-center gap-2"
          >
            <Upload className="w-4 h-4" /> Browse
          </button>
        </div>
      </div>

      {/* Depth */}
      <div className="card">
        <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
          <Layers className="w-4 h-4" /> Depth
        </h3>
        <div className="flex gap-2">
          <input
            type="text"
            value={settings.depthCustomModel}
            onChange={(e) => settings.setSetting('depthCustomModel', e.target.value)}
            placeholder="Select custom depth model..."
            className="input-field flex-1"
          />
          <button
            onClick={() => selectModelFile('depth')}
            className="btn-secondary flex items-center gap-2"
          >
            <Upload className="w-4 h-4" /> Browse
          </button>
        </div>
      </div>
    </div>
  );
};
