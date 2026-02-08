import React from 'react';
import { Zap, Info } from 'lucide-react';
import { useSettingsStore } from '../../stores/settingsStore';

const compileModes = ['default', 'max', 'max-graphs'];
const tileSizes = [128, 256, 384, 512];

export const PerformanceTab: React.FC = () => {
  const settings = useSettingsStore();

  return (
    <div className="space-y-6 max-w-4xl">
      {/* FP16 */}
      <div className="card">
        <h2 className="section-title">
          <Zap className="w-5 h-5 text-primary" />
          Half Precision (FP16)
        </h2>
        <label className="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={settings.half}
            onChange={(e) => settings.setSetting('half', e.target.checked)}
            className="checkbox-field"
          />
          <span className="text-text-primary">Enable FP16 (faster, less VRAM)</span>
        </label>
        <p className="text-xs text-text-muted mt-2 ml-8">
          Recommended for modern GPUs. May not work on older cards.
        </p>
      </div>

      {/* Decode Method */}
      <div className="card">
        <h2 className="section-title">Decode Method</h2>
        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-text-secondary">Video Decoder</label>
            <select
              value={settings.decodeMethod}
              onChange={(e) => settings.setSetting('decodeMethod', e.target.value)}
              className="select-field"
            >
              <option value="cpu">CPU (Safe)</option>
              <option value="nvdec">NVDEC (Fast, NVIDIA only)</option>
            </select>
          </div>

          {settings.decodeMethod === 'nvdec' && (
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.nvdecCompat}
                onChange={(e) => settings.setSetting('nvdecCompat', e.target.checked)}
                className="checkbox-field"
              />
              <span className="text-text-primary">NVDEC Compatibility Mode (slower)</span>
            </label>
          )}
        </div>
      </div>

      {/* Compile Mode */}
      <div className="card">
        <h2 className="section-title">Compile Mode</h2>
        <select
          value={settings.compileMode}
          onChange={(e) => settings.setSetting('compileMode', e.target.value)}
          className="select-field"
        >
          {compileModes.map((mode) => (
            <option key={mode} value={mode}>{mode}</option>
          ))}
        </select>
      </div>

      {/* Tile Rendering */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="section-title">Tile Rendering</h2>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.tileRendering}
              onChange={(e) => settings.setSetting('tileRendering', e.target.checked)}
              className="checkbox-field"
            />
            <span className="text-sm font-medium text-text-primary">Enable</span>
          </label>
        </div>

        {settings.tileRendering && (
          <div className="space-y-2">
            <label className="text-sm font-medium text-text-secondary">Tile Size</label>
            <select
              value={settings.tileSize}
              onChange={(e) => settings.setSetting('tileSize', parseInt(e.target.value))}
              className="select-field"
            >
              {tileSizes.map((size) => (
                <option key={size} value={size}>{size}px</option>
              ))}
            </select>
            <p className="text-xs text-text-muted mt-1">
              Enable if you run out of VRAM. Smaller tiles = less VRAM but may show seams.
            </p>
          </div>
        )}
      </div>

      {/* Static Chunk */}
      <div className="card">
        <label className="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={settings.static}
            onChange={(e) => settings.setSetting('static', e.target.checked)}
            className="checkbox-field"
          />
          <span className="text-text-primary">Static Chunk Size (TensorRT)</span>
        </label>
      </div>

      <div className="bg-info/10 border border-info/20 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-info mt-0.5" />
          <div>
            <h3 className="text-sm font-semibold text-info mb-1">Performance Tips</h3>
            <p className="text-sm text-text-secondary">
              FP16 is faster and uses less VRAM - enable it if you have a modern GPU.
              Use Tile Rendering if you get OOM errors. Try 256 or 384 if you see grid lines.
              NVDEC is faster but may be unstable on some systems.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
