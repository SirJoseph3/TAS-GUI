import React from 'react';
import { Target, Info } from 'lucide-react';
import { useSettingsStore } from '../../stores/settingsStore';
import { OBJ_DETECT_METHODS } from '../../utils/builtinModelLists';

export const ObjectDetectionTab: React.FC = () => {
  const settings = useSettingsStore();

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="section-title">
            <Target className="w-5 h-5 text-primary" />
            Object Detection (YOLOv9)
          </h2>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.objDetectEnabled}
              onChange={(e) => settings.setSetting('objDetectEnabled', e.target.checked)}
              className="checkbox-field"
            />
            <span className="text-sm font-medium text-text-primary">Enable</span>
          </label>
        </div>

        {settings.objDetectEnabled && (
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-text-secondary">Detection Method</label>
              <select
                value={settings.objDetectMethod}
                onChange={(e) => settings.setSetting('objDetectMethod', e.target.value)}
                className="select-field"
              >
                {OBJ_DETECT_METHODS.map((method) => (
                  <option key={method} value={method}>{method}</option>
                ))}
              </select>
            </div>

            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.objDetectDisableAnnotations}
                onChange={(e) => settings.setSetting('objDetectDisableAnnotations', e.target.checked)}
                className="checkbox-field"
              />
              <span className="text-text-primary">Disable labels/confidence on boxes</span>
            </label>
          </div>
        )}
      </div>

      <div className="bg-info/10 border border-info/20 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-info mt-0.5" />
          <div>
            <h3 className="text-sm font-semibold text-info mb-1">About Object Detection</h3>
            <p className="text-sm text-text-secondary">
              When enabled, TAS will run object detection on the video and annotate detected objects.
              DirectML models work on AMD/Intel GPUs, OpenVINO models work on Intel CPUs/GPUs.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
