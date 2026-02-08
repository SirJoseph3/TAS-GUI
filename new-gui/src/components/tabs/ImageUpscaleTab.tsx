import React, { useState, useEffect, useCallback } from 'react';
import { Image, FolderOpen, Save, Info, Maximize2, X, Play, Square } from 'lucide-react';
import { useSettingsStore } from '../../stores/settingsStore';
import { useProcessingStore } from '../../stores/processingStore';

// Task Progress: Fixed "Cannot read properties of undefined (reading 'exec')" error
// - [x] Identified the issue: subprocess API not exposed in preload
// - [x] Modified code to use existing processing.start API
// - [x] Fixed TypeScript error with proper error handling

// Task Progress: Fixed "Cannot read properties of undefined (reading 'exec')" error
// - [x] Identified the issue: subprocess API not exposed in preload
// - [x] Modified code to use existing processing.start API
// - [x] Fixed TypeScript error with proper error handling

// Task Progress: Fixed "Cannot read properties of undefined (reading 'exec')" error
// - [x] Identified the issue: subprocess API not exposed in preload
// - [x] Modified code to use existing processing.start API
// - [x] Fixed TypeScript error with proper error handling

// Task Progress: Fixed "Cannot read properties of undefined (reading 'exec')" error
// - [x] Identified the issue: subprocess API not exposed in preload
// - [x] Modified code to use existing processing.start API
// - [x] Fixed TypeScript error with proper error handling

export const ImageUpscaleTab: React.FC = () => {
  const settings = useSettingsStore();
  const { status, startProcessing, addLog, setError } = useProcessingStore();
  
  const [inputPreviewUrl, setInputPreviewUrl] = useState<string | null>(null);
  const [outputPreviewUrl, setOutputPreviewUrl] = useState<string | null>(null);
  const [isInputFullscreen, setIsInputFullscreen] = useState(false);
  const [isOutputFullscreen, setIsOutputFullscreen] = useState(false);

  const isProcessing = status === 'processing' || status === 'stopping';

  const buildAutoOutputPath = useCallback((inputPath: string) => {
    const baseName = inputPath.substring(0, inputPath.lastIndexOf('.'));
    const fileName = baseName.substring(baseName.lastIndexOf('\\') + 1).substring(baseName.lastIndexOf('/') + 1);
    
    // Get upscale method and remove backend suffix
    let method = settings.upscaleMethod || 'upscale';
    const backendSuffixes = ['-tensorrt', '-directml', '-ncnn'];
    for (const suffix of backendSuffixes) {
      if (method.endsWith(suffix)) {
        method = method.substring(0, method.length - suffix.length);
        break;
      }
    }
    
    const factor = settings.upscaleFactor || 2;
    const methodPart = method ? `_${method}` : '';
    const outExt = settings.imageOutputFormat === 'jpg' ? '.jpg' : '.png';
    
    // Build path in output directory
    const outputDir = baseName.substring(0, baseName.lastIndexOf('\\') >= 0 ? baseName.lastIndexOf('\\') : baseName.lastIndexOf('/'));
    return `${outputDir}${outputDir ? '\\' : ''}${fileName}_upscaled${methodPart}_${factor}x${outExt}`;
  }, [settings.upscaleMethod, settings.upscaleFactor, settings.imageOutputFormat]);

  useEffect(() => {
    if (settings.imageInputPath) {
      const loadInputPreview = async () => {
        if (window.electronAPI?.video?.readFile) {
          try {
            const result = await window.electronAPI.video.readFile(settings.imageInputPath);
            if (result.success) {
              setInputPreviewUrl(result.dataUrl);
            }
          } catch (err) {
            console.error('Failed to load input image preview:', err);
          }
        }
      };
      loadInputPreview();
    } else {
      setInputPreviewUrl(null);
    }
  }, [settings.imageInputPath]);

  useEffect(() => {
    if (settings.imageOutputPath) {
      const loadOutputPreview = async () => {
        if (window.electronAPI?.video?.readFile) {
          try {
            const result = await window.electronAPI.video.readFile(settings.imageOutputPath);
            if (result.success) {
              setOutputPreviewUrl(result.dataUrl);
            }
          } catch (err) {
            setOutputPreviewUrl(null);
          }
        }
      };
      loadOutputPreview();
    } else {
      setOutputPreviewUrl(null);
    }
  }, [settings.imageOutputPath]);

  // Update output path when model changes
  useEffect(() => {
    if (settings.imageInputPath) {
      const newPath = buildAutoOutputPath(settings.imageInputPath);
      settings.setSetting('imageOutputPath', newPath);
    }
  }, [settings.upscaleMethod, settings.upscaleFactor, settings.imageOutputFormat, buildAutoOutputPath]);

  const handleSelectImageInput = async () => {
    const result = await window.electronAPI?.dialog.selectFile({
      filters: [
        { name: 'Image Files', extensions: ['png', 'jpg', 'jpeg'] },
        { name: 'All Files', extensions: ['*'] }
      ]
    });
    
    if (result && !result.canceled && result.filePaths) {
      settings.setSetting('imageInputPath', result.filePaths[0]);
      
      // Auto-generate output path with model name
      if (!settings.imageOutputPath) {
        settings.setSetting('imageOutputPath', buildAutoOutputPath(result.filePaths[0]));
      }
    }
  };

  const handleSelectImageOutput = async () => {
    const result = await window.electronAPI?.dialog.selectOutput({
      filters: [
        { name: 'PNG Image', extensions: ['png'] },
        { name: 'JPG Image', extensions: ['jpg', 'jpeg'] }
      ]
    });
    
    if (result && !result.canceled && result.filePath) {
      settings.setSetting('imageOutputPath', result.filePath);
    }
  };

  const getFormatExt = () => settings.imageOutputFormat === 'jpg' ? '.jpg' : '.png';

  const handleStartImageUpscale = async () => {
    if (!settings.imageInputPath) {
      setError('Please select an input image');
      return;
    }

    if (!settings.imageOutputPath) {
      setError('Please specify an output path');
      return;
    }

    startProcessing();
    addLog({ type: 'info', message: 'Starting image upscale...', timestamp: Date.now() });

    try {
      const args = [
        '--input', settings.imageInputPath,
        '--output', settings.imageOutputPath,
        '--method', settings.upscaleMethod,
        '--factor', settings.upscaleFactor.toString(),
      ];

      if (settings.imageOutputFormat === 'jpg') {
        args.push('--jpeg_quality', settings.imageJpegQuality.toString());
      }

      // Add restore settings if enabled
      if (settings.restoreEnabled) {
        args.push('--restore');
        const restoreChain = settings.restoreChain.length > 0 
          ? settings.restoreChain 
          : [settings.restoreMethod];
        const hasCustom = restoreChain.includes('custom');
        const mappedMethods = restoreChain.map((m) => {
          if (m === 'custom' && settings.restoreCustomModel) {
            return settings.restoreCustomModel;
          }
          return m;
        });
        args.push('--restore_method', ...mappedMethods);
        
        if (hasCustom && settings.restoreCustomBackend !== 'default') {
          args.push('--custom_restore_backend', settings.restoreCustomBackend);
        }
      }

      // Add performance settings
      if (settings.half) {
        args.push('--half');
      }

      // Add custom model if using custom upscale method
      if (settings.upscaleMethod === 'custom' && settings.upscaleCustomModel) {
        args.push('--custom_model', settings.upscaleCustomModel);
        if (settings.upscaleCustomBackend !== 'default') {
          args.push('--custom_upscale_backend', settings.upscaleCustomBackend);
        }
      }

      // Add tile rendering settings
      if (settings.tileRendering) {
        args.push('--tile_size', settings.tileSize.toString());
      }

      // Add compile mode
      args.push('--compile_mode', settings.compileMode);

      // Execute the image upscale Python CLI
      const result = await window.electronAPI?.processing.startImageUpscale(args);

      if (!result?.success) {
        setError(result?.error || 'Failed to start image upscale');
      } else {
        // Update preview after successful processing
        const loadOutputPreview = async () => {
          if (window.electronAPI?.video?.readFile) {
            try {
              const result = await window.electronAPI.video.readFile(settings.imageOutputPath);
              if (result.success) {
                setOutputPreviewUrl(result.dataUrl);
              }
            } catch (err) {
              console.error('Failed to load output image preview:', err);
            }
          }
        };
        loadOutputPreview();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  };

  const handleStopImageUpscale = async () => {
    await window.electronAPI?.processing.stopImageUpscale();
  };

  return (
    <>
      {/* Input Fullscreen Modal */}
      {isInputFullscreen && inputPreviewUrl && (
        <div 
          className="fixed inset-0 z-50 bg-black flex items-center justify-center"
          onClick={() => setIsInputFullscreen(false)}
        >
          <img 
            src={inputPreviewUrl} 
            alt="Input fullscreen" 
            className="max-w-full max-h-full object-contain" 
          />
          <button
            onClick={() => setIsInputFullscreen(false)}
            className="absolute top-4 right-4 text-white hover:text-gray-300 transition-colors"
          >
            <X className="w-8 h-8" />
          </button>
        </div>
      )}

      {/* Output Fullscreen Modal */}
      {isOutputFullscreen && outputPreviewUrl && (
        <div 
          className="fixed inset-0 z-50 bg-black flex items-center justify-center"
          onClick={() => setIsOutputFullscreen(false)}
        >
          <img 
            src={outputPreviewUrl} 
            alt="Output fullscreen" 
            className="max-w-full max-h-full object-contain" 
          />
          <button
            onClick={() => setIsOutputFullscreen(false)}
            className="absolute top-4 right-4 text-white hover:text-gray-300 transition-colors"
          >
            <X className="w-8 h-8" />
          </button>
        </div>
      )}

      <div className="space-y-6 max-w-4xl">
      <div className="card">
        <h2 className="section-title">
          <Image className="w-5 h-5 text-primary" />
          Image Upscale
        </h2>

        {/* Input Image */}
        <div className="space-y-2 mb-4">
          <label className="text-sm font-medium text-text-secondary">Input Image</label>
          <div className="flex gap-2">
            <input
              type="text"
              value={settings.imageInputPath}
              onChange={(e) => settings.setSetting('imageInputPath', e.target.value)}
              placeholder="Select input image (PNG/JPG)..."
              className="input-field flex-1"
            />
            <button
              onClick={handleSelectImageInput}
              className="btn-secondary flex items-center gap-2"
            >
              <FolderOpen className="w-4 h-4" />
              Browse
            </button>
          </div>
        </div>

        {/* Output Image */}
        <div className="space-y-2 mb-4">
          <label className="text-sm font-medium text-text-secondary">Output Image</label>
          <div className="flex gap-2">
            <input
              type="text"
              value={settings.imageOutputPath}
              onChange={(e) => settings.setSetting('imageOutputPath', e.target.value)}
              placeholder="Leave empty for auto-generated name..."
              className="input-field flex-1"
            />
            <button
              onClick={handleSelectImageOutput}
              className="btn-secondary flex items-center gap-2"
            >
              <Save className="w-4 h-4" />
              Browse
            </button>
          </div>
        </div>

        {/* Output Format */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-text-secondary">Output Format</label>
            <select
              value={settings.imageOutputFormat}
              onChange={(e) => {
                const format = e.target.value as 'png' | 'jpg';
                settings.setSetting('imageOutputFormat', format);
                // Update output path extension
                if (settings.imageOutputPath) {
                  const baseName = settings.imageOutputPath.replace(/\.[^/.]+$/, '');
                  settings.setSetting('imageOutputPath', `${baseName}.${format}`);
                }
              }}
              className="select-field"
            >
              <option value="png">PNG (Lossless)</option>
              <option value="jpg">JPG (Compressed)</option>
            </select>
          </div>

          {settings.imageOutputFormat === 'jpg' && (
            <div className="space-y-2">
              <label className="text-sm font-medium text-text-secondary">JPEG Quality</label>
              <input
                type="number"
                min={1}
                max={100}
                value={settings.imageJpegQuality}
                onChange={(e) => settings.setSetting('imageJpegQuality', parseInt(e.target.value))}
                className="input-field"
              />
            </div>
          )}
        </div>
      </div>

      {/* Start Button */}
      <div className="card">
        <div className="flex gap-2">
          <button
            onClick={handleStartImageUpscale}
            disabled={isProcessing || !settings.imageInputPath || !settings.imageOutputPath}
            className="flex-1 btn-primary flex items-center justify-center gap-2 disabled:opacity-50 py-3"
          >
            <Play className="w-5 h-5" />
            Upscale Image
          </button>
          <button
            onClick={handleStopImageUpscale}
            disabled={!isProcessing}
            className="btn-danger flex items-center justify-center gap-2 disabled:opacity-50 px-4 py-3"
            title="Stop processing"
          >
            <Square className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Preview Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Preview */}
        <div className="card">
          <h3 className="text-sm font-semibold text-primary mb-3">Input Image Preview</h3>
          <div 
            className="relative w-full aspect-video rounded-lg overflow-hidden bg-background-secondary border border-border flex items-center justify-center group cursor-pointer"
            onClick={() => inputPreviewUrl && setIsInputFullscreen(true)}
          >
            {inputPreviewUrl ? (
              <>
                <img src={inputPreviewUrl} alt="Input preview" className="w-full h-full object-contain" />
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-all flex items-center justify-center">
                  <Maximize2 className="w-8 h-8 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              </>
            ) : (
              <span className="text-sm text-text-muted text-center px-4">
                Input image preview will appear here...
              </span>
            )}
          </div>
        </div>

        {/* Output Preview */}
        <div className="card">
          <h3 className="text-sm font-semibold text-secondary mb-3">Output Image Preview</h3>
          <div 
            className="relative w-full aspect-video rounded-lg overflow-hidden bg-background-secondary border border-border flex items-center justify-center group cursor-pointer"
            onClick={() => outputPreviewUrl && setIsOutputFullscreen(true)}
          >
            {outputPreviewUrl ? (
              <>
                <img src={outputPreviewUrl} alt="Output preview" className="w-full h-full object-contain" />
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-all flex items-center justify-center">
                  <Maximize2 className="w-8 h-8 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              </>
            ) : (
              <span className="text-sm text-text-muted text-center px-4">
                Upscaled image preview will appear here...
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Note Card */}
      <div className="bg-info/10 border border-info/20 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-info mt-0.5" />
          <div>
            <h3 className="text-sm font-semibold text-info mb-1">Important Notes</h3>
            <p className="text-sm text-text-secondary">
              Model / Factor settings are taken from the Upscaling tab.
              Restoration (if enabled) is taken from the Restoration tab.
              FP16 / Compile / Tile settings are taken from the Performance tab.
            </p>
          </div>
        </div>
      </div>
    </div>
    </>
  );
};
