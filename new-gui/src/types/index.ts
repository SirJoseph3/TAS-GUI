// Main types for TAS-GUI

export interface ProcessResult {
  success: boolean;
  error?: string;
  pid?: number;
}

export interface ElectronAPI {
  // Window controls
  window: {
    minimize: () => Promise<void>;
    maximize: () => Promise<void>;
    close: () => Promise<void>;
    isMaximized: () => Promise<boolean>;
  };

  // Dialogs
  dialog: {
    selectFile: (options?: { filters?: FileFilter[] }) => Promise<DialogResult>;
    selectOutput: (options?: { filters?: FileFilter[] }) => Promise<DialogResult>;
    selectFolder: () => Promise<DialogResult>;
  };

// Python processing
  processing: {
    start: (args: string[]) => Promise<ProcessResult>;
    stop: () => Promise<ProcessResult>;
    startImageUpscale: (args: string[]) => Promise<ProcessResult>;
    stopImageUpscale: () => Promise<ProcessResult>;
  };

  // Preset management
  presets: {
    list: () => Promise<string[]>;
    load: (name: string) => Promise<{ success: boolean; data?: Record<string, unknown>; error?: string }>;
    save: (name: string, data: Record<string, unknown>) => Promise<{ success: boolean; error?: string }>;
    delete: (name: string) => Promise<{ success: boolean; error?: string }>;
  };

  // Video operations
  video: {
    readFile: (path: string) => Promise<{ success: boolean; dataUrl?: string }>;
  };

  // Live preview (preview.jpg)
  preview: {
    getImage: () => Promise<string | null>;
  };

  // Process event listeners
  onProcessOutput: (callback: (data: { type: 'stdout' | 'stderr'; data: string }) => void) => void;
  onProcessFinished: (callback: (data: { code: number }) => void) => void;
  onProcessError: (callback: (data: { error: string }) => void) => void;
  removeAllListeners: (channel: string) => void;

  // Utilities
  openExternal: (url: string) => Promise<void>;
  getAppVersion: () => Promise<string>;
  getPlatform: () => Promise<string>;
}

export interface DialogResult {
  canceled: boolean;
  filePath?: string;
  filePaths?: string[];
}

export interface FileFilter {
  name: string;
  extensions: string[];
}

export interface SelectFilesOptions {
  properties?: ('openFile' | 'openDirectory' | 'multiSelections' | 'showHiddenFiles')[];
  filters?: FileFilter[];
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}

// Upscale Types
export interface UpscalePreset {
  upscaleFactor: number;
  interimRes: 'Same' | '2x' | '4x';
  customModel: string;
  modelType: 'ncnn' | 'ncnn-pytorch' | 'onnx' | 'directml' | 'onnx-dml' | 'tensorrt' | 'onnx-tensorrt';
  architecture: 'SPAN' | 'Shufflecugan' | 'cugan' | 'compact' | 'ultracompact' | 'superultracompact' | 'custom';
  fp16: boolean;
  onnxOptimization: string;
  upscaleWidth: number;
  tileWidth: number;
  tileHeight: number;
  rifeStride: number;
  rifeDenoise: boolean;
  rifeFP16: boolean;
}

// Interpolate Types  
export interface InterpolatePreset {
  interpolationFactor: number;
  rifeModel: 
    | 'rife46' | 'rife47' | 'rife48'
    | 'rife49' | 'rife410' | 'rife411'
    | 'rife411-tensorrt' | 'rife412' | 'rife413-lite';
  ensembleMode: 'Disabled' | 'Enabled' | 'Fast';
  fp16: boolean;
  sceneChangeDetection: boolean;
  sensitivity: number;
  dynamicSensitivity: boolean;
}

// Restore Types
export interface RestorePreset {
  restore: boolean;
  denoise: boolean;
  derain: boolean;
  dehaze: boolean;
  deblur: boolean;
  decontour: boolean;
  removeObjects: boolean;
  customModels: boolean;
  customModel: string;
  sceneChangeDetection: boolean;
  autoTune: boolean;
  modelType: 'spandrel' | 'onnx';
  fp16: boolean;
}

// Settings Types
export interface AppSettings {
  inputPaths: string[];
  outputPath: string;
  recursiveFolder: boolean;
  preserveFolderStructure: boolean;
  encoder: 'x264' | 'x265' | 'av1' | 'vp9' | 'prores' | 'nvenc_h264' | 'nvenc_h265' | 'nvenc_av1' | 'qsv_h264' | 'qsv_h265' | 'qsv_av1' | 'vvc' | 'x264_animation' | 'x265_animation';
  pixelFormat: 'yuv420p' | 'yuv420p10le' | 'yuv422p' | 'yuv422p10le' | 'yuv444p' | 'yuv444p10le';
  quality: 'low' | 'medium' | 'high' | 'ultra';
  compression: 'low' | 'medium' | 'high' | 'ultra';
  customEncoderArgs: string;
}

// Queue Types
export interface QueueItem {
  id: string;
  inputPath: string;
  outputPath: string;
  status: 'pending' | 'preparing' | 'processing' | 'paused' | 'completed' | 'error';
  progress: number;
  currentStage: string;
  fps: number;
  frame: number;
  totalFrames: number;
  eta: string;
  error?: string;
}

// Settings categories
export type SettingsCategory = 
  | 'presets' 
  | 'upscale' 
  | 'interpolate' 
  | 'segment' 
  | 'restore' 
  | 'output' 
  | 'app';
