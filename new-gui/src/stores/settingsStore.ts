import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export interface CustomModel {
  path: string;
  backend: string;
}

export interface SettingsState {
  // General
  inputPath: string;
  outputPath: string;
  youtubeUrl: string;
  audioCopy: boolean;
  benchmark: boolean;
  
  // Upscale
  upscaleEnabled: boolean;
  upscaleFactor: number;
  upscaleMethod: string;
  upscaleCustomModel: string;
  upscaleCustomBackend: string;
  useCustomUpscaleModel: boolean;
  
  // Image Upscale
  imageInputPath: string;
  imageOutputPath: string;
  imageOutputFormat: 'png' | 'jpg';
  imageJpegQuality: number;
  
  // Interpolate
  interpolateEnabled: boolean;
  interpolateFactor: number;
  interpolateMethod: string;
  interpolateCustomModel: string;
  interpolateCustomBackend: string;
  useCustomInterpolateModel: boolean;
  ensemble: boolean;
  dynamicScale: boolean;
  slowmo: boolean;
  
  // Restore
  restoreEnabled: boolean;
  restoreMethod: string;
  restoreChain: string[];
  restoreCustomModel: string;
  restoreCustomBackend: string;
  useCustomRestoreModel: boolean;
  sharpenEnabled: boolean;
  sharpenSens: number;
  
  // Scene Detection
  scenechangeEnabled: boolean;
  scenechangeMethod: string;
  scenechangeSens: number;
  scenechangeCustomModel: string;
  scenechangeCustomBackend: string;
  useCustomScenechangeModel: boolean;
  autoclip: boolean;
  
  // Deduplication
  dedupEnabled: boolean;
  dedupMethod: string;
  dedupSens: number;
  dedupCustomModel: string;
  useCustomDedupModel: boolean;
  
  // Segmentation
  segmentEnabled: boolean;
  segmentMethod: string;
  segmentCustomModel: string;
  segmentCustomBackend: string;
  useCustomSegmentModel: boolean;
  
  // Depth
  depthEnabled: boolean;
  depthMethod: string;
  depthQuality: string;
  depthCustomModel: string;
  depthCustomBackend: string;
  useCustomDepthModel: boolean;
  
  // Object Detection
  objDetectEnabled: boolean;
  objDetectMethod: string;
  objDetectDisableAnnotations: boolean;
  
  // Encoding
  encodeMethod: string;
  bitDepth: string;
  resizeEnabled: boolean;
  resizeFactor: number;
  
  // Performance
  half: boolean;
  decodeMethod: string;
  nvdecCompat: boolean;
  compileMode: string;
  static: boolean;
  tileRendering: boolean;
  tileSize: number;
  
  // Preview
  livePreview: boolean;
  previewStart: number;
  previewDuration: number;
  
  // Custom Models Storage
  customModels: Record<string, CustomModel>;
}

interface SettingsStore extends SettingsState {
  setSetting: <K extends keyof SettingsState>(key: K, value: SettingsState[K]) => void;
  setSettings: (settings: Partial<SettingsState>) => void;
  resetSettings: () => void;
  addRestoreChainItem: (item: string) => void;
  removeRestoreChainItem: (index: number) => void;
  moveRestoreChainItem: (from: number, to: number) => void;
  clearRestoreChain: () => void;
  setCustomModel: (type: string, path: string, backend: string) => void;
}

const defaultSettings: SettingsState = {
  // General
  inputPath: '',
  outputPath: '',
  youtubeUrl: '',
  audioCopy: true,
  benchmark: false,
  
  // Upscale
  upscaleEnabled: false,
  upscaleFactor: 2,
  upscaleMethod: 'shufflecugan',
  upscaleCustomModel: '',
  upscaleCustomBackend: 'default',
  useCustomUpscaleModel: false,
  
  // Image Upscale
  imageInputPath: '',
  imageOutputPath: '',
  imageOutputFormat: 'png',
  imageJpegQuality: 95,
  
  // Interpolate
  interpolateEnabled: false,
  interpolateFactor: 2,
  interpolateMethod: 'rife4.22',
  interpolateCustomModel: '',
  interpolateCustomBackend: 'default',
  useCustomInterpolateModel: false,
  ensemble: false,
  dynamicScale: false,
  slowmo: false,
  
  // Restore
  restoreEnabled: false,
  restoreMethod: 'scunet',
  restoreChain: [],
  restoreCustomModel: '',
  restoreCustomBackend: 'default',
  useCustomRestoreModel: false,
  sharpenEnabled: false,
  sharpenSens: 50,
  
  // Scene Detection
  scenechangeEnabled: false,
  scenechangeMethod: 'maxxvit-tensorrt',
  scenechangeSens: 50,
  scenechangeCustomModel: '',
  scenechangeCustomBackend: 'default',
  useCustomScenechangeModel: false,
  autoclip: false,
  
  // Deduplication
  dedupEnabled: false,
  dedupMethod: 'ssim',
  dedupSens: 35,
  dedupCustomModel: '',
  useCustomDedupModel: false,
  
  // Segmentation
  segmentEnabled: false,
  segmentMethod: 'anime',
  segmentCustomModel: '',
  segmentCustomBackend: 'default',
  useCustomSegmentModel: false,
  
  // Depth
  depthEnabled: false,
  depthMethod: 'small_v2',
  depthQuality: 'high',
  depthCustomModel: '',
  depthCustomBackend: 'default',
  useCustomDepthModel: false,
  
  // Object Detection
  objDetectEnabled: false,
  objDetectMethod: 'yolov9_small-directml',
  objDetectDisableAnnotations: false,
  
  // Encoding
  encodeMethod: 'x264',
  bitDepth: '8bit',
  resizeEnabled: false,
  resizeFactor: 1.0,
  
  // Performance
  half: true,
  decodeMethod: 'cpu',
  nvdecCompat: false,
  compileMode: 'default',
  static: false,
  tileRendering: false,
  tileSize: 128,
  
  // Preview
  livePreview: false,
  previewStart: 0,
  previewDuration: 5,
  
  // Custom Models
  customModels: {},
};

export const useSettingsStore = create<SettingsStore>()(
  persist(
    (set) => ({
      ...defaultSettings,
      
      setSetting: (key, value) => set((state) => ({ ...state, [key]: value })),
      
      setSettings: (settings) => set((state) => ({ ...state, ...settings })),
      
      resetSettings: () => set(defaultSettings),
      
      addRestoreChainItem: (item) => set((state) => ({
        ...state,
        restoreChain: [...state.restoreChain, item]
      })),
      
      removeRestoreChainItem: (index) => set((state) => ({
        ...state,
        restoreChain: state.restoreChain.filter((_, i) => i !== index)
      })),
      
      moveRestoreChainItem: (from, to) => set((state) => {
        const newChain = [...state.restoreChain];
        const [moved] = newChain.splice(from, 1);
        newChain.splice(to, 0, moved);
        return { ...state, restoreChain: newChain };
      }),
      
      clearRestoreChain: () => set((state) => ({ ...state, restoreChain: [] })),
      
      setCustomModel: (type, path, backend) => set((state) => ({
        ...state,
        customModels: {
          ...state.customModels,
          [type]: { path, backend }
        }
      })),
    }),
    {
      name: 'tas-settings',
      storage: createJSONStorage(() => localStorage),
      version: 1,
    }
  )
);
