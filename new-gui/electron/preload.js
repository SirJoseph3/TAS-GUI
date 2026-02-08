const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Window controls
  window: {
    minimize: () => ipcRenderer.invoke('window-minimize'),
    maximize: () => ipcRenderer.invoke('window-maximize'),
    close: () => ipcRenderer.invoke('window-close'),
    isMaximized: () => ipcRenderer.invoke('window-is-maximized'),
  },

  // Dialogs
  dialog: {
    selectFile: (options) => ipcRenderer.invoke('dialog-select-file', options),
    selectOutput: (options) => ipcRenderer.invoke('dialog-select-output', options),
    selectFolder: () => ipcRenderer.invoke('dialog-select-folder'),
  },

  // Python processing
  processing: {
    start: (args) => ipcRenderer.invoke('start-processing', args),
    stop: () => ipcRenderer.invoke('stop-processing'),
    startImageUpscale: (args) => ipcRenderer.invoke('start-image-upscale', args),
    stopImageUpscale: () => ipcRenderer.invoke('stop-image-upscale'),
  },

  // Preset management
  presets: {
    list: () => ipcRenderer.invoke('presets-list'),
    load: (name) => ipcRenderer.invoke('presets-load', name),
    save: (name, data) => ipcRenderer.invoke('presets-save', name, data),
    delete: (name) => ipcRenderer.invoke('presets-delete', name),
  },

  // Live preview
  preview: {
    getImage: () => ipcRenderer.invoke('get-preview-image'),
  },

  // Video file reading
  video: {
    readFile: (filePath) => ipcRenderer.invoke('read-video-file', filePath),
  },

  // Event listeners
  onProcessOutput: (callback) => {
    ipcRenderer.on('process-output', (event, data) => callback(data));
  },
  onProcessFinished: (callback) => {
    ipcRenderer.on('process-finished', (event, data) => callback(data));
  },
  onProcessError: (callback) => {
    ipcRenderer.on('process-error', (event, data) => callback(data));
  },
  removeAllListeners: (channel) => {
    ipcRenderer.removeAllListeners(channel);
  },

  // Utilities
  openExternal: (url) => ipcRenderer.invoke('open-external', url),
  getAppVersion: () => ipcRenderer.invoke('get-app-version'),
  getPlatform: () => ipcRenderer.invoke('get-platform'),
});

// Notify when DOM is ready
window.addEventListener('DOMContentLoaded', () => {
  console.log('Preload script loaded successfully');
});