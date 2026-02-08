const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');

// Keep a global reference of the window object
let mainWindow;
let pythonProcess = null;
let imageUpscaleProcess = null;

function getPreviewPath() {
  // Python writes preview.jpg under cs.MAINPATH, which is an AppData/XDG config dir
  // (see main.py: cs.MAINPATH = %APPDATA%/TheAnimeScripter on Windows).
  const platform = process.platform;

  if (platform === 'win32') {
    const appdata = process.env.APPDATA || process.env.LOCALAPPDATA;
    const base = appdata || path.join(os.homedir(), 'AppData', 'Roaming');
    return path.join(base, 'TheAnimeScripter', 'preview.jpg');
  }

  const xdgConfig = process.env.XDG_CONFIG_HOME || path.join(os.homedir(), '.config');
  return path.join(xdgConfig, 'TheAnimeScripter', 'preview.jpg');
}

// Python executable path detection
function getPythonPath() {
  const isDev = !app.isPackaged;
  
  if (isDev) {
    // Development mode - use parent folder python.exe
    const devPython = path.join(__dirname, '..', '..', 'python.exe');
    if (fs.existsSync(devPython)) {
      return devPython;
    }
    return 'python';
  } else {
    // Production mode - use bundled Python in gerekenler folder
    const pythonExe = path.join(process.resourcesPath, 'gerekenler', 'python.exe');
    
    if (fs.existsSync(pythonExe)) {
      return pythonExe;
    }
    
    // Fallback to system Python
    return 'python';
  }
}

// Get main.py path
function getMainScriptPath() {
  const isDev = !app.isPackaged;
  
  if (isDev) {
    return path.join(__dirname, '..', '..', 'main.py');
  } else {
    return path.join(process.resourcesPath, 'gerekenler', 'main.py');
  }
}

// Get image_upscale_cli.py path
function getImageUpscaleScriptPath() {
  const isDev = !app.isPackaged;
  
  if (isDev) {
    return path.join(__dirname, '..', '..', 'image_upscale_cli.py');
  } else {
    return path.join(process.resourcesPath, 'gerekenler', 'image_upscale_cli.py');
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 1000,
    minWidth: 1200,
    minHeight: 800,
    frame: false, // Remove default frame for custom title bar
    titleBarStyle: 'hidden',
    backgroundColor: '#0f0f0f',
    show: false, // Don't show until ready
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      webSecurity: false
    },
    icon: path.join(__dirname, '..', 'assets', 'icon.png')
  });

// Live preview image (preview.jpg written by the Python pipeline when --preview is enabled)
ipcMain.handle('get-preview-image', async () => {
  try {
    const previewPath = getPreviewPath();
    if (!fs.existsSync(previewPath)) {
      return null;
    }

    const buf = fs.readFileSync(previewPath);
    return `data:image/jpeg;base64,${buf.toString('base64')}`;
  } catch {
    return null;
  }
});

// Read video file and return as blob URL compatible format
ipcMain.handle('read-video-file', async (event, filePath) => {
  try {
    if (!filePath || !fs.existsSync(filePath)) {
      return { success: false, error: 'File not found' };
    }

    const ext = path.extname(filePath).toLowerCase();
    const mimeTypes = {
      '.mp4': 'video/mp4',
      '.webm': 'video/webm',
      '.mkv': 'video/x-matroska',
      '.avi': 'video/x-msvideo',
      '.mov': 'video/quicktime',
      '.m4v': 'video/mp4'
    };

    const mimeType = mimeTypes[ext] || 'video/mp4';
    const buffer = fs.readFileSync(filePath);
    const base64 = buffer.toString('base64');

    return {
      success: true,
      dataUrl: `data:${mimeType};base64,${base64}`
    };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

  // Load the app
  const isDev = !app.isPackaged;
  if (isDev) {
    mainWindow.loadURL('http://localhost:3000');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'dist-renderer', 'index.html'));
  }

  // Show window when ready to prevent visual flash
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    
    // Maximize on startup for better UX
    mainWindow.maximize();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
    // Kill Python process if running
    if (pythonProcess) {
      pythonProcess.kill();
      pythonProcess = null;
    }
  });
}

// IPC Handlers

// Window controls
ipcMain.handle('window-minimize', () => {
  if (mainWindow) mainWindow.minimize();
});

ipcMain.handle('window-maximize', () => {
  if (mainWindow) {
    if (mainWindow.isMaximized()) {
      mainWindow.unmaximize();
    } else {
      mainWindow.maximize();
    }
  }
});

ipcMain.handle('window-close', () => {
  if (mainWindow) mainWindow.close();
});

ipcMain.handle('window-is-maximized', () => {
  return mainWindow ? mainWindow.isMaximized() : false;
});

// File dialogs
ipcMain.handle('dialog-select-file', async (event, options) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: options.filters || [
      { name: 'Video Files', extensions: ['mp4', 'mkv', 'avi', 'mov', 'webm', 'm4v', 'gif'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });
  return result;
});

ipcMain.handle('dialog-select-output', async (event, options) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    filters: options.filters || [
      { name: 'MP4 Video', extensions: ['mp4'] },
      { name: 'MKV Video', extensions: ['mkv'] },
      { name: 'GIF', extensions: ['gif'] }
    ]
  });
  return result;
});

ipcMain.handle('dialog-select-folder', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory']
  });
  return result;
});

// Python process management
ipcMain.handle('start-processing', async (event, args) => {
  if (pythonProcess) {
    return { success: false, error: 'Processing already running' };
  }

  const pythonPath = getPythonPath();
  const mainScript = getMainScriptPath();
  
  // args is now a string array
  const cmdArgs = Array.isArray(args) ? args : [];

  console.log('Starting Python process:', pythonPath, mainScript, cmdArgs);

  try {
    pythonProcess = spawn(pythonPath, [mainScript, ...cmdArgs], {
      cwd: path.dirname(mainScript),
      env: {
        ...process.env,
        'PYTHONUNBUFFERED': '1',
        'TAS_GUI_PROGRESS': '1'
      }
    });

    let stdoutBuffer = '';
    let stderrBuffer = '';

    pythonProcess.stdout.on('data', (data) => {
      const text = data.toString();
      stdoutBuffer += text;
      
      // Process line by line
      const lines = stdoutBuffer.split(/\r\n|\n|\r/);
      stdoutBuffer = lines.pop(); // Keep incomplete line
      
      lines.forEach(line => {
        if (line.trim()) {
          mainWindow?.webContents.send('process-output', { type: 'stdout', data: line });
        }
      });
    });

    pythonProcess.stderr.on('data', (data) => {
      const text = data.toString();
      stderrBuffer += text;
      
      const lines = stderrBuffer.split(/\r\n|\n|\r/);
      stderrBuffer = lines.pop();
      
      lines.forEach(line => {
        if (line.trim()) {
          mainWindow?.webContents.send('process-output', { type: 'stderr', data: line });
        }
      });
    });

    pythonProcess.on('close', (code) => {
      mainWindow?.webContents.send('process-finished', { code });
      pythonProcess = null;
    });

    pythonProcess.on('error', (error) => {
      mainWindow?.webContents.send('process-error', { error: error.message });
      pythonProcess = null;
    });

    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('stop-processing', async () => {
  if (pythonProcess) {
    pythonProcess.kill('SIGTERM');
    
    // Force kill after 5 seconds if still running
    setTimeout(() => {
      if (pythonProcess) {
        pythonProcess.kill('SIGKILL');
        pythonProcess = null;
      }
    }, 5000);
    
    return { success: true };
  }
  return { success: false, error: 'No process running' };
});

// Preset management
function getPresetsDir() {
  const isDev = !app.isPackaged;
  if (isDev) {
    return path.join(__dirname, '..', '..', 'presets');
  }
  return path.join(process.resourcesPath, 'gerekenler', 'presets');
}

ipcMain.handle('presets-list', async () => {
  try {
    const presetsDir = getPresetsDir();
    if (!fs.existsSync(presetsDir)) {
      fs.mkdirSync(presetsDir, { recursive: true });
      return [];
    }
    const files = fs.readdirSync(presetsDir);
    return files.filter(f => f.endsWith('.json')).map(f => f.replace('.json', ''));
  } catch (error) {
    console.error('Failed to list presets:', error);
    return [];
  }
});

ipcMain.handle('presets-load', async (event, name) => {
  try {
    const presetsDir = getPresetsDir();
    const filePath = path.join(presetsDir, `${name}.json`);
    if (!fs.existsSync(filePath)) {
      return { success: false, error: 'Preset not found' };
    }
    const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    return { success: true, data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('presets-save', async (event, name, data) => {
  try {
    const presetsDir = getPresetsDir();
    if (!fs.existsSync(presetsDir)) {
      fs.mkdirSync(presetsDir, { recursive: true });
    }
    const safeName = name.replace(/[^a-zA-Z0-9_-]/g, '_');
    const filePath = path.join(presetsDir, `${safeName}.json`);
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('presets-delete', async (event, name) => {
  try {
    const presetsDir = getPresetsDir();
    const filePath = path.join(presetsDir, `${name}.json`);
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// Image Upscale process management
ipcMain.handle('start-image-upscale', async (event, args) => {
  if (imageUpscaleProcess) {
    return { success: false, error: 'Image upscale already running' };
  }

  const pythonPath = getPythonPath();
  const imageUpscaleScript = getImageUpscaleScriptPath();
  
  // args is now a string array
  const cmdArgs = Array.isArray(args) ? args : [];

  console.log('Starting Image Upscale process:', pythonPath, imageUpscaleScript, cmdArgs);

  try {
    imageUpscaleProcess = spawn(pythonPath, ['-u', imageUpscaleScript, ...cmdArgs], {
      cwd: path.dirname(imageUpscaleScript),
      env: {
        ...process.env,
        'PYTHONUNBUFFERED': '1',
        'TAS_GUI_PROGRESS': '1'
      }
    });

    let stdoutBuffer = '';
    let stderrBuffer = '';

    imageUpscaleProcess.stdout.on('data', (data) => {
      const text = data.toString();
      stdoutBuffer += text;
      
      // Process line by line
      const lines = stdoutBuffer.split(/\r\n|\n|\r/);
      stdoutBuffer = lines.pop(); // Keep incomplete line
      
      lines.forEach(line => {
        if (line.trim()) {
          mainWindow?.webContents.send('process-output', { type: 'stdout', data: line });
        }
      });
    });

    imageUpscaleProcess.stderr.on('data', (data) => {
      const text = data.toString();
      stderrBuffer += text;
      
      const lines = stderrBuffer.split(/\r\n|\n|\r/);
      stderrBuffer = lines.pop();
      
      lines.forEach(line => {
        if (line.trim()) {
          mainWindow?.webContents.send('process-output', { type: 'stderr', data: line });
        }
      });
    });

    imageUpscaleProcess.on('close', (code) => {
      mainWindow?.webContents.send('process-finished', { code });
      imageUpscaleProcess = null;
    });

    imageUpscaleProcess.on('error', (error) => {
      mainWindow?.webContents.send('process-error', { error: error.message });
      imageUpscaleProcess = null;
    });

    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('stop-image-upscale', async () => {
  if (imageUpscaleProcess) {
    imageUpscaleProcess.kill('SIGTERM');
    
    // Force kill after 5 seconds if still running
    setTimeout(() => {
      if (imageUpscaleProcess) {
        imageUpscaleProcess.kill('SIGKILL');
        imageUpscaleProcess = null;
      }
    }, 5000);
    
    return { success: true };
  }
  return { success: false, error: 'No image upscale process running' };
});

// External links
ipcMain.handle('open-external', async (event, url) => {
  await shell.openExternal(url);
});

// App info
ipcMain.handle('get-app-version', () => {
  return app.getVersion();
});

ipcMain.handle('get-platform', () => {
  return process.platform;
});

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// Security: Prevent new window creation
app.on('web-contents-created', (event, contents) => {
  contents.on('new-window', (event, navigationUrl) => {
    event.preventDefault();
    shell.openExternal(navigationUrl);
  });
});