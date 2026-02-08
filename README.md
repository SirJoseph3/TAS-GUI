# TAS-GUI - The Anime Scripter Modern Interface

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)
![License](https://img.shields.io/badge/license-AGPL--3.0-green)

Modern Electron + React interface for AI-powered video enhancement toolkit specialized for anime and general video content.

## ✨ Features

- **🚀 AI Video Upscaling** - RTMOSR, ShuffleCugan, RealESRGAN, and more
- **🎬 Video Interpolation** - RIFE-based frame interpolation
- **🔧 Multiple Decode Support** - NVDEC (GPU) and CPU decoding
- **⚡ TensorRT Optimization** - Maximum performance
- **🎨 Modern UI** - User-friendly dark theme interface

## 📦 Installation

### Ready-to-Use Installer (Recommended)
1. Download `TAS-NewGUI-Setup-1.0.0.exe` from [Releases](https://github.com/test4373/TAS-GUI/releases)
2. Run the installer
3. Dependencies will be installed automatically on first launch

### Developer Setup
```bash
cd new-gui
npm install
npm run dev
```

## 💻 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 (64-bit) | Windows 11 |
| RAM | 8 GB | 16 GB+ |
| GPU | - | NVIDIA RTX Series |
| Storage | 2 GB | 5 GB+ |

## 🎮 Usage

1. Launch the application
2. Drag and drop your video file or click to select
3. Choose your desired operations (Upscale, Interpolate, etc.)
4. Configure settings
5. Click "Start Processing"

## 🛠️ Supported AI Models

### Upscaling
- RTMOSR (2x, 4x)
- ShuffleCugan
- RealESRGAN
- Compact / UltraCompact
- And more...

### Interpolation
- RIFE (4.6, 4.15, 4.22, 4.25)
- GMFSS

## 🐛 Troubleshooting

### Video decode error
- Try using `--decode_method cpu`
- Update your NVIDIA drivers

### Dependency installation failed
- Check your internet connection
- Temporarily disable antivirus software

## � License & Attribution

This project is a fork/derivative of [TheAnimeScripter](https://github.com/NevermindNilas/TheAnimeScripter) 
by [NevermindNilas](https://github.com/NevermindNilas), licensed under [AGPL-3.0](LICENSE).

**Modifications made:**
- New Electron + React GUI interface
- Build system improvements
- CeLux dependency fixes

## 🙏 Credits

- [TheAnimeScripter](https://github.com/NevermindNilas/TheAnimeScripter) - Original project by NevermindNilas
- All AI model developers
