"""
The Anime Scripter - Desktop GUI Application

A beautiful desktop GUI interface for The Anime Scripter video enhancement toolkit.
Built with PyQt5 for native performance and easy EXE conversion.

Copyright (C) 2023-present Nilas Tiago
"""

import os
import sys
import json
import re
import subprocess
from pathlib import Path
import time

# --- Windows DLL search path fix ---
# Some environments (especially when launching via Explorer/shortcuts) can miss DLL search
# paths needed by PyTorch / CeLux. We proactively add the app folder and torch\lib.
_GUI_DLL_DIR_HANDLES = []


def _gui_add_dll_directory(path: str) -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    if not path or not os.path.isdir(path):
        return
    try:
        handle = os.add_dll_directory(path)
        _GUI_DLL_DIR_HANDLES.append(handle)
    except Exception:
        # Non-fatal; if this fails, the import error will surface later.
        pass


_GUI_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_gui_add_dll_directory(_GUI_BASE_DIR)

_GUI_TORCH_LIB = os.path.join(_GUI_BASE_DIR, "Lib", "site-packages", "torch", "lib")
_gui_add_dll_directory(_GUI_TORCH_LIB)

if os.name == "nt":
    # Help subprocesses too (some libs still rely on PATH search).
    os.environ["PATH"] = os.pathsep.join(
        [p for p in [_GUI_BASE_DIR, _GUI_TORCH_LIB, os.environ.get("PATH", "")] if p]
    )

# Windows: set a stable AppUserModelID so taskbar grouping/icon uses our app identity
# (especially important when launched via pythonw.exe).
if os.name == "nt":
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "TheAnimeScripter.GUI"
        )
    except Exception:
        # Not fatal; affects taskbar grouping/icon only.
        pass

_GUI_ICON_PATH = os.path.join(_GUI_BASE_DIR, "src", "assets", "icon.ico")

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSizePolicy, QPushButton, QLabel, QLineEdit, QCheckBox, QComboBox, QSlider,
    QFileDialog, QTabWidget, QGroupBox, QGridLayout, QTextEdit,
    QProgressBar, QMessageBox, QDoubleSpinBox, QSpinBox, QScrollArea,
    QFrame, QDialog, QListWidget, QProgressDialog, QInputDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl, QProcess
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QPixmap
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest


TRANSLATIONS = {
    'en': {
        'app_title': 'The Anime Scripter - AI Video Enhancement',
        'live_preview': 'Live Preview',
        'enable_live_preview': 'Enable Live Preview (slower)',
        'live_preview_disabled': 'Live Preview is disabled',
        'live_preview_ready': 'Live Preview enabled — start processing to see frames.',
        'live_preview_help': 'Shows an in-progress preview during processing (may reduce FPS).',
        'fullscreen': 'Fullscreen',
        'image_upscale': 'Image Upscale',
        'input_image': 'Input Image:',
        'output_image': 'Output Image:',
        'image_output_format': 'Output Format:',
        'image_jpeg_quality': 'JPG Quality:',
        'start_image_upscale': 'Upscale Image',
        'image_upscale_note': '💡 Model / Factor are taken from the Upscaling tab. Restoration (if enabled) is taken from the Restoration tab. FP16 / compile / tile settings are taken from the Performance tab.',
        'image_input_preview_placeholder': 'Input image preview will appear here...',
        'image_output_preview_placeholder': 'Upscaled image preview will appear here...',
        'input_output': 'Input / Output',
        'youtube_tab': 'YouTube',
        'input_tab': 'Input',
        'youtube_url': 'YouTube URL:',
        'enter_youtube_url': 'Enter YouTube video URL...',
        'input_video': 'Input Video:',
        'output_video': 'Output Video:',
        'browse': 'Browse',
        'select_input': 'Select input video file...',
        'auto_output': 'Leave empty for auto-generated name...',
        'preview_segment': 'Preview Segment',
        'preview_start': 'Start Time:',
        'preview_duration': 'Duration:',
        'create_preview': 'Create Preview',
        'loading_video_info': 'Loading video info...',
        'preview_not_ready': 'Select a local input video first.',
        'preview_requires_local': 'Preview currently supports local video files only.',
        'timeline_preview': 'Timeline Preview',
        'timeline_preview_placeholder': 'Selected frame will appear here...',
        'loading_frame': 'Loading frame...',
        'preview_open_failed': 'Could not open preview file:',
        'options': 'Options',
        'benchmark_mode': 'Benchmark Mode (No Output)',
        'obj_detect': 'Object Detection (YOLOv9)',
        'obj_detect_disable_annotations': 'Disable labels/confidence on boxes',
        'presets': 'Presets',
        'load_preset': 'Load Preset',
        'save_preset': 'Save Preset',
        'general': 'General',
        'upscaling': 'Upscaling',
        'interpolation': 'Interpolation',
        'restoration': 'Restoration',
        'scene_detection': 'Scene Detection',
        'deduplication': 'Deduplication',
        'segmentation': 'Segmentation',
        'depth': 'Depth',
        'encoding': 'Encoding',
        'performance': 'Performance',
        'enable_upscaling': 'Enable AI Upscaling',
        'upscale_factor': 'Upscale Factor:',
        'upscale_method': 'Upscaling Method:',
        'settings': 'Settings',
        'enable_interpolation': 'Enable Frame Interpolation',
        'interpolation_factor': 'Interpolation Factor:',
        'method': 'Method:',
        'ensemble': 'Ensemble (Better Quality)',
        'dynamic_scale': 'Dynamic Scale',
        'slowmo': 'Slow Motion Mode',
        'enable_restoration': 'Enable Video Restoration',
        'restoration_method': 'Restoration Method:',
        'enable_sharpening': 'Enable Sharpening',
        'sharpening_sensitivity': 'Sharpening Sensitivity:',
        'enable_scene_detection': 'Enable Scene Change Detection',
        'scene_method': 'Detection Method:',
        'scene_sensitivity': 'Scene Change Sensitivity:',
        'autoclip': 'Auto Clip Scenes',
        'enable_deduplication': 'Enable Frame Deduplication',
        'dedup_method': 'Deduplication Method:',
        'dedup_sensitivity': 'Deduplication Sensitivity:',
        'enable_segmentation': 'Enable Video Segmentation',
        'segment_method': 'Segmentation Method:',
        'enable_depth': 'Enable Depth Map Generation',
        'depth_method': 'Depth Method:',
        'depth_quality': 'Depth Quality:',
        'encode_method': 'Encoding Method:',
        'bit_depth': 'Bit Depth:',
        'enable_resize': 'Enable Output Resize',
        'resize_factor': 'Resize Factor:',
        'half_precision': 'Half Precision (FP16)',
        'decode_method': 'Decode Method:',
        'decode_note': "⚠ NVDEC requires NVIDIA drivers + CUDA-enabled CeLux (bundled as celux_cuda). If you see decoder/DLL errors, use 'cpu'.",
        'nvdec_compat': 'NVDEC Compatibility Mode (slower)',
        'nvdec_compat_help': "Use this if NVDEC crashes or you see CUDA/PTX errors. It forces the safer CPU-frame path and is slower.",
        'compile_mode': 'Compile Mode:',
        'static_chunk': 'Static Chunk Size',
        'tile_rendering': 'Tile Rendering',
        'enable_tile_size': 'Enable Tile Rendering',
        'tile_size': 'Tile Size:',
        'tile_help': "Enable Tile Rendering if you run out of VRAM. Smaller tiles use less VRAM but may introduce seams/grid lines; try 256/384 if you see artifacts.",
        'oom_guidance': "Looks like an OOM (out of memory). Enable Tile Rendering in the Performance tab and try a smaller Tile Size, or choose a lighter model.",
        'start_processing': 'Start Processing',
        'stop': 'Stop',
        'processing_log': 'Processing Log',
        'frames': 'Frames:',
        'fps': 'FPS:',
        'elapsed': 'Elapsed:',
        'eta': 'ETA:',
        'processing': 'Processing...',
        'complete': 'Complete!',
        'failed': 'Failed',
        'stopping': 'Stopping...',
        'loading_preview': 'Loading preview...',
        'success': 'Success',
        'error': 'Error',
        'input_required': 'Input video is required',
        'processing_complete': 'Processing completed successfully!',
        'processing_failed': 'Processing failed with code',
        'starting_processing': 'Starting processing...',
        'command': 'Command:',
        'recommended': 'Recommended:',
        'best_quality': 'Best Quality:',
        'fastest': 'Fastest:',
        'balanced': 'balanced',
        'select_quality': 'Select Video Quality',
        'quality_prompt': 'Choose the quality for downloading:',
        'select': 'Select',
        'cancel': 'Cancel',
        'fetching_qualities': 'Fetching Available Qualities',
        'please_wait': 'Please wait, processing will continue shortly...',
        'app_may_freeze': 'The app may freeze for a few seconds, don\'t worry',
        'youtube': 'YouTube',
        'idle': 'Ready',
        'browse_custom_model': 'Browse Custom Model',
        'use_custom_model': 'Use Custom Model',
        'custom_model': 'Custom Model',
        'select_backend': 'Select Backend',
        'backend_prompt': 'Choose which backend to use with this custom model:',
        'backend_info': '💡 Select the backend, then choose the corresponding model file (.onnx for TensorRT/Default, -directml.onnx for DirectML, -ncnn.param for NCNN)',
        'backend_default': 'Default/PyTorch (CPU or CUDA if available)',
        'backend_cuda': 'CUDA (Requires NVIDIA GPU)',
        'backend_tensorrt': 'TensorRT (Requires NVIDIA RTX GPU - Fastest)',
        'backend_directml': 'DirectML (For AMD/Intel GPUs)',
        'backend_ncnn': 'NCNN (CPU optimized, cross-platform)',
        'warning_no_nvidia': '⚠️ Warning: No NVIDIA GPU detected. CUDA may not work.',
        'warning_no_rtx': '⚠️ Warning: TensorRT requires RTX series GPU (20xx, 30xx, 40xx).',
        'gpu_detected': '✓ GPU detected:',
        'no_gpu_detected': '⚠️ No compatible GPU detected',
        
        'info_general': '📖 This is where you select your input video and choose where to save the output.\n💡 You can use a YouTube link OR select a local video file.\n💡 If you leave the output empty, the program will create a name automatically.\n⚠️ Benchmark mode processes the video without saving - useful for testing speed.',
        'info_upscaling': '📖 Upscaling makes your video resolution higher (e.g., 720p → 1440p).\n💡 2x means double the resolution, 4x means quadruple.\n🎯 TensorRT models are fastest but require NVIDIA GPU.\n⚡ Start with shufflecugan-tensorrt - it\'s fast and good quality.\n📊 More advanced models like span give better quality but are slower.',
        'info_interpolation': '📖 Interpolation adds frames between existing ones to make motion smoother.\n💡 2x factor turns 30fps into 60fps, 3x turns it into 90fps.\n🎯 Higher factor = smoother but also larger file size.\n⚡ RIFE models are popular - rife4.22 is reliable, rife4.25-heavy gives best quality.\n🔧 Ensemble mode uses multiple models for better quality but is much slower.\n🎬 Slow Motion mode slows down the video while keeping it smooth.',
        'info_restoration': '📖 Restoration removes noise, compression artifacts, and improves overall quality.\n💡 Use this if your video looks blocky, blurry, or has compression issues.\n🎯 SCUNet is good for general cleanup, NAFNet for heavy noise.\n✨ Sharpening makes edges more defined - don\'t overdo it or it looks unnatural.\n⚠️ Too much sharpening (>70) can create weird halos around objects.',
        'info_scene': '📖 Scene detection finds where scenes change in your video.\n💡 Useful to prevent interpolation from blending different scenes together.\n🎯 MaxxVIT is most accurate, Differential is fastest.\n🎬 Auto-Clip splits your video into separate files at scene changes.\n⚠️ Lower sensitivity = only obvious scene changes detected.',
        'info_dedup': '📖 Deduplication removes duplicate frames (useful for anime with still shots).\n💡 Many anime have the same frame repeated - this finds and removes them.\n🎯 SSIM-CUDA is accurate and fast with GPU.\n📊 Lower threshold = more aggressive removal (might remove too much).\n⚠️ Use carefully - too aggressive can make motion look stuttery.',
        'info_segmentation': '📖 Segmentation separates characters from background.\n💡 Useful if you want to apply different effects to characters vs background.\n🎯 Anime model works best for anime/cartoon content.\n⚠️ This is an advanced feature - most users don\'t need it.',
        'info_depth': '📖 Depth map generation creates a grayscale map showing distance.\n💡 Brighter = closer to camera, darker = farther away.\n🎯 Use large_v2 for best quality, small_v2 for speed.\n⚠️ This is for 3D effects or advanced editing - most users don\'t need it.',
        'info_encoding': '📖 Encoding determines how your final video is saved.\n💡 x264 is widely compatible, x265 gives smaller files but slower.\n🎯 10bit keeps more color information than 8bit (less banding).\n⚡ NVENC uses your NVIDIA GPU for fast encoding.\n📏 Resize lets you make the output smaller (e.g., 0.5 = half size).',
        'info_performance': '📖 Performance settings control speed vs quality tradeoffs.\n💡 FP16 (Half Precision) is faster and uses less VRAM - recommended for modern GPUs.\n🎯 CPU decode is safer, NVDEC is faster but might have issues.\n⚡ Compile mode can speed things up but is experimental.\n⚠️ Static mode is for TensorRT - only enable if you know what you\'re doing.',
    },
    'tr': {
        'app_title': 'The Anime Scripter - Yapay Zeka Video İyileştirme',
        'live_preview': 'Canlı Önizleme',
        'enable_live_preview': 'Canlı Önizlemeyi Aç (yavaş)',
        'live_preview_disabled': 'Canlı Önizleme kapalı',
        'live_preview_ready': 'Canlı Önizleme açık — başlatınca görüntü gelir.',
        'live_preview_help': 'İşlem sırasında canlı önizleme gösterir (FPS düşebilir).',
        'fullscreen': 'Tam Ekran',
        'image_upscale': 'Resim Upscale',
        'input_image': 'Girdi Resmi:',
        'output_image': 'Çıktı Resmi:',
        'image_output_format': 'Çıktı Formatı:',
        'image_jpeg_quality': 'JPG Kalite:',
        'start_image_upscale': 'Resmi Upscale Et',
        'image_upscale_note': '💡 Model / Factor Upscaling sekmesinden alınır. Temizleme (açıksa) Temizleme sekmesinden alınır. FP16 / compile / tile ayarları Performance sekmesinden alınır.',
        'image_input_preview_placeholder': 'Girdi resmi burada görünecek...',
        'image_output_preview_placeholder': 'Upscale edilmiş resim burada görünecek...',
        'input_output': 'Girdi / Çıktı',
        'youtube_tab': 'YouTube',
        'input_tab': 'Girdi',
        'youtube_url': 'YouTube Linki:',
        'enter_youtube_url': 'YouTube video linkini girin...',
        'input_video': 'Girdi Videosu:',
        'output_video': 'Çıktı Videosu:',
        'browse': 'Gözat',
        'select_input': 'Girdi video dosyası seçin...',
        'auto_output': 'Otomatik isim için boş bırakın...',
        'preview_segment': 'Önizleme',
        'preview_start': 'Başlangıç:',
        'preview_duration': 'Süre:',
        'create_preview': 'Önizleme Oluştur',
        'loading_video_info': 'Video bilgisi okunuyor...',
        'preview_not_ready': 'Önce yerel bir video seçin.',
        'preview_requires_local': 'Önizleme şimdilik sadece yerel video dosyalarında çalışıyor.',
        'timeline_preview': 'Zaman Çizgisi Önizlemesi',
        'timeline_preview_placeholder': 'Seçili kare burada görünecek...',
        'loading_frame': 'Kare yükleniyor...',
        'preview_open_failed': 'Önizleme dosyası açılamadı:',
        'options': 'Seçenekler',
        'benchmark_mode': 'Kıyaslama Modu (Çıktı Yok)',
        'obj_detect': 'Nesne Tespiti (YOLOv9)',
        'obj_detect_disable_annotations': 'Kutu üstü etiket/güven değerini gizle',
        'presets': 'Ön Ayarlar',
        'load_preset': 'Ön Ayar Yükle',
        'save_preset': 'Ön Ayar Kaydet',
        'general': 'Genel',
        'upscaling': 'Çözünürlük artırma',
        'interpolation': 'FPS artırma',
        'restoration': 'Temizleme',
        'scene_detection': 'Sahne bulma',
        'deduplication': 'Aynı kareleri sil',
        'segmentation': 'Karakter ayırma',
        'depth': 'Derinlik haritası',
        'encoding': 'Kaydetme',
        'performance': 'Performans',
        'enable_upscaling': 'Çözünürlük artırmayı aç',
        'upscale_factor': 'Çözünürlük çarpanı:',
        'upscale_method': 'Model:',
        'settings': 'Ayarlar',
        'enable_interpolation': 'FPS artırmayı aç',
        'interpolation_factor': 'FPS çarpanı:',
        'method': 'Yöntem:',
        'ensemble': 'Topluluk (Daha İyi Kalite)',
        'dynamic_scale': 'Dinamik Ölçek',
        'slowmo': 'Ağır Çekim Modu',
        'enable_restoration': 'Temizlemeyi aç',
        'restoration_method': 'Temizleme yöntemi:',
        'enable_sharpening': 'Keskinleştirmeyi Etkinleştir',
        'sharpening_sensitivity': 'Keskinleştirme Hassasiyeti:',
        'enable_scene_detection': 'Sahne değişimini bulmayı aç',
        'scene_method': 'Yöntem:',
        'scene_sensitivity': 'Hassasiyet:',
        'autoclip': 'Sahneleri Otomatik Kes',
        'enable_deduplication': 'Aynı kareleri silmeyi aç',
        'dedup_method': 'Silme yöntemi:',
        'dedup_sensitivity': 'Hassasiyet:',
        'enable_segmentation': 'Karakter ayırmayı aç',
        'segment_method': 'Ayırma yöntemi:',
        'enable_depth': 'Derinlik haritasını aç',
        'depth_method': 'Yöntem:',
        'depth_quality': 'Kalite:',
        'encode_method': 'Kayıt yöntemi:',
        'bit_depth': 'Renk (bit):',
        'enable_resize': 'Çıktı Yeniden Boyutlandırmayı Etkinleştir',
        'resize_factor': 'Yeniden Boyutlandırma Faktörü:',
        'half_precision': 'Yarım Hassasiyet (FP16)',
        'decode_method': 'Kod Çözme Yöntemi:',
        'decode_note': "⚠ NVDEC için NVIDIA driver + CUDA destekli CeLux gerekir (bundle: celux_cuda). DLL/decoder hatası alırsanız 'cpu' seçin.",
        'nvdec_compat': 'NVDEC Uyumluluk Modu (Yavaş)',
        'nvdec_compat_help': 'NVDEC crash / CUDA/PTX hatası görürsen bunu kullan. Daha güvenli CPU-frame yolunu zorlar ve daha yavaştır.',
        'compile_mode': 'Derleme Modu:',
        'static_chunk': 'Sabit Parça Boyutu',
        'tile_rendering': 'Parçalı İşleme (Tile)',
        'enable_tile_size': 'Parçalı İşlemeyi Etkinleştir',
        'tile_size': 'Parça Boyutu:',
        'tile_help': "VRAM yetmiyorsa Parçalı İşleme'yi açın. Küçük parça daha az VRAM kullanır ama dikiş/ızgara çizgileri yapabilir; çizgi görüyorsanız 256/384 deneyin.",
        'oom_guidance': "OOM hatası gibi görünüyor. Performans sekmesinden Parçalı İşleme'yi açıp daha küçük bir Parça Boyutu deneyin veya daha hafif bir model seçin.",
        'start_processing': 'İşlemeyi Başlat',
        'stop': 'Durdur',
        'processing_log': 'İşleme Günlüğü',
        'frames': 'Kareler:',
        'fps': 'FPS:',
        'elapsed': 'Geçen:',
        'eta': 'Kalan:',
        'processing': 'İşleniyor...',
        'complete': 'Tamamlandı!',
        'failed': 'Başarısız',
        'stopping': 'Durduruluyor...',
        'loading_preview': 'Önizleme yükleniyor...',
        'success': 'Başarılı',
        'error': 'Hata',
        'input_required': 'Girdi videosu gereklidir',
        'processing_complete': 'İşleme başarıyla tamamlandı!',
        'processing_failed': 'İşleme başarısız oldu, kod:',
        'starting_processing': 'İşleme başlatılıyor...',
        'command': 'Komut:',
        'recommended': 'Önerilen:',
        'best_quality': 'En İyi Kalite:',
        'fastest': 'En Hızlı:',
        'balanced': 'dengeli',
        'select_quality': 'Video Kalitesi Seç',
        'quality_prompt': 'İndirme için kalite seçin:',
        'select': 'Seç',
        'cancel': 'İptal',
        'fetching_qualities': 'Mevcut Kaliteler Alınıyor',
        'please_wait': 'Lütfen bekleyin, birazdan işleme devam edebileceksiniz...',
        'app_may_freeze': 'Uygulama birkaç saniye donuk olacaktır, endişelenmeyin',
        'youtube': 'YouTube',
        'idle': 'Hazır',
        'browse_custom_model': 'Özel Model Seç',
        'use_custom_model': 'Özel Model Kullan',
        'custom_model': 'Özel Model',
        'select_backend': 'Backend Seç',
        'backend_prompt': 'Bu özel model ile hangi backend\'i kullanmak istersiniz:',
        'backend_info': '💡 Backend\'i seçin, ardından ilgili model dosyasını seçin (.onnx TensorRT/Varsayılan için, -directml.onnx DirectML için, -ncnn.param NCNN için)',
        'backend_default': 'Varsayılan/PyTorch (CPU veya mevcut ise CUDA)',
        'backend_cuda': 'CUDA (NVIDIA GPU Gerektirir)',
        'backend_tensorrt': 'TensorRT (NVIDIA RTX GPU Gerektirir - En Hızlı)',
        'backend_directml': 'DirectML (AMD/Intel GPU\'lar için)',
        'backend_ncnn': 'NCNN (CPU optimize edilmiş, çapraz platform)',
        'warning_no_nvidia': '⚠️ Uyarı: NVIDIA GPU bulunamadı. CUDA çalışmayabilir.',
        'warning_no_rtx': '⚠️ Uyarı: TensorRT, RTX serisi GPU gerektirir (20xx, 30xx, 40xx).',
        'gpu_detected': '✓ GPU tespit edildi:',
        'no_gpu_detected': '⚠️ Uyumlu GPU bulunamadı',
        
        'info_general': '📖 Buradan giriş videonu seçer ve çıktı nereye kaydedilecek ayarlarsın.\n💡 YouTube linki de kullanabilirsin, dosya da seçebilirsin.\n💡 Çıktı yolunu boş bırakırsan otomatik isim verir.\n⚠️ Kıyaslama modu dosya kaydetmeden sadece hız testi yapar.',
        'info_upscaling': '📖 Çözünürlük artırma videoyu daha net yapar (örn: 720p → 1080p).\n💡 2x iki kat, 4x dört kat çözünürlük demek.\n🎯 Başlangıç için 2x bir model genelde yeterlidir.\n⚠️ Daha büyük modeller daha çok VRAM ister.',
        'info_interpolation': '📖 FPS artırma (kare ekleme) hareketi daha akıcı yapar.\n💡 2x: 30 → 60 FPS, 3x: 30 → 90 FPS.\n🎯 FPS arttıkça işlem süresi ve dosya boyutu artar.\n🎬 Ağır çekim modu videoyu yavaşlatır ama akıcı kalır.',
        'info_restoration': '📖 Temizleme; gürültü, bloklanma ve sıkıştırma izlerini azaltmaya çalışır.\n💡 Video bulanıksa veya eskiyse işe yarar.\n⚠️ Çok yüksek ayarlar görüntüyü yapay gösterebilir.',
        'info_scene': '📖 Sahne bulma videodaki sahne değişimlerini yakalar.\n💡 FPS artırma yaparken sahneler birbirine karışmasın diye önerilir.\n⚠️ Hassasiyet düşükse sadece belirgin sahneleri yakalar.',
        'info_dedup': '📖 Aynı kareleri silme tekrarlanan kareleri kaldırır.\n💡 Anime videolarında sık işe yarar.\n⚠️ Fazla agresif olursa hareket kesik kesik görünebilir.',
        'info_segmentation': '📖 Karakter ayırma; karakteri arka plandan ayırır.\n💡 İleri seviye bir özellik, çoğu kullanıcı için şart değil.',
        'info_depth': '📖 Derinlik haritası, görüntüden yaklaşık uzaklık çıkarır.\n💡 Açık renk yakını, koyu renk uzağı gösterir.\n⚠️ 3D efekt gibi ileri işler için.',
        'info_encoding': '📖 Kaydetme ayarları çıktı videonun boyutunu ve uyumluluğunu belirler.\n💡 x264 en uyumlu; x265 daha küçük dosya verir ama daha yavaştır.\n⚡ NVENC GPU ile daha hızlı kaydetmeye yardım eder (uyumlu kart varsa).\n📏 Yeniden boyutlandırma çıktı çözünürlüğünü küçültür (örn: 0.5 = yarı boyut).',
        'info_performance': '📖 Performans ayarları hız ve VRAM kullanımını etkiler.\n💡 FP16 genelde daha hızlı ve daha az VRAM kullanır.\n💡 VRAM yetmiyorsa Parçalı İşleme (Tile) açıp Parça Boyutunu küçült.\n⚠️ Derleme modu deneyseldir; sorun görürsen kapat.',
    }
}

class LanguageManager:
    current_language = 'tr'
    
    @classmethod
    def set_language(cls, lang):
        cls.current_language = lang
    
    @classmethod
    def get_language(cls):
        return cls.current_language

def t(key):
    """Get translated text for the current language"""
    return TRANSLATIONS.get(LanguageManager.get_language(), TRANSLATIONS['en']).get(key, key)


def sanitize_filename(filename):
    """Remove or replace problematic characters from filename."""
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    parent = Path(filename).parent
    
    # Replace problematic characters with underscore
    stem = re.sub(r'[#<>:"|?*]', '_', stem)
    
    # Limit length to prevent Windows MAX_PATH issues
    if len(stem) > 100:
        stem = stem[:100]
    
    return str(parent / f"{stem}{suffix}")


class QualitySelectionDialog(QDialog):
    def __init__(self, video_title, quality_options, parent=None):
        super().__init__(parent)
        self.selected_index = 0
        self.quality_options = quality_options
        
        self.setWindowTitle(t('select_quality'))
        self.setModal(True)
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        if video_title:
            title_label = QLabel(f"📹 {video_title}")
            title_label.setStyleSheet("font-weight: bold; font-size: 11pt; color: #89b4fa; padding: 10px;")
            title_label.setWordWrap(True)
            layout.addWidget(title_label)
        
        info_label = QLabel(t('quality_prompt'))
        info_label.setStyleSheet("color: #cdd6f4; padding: 5px;")
        layout.addWidget(info_label)
        
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 2px solid #313244;
                border-radius: 5px;
                padding: 5px;
                font-size: 10pt;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 3px;
            }
            QListWidget::item:selected {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QListWidget::item:hover {
                background-color: #313244;
            }
        """)
        
        for option in quality_options:
            self.list_widget.addItem(option)
        
        self.list_widget.setCurrentRow(0)
        layout.addWidget(self.list_widget)
        
        button_layout = QHBoxLayout()
        
        select_btn = QPushButton(t('select'))
        select_btn.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #74c7ec;
            }
        """)
        select_btn.clicked.connect(self.accept)
        button_layout.addWidget(select_btn)
        
        cancel_btn = QPushButton(t('cancel'))
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 2px solid #45475a;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45475a;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.setStyleSheet("background-color: #181825;")
    
    def get_selected_index(self):
        return self.list_widget.currentRow()


class FullscreenPreviewDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t('fullscreen'))
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setStyleSheet("background-color: #000000;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #000000; color: #cdd6f4;")
        layout.addWidget(self.image_label)

        self._pixmap = None

    def set_pixmap(self, pixmap):
        self._pixmap = pixmap
        self._refresh_pixmap()

    def _refresh_pixmap(self):
        if not self._pixmap:
            self.image_label.setText(t('loading_preview'))
            self.image_label.setPixmap(QPixmap())
            return

        if self.image_label.width() <= 1 or self.image_label.height() <= 1:
            return

        self.image_label.setText("")
        scaled = self._pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        self.close()


def detect_gpu_capabilities():
    """Detect GPU capabilities for backend recommendations"""
    gpu_info = {
        'has_nvidia': False,
        'has_rtx': False,
        'gpu_name': '',
        'recommended_backend': 'default'
    }
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=3
        )
        
        if result.returncode == 0 and result.stdout.strip():
            gpu_info['has_nvidia'] = True
            gpu_info['gpu_name'] = result.stdout.strip()
            
            gpu_name_lower = gpu_info['gpu_name'].lower()
            if any(rtx in gpu_name_lower for rtx in ['rtx', 'a100', 'a40', 'a30', 'a10', 't4']):
                gpu_info['has_rtx'] = True
                gpu_info['recommended_backend'] = 'tensorrt'
            else:
                gpu_info['recommended_backend'] = 'cuda'
        else:
            gpu_info['recommended_backend'] = 'directml'
    except (FileNotFoundError, subprocess.TimeoutExpired):
        gpu_info['recommended_backend'] = 'directml'
    
    return gpu_info



class ProcessThread(QThread):
    progress_update = pyqtSignal(int, str, dict)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, command, quality_index=None):
        super().__init__()
        self.command = command
        self.process = None
        self.last_progress = 0
        self.start_time = time.time()
        self.quality_index = quality_index
        self.error_message = None
        self.seen_oom = False
        
    def run(self):
        try:
            self.start_time = time.time()
            
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            # Signal the child process to emit GUI-friendly progress updates (per-frame, \r overwrite style).
            env['TAS_GUI_PROGRESS'] = '1'
            
            if self.quality_index is not None:
                env['YTDLP_QUALITY_INDEX'] = str(self.quality_index)
            
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                env=env,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )

            buffer = b""
            while True:
                chunk = self.process.stdout.read(4096)
                if not chunk:
                    break

                buffer += chunk

                while True:
                    r_idx = buffer.find(b"\r")
                    n_idx = buffer.find(b"\n")

                    if r_idx == -1 and n_idx == -1:
                        break

                    if r_idx != -1 and (n_idx == -1 or r_idx < n_idx):
                        consume = 1
                        overwrite = True
                        if len(buffer) > r_idx + 1 and buffer[r_idx + 1] == 0x0A:
                            consume = 2
                            overwrite = False

                        raw = buffer[:r_idx]
                        buffer = buffer[r_idx + consume:]
                        self._emit_output_update(raw, overwrite=overwrite)
                    else:
                        raw = buffer[:n_idx]
                        buffer = buffer[n_idx + 1:]
                        self._emit_output_update(raw, overwrite=False)

            if buffer:
                self._emit_output_update(buffer, overwrite=True)
            
            self.process.wait()
            
            # Prefer a dedicated OOM guidance message if we detected OOM in logs.
            # Note: some failures may still exit with code 0; treat detected errors as failure.
            if self.seen_oom:
                self.error_message = t('oom_guidance')

            if self.error_message:
                self.finished.emit(False, self.error_message)
            elif self.process.returncode == 0:
                self.finished.emit(True, t('processing_complete'))
            else:
                self.finished.emit(False, f"{t('processing_failed')} {self.process.returncode}")
                
        except Exception as e:
            self.finished.emit(False, f"{t('error')}: {str(e)}")

    def _emit_output_update(self, raw_bytes: bytes, overwrite: bool):
        if not raw_bytes:
            return

        try:
            line = raw_bytes.decode("utf-8", errors="replace")
        except Exception:
            line = str(raw_bytes)

        line = line.strip()
        if not line:
            return

        clean_line = self.clean_ansi(line)

        # Detect OOM early so we can show a helpful dialog even if the final error is generic.
        lowered = clean_line.lower()
        if (
            'out of memory' in lowered
            or 'oom' in lowered
            or 'cuda oom' in lowered
            or 'outofmemoryerror' in lowered
        ):
            self.seen_oom = True

        if "Bu model video için" in clean_line or "çok güçlü" in clean_line:
            self.error_message = clean_line
        elif "TENSORRT ENGINE BUILD FAILED" in clean_line:
            self.error_message = "TensorRT engine build başarısız oldu. Model ekran kartınız için çok büyük olabilir. Farklı bir model deneyebilirsiniz."
        elif "Error processing video" in clean_line:
            self.error_message = clean_line

        progress_data = self.parse_progress_line(clean_line)
        if overwrite:
            progress_data['overwrite'] = True

        progress = progress_data.get('progress_percent', self.last_progress)

        if progress > self.last_progress:
            self.last_progress = progress

        self.progress_update.emit(progress, clean_line, progress_data)
    
    def clean_ansi(self, text):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)
    
    def parse_progress_line(self, line):
        data = {}
        clean_line = self.clean_ansi(line)
        
        try:
            if '[PROGRESS]' in clean_line:
                parts = clean_line.split('|')
                for part in parts:
                    part = part.strip()
                    
                    if '%' in part:
                        pct_match = re.search(r'(\d{1,3})%', part)
                        if pct_match:
                            data['progress_percent'] = int(pct_match.group(1))
                    
                    if 'Frames:' in part:
                        frames_match = re.search(r'(\d+)/(\d+)', part)
                        if frames_match:
                            data['current_frame'] = int(frames_match.group(1))
                            data['total_frames'] = int(frames_match.group(2))
                    
                    if 'FPS:' in part:
                        fps_match = re.search(r'(\d+\.?\d*)', part)
                        if fps_match:
                            data['fps'] = float(fps_match.group(1))
                    
                    if 'Elapsed:' in part:
                        elapsed_match = re.search(r'(\d{1,2}:\d{2}:\d{2})', part)
                        if elapsed_match:
                            data['elapsed'] = elapsed_match.group(1)
                    
                    if 'ETA:' in part:
                        eta_match = re.search(r'(\d{1,2}:\d{2}:\d{2})', part)
                        if eta_match:
                            data['eta'] = eta_match.group(1)
            else:
                lowered = clean_line.lower()
                is_download_line = 'download' in lowered or 'downloading' in lowered
                is_processing_line = 'processing' in lowered or 'frames' in lowered or 'fps' in lowered or 'eta' in lowered

                percent_match = re.search(r'(\d{1,3})%', clean_line)
                if percent_match:
                    pct = int(percent_match.group(1))
                    if 0 <= pct <= 100:
                        # Don't treat model download percent as the main processing percent.
                        if is_download_line and not is_processing_line:
                            data['download_percent'] = pct
                        elif is_processing_line:
                            data['progress_percent'] = pct

                # If we're in a download phase but the line doesn't contain a percent (e.g. unknown total),
                # mark it so the GUI can show an indeterminate/busy progress bar.
                if is_download_line and not is_processing_line and 'download_percent' not in data:
                    if clean_line.startswith('[DOWNLOAD]') or 'downloading' in lowered:
                        data['download_indeterminate'] = True
                
                fps_match = re.search(r'FPS:\s*(\d+\.?\d*)', clean_line, re.IGNORECASE)
                if fps_match:
                    data['fps'] = float(fps_match.group(1))
                
                eta_match = re.search(r'ETA:\s*([\d:]+)', clean_line, re.IGNORECASE)
                if eta_match:
                    data['eta'] = eta_match.group(1)
                
                frames_match = re.search(r'Frames?:\s*(\d+)/(\d+)', clean_line, re.IGNORECASE)
                if frames_match:
                    current = int(frames_match.group(1))
                    total = int(frames_match.group(2))
                    data['current_frame'] = current
                    data['total_frames'] = total
                    if 'progress_percent' not in data and total > 0:
                        data['progress_percent'] = min(100, int((current / total) * 100))
                
                elapsed_match = re.search(r'Elapsed.*?(\d{1,2}:\d{2}:\d{2})', clean_line, re.IGNORECASE)
                if not elapsed_match:
                    elapsed_match = re.search(r'(\d{1,2}:\d{2}:\d{2})', clean_line)
                if elapsed_match:
                    data['elapsed'] = elapsed_match.group(1)
                
        except Exception as e:
            pass
        
        # Only synthesize elapsed once actual processing progress exists.
        if (
            not data.get('elapsed')
            and ('current_frame' in data or 'progress_percent' in data)
        ):
            try:
                elapsed = int(time.time() - self.start_time)
                hours = elapsed // 3600
                minutes = (elapsed % 3600) // 60
                seconds = elapsed % 60
                data['elapsed'] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            except Exception:
                pass
        
        return data
    
    def send_input(self, choice):
        if self.process:
            if self.process.stdin:
                try:
                    print(f"Attempting to send choice: {choice}")
                    
                    choice_str = str(choice)
                    payload = (choice_str + "\r\n").encode("utf-8", errors="replace")
                    self.process.stdin.write(payload)
                    self.process.stdin.flush()
                    
                    print(f"Sent choice '{choice_str}' with CR+LF")
                    
                    import os
                    os.fsync(self.process.stdin.fileno())
                    print("fsync completed")
                    
                except Exception as e:
                    print(f"Error sending input: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("ERROR: process.stdin is None!")
        else:
            print("ERROR: process is None!")
    
    def stop(self):
        if self.process:
            self.process.terminate()


class VideoInfoThread(QThread):
    """Fetch lightweight video info (duration/fps/size) using ffprobe.

    We intentionally avoid importing celux/torch inside the GUI process because on some
    Windows setups torch DLL initialization can fail inside the PyQt app (WinError 1114).
    """

    result = pyqtSignal(bool, dict, str)

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path

    def _ensure_ffprobe(self) -> str:
        import os
        from platform import system as platform_system

        import src.constants as cs

        # Mirror main.py init so the ffmpeg/ffprobe downloader knows where to cache binaries.
        if not getattr(cs, 'SYSTEM', ''):
            cs.SYSTEM = platform_system()

        if not getattr(cs, 'MAINPATH', ''):
            if cs.SYSTEM == "Windows":
                appdata = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA")
                if not appdata:
                    appdata = os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
                cs.MAINPATH = os.path.join(appdata, "TheAnimeScripter")
            else:
                xdg_config = os.getenv("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
                cs.MAINPATH = os.path.join(xdg_config, "TheAnimeScripter")
            os.makedirs(cs.MAINPATH, exist_ok=True)

        if not getattr(cs, 'WHEREAMIRUNFROM', ''):
            cs.WHEREAMIRUNFROM = os.path.dirname(os.path.abspath(__file__))

        if not getattr(cs, 'FFMPEGPATH', ''):
            cs.FFMPEGPATH = os.path.join(
                cs.MAINPATH,
                "ffmpeg",
                "ffmpeg.exe" if cs.SYSTEM == "Windows" else "ffmpeg",
            )

        if not getattr(cs, 'FFPROBEPATH', ''):
            cs.FFPROBEPATH = os.path.join(
                cs.MAINPATH,
                "ffmpeg",
                "ffprobe.exe" if cs.SYSTEM == "Windows" else "ffprobe",
            )

        ffprobe_dir = os.path.dirname(cs.FFPROBEPATH)
        if ffprobe_dir and ffprobe_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + ffprobe_dir

        if not os.path.exists(cs.FFMPEGPATH) or not os.path.exists(cs.FFPROBEPATH):
            from src.utils.getFFMPEG import getFFMPEG

            getFFMPEG()

        return cs.FFPROBEPATH

    def run(self):
        try:
            import json
            import subprocess

            ffprobe = self._ensure_ffprobe()
            cmd = [
                ffprobe,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                self.video_path,
            ]

            proc = subprocess.run(
                cmd,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )

            if not proc.stdout:
                raise Exception("No output received from ffprobe")

            probe = json.loads(proc.stdout)
            video_stream = next(
                stream
                for stream in probe.get("streams", [])
                if stream.get("codec_type") == "video"
            )

            fps_str = str(video_stream.get("r_frame_rate") or "0/1")
            if "/" in fps_str:
                num_s, den_s = fps_str.split("/", 1)
            else:
                num_s, den_s = fps_str, "1"

            try:
                den = float(den_s)
                fps = float(num_s) / den if den else 0.0
            except Exception:
                fps = 0.0

            duration = 0.0
            try:
                duration = float((probe.get("format", {}) or {}).get("duration") or 0.0)
            except Exception:
                duration = 0.0

            info = {
                'duration': duration,
                'fps': fps,
                'width': int(video_stream.get('width') or 0),
                'height': int(video_stream.get('height') or 0),
            }

            self.result.emit(True, info, "")
        except Exception as e:
            self.result.emit(False, {}, str(e))


class SourceFrameThread(QThread):
    """Extract a single frame from the input video using ffmpeg.

    This powers the timeline scrub preview (separate from Live Preview).
    """

    # ok, video_path, timestamp_s, jpg_bytes, error
    result = pyqtSignal(bool, str, int, object, str)

    def __init__(self, video_path: str, timestamp_s: int, target_width: int = 500):
        super().__init__()
        self.video_path = video_path
        self.timestamp_s = int(timestamp_s)
        self.target_width = int(target_width)

    def _ensure_ffmpeg(self) -> str:
        import os
        from platform import system as platform_system

        import src.constants as cs

        if not getattr(cs, 'SYSTEM', ''):
            cs.SYSTEM = platform_system()

        if not getattr(cs, 'MAINPATH', ''):
            if cs.SYSTEM == "Windows":
                appdata = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA")
                if not appdata:
                    appdata = os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
                cs.MAINPATH = os.path.join(appdata, "TheAnimeScripter")
            else:
                xdg_config = os.getenv("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
                cs.MAINPATH = os.path.join(xdg_config, "TheAnimeScripter")
            os.makedirs(cs.MAINPATH, exist_ok=True)

        if not getattr(cs, 'WHEREAMIRUNFROM', ''):
            cs.WHEREAMIRUNFROM = os.path.dirname(os.path.abspath(__file__))

        if not getattr(cs, 'FFMPEGPATH', ''):
            cs.FFMPEGPATH = os.path.join(
                cs.MAINPATH,
                "ffmpeg",
                "ffmpeg.exe" if cs.SYSTEM == "Windows" else "ffmpeg",
            )

        if not getattr(cs, 'FFPROBEPATH', ''):
            cs.FFPROBEPATH = os.path.join(
                cs.MAINPATH,
                "ffmpeg",
                "ffprobe.exe" if cs.SYSTEM == "Windows" else "ffprobe",
            )

        ffmpeg_dir = os.path.dirname(cs.FFMPEGPATH)
        if ffmpeg_dir and ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + ffmpeg_dir

        if not os.path.exists(cs.FFMPEGPATH) or not os.path.exists(cs.FFPROBEPATH):
            from src.utils.getFFMPEG import getFFMPEG

            getFFMPEG()

        return cs.FFMPEGPATH

    def run(self):
        try:
            import os
            import subprocess

            ffmpeg = self._ensure_ffmpeg()
            ts = max(0, int(self.timestamp_s))
            width = max(64, int(self.target_width))
            scale = f"scale={width}:-2"

            cmd = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(ts),
                "-i",
                self.video_path,
                "-frames:v",
                "1",
                "-vf",
                scale,
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "-q:v",
                "3",
                "pipe:1",
            ]

            creationflags = 0
            if os.name == 'nt' and hasattr(subprocess, 'CREATE_NO_WINDOW'):
                creationflags = subprocess.CREATE_NO_WINDOW

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                creationflags=creationflags,
            )

            if proc.returncode != 0 or not proc.stdout:
                try:
                    err = proc.stderr.decode('utf-8', errors='replace').strip()
                except Exception:
                    err = str(proc.stderr)
                err = (err or f"ffmpeg failed with code {proc.returncode}")
                raise Exception(err[:400])

            self.result.emit(True, self.video_path, ts, proc.stdout, "")
        except Exception as e:
            self.result.emit(False, self.video_path, int(self.timestamp_s), b"", str(e))


class TheAnimeScripterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(t('app_title'))
        self.setGeometry(100, 100, 1400, 900)

        # Ensure the app has a consistent icon (window + taskbar).
        self.setWindowIcon(QIcon(_GUI_ICON_PATH))
        
        self.custom_models_base_dir = Path(__file__).parent / "custom_models"
        self.custom_models_base_dir.mkdir(exist_ok=True)
        
        self.custom_models_dirs = {}
        for model_type in ['upscale', 'interpolate', 'restore', 'scenechange', 'dedup', 'segment', 'depth']:
            type_dir = self.custom_models_base_dir / model_type
            type_dir.mkdir(exist_ok=True)
            self.custom_models_dirs[model_type] = type_dir
        
        self.custom_models = {
            'upscale': {'path': '', 'backend': 'default'},
            'interpolate': {'path': '', 'backend': 'default'},
            'restore': {'path': '', 'backend': 'default'},
            'scenechange': {'path': '', 'backend': 'default'},
            'dedup': {'path': '', 'backend': 'default'},
            'segment': {'path': '', 'backend': 'default'},
            'depth': {'path': '', 'backend': 'default'}
        }
        
        # Output path auto-naming: stay in sync until user edits the path manually.
        self.output_auto = False
        
        self.setup_style()
        self.init_ui()
        
        from src.utils.logAndPrint import set_error_dialog_callback
        set_error_dialog_callback(self.show_backend_error_dialog)
        
        self.process_thread = None
        self.image_upscale_process = None
        self.stop_requested = False
        
        self.network_manager = QNetworkAccessManager()
        self.network_manager.finished.connect(self.on_preview_loaded)
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.refresh_preview)
        self.preview_url = "http://127.0.0.1:5000/image"
        # Live preview is written by the CLI/ffmpeg as a local file (see src/utils/ffmpegSettings.py).
        # Poll it directly instead of relying on an HTTP preview server.
        self.preview_file_path = os.path.join(_GUI_BASE_DIR, "preview.jpg")

        # Segment preview state
        self._video_info_thread = None
        self._input_video_info = None
        self._input_video_duration_s = None
        self._is_segment_preview_run = False
        self._segment_preview_output_path = None

        # Timeline (scrub) preview state - separate from Live Preview
        self._source_frame_thread = None
        self._pending_source_frame_key = None
        self._last_source_frame_pixmap = None
        self._source_frame_fullscreen_dialog = None

        self._source_frame_debounce_timer = QTimer()
        self._source_frame_debounce_timer.setSingleShot(True)
        self._source_frame_debounce_timer.timeout.connect(self._request_source_frame_preview)

        self._last_preview_pixmap = None
        self._fullscreen_preview_dialog = None
        
    def setup_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 10pt;
            }
            QTabWidget::pane {
                border: 1px solid #45475a;
                background-color: #181825;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #313244;
                color: #cdd6f4;
                padding: 10px 20px;
                margin: 2px;
                border-radius: 5px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QTabBar::tab:hover {
                background-color: #45475a;
            }
            QGroupBox {
                border: 2px solid #45475a;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #89b4fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #313244;
                border: 2px solid #45475a;
                border-radius: 5px;
                padding: 8px;
                color: #cdd6f4;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 2px solid #89b4fa;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #74c7ec;
            }
            QPushButton:pressed {
                background-color: #7287fd;
            }
            QPushButton:disabled {
                background-color: #45475a;
                color: #6c7086;
            }
            QPushButton#startBtn {
                background-color: #a6e3a1;
                color: #1e1e2e;
            }
            QPushButton#startBtn:hover {
                background-color: #94e2d5;
            }
            QPushButton#startBtn[active="true"], QPushButton#startBtn[active="true"]:disabled {
                background-color: #3f8f3a;
                color: #1e1e2e;
            }
            QPushButton#stopBtn {
                background-color: #f38ba8;
                color: #1e1e2e;
            }
            QPushButton#stopBtn:hover {
                background-color: #eba0ac;
            }
            QPushButton#stopBtn[active="true"], QPushButton#stopBtn[active="true"]:disabled {
                background-color: #b23a57;
                color: #1e1e2e;
            }
            QPushButton#browseBtn {
                background-color: #f9e2af;
                color: #1e1e2e;
            }
            QCheckBox {
                spacing: 8px;
                color: #cdd6f4;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 4px;
                border: 2px solid #45475a;
                background-color: #313244;
            }
            QCheckBox::indicator:checked {
                background-color: #89b4fa;
                border: 2px solid #89b4fa;
            }
            QProgressBar {
                border: 2px solid #45475a;
                border-radius: 5px;
                text-align: center;
                background-color: #313244;
                color: #cdd6f4;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #89b4fa, stop:1 #cba6f7);
                border-radius: 3px;
            }
            QTextEdit {
                background-color: #181825;
                border: 2px solid #45475a;
                border-radius: 5px;
                padding: 5px;
                color: #cdd6f4;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            QSlider::groove:horizontal {
                border: 1px solid #45475a;
                height: 8px;
                background: #313244;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #89b4fa;
                border: 1px solid #74c7ec;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #74c7ec;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QLabel {
                color: #cdd6f4;
            }
        """)
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        title_bar = QWidget()
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(10)
        
        title_label = QLabel(f"🎬 {t('app_title')}")
        title_font = QFont("Segoe UI", 24, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #89b4fa; padding: 15px; background-color: #1e1e2e;")
        title_layout.addWidget(title_label, 1)
        
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("🌐 TR", "tr")
        self.lang_combo.addItem("🌐 ENG", "en")
        self.lang_combo.setCurrentIndex(0 if LanguageManager.get_language() == 'tr' else 1)
        self.lang_combo.setFixedSize(100, 40)
        self.lang_combo.setStyleSheet("""
            QComboBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 2px solid #45475a;
                border-radius: 5px;
                font-weight: bold;
                font-size: 11pt;
                padding: 5px;
            }
            QComboBox:hover {
                background-color: #45475a;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #313244;
                color: #cdd6f4;
                selection-background-color: #45475a;
            }
        """)
        self.lang_combo.currentIndexChanged.connect(self.update_language)
        title_layout.addWidget(self.lang_combo, alignment=Qt.AlignRight)
        
        title_bar.setStyleSheet("background-color: #1e1e2e;")
        main_layout.addWidget(title_bar)
        
        content_layout = QHBoxLayout()
        content_layout.setSpacing(0)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setSpacing(0)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.toggle_btn = QPushButton("✕")
        self.toggle_btn.setObjectName("toggleBtn")
        self.toggle_btn.setFixedSize(40, 40)
        self.toggle_btn.setStyleSheet("""
            QPushButton#toggleBtn {
                background-color: #313244;
                color: #cdd6f4;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 16pt;
                font-weight: bold;
            }
            QPushButton#toggleBtn:hover {
                background-color: #45475a;
            }
        """)
        self.toggle_btn.clicked.connect(self.toggle_sidebar)
        left_layout.addWidget(self.toggle_btn, alignment=Qt.AlignTop)
        
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(250)
        self.sidebar.setStyleSheet("background-color: #181825; border-right: 2px solid #45475a;")
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setSpacing(5)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        
        self.sidebar_buttons = []
        menu_items = [
            ("📁", t("general")),
            ("🎯", t("obj_detect")),
            ("⬆", t("upscaling")),
            ("🖼", t("image_upscale")),
            ("⏱", t("interpolation")),
            ("🔧", t("restoration")),
            ("🎬", t("scene_detection")),
            ("🔄", t("deduplication")),
            ("✂", t("segmentation")),
            ("📊", t("depth")),
            ("🎞", t("encoding")),
            ("⚡", t("performance"))
        ]
        
        for icon, name in menu_items:
            btn = QPushButton(f"{icon} {name}")
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #313244;
                    color: #cdd6f4;
                    border: none;
                    border-radius: 5px;
                    padding: 12px;
                    text-align: left;
                    font-size: 11pt;
                }
                QPushButton:hover {
                    background-color: #45475a;
                }
                QPushButton:checked {
                    background-color: #89b4fa;
                    color: #1e1e2e;
                    font-weight: bold;
                }
            """)
            btn.clicked.connect(lambda checked, n=name: self.change_content(n))
            sidebar_layout.addWidget(btn)
            self.sidebar_buttons.append(btn)
        
        sidebar_layout.addStretch()
        left_layout.addWidget(self.sidebar)
        
        content_layout.addWidget(left_container)
        
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(15, 15, 15, 15)
        
        preview_and_settings = QHBoxLayout()
        preview_and_settings.setSpacing(10)

        preview_container = QWidget()
        preview_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        preview_container_layout = QVBoxLayout(preview_container)
        preview_container_layout.setContentsMargins(0, 0, 0, 0)
        preview_container_layout.setSpacing(0)

        self.live_preview_group = QGroupBox(f"🎥 {t('live_preview')}")
        self.live_preview_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.setSpacing(6)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(500, 280)
        self.preview_label.setMaximumSize(500, 280)
        self.preview_label.setStyleSheet("background-color: #181825; border: 2px solid #45475a; border-radius: 5px;")
        self.preview_label.setText(t('live_preview_disabled'))
        self.preview_label.setScaledContents(False)
        preview_layout.addWidget(self.preview_label)

        self.live_preview_check = QCheckBox(t('enable_live_preview'))
        self.live_preview_check.setChecked(False)
        self.live_preview_check.setToolTip(t('live_preview_help'))
        self.live_preview_check.stateChanged.connect(self._on_live_preview_checkbox_changed)
        preview_layout.addWidget(self.live_preview_check)

        self.preview_fullscreen_btn = QPushButton(f"⛶ {t('fullscreen')}")
        self.preview_fullscreen_btn.setToolTip("ESC")
        self.preview_fullscreen_btn.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 2px solid #45475a;
                border-radius: 5px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #45475a;
            }
        """)
        self.preview_fullscreen_btn.clicked.connect(self.open_preview_fullscreen)

        # Segment preview controls (Topaz-like): choose a start timestamp and preview duration
        self.preview_segment_label = QLabel(t('preview_not_ready'))
        self.preview_segment_label.setStyleSheet("color: #a6adc8; font-size: 9pt;")
        self.preview_segment_label.setWordWrap(True)

        preview_segment_row = QHBoxLayout()
        preview_segment_row.addWidget(self.preview_segment_label, 1)
        preview_segment_row.addWidget(self.preview_fullscreen_btn)
        preview_layout.addLayout(preview_segment_row)

        self.preview_start_slider = QSlider(Qt.Horizontal)
        self.preview_start_slider.setRange(0, 0)
        self.preview_start_slider.setEnabled(False)
        self.preview_start_slider.valueChanged.connect(self.on_preview_start_changed)
        preview_layout.addWidget(self.preview_start_slider)

        # Timeline (scrub) preview - shows the *source* frame at the selected timestamp
        self.timeline_preview_title = QLabel(f"🎞 {t('timeline_preview')}")
        self.timeline_preview_title.setStyleSheet("font-weight: bold; color: #a6adc8;")
        preview_layout.addWidget(self.timeline_preview_title)

        self.timeline_preview_label = QLabel()
        self.timeline_preview_label.setAlignment(Qt.AlignCenter)
        self.timeline_preview_label.setMinimumSize(500, 160)
        self.timeline_preview_label.setMaximumSize(500, 160)
        self.timeline_preview_label.setStyleSheet("background-color: #181825; border: 2px solid #45475a; border-radius: 5px;")
        self.timeline_preview_label.setText(t('timeline_preview_placeholder'))
        self.timeline_preview_label.setScaledContents(False)
        preview_layout.addWidget(self.timeline_preview_label)

        self.timeline_fullscreen_btn = QPushButton(f"⛶ {t('fullscreen')}")
        self.timeline_fullscreen_btn.setToolTip("ESC")
        self.timeline_fullscreen_btn.setEnabled(False)
        self.timeline_fullscreen_btn.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 2px solid #45475a;
                border-radius: 5px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #45475a;
            }
            QPushButton:disabled {
                background-color: #181825;
                color: #6c7086;
            }
        """)
        self.timeline_fullscreen_btn.clicked.connect(self.open_timeline_fullscreen)

        timeline_fullscreen_row = QHBoxLayout()
        timeline_fullscreen_row.addStretch()
        timeline_fullscreen_row.addWidget(self.timeline_fullscreen_btn)
        preview_layout.addLayout(timeline_fullscreen_row)

        preview_controls_row = QHBoxLayout()
        preview_controls_row.addWidget(QLabel(t('preview_duration')))

        self.preview_duration_combo = QComboBox()
        self.preview_duration_combo.addItem("1s", 1)
        self.preview_duration_combo.addItem("2s", 2)
        self.preview_duration_combo.addItem("3s", 3)
        self.preview_duration_combo.addItem("5s", 5)
        self.preview_duration_combo.addItem("10s", 10)
        self.preview_duration_combo.setCurrentIndex(3)
        self.preview_duration_combo.setEnabled(False)
        self.preview_duration_combo.currentIndexChanged.connect(self.on_preview_duration_changed)
        preview_controls_row.addWidget(self.preview_duration_combo, 1)

        self.preview_create_btn = QPushButton(f"⚡ {t('create_preview')}")
        self.preview_create_btn.setEnabled(False)
        self.preview_create_btn.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 5px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #74c7ec;
            }
            QPushButton:disabled {
                background-color: #313244;
                color: #6c7086;
            }
        """)
        self.preview_create_btn.clicked.connect(self.start_preview_segment)
        preview_controls_row.addWidget(self.preview_create_btn)

        preview_layout.addLayout(preview_controls_row)
        self.live_preview_group.setLayout(preview_layout)

        self.image_preview_group = QGroupBox(f"🖼 {t('image_upscale')}")
        self.image_preview_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        image_preview_layout = QVBoxLayout()
        image_preview_layout.setContentsMargins(12, 12, 12, 12)
        image_preview_layout.setSpacing(6)

        image_in_title = QLabel(t('input_image'))
        image_in_title.setStyleSheet("font-weight: bold; color: #89b4fa;")
        image_preview_layout.addWidget(image_in_title)

        self.image_input_preview_label = QLabel()
        self.image_input_preview_label.setAlignment(Qt.AlignCenter)
        self.image_input_preview_label.setMinimumSize(500, 130)
        self.image_input_preview_label.setMaximumSize(500, 130)
        self.image_input_preview_label.setStyleSheet("background-color: #181825; border: 2px solid #45475a; border-radius: 5px;")
        self.image_input_preview_label.setText(t('image_input_preview_placeholder'))
        self.image_input_preview_label.setScaledContents(False)
        image_preview_layout.addWidget(self.image_input_preview_label)

        self.image_input_fullscreen_btn = QPushButton(f"⛶ {t('fullscreen')}")
        self.image_input_fullscreen_btn.setToolTip("ESC")
        self.image_input_fullscreen_btn.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 2px solid #45475a;
                border-radius: 5px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #45475a;
            }
        """)
        self.image_input_fullscreen_btn.clicked.connect(self.open_image_input_fullscreen)

        image_in_fullscreen_row = QHBoxLayout()
        image_in_fullscreen_row.addStretch()
        image_in_fullscreen_row.addWidget(self.image_input_fullscreen_btn)
        image_preview_layout.addLayout(image_in_fullscreen_row)

        image_out_title = QLabel(t('output_image'))
        image_out_title.setStyleSheet("font-weight: bold; color: #cba6f7;")
        image_preview_layout.addWidget(image_out_title)

        self.image_output_preview_label = QLabel()
        self.image_output_preview_label.setAlignment(Qt.AlignCenter)
        self.image_output_preview_label.setMinimumSize(500, 130)
        self.image_output_preview_label.setMaximumSize(500, 130)
        self.image_output_preview_label.setStyleSheet("background-color: #181825; border: 2px solid #45475a; border-radius: 5px;")
        self.image_output_preview_label.setText(t('image_output_preview_placeholder'))
        self.image_output_preview_label.setScaledContents(False)
        image_preview_layout.addWidget(self.image_output_preview_label)

        self.image_output_fullscreen_btn = QPushButton(f"⛶ {t('fullscreen')}")
        self.image_output_fullscreen_btn.setToolTip("ESC")
        self.image_output_fullscreen_btn.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 2px solid #45475a;
                border-radius: 5px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #45475a;
            }
        """)
        self.image_output_fullscreen_btn.clicked.connect(self.open_image_output_fullscreen)

        image_out_fullscreen_row = QHBoxLayout()
        image_out_fullscreen_row.addStretch()
        image_out_fullscreen_row.addWidget(self.image_output_fullscreen_btn)
        image_preview_layout.addLayout(image_out_fullscreen_row)

        self.image_preview_group.setLayout(image_preview_layout)
        self.image_preview_group.hide()

        preview_container_layout.addWidget(self.live_preview_group)
        preview_container_layout.addWidget(self.image_preview_group)
        preview_and_settings.addWidget(preview_container, 0, Qt.AlignTop)
        
        self.content_stack = QWidget()
        self.stack_layout = QVBoxLayout(self.content_stack)
        self.stack_layout.setContentsMargins(0, 0, 0, 0)
        
        self.content_widgets = {}
        self.create_general_tab()
        self.create_object_detection_tab()
        self.create_upscaling_tab()
        self.create_image_upscale_tab()
        self.create_interpolation_tab()
        self.create_restoration_tab()
        self.create_scene_tab()
        self.create_dedup_tab()
        self.create_segmentation_tab()
        self.create_depth_tab()
        self.create_encoding_tab()
        self.create_performance_tab()
        
        for widget in self.content_widgets.values():
            widget.hide()
        
        if t("general") in self.content_widgets:
            self.content_widgets[t("general")].show()
            self.sidebar_buttons[0].setChecked(True)
            self.active_content_name = t("general")
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.content_stack)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        preview_and_settings.addWidget(scroll, 1)
        
        right_layout.addLayout(preview_and_settings)
        content_layout.addWidget(right_container, 1)
        
        main_layout.addLayout(content_layout, 1)
        
        bottom_frame = QFrame()
        bottom_layout = QGridLayout(bottom_frame)
        bottom_layout.setSpacing(10)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        self.start_btn = QPushButton(f"▶ {t('start_processing')}")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self.on_start_clicked)
        self.start_btn.setMinimumHeight(45)
        bottom_layout.addWidget(self.start_btn, 0, 0, alignment=Qt.AlignVCenter)

        self.stop_btn = QPushButton(f"⏹ {t('stop')}")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(45)
        bottom_layout.addWidget(self.stop_btn, 0, 1, alignment=Qt.AlignVCenter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(30)
        self.progress_bar.setValue(0)
        bottom_layout.addWidget(self.progress_bar, 0, 2, alignment=Qt.AlignVCenter)

        self.status_label = QLabel(t('idle'))
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #a6e3a1;")
        self.status_label.setMinimumWidth(90)
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        bottom_layout.addWidget(self.status_label, 0, 3, alignment=Qt.AlignVCenter)

        self.progress_details_label = QLabel("")
        self.progress_details_label.setStyleSheet("color: #a6adc8; font-size: 9pt; padding: 2px;")
        self.progress_details_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        bottom_layout.addWidget(self.progress_details_label, 1, 2, 1, 2)

        bottom_layout.setColumnStretch(2, 1)

        main_layout.addWidget(bottom_frame)
        
        log_group = QGroupBox(t('processing_log'))
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        self.loading_overlay = QLabel(central_widget)
        self.loading_overlay.setAlignment(Qt.AlignCenter)
        self.loading_overlay.setStyleSheet("""
            background-color: rgba(24, 24, 37, 230);
            color: #cdd6f4;
            font-size: 16pt;
            font-weight: bold;
            border-radius: 10px;
        """)
        
        overlay_text = f"""
            <div style='text-align: center; padding: 40px;'>
                <p style='font-size: 18pt; color: #89b4fa; margin-bottom: 20px;'>
                    ⏳ {t('fetching_qualities')}
                </p>
                <p style='font-size: 14pt; color: #a6adc8; margin-bottom: 10px;'>
                    {t('please_wait')}
                </p>
                <p style='font-size: 11pt; color: #f9e2af; font-style: italic;'>
                    ⚠️ {t('app_may_freeze')}
                </p>
            </div>
        """
        self.loading_overlay.setText(overlay_text)
        self.loading_overlay.hide()
        self.loading_overlay.setGeometry(central_widget.rect())
        
        # Output auto-name wiring (only updates when auto generated).
        self.output_path.textEdited.connect(self._on_output_path_user_edited)
        self._connect_output_auto_signals()
        self.update_output_auto_name()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.setGeometry(self.centralWidget().rect())
    
    def toggle_sidebar(self):
        if self.sidebar.isVisible():
            self.sidebar.hide()
            self.toggle_btn.setText("☰")
        else:
            self.sidebar.show()
            self.toggle_btn.setText("✕")
    
    def change_content(self, name):
        for btn in self.sidebar_buttons:
            btn.setChecked(False)
        
        for btn in self.sidebar_buttons:
            if name in btn.text():
                btn.setChecked(True)
                break
        
        for widget in self.content_widgets.values():
            widget.hide()
        
        if name in self.content_widgets:
            self.content_widgets[name].show()

        # Image Upscale tab: show image previews instead of the live preview panel.
        if hasattr(self, 'live_preview_group') and hasattr(self, 'image_preview_group'):
            is_image_upscale = name == t('image_upscale')
            self.live_preview_group.setVisible(not is_image_upscale)
            self.image_preview_group.setVisible(is_image_upscale)

        self.active_content_name = name
        if hasattr(self, 'start_btn'):
            if name == t('image_upscale'):
                self.start_btn.setText(f"▶ 🖼 {t('start_image_upscale')}")
            else:
                self.start_btn.setText(f"▶ {t('start_processing')}")
    
    def create_general_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        io_group = QGroupBox(f"📁 {t('input_output')}")
        io_main_layout = QVBoxLayout()
        
        self.input_tabs = QTabWidget()
        
        youtube_tab = QWidget()
        youtube_layout = QGridLayout(youtube_tab)
        youtube_layout.addWidget(QLabel(t('youtube_url')), 0, 0)
        self.youtube_url = QLineEdit()
        self.youtube_url.setPlaceholderText(t('enter_youtube_url'))
        self.youtube_url.textChanged.connect(self.on_input_video_changed)
        youtube_layout.addWidget(self.youtube_url, 0, 1)
        
        input_tab_widget = QWidget()
        input_tab_layout = QGridLayout(input_tab_widget)
        input_tab_layout.addWidget(QLabel(t('input_video')), 0, 0)
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText(t('select_input'))
        self.input_path.editingFinished.connect(self.on_input_video_changed)
        input_tab_layout.addWidget(self.input_path, 0, 1)
        
        input_browse = QPushButton(t('browse'))
        input_browse.setObjectName("browseBtn")
        input_browse.clicked.connect(self.browse_input)
        input_tab_layout.addWidget(input_browse, 0, 2)
        
        self.input_tabs.addTab(youtube_tab, t('youtube_tab'))
        self.input_tabs.addTab(input_tab_widget, t('input_tab'))
        self.input_tabs.currentChanged.connect(self.on_input_video_changed)

        io_main_layout.addWidget(self.input_tabs)
        
        output_layout = QGridLayout()
        output_layout.addWidget(QLabel(t('output_video')), 0, 0)
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText(t('auto_output'))
        output_layout.addWidget(self.output_path, 0, 1)
        
        output_browse = QPushButton(t('browse'))
        output_browse.setObjectName("browseBtn")
        output_browse.clicked.connect(self.browse_output)
        output_layout.addWidget(output_browse, 0, 2)
        
        io_main_layout.addLayout(output_layout)
        
        io_group.setLayout(io_main_layout)
        layout.addWidget(io_group)
        
        options_group = QGroupBox(f"⚙ {t('options')}")
        options_layout = QVBoxLayout()
        
        self.benchmark_check = QCheckBox(t('benchmark_mode'))
        options_layout.addWidget(self.benchmark_check)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        preset_group = QGroupBox(f"💾 {t('presets')}")
        preset_layout = QHBoxLayout()
        
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(250)
        self.preset_combo.currentIndexChanged.connect(self.on_preset_selected)
        preset_layout.addWidget(self.preset_combo)
        
        save_preset_btn = QPushButton(t('save_preset'))
        save_preset_btn.clicked.connect(self.save_preset)
        preset_layout.addWidget(save_preset_btn)
        
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        self.refresh_presets_dropdown()
        
        info_label = QLabel(t('info_general'))
        info_label.setStyleSheet("color: #a6adc8; font-size: 9pt; padding: 15px; background-color: #181825; border: 2px solid #313244; border-radius: 5px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        self.stack_layout.addWidget(tab)
        self.content_widgets[t("general")] = tab

    def create_object_detection_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.obj_detect_check = QCheckBox(f"🎯 {t('obj_detect')}")
        self.obj_detect_check.setStyleSheet(
            "font-size: 12pt; font-weight: bold; color: #89b4fa;"
        )
        layout.addWidget(self.obj_detect_check)

        settings_group = QGroupBox(t('settings'))
        settings_layout = QVBoxLayout()

        obj_detect_method_row = QHBoxLayout()
        obj_detect_method_row.addWidget(QLabel(t('method')))

        self.obj_detect_method = QComboBox()
        self.obj_detect_method.addItems(
            [
                "yolov9_small-directml",
                "yolov9_medium-directml",
                "yolov9_large-directml",
                "yolov9_small-openvino",
                "yolov9_medium-openvino",
                "yolov9_large-openvino",
            ]
        )
        self.obj_detect_method.setCurrentText("yolov9_small-directml")
        obj_detect_method_row.addWidget(self.obj_detect_method)
        obj_detect_method_row.addStretch()

        self.obj_detect_method_widget = QWidget()
        self.obj_detect_method_widget.setLayout(obj_detect_method_row)
        settings_layout.addWidget(self.obj_detect_method_widget)

        self.obj_detect_disable_annotations_check = QCheckBox(
            t('obj_detect_disable_annotations')
        )
        settings_layout.addWidget(self.obj_detect_disable_annotations_check)

        # Disable controls unless obj-detect is enabled.
        self.obj_detect_method_widget.setEnabled(False)
        self.obj_detect_disable_annotations_check.setEnabled(False)
        self.obj_detect_check.toggled.connect(self.obj_detect_method_widget.setEnabled)
        self.obj_detect_check.toggled.connect(
            self.obj_detect_disable_annotations_check.setEnabled
        )

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        info_label = QLabel(
            "📖 "
            + t('obj_detect')
            + "\n\n"
            + "💡 When enabled, TAS will run object detection on the video and annotate detected objects."
        )
        if LanguageManager.get_language() == 'tr':
            info_label.setText(
                "📖 "
                + t('obj_detect')
                + "\n\n"
                + "💡 Açıksa, video üzerinde nesne tespiti çalışır ve algılanan nesneleri kutularla işaretler."
            )

        info_label.setStyleSheet(
            "color: #a6adc8; font-size: 9pt; padding: 15px; background-color: #181825; border: 2px solid #313244; border-radius: 5px;"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addStretch()

        self.stack_layout.addWidget(tab)
        self.content_widgets[t('obj_detect')] = tab
    
    def create_upscaling_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.upscale_check = QCheckBox(f"⬆ {t('enable_upscaling')}")
        self.upscale_check.setStyleSheet("font-size: 12pt; font-weight: bold; color: #89b4fa;")
        layout.addWidget(self.upscale_check)
        
        settings_group = QGroupBox(t('settings'))
        settings_layout = QGridLayout()
        
        settings_layout.addWidget(QLabel(t('upscale_factor')), 0, 0)
        self.upscale_factor = QComboBox()
        self.upscale_factor.addItems(["1x", "2x", "3x", "4x"])
        self.upscale_factor.setCurrentText("2x")
        settings_layout.addWidget(self.upscale_factor, 0, 1)
        
        settings_layout.addWidget(QLabel(t('upscale_method')), 1, 0)
        self.upscale_method = QComboBox()
        self.upscale_method.addItems([
            "shufflecugan", "shufflecugan-tensorrt", "shufflecugan-ncnn",
            "fallin_soft", "fallin_soft-tensorrt", "fallin_soft-directml",
            "fallin_strong", "fallin_strong-tensorrt", "fallin_strong-directml",
            "span", "span-tensorrt", "span-directml", "span-ncnn",
            "compact", "compact-tensorrt", "compact-directml",
            "ultracompact", "ultracompact-tensorrt", "ultracompact-directml",
            "superultracompact", "superultracompact-tensorrt", "superultracompact-directml",
            "open-proteus", "open-proteus-tensorrt", "open-proteus-directml",
            "aniscale2", "aniscale2-tensorrt", "aniscale2-directml",
            "rtmosr", "rtmosr-tensorrt", "rtmosr-directml",
            "saryn", "saryn-tensorrt", "saryn-directml",
            "animesr", "animesr-tensorrt", "animesr-directml",
            "gauss", "gauss-tensorrt", "gauss-directml", "gauss-openvino"
        ])
        settings_layout.addWidget(self.upscale_method, 1, 1, 1, 2)
        
        self.custom_upscale_check = QCheckBox("🎨 " + t('use_custom_model'))
        self.custom_upscale_check.stateChanged.connect(lambda: self.toggle_custom_model('upscale'))
        settings_layout.addWidget(self.custom_upscale_check, 2, 0, 1, 3)
        
        backend_label = QLabel("🔧 Backend:")
        settings_layout.addWidget(backend_label, 3, 0)
        self.global_backend_combo = QComboBox()
        self.global_backend_combo.addItems(['Default', 'CUDA', 'TensorRT', 'DirectML', 'NCNN'])
        settings_layout.addWidget(self.global_backend_combo, 3, 1, 1, 2)
        backend_label.hide()
        self.global_backend_combo.hide()
        self.global_backend_label = backend_label
        
        settings_layout.addWidget(QLabel(t('custom_model')), 4, 0)
        self.custom_upscale_combo = QComboBox()
        self.custom_upscale_combo.addItem("None")
        settings_layout.addWidget(self.custom_upscale_combo, 4, 1)
        
        upscale_browse = QPushButton("📂 " + t('browse_custom_model'))
        upscale_browse.setObjectName("browseModelBtn")
        upscale_browse.setMinimumWidth(150)
        upscale_browse.setFixedHeight(32)
        upscale_browse.setStyleSheet("QPushButton { font-weight: bold; padding: 5px; }")
        upscale_browse.setToolTip(t('browse_custom_model'))
        upscale_browse.clicked.connect(lambda: self.browse_custom_model('upscale'))
        settings_layout.addWidget(upscale_browse, 4, 2)
        
        self.custom_upscale_label = settings_layout.itemAtPosition(4, 0).widget()
        self.custom_upscale_label.hide()
        self.custom_upscale_combo.hide()
        upscale_browse.hide()
        
        self.custom_upscale_browse_btn = upscale_browse
        
        self.load_saved_custom_models('upscale')
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        info_label = QLabel(t('info_upscaling'))
        info_label.setStyleSheet("color: #a6adc8; font-size: 9pt; padding: 15px; background-color: #181825; border: 2px solid #313244; border-radius: 5px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        self.stack_layout.addWidget(tab)
        self.content_widgets[t("upscaling")] = tab

    def create_image_upscale_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        title = QLabel(f"🖼 {t('image_upscale')}")
        title.setStyleSheet("font-size: 12pt; font-weight: bold; color: #89b4fa;")
        layout.addWidget(title)

        io_group = QGroupBox(f"📁 {t('input_output')}")
        io_layout = QGridLayout()

        io_layout.addWidget(QLabel(t('input_image')), 0, 0)
        self.image_input_path = QLineEdit()
        self.image_input_path.setPlaceholderText(".png / .jpg / .jpeg")
        io_layout.addWidget(self.image_input_path, 0, 1)
        browse_in = QPushButton(t('browse'))
        browse_in.setObjectName("browseBtn")
        browse_in.clicked.connect(self.browse_image_input)
        io_layout.addWidget(browse_in, 0, 2)

        io_layout.addWidget(QLabel(t('output_image')), 1, 0)
        self.image_output_path = QLineEdit()
        self.image_output_path.setPlaceholderText(t('auto_output'))
        io_layout.addWidget(self.image_output_path, 1, 1)
        browse_out = QPushButton(t('browse'))
        browse_out.setObjectName("browseBtn")
        browse_out.clicked.connect(self.browse_image_output)
        io_layout.addWidget(browse_out, 1, 2)

        io_group.setLayout(io_layout)
        layout.addWidget(io_group)

        settings_group = QGroupBox(t('settings'))
        settings_layout = QVBoxLayout()

        note = QLabel(t('image_upscale_note'))
        note.setWordWrap(True)
        note.setStyleSheet("color: #a6adc8; font-size: 9pt;")
        settings_layout.addWidget(note)

        fmt_layout = QHBoxLayout()
        fmt_layout.addWidget(QLabel(t('image_output_format')))
        self.image_output_format = QComboBox()
        self.image_output_format.addItems(["PNG", "JPG"])
        self.image_output_format.setCurrentText("PNG")
        self.image_output_format.currentTextChanged.connect(self.on_image_output_format_changed)
        fmt_layout.addWidget(self.image_output_format)
        fmt_layout.addStretch()
        settings_layout.addLayout(fmt_layout)

        self.image_jpeg_quality_row = QWidget()
        quality_layout = QHBoxLayout(self.image_jpeg_quality_row)
        quality_layout.setContentsMargins(0, 0, 0, 0)

        self.image_jpeg_quality_label = QLabel(t('image_jpeg_quality'))
        quality_layout.addWidget(self.image_jpeg_quality_label)

        self.image_jpeg_quality = QSpinBox()
        self.image_jpeg_quality.setRange(1, 100)
        self.image_jpeg_quality.setValue(95)
        quality_layout.addWidget(self.image_jpeg_quality)
        quality_layout.addStretch()
        settings_layout.addWidget(self.image_jpeg_quality_row)

        # Initialize enabled/disabled state and ensure output extension matches selected format.
        self.on_image_output_format_changed(self.image_output_format.currentText())

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        layout.addStretch()

        self.stack_layout.addWidget(tab)
        self.content_widgets[t('image_upscale')] = tab
    
    def create_interpolation_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.interpolate_check = QCheckBox(f"⏱ {t('enable_interpolation')}")
        self.interpolate_check.setStyleSheet("font-size: 12pt; font-weight: bold; color: #89b4fa;")
        layout.addWidget(self.interpolate_check)
        
        settings_group = QGroupBox(t('settings'))
        settings_layout = QVBoxLayout()
        
        factor_layout = QHBoxLayout()
        factor_layout.addWidget(QLabel(t('interpolation_factor')))
        self.interpolate_factor = QDoubleSpinBox()
        self.interpolate_factor.setMinimum(2.0)
        self.interpolate_factor.setMaximum(8.0)
        self.interpolate_factor.setSingleStep(0.1)
        self.interpolate_factor.setValue(2.0)
        factor_layout.addWidget(self.interpolate_factor)
        factor_layout.addWidget(QLabel("x"))
        factor_layout.addStretch()
        settings_layout.addLayout(factor_layout)
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel(t('method')))
        self.interpolate_method = QComboBox()
        self.interpolate_method.addItems([
            "distildrba", "distildrba-lite", "distildrba-lite-tensorrt",
            "rife4.6", "rife4.15-lite", "rife4.16-lite", "rife4.17", "rife4.18", "rife4.20", "rife4.21",
            "rife4.22", "rife4.22-lite", "rife4.25", "rife4.25-lite", "rife4.25-heavy",
            "rife4.6-tensorrt", "rife4.15-tensorrt", "rife4.17-tensorrt", "rife4.18-tensorrt",
            "rife4.20-tensorrt", "rife4.21-tensorrt", "rife4.22-tensorrt", "rife4.22-lite-tensorrt",
            "rife4.25-tensorrt", "rife4.25-lite-tensorrt", "rife4.25-heavy-tensorrt",
            "rife4.6-ncnn", "rife4.15-lite-ncnn", "rife4.16-lite-ncnn", "rife4.17-ncnn", "rife4.22-ncnn",
            "gmfss", "gmfss-tensorrt", "rife_elexor", "rife_elexor-tensorrt"
        ])
        method_layout.addWidget(self.interpolate_method)
        settings_layout.addLayout(method_layout)
        
        self.custom_interpolate_check = QCheckBox("🎨 " + t('use_custom_model'))
        self.custom_interpolate_check.stateChanged.connect(lambda: self.toggle_custom_model('interpolate'))
        settings_layout.addWidget(self.custom_interpolate_check)
        
        custom_interp_layout = QHBoxLayout()
        custom_interp_layout.addWidget(QLabel(t('custom_model')))
        self.custom_interpolate_combo = QComboBox()
        self.custom_interpolate_combo.addItem("None")
        custom_interp_layout.addWidget(self.custom_interpolate_combo)

        backend_label = QLabel("🔧 Backend:")
        self.custom_interpolate_backend_combo = QComboBox()
        self.custom_interpolate_backend_combo.addItems(['Default', 'CUDA', 'TensorRT', 'DirectML', 'NCNN'])
        custom_interp_layout.addWidget(backend_label)
        custom_interp_layout.addWidget(self.custom_interpolate_backend_combo)
        self.custom_interpolate_backend_label = backend_label
        
        interpolate_browse = QPushButton("📂 " + t('browse_custom_model'))
        interpolate_browse.setObjectName("browseModelBtn")
        interpolate_browse.setMinimumWidth(150)
        interpolate_browse.setFixedHeight(32)
        interpolate_browse.setStyleSheet("QPushButton { font-weight: bold; padding: 5px; }")
        interpolate_browse.setToolTip(t('browse_custom_model'))
        interpolate_browse.clicked.connect(lambda: self.browse_custom_model('interpolate'))
        custom_interp_layout.addWidget(interpolate_browse)
        
        settings_layout.addLayout(custom_interp_layout)
        
        self.custom_interpolate_label = custom_interp_layout.itemAt(0).widget()
        self.custom_interpolate_browse_btn = interpolate_browse
        self.custom_interpolate_layout = custom_interp_layout
        
        for i in range(custom_interp_layout.count()):
            widget = custom_interp_layout.itemAt(i).widget()
            if widget:
                widget.hide()
        
        self.load_saved_custom_models('interpolate')
        
        self.ensemble_check = QCheckBox(t('ensemble'))
        settings_layout.addWidget(self.ensemble_check)
        
        self.dynamic_scale_check = QCheckBox(t('dynamic_scale'))
        settings_layout.addWidget(self.dynamic_scale_check)
        
        self.slowmo_check = QCheckBox(t('slowmo'))
        settings_layout.addWidget(self.slowmo_check)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        info_label = QLabel(t('info_interpolation'))
        info_label.setStyleSheet("color: #a6adc8; font-size: 9pt; padding: 15px; background-color: #181825; border: 2px solid #313244; border-radius: 5px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        self.stack_layout.addWidget(tab)
        self.content_widgets[t("interpolation")] = tab
    
    def create_restoration_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.restore_check = QCheckBox(f"🔧 {t('enable_restoration')}")
        self.restore_check.setStyleSheet("font-size: 12pt; font-weight: bold; color: #89b4fa;")
        layout.addWidget(self.restore_check)
        
        restore_group = QGroupBox(t('restoration_method'))
        restore_layout = QVBoxLayout()
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel(t('method')))
        self.restore_method = QComboBox()
        self.restore_method.addItems([
            "scunet", "scunet-tensorrt", "scunet-directml",
            "nafnet", "dpir", "real-plksr",
            "anime1080fixer", "anime1080fixer-tensorrt", "anime1080fixer-directml",
            "fastlinedarken", "fastlinedarken-tensorrt",
            "gater3", "gater3-directml",
            "deh264_real", "deh264_real-tensorrt", "deh264_real-directml"
        ])
        method_layout.addWidget(self.restore_method)
        add_builtin_btn = QPushButton("➕ Add")
        add_builtin_btn.setObjectName("browseModelBtn")
        add_builtin_btn.setMinimumWidth(100)
        add_builtin_btn.setFixedHeight(32)
        add_builtin_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 5px; }")
        add_builtin_btn.clicked.connect(self.add_restore_builtin)
        method_layout.addWidget(add_builtin_btn)
        restore_layout.addLayout(method_layout)
        
        self.custom_restore_check = QCheckBox("🎨 " + t('use_custom_model'))
        self.custom_restore_check.stateChanged.connect(lambda: self.toggle_custom_model('restore'))
        restore_layout.addWidget(self.custom_restore_check)
        
        custom_restore_layout = QHBoxLayout()
        custom_restore_layout.addWidget(QLabel(t('custom_model')))
        self.custom_restore_combo = QComboBox()
        self.custom_restore_combo.addItem("None")
        custom_restore_layout.addWidget(self.custom_restore_combo)

        backend_label = QLabel("🔧 Backend:")
        self.custom_restore_backend_combo = QComboBox()
        self.custom_restore_backend_combo.addItems(['Default', 'CUDA', 'TensorRT', 'DirectML', 'NCNN'])
        custom_restore_layout.addWidget(backend_label)
        custom_restore_layout.addWidget(self.custom_restore_backend_combo)
        self.custom_restore_backend_label = backend_label
        
        restore_browse = QPushButton("📂 " + t('browse_custom_model'))
        restore_browse.setObjectName("browseModelBtn")
        restore_browse.setMinimumWidth(150)
        restore_browse.setFixedHeight(32)
        restore_browse.setStyleSheet("QPushButton { font-weight: bold; padding: 5px; }")
        restore_browse.setToolTip(t('browse_custom_model'))
        restore_browse.clicked.connect(lambda: self.browse_custom_model('restore'))
        custom_restore_layout.addWidget(restore_browse)

        add_custom_btn = QPushButton("➕ Add")
        add_custom_btn.setObjectName("browseModelBtn")
        add_custom_btn.setMinimumWidth(100)
        add_custom_btn.setFixedHeight(32)
        add_custom_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 5px; }")
        add_custom_btn.clicked.connect(self.add_restore_custom)
        custom_restore_layout.addWidget(add_custom_btn)
        
        restore_layout.addLayout(custom_restore_layout)
        
        self.custom_restore_label = custom_restore_layout.itemAt(0).widget()
        self.custom_restore_browse_btn = restore_browse
        self.custom_restore_layout = custom_restore_layout
        
        for i in range(custom_restore_layout.count()):
            widget = custom_restore_layout.itemAt(i).widget()
            if widget:
                widget.hide()
        
        self.load_saved_custom_models('restore')

        selected_group = QGroupBox("Selected Restoration Models (order matters)")
        selected_layout = QVBoxLayout()

        self.restore_chain_list = QListWidget()
        self.restore_chain_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 2px solid #313244;
                border-radius: 5px;
                padding: 5px;
                font-size: 10pt;
            }
            QListWidget::item {
                padding: 6px;
                border-radius: 3px;
            }
            QListWidget::item:selected {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
        """)
        selected_layout.addWidget(self.restore_chain_list)

        btn_row = QHBoxLayout()
        remove_btn = QPushButton("🗑 Remove")
        remove_btn.clicked.connect(self.remove_restore_selected)
        up_btn = QPushButton("⬆ Up")
        up_btn.clicked.connect(self.move_restore_up)
        down_btn = QPushButton("⬇ Down")
        down_btn.clicked.connect(self.move_restore_down)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_restore_list)

        btn_row.addWidget(remove_btn)
        btn_row.addWidget(up_btn)
        btn_row.addWidget(down_btn)
        btn_row.addWidget(clear_btn)
        selected_layout.addLayout(btn_row)

        selected_group.setLayout(selected_layout)
        restore_layout.addWidget(selected_group)
        
        restore_group.setLayout(restore_layout)
        layout.addWidget(restore_group)
        
        sharpen_group = QGroupBox(t('enable_sharpening'))
        sharpen_layout = QVBoxLayout()
        
        self.sharpen_check = QCheckBox(t('enable_sharpening'))
        sharpen_layout.addWidget(self.sharpen_check)
        
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel(t('sharpening_sensitivity')))
        self.sharpen_slider = QSlider(Qt.Horizontal)
        self.sharpen_slider.setMinimum(0)
        self.sharpen_slider.setMaximum(100)
        self.sharpen_slider.setValue(50)
        slider_layout.addWidget(self.sharpen_slider)
        self.sharpen_value_label = QLabel("50")
        self.sharpen_slider.valueChanged.connect(
            lambda v: self.sharpen_value_label.setText(str(v))
        )
        slider_layout.addWidget(self.sharpen_value_label)
        sharpen_layout.addLayout(slider_layout)
        
        sharpen_group.setLayout(sharpen_layout)
        layout.addWidget(sharpen_group)
        
        info_label = QLabel(t('info_restoration'))
        info_label.setStyleSheet("color: #a6adc8; font-size: 9pt; padding: 15px; background-color: #181825; border: 2px solid #313244; border-radius: 5px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        self.stack_layout.addWidget(tab)
        self.content_widgets[t("restoration")] = tab
    
    def create_scene_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.scenechange_check = QCheckBox(f"🎬 {t('enable_scene_detection')}")
        self.scenechange_check.setStyleSheet("font-size: 12pt; font-weight: bold; color: #89b4fa;")
        layout.addWidget(self.scenechange_check)
        
        settings_group = QGroupBox(t('settings'))
        settings_layout = QVBoxLayout()
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel(t('scene_method')))
        self.scenechange_method = QComboBox()
        self.scenechange_method.addItems([
            "maxxvit-tensorrt", "maxxvit-directml",
            "differential", "differential-tensorrt",
            "shift_lpips-tensorrt", "shift_lpips-directml"
        ])
        method_layout.addWidget(self.scenechange_method)
        settings_layout.addLayout(method_layout)
        
        self.custom_scenechange_check = QCheckBox("🎨 " + t('use_custom_model'))
        self.custom_scenechange_check.stateChanged.connect(lambda: self.toggle_custom_model('scenechange'))
        settings_layout.addWidget(self.custom_scenechange_check)
        
        custom_scene_layout = QHBoxLayout()
        custom_scene_layout.addWidget(QLabel(t('custom_model')))
        self.custom_scenechange_combo = QComboBox()
        self.custom_scenechange_combo.addItem("None")
        custom_scene_layout.addWidget(self.custom_scenechange_combo)

        backend_label = QLabel("🔧 Backend:")
        self.custom_scenechange_backend_combo = QComboBox()
        self.custom_scenechange_backend_combo.addItems(['Default', 'CUDA', 'TensorRT', 'DirectML', 'NCNN'])
        custom_scene_layout.addWidget(backend_label)
        custom_scene_layout.addWidget(self.custom_scenechange_backend_combo)
        self.custom_scenechange_backend_label = backend_label
        
        scenechange_browse = QPushButton("📂 " + t('browse_custom_model'))
        scenechange_browse.setObjectName("browseModelBtn")
        scenechange_browse.setMinimumWidth(150)
        scenechange_browse.setFixedHeight(32)
        scenechange_browse.setStyleSheet("QPushButton { font-weight: bold; padding: 5px; }")
        scenechange_browse.setToolTip(t('browse_custom_model'))
        scenechange_browse.clicked.connect(lambda: self.browse_custom_model('scenechange'))
        custom_scene_layout.addWidget(scenechange_browse)
        
        settings_layout.addLayout(custom_scene_layout)
        
        self.custom_scenechange_label = custom_scene_layout.itemAt(0).widget()
        self.custom_scenechange_browse_btn = scenechange_browse
        self.custom_scenechange_layout = custom_scene_layout
        
        for i in range(custom_scene_layout.count()):
            widget = custom_scene_layout.itemAt(i).widget()
            if widget:
                widget.hide()
        
        self.load_saved_custom_models('scenechange')
        
        sens_layout = QHBoxLayout()
        sens_layout.addWidget(QLabel(t('scene_sensitivity')))
        self.scenechange_slider = QSlider(Qt.Horizontal)
        self.scenechange_slider.setMinimum(0)
        self.scenechange_slider.setMaximum(100)
        self.scenechange_slider.setValue(50)
        sens_layout.addWidget(self.scenechange_slider)
        self.scenechange_value_label = QLabel("50")
        self.scenechange_slider.valueChanged.connect(
            lambda v: self.scenechange_value_label.setText(str(v))
        )
        sens_layout.addWidget(self.scenechange_value_label)
        settings_layout.addLayout(sens_layout)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        self.autoclip_check = QCheckBox(f"✂ {t('autoclip')}")
        self.autoclip_check.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.autoclip_check)
        
        info_label = QLabel(t('info_scene'))
        info_label.setStyleSheet("color: #a6adc8; font-size: 9pt; padding: 15px; background-color: #181825; border: 2px solid #313244; border-radius: 5px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        self.stack_layout.addWidget(tab)
        self.content_widgets[t("scene_detection")] = tab
    
    def create_dedup_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.dedup_check = QCheckBox(f"🔄 {t('enable_deduplication')}")
        self.dedup_check.setStyleSheet("font-size: 12pt; font-weight: bold; color: #89b4fa;")
        layout.addWidget(self.dedup_check)
        
        settings_group = QGroupBox(t('settings'))
        settings_layout = QVBoxLayout()
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel(t('dedup_method')))
        self.dedup_method = QComboBox()
        self.dedup_method.addItems(["ssim", "ssim-cuda", "mse", "mse-cuda", "flownets", "vmaf", "vmaf-cuda"])
        method_layout.addWidget(self.dedup_method)
        settings_layout.addLayout(method_layout)
        
        self.custom_dedup_check = QCheckBox("🎨 " + t('use_custom_model'))
        self.custom_dedup_check.stateChanged.connect(lambda: self.toggle_custom_model('dedup'))
        settings_layout.addWidget(self.custom_dedup_check)
        
        custom_dedup_layout = QHBoxLayout()
        custom_dedup_layout.addWidget(QLabel(t('custom_model')))
        self.custom_dedup_combo = QComboBox()
        self.custom_dedup_combo.addItem("None")
        custom_dedup_layout.addWidget(self.custom_dedup_combo)
        
        dedup_browse = QPushButton("📂 " + t('browse_custom_model'))
        dedup_browse.setObjectName("browseModelBtn")
        dedup_browse.setMinimumWidth(150)
        dedup_browse.setFixedHeight(32)
        dedup_browse.setStyleSheet("QPushButton { font-weight: bold; padding: 5px; }")
        dedup_browse.setToolTip(t('browse_custom_model'))
        dedup_browse.clicked.connect(lambda: self.browse_custom_model('dedup'))
        custom_dedup_layout.addWidget(dedup_browse)
        
        settings_layout.addLayout(custom_dedup_layout)
        
        self.custom_dedup_label = custom_dedup_layout.itemAt(0).widget()
        self.custom_dedup_browse_btn = dedup_browse
        self.custom_dedup_layout = custom_dedup_layout
        
        for i in range(custom_dedup_layout.count()):
            widget = custom_dedup_layout.itemAt(i).widget()
            if widget:
                widget.hide()
        
        self.load_saved_custom_models('dedup')
        
        sens_layout = QHBoxLayout()
        sens_layout.addWidget(QLabel(t('dedup_sensitivity')))
        self.dedup_slider = QSlider(Qt.Horizontal)
        self.dedup_slider.setMinimum(0)
        self.dedup_slider.setMaximum(100)
        self.dedup_slider.setValue(35)
        sens_layout.addWidget(self.dedup_slider)
        self.dedup_value_label = QLabel("35")
        self.dedup_slider.valueChanged.connect(
            lambda v: self.dedup_value_label.setText(str(v))
        )
        sens_layout.addWidget(self.dedup_value_label)
        settings_layout.addLayout(sens_layout)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        info_label = QLabel(t('info_dedup'))
        info_label.setStyleSheet("color: #a6adc8; font-size: 9pt; padding: 15px; background-color: #181825; border: 2px solid #313244; border-radius: 5px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        self.stack_layout.addWidget(tab)
        self.content_widgets[t("deduplication")] = tab
    
    def create_segmentation_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.segment_check = QCheckBox(f"✂ {t('enable_segmentation')}")
        self.segment_check.setStyleSheet("font-size: 12pt; font-weight: bold; color: #89b4fa;")
        layout.addWidget(self.segment_check)
        
        settings_group = QGroupBox(t('settings'))
        settings_layout = QVBoxLayout()
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel(t('segment_method')))
        self.segment_method = QComboBox()
        self.segment_method.addItems(["anime", "anime-tensorrt", "anime-directml", "cartoon"])
        method_layout.addWidget(self.segment_method)
        settings_layout.addLayout(method_layout)
        
        self.custom_segment_check = QCheckBox("🎨 " + t('use_custom_model'))
        self.custom_segment_check.stateChanged.connect(lambda: self.toggle_custom_model('segment'))
        settings_layout.addWidget(self.custom_segment_check)
        
        custom_segment_layout = QHBoxLayout()
        custom_segment_layout.addWidget(QLabel(t('custom_model')))
        self.custom_segment_combo = QComboBox()
        self.custom_segment_combo.addItem("None")
        custom_segment_layout.addWidget(self.custom_segment_combo)

        backend_label = QLabel("🔧 Backend:")
        self.custom_segment_backend_combo = QComboBox()
        self.custom_segment_backend_combo.addItems(['Default', 'CUDA', 'TensorRT', 'DirectML', 'NCNN'])
        custom_segment_layout.addWidget(backend_label)
        custom_segment_layout.addWidget(self.custom_segment_backend_combo)
        self.custom_segment_backend_label = backend_label
        
        segment_browse = QPushButton("📂 " + t('browse_custom_model'))
        segment_browse.setObjectName("browseModelBtn")
        segment_browse.setMinimumWidth(150)
        segment_browse.setFixedHeight(32)
        segment_browse.setStyleSheet("QPushButton { font-weight: bold; padding: 5px; }")
        segment_browse.setToolTip(t('browse_custom_model'))
        segment_browse.clicked.connect(lambda: self.browse_custom_model('segment'))
        custom_segment_layout.addWidget(segment_browse)
        
        settings_layout.addLayout(custom_segment_layout)
        
        self.custom_segment_label = custom_segment_layout.itemAt(0).widget()
        self.custom_segment_browse_btn = segment_browse
        self.custom_segment_layout = custom_segment_layout
        
        for i in range(custom_segment_layout.count()):
            widget = custom_segment_layout.itemAt(i).widget()
            if widget:
                widget.hide()
        
        self.load_saved_custom_models('segment')
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        info_label = QLabel(t('info_segmentation'))
        info_label.setStyleSheet("color: #a6adc8; font-size: 9pt; padding: 15px; background-color: #181825; border: 2px solid #313244; border-radius: 5px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        self.stack_layout.addWidget(tab)
        self.content_widgets[t("segmentation")] = tab
    
    def create_depth_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.depth_check = QCheckBox(f"📊 {t('enable_depth')}")
        self.depth_check.setStyleSheet("font-size: 12pt; font-weight: bold; color: #89b4fa;")
        layout.addWidget(self.depth_check)
        
        settings_group = QGroupBox(t('settings'))
        settings_layout = QGridLayout()
        
        settings_layout.addWidget(QLabel(t('depth_method')), 0, 0)
        self.depth_method = QComboBox()
        self.depth_method.addItems([
            "small_v2", "base_v2", "large_v2",
            "distill_small_v2", "distill_base_v2", "distill_large_v2",
            "small_v2-tensorrt", "base_v2-tensorrt", "large_v2-tensorrt"
        ])
        settings_layout.addWidget(self.depth_method, 0, 1, 1, 2)
        
        self.custom_depth_check = QCheckBox("🎨 " + t('use_custom_model'))
        self.custom_depth_check.stateChanged.connect(lambda: self.toggle_custom_model('depth'))
        settings_layout.addWidget(self.custom_depth_check, 1, 0, 1, 3)
        
        settings_layout.addWidget(QLabel(t('custom_model')), 2, 0)
        self.custom_depth_combo = QComboBox()
        self.custom_depth_combo.addItem("None")
        settings_layout.addWidget(self.custom_depth_combo, 2, 1)
        
        depth_browse = QPushButton("📂 " + t('browse_custom_model'))
        depth_browse.setObjectName("browseModelBtn")
        depth_browse.setMinimumWidth(150)
        depth_browse.setFixedHeight(32)
        depth_browse.setStyleSheet("QPushButton { font-weight: bold; padding: 5px; }")
        depth_browse.setToolTip(t('browse_custom_model'))
        depth_browse.clicked.connect(lambda: self.browse_custom_model('depth'))
        settings_layout.addWidget(depth_browse, 2, 2)

        backend_label = QLabel("🔧 Backend:")
        self.custom_depth_backend_combo = QComboBox()
        self.custom_depth_backend_combo.addItems(['Default', 'CUDA', 'TensorRT', 'DirectML', 'NCNN'])
        settings_layout.addWidget(backend_label, 3, 0)
        settings_layout.addWidget(self.custom_depth_backend_combo, 3, 1, 1, 2)
        backend_label.hide()
        self.custom_depth_backend_combo.hide()
        self.custom_depth_backend_label = backend_label
        
        self.custom_depth_label = settings_layout.itemAtPosition(2, 0).widget()
        self.custom_depth_label.hide()
        self.custom_depth_combo.hide()
        depth_browse.hide()
        
        self.custom_depth_browse_btn = depth_browse
        
        self.load_saved_custom_models('depth')
        
        settings_layout.addWidget(QLabel(t('depth_quality')), 4, 0)
        self.depth_quality = QComboBox()
        self.depth_quality.addItems(["low", "medium", "high"])
        self.depth_quality.setCurrentText("high")
        settings_layout.addWidget(self.depth_quality, 4, 1, 1, 2)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        info_label = QLabel(t('info_depth'))
        info_label.setStyleSheet("color: #a6adc8; font-size: 9pt; padding: 15px; background-color: #181825; border: 2px solid #313244; border-radius: 5px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        self.stack_layout.addWidget(tab)
        self.content_widgets[t("depth")] = tab
    
    def create_encoding_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        encoding_group = QGroupBox(f"🎞 {t('encoding')}")
        encoding_layout = QGridLayout()
        
        encoding_layout.addWidget(QLabel(t('encode_method')), 0, 0)
        self.encode_method = QComboBox()
        # Keep GUI options in sync with CLI (--encode_method).
        # Source of truth: src/utils/argumentsChecker.py (encodeMethods).
        self.encode_method.addItems([
            "x264",
            "slow_x264",
            "x264_10bit",
            "x264_animation",
            "x264_animation_10bit",
            "x265",
            "slow_x265",
            "x265_10bit",
            "nvenc_h264",
            "slow_nvenc_h264",
            "nvenc_h265",
            "slow_nvenc_h265",
            "nvenc_h265_10bit",
            "nvenc_av1",
            "slow_nvenc_av1",
            "qsv_h264",
            "qsv_h265",
            "qsv_h265_10bit",
            "av1",
            "slow_av1",
            "h264_amf",
            "hevc_amf",
            "hevc_amf_10bit",
            "prores",
            "prores_segment",
            "gif",
            "vp9",
            "qsv_vp9",
            "lossless",
            "lossless_nvenc",
            "png",
        ])
        encoding_layout.addWidget(self.encode_method, 0, 1)
        
        encoding_layout.addWidget(QLabel(t('bit_depth')), 1, 0)
        self.bit_depth = QComboBox()
        # Keep GUI options in sync with CLI (--bit_depth).
        self.bit_depth.addItems(["8bit", "16bit"])
        encoding_layout.addWidget(self.bit_depth, 1, 1)
        
        encoding_group.setLayout(encoding_layout)
        layout.addWidget(encoding_group)
        
        resize_group = QGroupBox(t('enable_resize'))
        resize_layout = QVBoxLayout()
        
        self.resize_check = QCheckBox(t('enable_resize'))
        resize_layout.addWidget(self.resize_check)
        
        factor_layout = QHBoxLayout()
        factor_layout.addWidget(QLabel(t('resize_factor')))
        self.resize_factor = QDoubleSpinBox()
        self.resize_factor.setMinimum(0.1)
        self.resize_factor.setMaximum(10.0)
        self.resize_factor.setSingleStep(0.1)
        self.resize_factor.setValue(1.0)
        factor_layout.addWidget(self.resize_factor)
        factor_layout.addStretch()
        resize_layout.addLayout(factor_layout)
        
        resize_group.setLayout(resize_layout)
        layout.addWidget(resize_group)
        
        info_label = QLabel(t('info_encoding'))
        info_label.setStyleSheet("color: #a6adc8; font-size: 9pt; padding: 15px; background-color: #181825; border: 2px solid #313244; border-radius: 5px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        self.stack_layout.addWidget(tab)
        self.content_widgets[t("encoding")] = tab
    
    def create_performance_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        precision_group = QGroupBox(f"⚡ {t('performance')}")
        precision_layout = QVBoxLayout()
        
        self.half_check = QCheckBox(t('half_precision'))
        self.half_check.setChecked(True)
        precision_layout.addWidget(self.half_check)
        
        precision_group.setLayout(precision_layout)
        layout.addWidget(precision_group)
        
        decode_group = QGroupBox(t('decode_method'))
        decode_layout = QVBoxLayout()
        
        decode_layout.addWidget(QLabel(t('decode_method')))
        self.decode_method = QComboBox()
        self.decode_method.addItem("cpu")

        # NVDEC can be used only on NVIDIA systems, but the CUDA-enabled CeLux build
        # might fail to import before FFmpeg shared DLLs are downloaded. For better UX,
        # show the NVDEC option as soon as we can detect an NVIDIA GPU.
        nvdec_available = False
        nvdec_supported = False

        try:
            from src.utils.isCudaInit import detectNVidiaGPU

            nvdec_supported = bool(detectNVidiaGPU())
        except Exception:
            nvdec_supported = False

        if not nvdec_supported:
            try:
                import torch

                nvdec_supported = bool(torch.cuda.is_available())
            except Exception:
                nvdec_supported = False

        try:
            import celux_cuda

            nvdec_available = bool(getattr(celux_cuda, "__cuda_support__", False))
        except Exception:
            nvdec_available = False

        if nvdec_supported or nvdec_available:
            self.decode_method.addItem("nvdec")

        self.decode_method.setCurrentText("cpu")  # Set CPU as default (safer)
        decode_layout.addWidget(self.decode_method)

        self.nvdec_compat_check = QCheckBox(t('nvdec_compat'))
        self.nvdec_compat_check.setChecked(False)
        self.nvdec_compat_check.setToolTip(t('nvdec_compat_help'))
        decode_layout.addWidget(self.nvdec_compat_check)

        def _update_nvdec_compat_visibility():
            is_nvdec = self.decode_method.currentText() == "nvdec"
            self.nvdec_compat_check.setVisible(is_nvdec)

        self.decode_method.currentTextChanged.connect(
            lambda _text: _update_nvdec_compat_visibility()
        )
        _update_nvdec_compat_visibility()
        
        decode_note = QLabel(t('decode_note'))
        decode_note.setStyleSheet("color: #fab387; font-size: 8pt; font-style: italic;")
        decode_note.setWordWrap(True)
        if not (nvdec_supported or nvdec_available):
            decode_note.hide()
        decode_layout.addWidget(decode_note)
        
        decode_group.setLayout(decode_layout)
        layout.addWidget(decode_group)
        
        compile_group = QGroupBox(t('compile_mode'))
        compile_layout = QVBoxLayout()
        
        self.compile_mode = QComboBox()
        self.compile_mode.addItems(["default", "max", "max-graphs"])
        compile_layout.addWidget(self.compile_mode)
        
        compile_group.setLayout(compile_layout)
        layout.addWidget(compile_group)

        tile_group = QGroupBox(t('tile_rendering'))
        tile_layout = QVBoxLayout()

        self.tile_size_check = QCheckBox(t('enable_tile_size'))
        self.tile_size_check.setChecked(False)
        tile_layout.addWidget(self.tile_size_check)

        tile_size_row = QHBoxLayout()
        tile_size_row.addWidget(QLabel(t('tile_size')))
        self.tile_size_combo = QComboBox()
        self.tile_size_combo.addItems(["128", "256", "384", "512"])
        self.tile_size_combo.setCurrentText("128")
        self.tile_size_combo.setEnabled(False)
        tile_size_row.addWidget(self.tile_size_combo)
        tile_size_row.addStretch()
        tile_layout.addLayout(tile_size_row)

        tile_help = QLabel(t('tile_help'))
        tile_help.setWordWrap(True)
        tile_help.setStyleSheet("color: #bac2de; font-size: 9pt;")
        tile_layout.addWidget(tile_help)

        self.tile_size_check.setToolTip(t('tile_help'))
        self.tile_size_combo.setToolTip(t('tile_help'))

        self.tile_size_check.toggled.connect(self.tile_size_combo.setEnabled)

        tile_group.setLayout(tile_layout)
        layout.addWidget(tile_group)
        
        self.static_check = QCheckBox(t('static_chunk'))
        layout.addWidget(self.static_check)
        
        info_label = QLabel(t('info_performance'))
        info_label.setStyleSheet("color: #a6adc8; font-size: 9pt; padding: 15px; background-color: #181825; border: 2px solid #313244; border-radius: 5px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        self.stack_layout.addWidget(tab)
        self.content_widgets[t("performance")] = tab
    
    def update_language(self, index):
        new_lang = self.lang_combo.itemData(index)
        
        if new_lang == LanguageManager.get_language():
            return
        
        was_fullscreen = self.isFullScreen()
        geometry = self.geometry()
        
        saved_state = {}
        saved_settings = self.get_settings()
        
        if hasattr(self, 'process_thread') and self.process_thread and self.process_thread.isRunning():
            saved_state['is_processing'] = True
            saved_state['log_content'] = self.log_text.toPlainText()
            saved_state['progress_value'] = self.progress_bar.value()
            saved_state['progress_details'] = self.progress_details_label.text()
            saved_state['status_text'] = self.status_label.text()
            saved_state['status_style'] = self.status_label.styleSheet()
            saved_state['thread'] = self.process_thread
        
        LanguageManager.set_language(new_lang)
        
        self.close()
        self.__init__()
        
        self.restore_settings(saved_settings)
        
        if saved_state.get('is_processing'):
            self.log_text.setPlainText(saved_state['log_content'])
            self.progress_bar.setValue(saved_state['progress_value'])
            self.progress_details_label.setText(saved_state['progress_details'])
            self.status_label.setText(saved_state['status_text'])
            self.status_label.setStyleSheet(saved_state['status_style'])
            
            self.process_thread = saved_state['thread']
            self.process_thread.blockSignals(True)
            try:
                self.process_thread.progress_update.disconnect()
            except:
                pass
            try:
                self.process_thread.finished.disconnect()
            except:
                pass
            self.process_thread.blockSignals(False)
            self.process_thread.progress_update.connect(self.update_progress)
            self.process_thread.finished.connect(self.processing_finished)
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

            if hasattr(self, 'live_preview_check'):
                self.live_preview_check.setEnabled(False)

            if "--preview" in getattr(self.process_thread, 'command', []):
                self.preview_timer.start(1000)
                self.preview_label.setText(t('loading_preview'))
            else:
                self.preview_timer.stop()
                self.preview_label.setPixmap(QPixmap())
                self.preview_label.setText(t('live_preview_disabled'))
        
        if was_fullscreen:
            self.showFullScreen()
        else:
            self.setGeometry(geometry)
            self.show()

    def _sanitize_filename_part(self, text: str) -> str:
        text = str(text).strip() if text is not None else ""
        if not text:
            return ""
        text = re.sub(r'[<>:\"/\\\\|?*]', '_', text)
        text = re.sub(r'\s+', '_', text)
        text = text.strip(" ._")
        if len(text) > 80:
            text = text[:80]
        return text

    def _get_output_suffix_parts(self):
        suffix_parts = []

        if self.upscale_check.isChecked():
            part = self.upscale_method.currentText()
            if hasattr(self, 'custom_upscale_check') and self.custom_upscale_check.isChecked():
                selected = self.custom_upscale_combo.currentText().strip()
                if selected and selected != "None":
                    part = selected
            part = self._sanitize_filename_part(part)
            if part:
                suffix_parts.append(part)

        if self.interpolate_check.isChecked():
            part = self.interpolate_method.currentText()
            if hasattr(self, 'custom_interpolate_check') and self.custom_interpolate_check.isChecked():
                selected = self.custom_interpolate_combo.currentText().strip()
                if selected and selected != "None":
                    part = selected
            part = self._sanitize_filename_part(part)
            if part:
                suffix_parts.append(part)

        if self.restore_check.isChecked():
            restore_suffix = self.restore_method.currentText()
            if hasattr(self, 'restore_chain_list') and self.restore_chain_list.count() > 0:
                first = self.restore_chain_list.item(0).text()
                if first:
                    restore_suffix = Path(first).name if os.path.exists(first) else first
                if self.restore_chain_list.count() > 1:
                    restore_suffix = f"{restore_suffix}+{self.restore_chain_list.count()}"
            else:
                if hasattr(self, 'custom_restore_check') and self.custom_restore_check.isChecked():
                    selected = self.custom_restore_combo.currentText().strip()
                    if selected and selected != "None":
                        restore_suffix = selected

            restore_suffix = self._sanitize_filename_part(restore_suffix)
            if restore_suffix:
                suffix_parts.append(restore_suffix)

        return suffix_parts

    def _build_auto_output_path(self, input_filename: str):
        input_file = Path(input_filename)
        safe_stem = re.sub(r'[#<>:\"|?*]', '_', input_file.stem)
        if len(safe_stem) > 100:
            safe_stem = safe_stem[:100]

        suffix_parts = self._get_output_suffix_parts()
        model_suffix = "_".join(suffix_parts) if suffix_parts else "enhanced"
        return input_file.parent / f"{safe_stem}_{model_suffix}{input_file.suffix}"

    def _on_output_path_user_edited(self, _text: str):
        self.output_auto = False

    def _on_output_settings_changed(self, *args, **kwargs):
        self.update_output_auto_name()

    def _connect_output_auto_signals(self):
        if getattr(self, "_output_auto_signals_connected", False):
            return
        self._output_auto_signals_connected = True

        self.upscale_check.toggled.connect(self._on_output_settings_changed)
        self.upscale_factor.currentTextChanged.connect(self._on_output_settings_changed)
        self.upscale_method.currentTextChanged.connect(self._on_output_settings_changed)
        if hasattr(self, 'custom_upscale_check'):
            self.custom_upscale_check.stateChanged.connect(self._on_output_settings_changed)
        if hasattr(self, 'custom_upscale_combo'):
            self.custom_upscale_combo.currentTextChanged.connect(self._on_output_settings_changed)
        if hasattr(self, 'global_backend_combo'):
            self.global_backend_combo.currentTextChanged.connect(self._on_output_settings_changed)

        self.interpolate_check.toggled.connect(self._on_output_settings_changed)
        self.interpolate_factor.valueChanged.connect(self._on_output_settings_changed)
        self.interpolate_method.currentTextChanged.connect(self._on_output_settings_changed)
        if hasattr(self, 'custom_interpolate_check'):
            self.custom_interpolate_check.stateChanged.connect(self._on_output_settings_changed)
        if hasattr(self, 'custom_interpolate_combo'):
            self.custom_interpolate_combo.currentTextChanged.connect(self._on_output_settings_changed)
        if hasattr(self, 'custom_interpolate_backend_combo'):
            self.custom_interpolate_backend_combo.currentTextChanged.connect(self._on_output_settings_changed)

        self.restore_check.toggled.connect(self._on_output_settings_changed)
        self.restore_method.currentTextChanged.connect(self._on_output_settings_changed)
        if hasattr(self, 'custom_restore_check'):
            self.custom_restore_check.stateChanged.connect(self._on_output_settings_changed)
        if hasattr(self, 'custom_restore_combo'):
            self.custom_restore_combo.currentTextChanged.connect(self._on_output_settings_changed)
        if hasattr(self, 'custom_restore_backend_combo'):
            self.custom_restore_backend_combo.currentTextChanged.connect(self._on_output_settings_changed)

        if hasattr(self, 'restore_chain_list'):
            model = self.restore_chain_list.model()
            if model is not None:
                model.rowsInserted.connect(self._on_output_settings_changed)
                model.rowsRemoved.connect(self._on_output_settings_changed)
                if hasattr(model, 'rowsMoved'):
                    model.rowsMoved.connect(self._on_output_settings_changed)
                model.modelReset.connect(self._on_output_settings_changed)

    def update_output_auto_name(self):
        input_filename = self.input_path.text().strip()
        if not input_filename:
            return

        # If the user left output empty, treat that as "auto".
        if not self.output_path.text().strip():
            self.output_auto = True
        if not getattr(self, "output_auto", False):
            return

        output_file = self._build_auto_output_path(input_filename)
        self.output_path.setText(str(output_file))
    
    def browse_input(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Video",
            "",
            "Video Files (*.mp4 *.mkv *.avi *.mov *.webm *.m4v *.gif);;All Files (*.*)"
        )
        if filename:
            self.input_path.setText(filename)

            # GIF input is CPU-decoded; NVDEC is intended for video codecs and can be unstable on GIF.
            # Keep UI in sync with backend defaults.
            if filename.lower().endswith(".gif"):
                if hasattr(self, "encode_method"):
                    idx = self.encode_method.findText("gif")
                    if idx >= 0:
                        self.encode_method.setCurrentIndex(idx)

                if hasattr(self, "decode_method"):
                    idx = self.decode_method.findText("cpu")
                    if idx < 0:
                        idx = self.decode_method.findText("CPU")
                    if idx >= 0:
                        self.decode_method.setCurrentIndex(idx)

            if (not self.output_path.text().strip()) or getattr(self, "output_auto", False):
                self.output_auto = True
                self.update_output_auto_name()
            self.on_input_video_changed()
    
    def browse_output(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Output Video",
            "",
            "MP4 Files (*.mp4);;MKV Files (*.mkv);;GIF Files (*.gif);;All Files (*.*)"
        )
        if filename:
            self.output_auto = False
            self.output_path.setText(filename)

    def _set_segment_preview_enabled(self, enabled: bool) -> None:
        """Enable/disable the segment preview controls."""
        if hasattr(self, 'preview_start_slider'):
            self.preview_start_slider.setEnabled(bool(enabled))
        if hasattr(self, 'preview_duration_combo'):
            self.preview_duration_combo.setEnabled(bool(enabled))
        if hasattr(self, 'preview_create_btn'):
            self.preview_create_btn.setEnabled(bool(enabled))

    def _format_ts(self, seconds: float) -> str:
        try:
            seconds = float(seconds)
        except Exception:
            seconds = 0.0
        seconds_i = max(0, int(seconds))

        h = seconds_i // 3600
        m = (seconds_i % 3600) // 60
        s = seconds_i % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _get_preview_duration_s(self) -> int:
        if not hasattr(self, 'preview_duration_combo'):
            return 5
        try:
            data = self.preview_duration_combo.currentData()
            if data is not None:
                return int(data)
            txt = str(self.preview_duration_combo.currentText()).strip().lower()
            txt = txt.replace('sn', '').replace('sec', '').replace('s', '').strip()
            return int(txt)
        except Exception:
            return 5

    def _update_segment_preview_ui(self) -> None:
        if not hasattr(self, 'preview_segment_label'):
            return

        duration = self._input_video_duration_s
        if not duration or duration <= 0:
            # Keep whatever status message we set in on_input_video_changed.
            return

        start_s = 0
        if hasattr(self, 'preview_start_slider'):
            try:
                start_s = int(self.preview_start_slider.value())
            except Exception:
                start_s = 0

        info = self._input_video_info or {}
        dims = ""
        try:
            w = int(info.get('width') or 0)
            h = int(info.get('height') or 0)
            fps = float(info.get('fps') or 0.0)
            if w > 0 and h > 0:
                dims = f" • {w}x{h}"
            if fps > 0:
                dims += f" • {fps:.2f}fps"
        except Exception:
            pass

        self.preview_segment_label.setText(
            f"{t('preview_start')} {self._format_ts(start_s)} / {self._format_ts(duration)}{dims}"
        )

    def _update_segment_preview_slider_range(self) -> None:
        if not hasattr(self, 'preview_start_slider'):
            return

        duration = self._input_video_duration_s
        if not duration or duration <= 0:
            self.preview_start_slider.setRange(0, 0)
            self.preview_start_slider.setValue(0)
            self._set_segment_preview_enabled(False)
            return

        # Avoid starting duplicate info threads for the same path.
        try:
            if (
                getattr(self, '_video_info_thread', None) is not None
                and self._video_info_thread.isRunning()
            ):
                pending = getattr(self, '_pending_video_info_path', None)
                if pending and os.path.normpath(pending) == os.path.normpath(video_path):
                    return
        except Exception:
            pass

        clip_len = max(1, int(self._get_preview_duration_s()))
        max_start = max(0, int(duration) - clip_len)

        self.preview_start_slider.blockSignals(True)
        try:
            self.preview_start_slider.setRange(0, max_start)
            self.preview_start_slider.setValue(min(self.preview_start_slider.value(), max_start))
        finally:
            self.preview_start_slider.blockSignals(False)

        self._set_segment_preview_enabled(True)
        self._update_segment_preview_ui()

    def _is_youtube_tab_active(self) -> bool:
        tabs = getattr(self, 'input_tabs', None)
        if tabs is None:
            return False
        return int(tabs.currentIndex()) == 0

    def on_input_video_changed(self) -> None:
        """Called when input source changes; refreshes segment preview controls."""
        youtube_url = self.youtube_url.text().strip() if hasattr(self, 'youtube_url') else ''
        if self._is_youtube_tab_active():
            self._input_video_info = None
            self._input_video_duration_s = None
            self._pending_source_frame_key = None
            self._last_source_frame_pixmap = None
            try:
                if hasattr(self, '_source_frame_debounce_timer') and self._source_frame_debounce_timer is not None:
                    self._source_frame_debounce_timer.stop()
            except Exception:
                pass
            if hasattr(self, 'timeline_preview_label'):
                self.timeline_preview_label.setPixmap(QPixmap())
                self.timeline_preview_label.setText(t('timeline_preview_placeholder'))
            if hasattr(self, 'timeline_fullscreen_btn'):
                self.timeline_fullscreen_btn.setEnabled(False)
            if hasattr(self, 'preview_segment_label'):
                if youtube_url:
                    self.preview_segment_label.setText(t('preview_requires_local'))
                else:
                    self.preview_segment_label.setText(t('preview_not_ready'))
            self._set_segment_preview_enabled(False)
            return

        video_path = self.input_path.text().strip() if hasattr(self, 'input_path') else ''
        if not video_path or not os.path.exists(video_path):
            self._input_video_info = None
            self._input_video_duration_s = None
            self._pending_source_frame_key = None
            self._last_source_frame_pixmap = None
            try:
                if hasattr(self, '_source_frame_debounce_timer') and self._source_frame_debounce_timer is not None:
                    self._source_frame_debounce_timer.stop()
            except Exception:
                pass
            if hasattr(self, 'timeline_preview_label'):
                self.timeline_preview_label.setPixmap(QPixmap())
                self.timeline_preview_label.setText(t('timeline_preview_placeholder'))
            if hasattr(self, 'timeline_fullscreen_btn'):
                self.timeline_fullscreen_btn.setEnabled(False)
            if hasattr(self, 'preview_segment_label'):
                self.preview_segment_label.setText(t('preview_not_ready'))
            self._set_segment_preview_enabled(False)
            return

        self._pending_video_info_path = video_path
        if hasattr(self, 'preview_segment_label'):
            self.preview_segment_label.setText(t('loading_video_info'))
        self._set_segment_preview_enabled(False)

        # Reset timeline preview until the first frame is extracted.
        self._pending_source_frame_key = None
        self._last_source_frame_pixmap = None
        if hasattr(self, 'timeline_preview_label'):
            self.timeline_preview_label.setPixmap(QPixmap())
            self.timeline_preview_label.setText(t('loading_frame'))
        if hasattr(self, 'timeline_fullscreen_btn'):
            self.timeline_fullscreen_btn.setEnabled(False)

        try:
            # Fire a background thread to avoid freezing the UI on slow disks.
            self._video_info_thread = VideoInfoThread(video_path)
            self._video_info_thread.result.connect(self._on_video_info_result)
            self._video_info_thread.start()
        except Exception as e:
            self._input_video_info = None
            self._input_video_duration_s = None
            if hasattr(self, 'preview_segment_label'):
                self.preview_segment_label.setText(f"{t('error')}: {str(e)}")
            self._set_segment_preview_enabled(False)

    def _on_video_info_result(self, ok: bool, info: dict, error: str) -> None:
        # Ignore stale results (user changed input while the thread was running)
        current_path = self.input_path.text().strip() if hasattr(self, 'input_path') else ''
        pending = getattr(self, '_pending_video_info_path', None)
        if pending and current_path and os.path.normpath(pending) != os.path.normpath(current_path):
            return

        if not ok:
            self._input_video_info = None
            self._input_video_duration_s = None
            if hasattr(self, 'preview_segment_label'):
                self.preview_segment_label.setText(f"{t('error')}: {error}")
            self._set_segment_preview_enabled(False)
            return

        self._input_video_info = info or {}
        try:
            self._input_video_duration_s = float((info or {}).get('duration') or 0.0)
        except Exception:
            self._input_video_duration_s = 0.0

        self._update_segment_preview_slider_range()
        # Kick the initial timeline thumbnail fetch.
        self._schedule_timeline_frame_update(immediate=True)

    def on_preview_start_changed(self, _value: int) -> None:
        self._update_segment_preview_ui()
        self._schedule_timeline_frame_update()

    def on_preview_duration_changed(self, _index: int) -> None:
        self._update_segment_preview_slider_range()

    def _schedule_timeline_frame_update(self, immediate: bool = False) -> None:
        if not hasattr(self, '_source_frame_debounce_timer') or self._source_frame_debounce_timer is None:
            return

        try:
            self._source_frame_debounce_timer.stop()
        except Exception:
            pass

        self._source_frame_debounce_timer.start(0 if immediate else 150)

    def _request_source_frame_preview(self) -> None:
        if not hasattr(self, 'timeline_preview_label'):
            return

        if self._is_youtube_tab_active():
            return

        video_path = self.input_path.text().strip() if hasattr(self, 'input_path') else ''
        if not video_path or not os.path.exists(video_path):
            return

        ts_s = int(self.preview_start_slider.value()) if hasattr(self, 'preview_start_slider') else 0
        self._pending_source_frame_key = (video_path, ts_s)

        # Avoid flooding: only one running extraction at a time.
        try:
            if self._source_frame_thread is not None and self._source_frame_thread.isRunning():
                return
        except Exception:
            pass

        if hasattr(self, 'timeline_preview_label'):
            self.timeline_preview_label.setText(t('loading_frame'))
            self.timeline_preview_label.setPixmap(QPixmap())
        if hasattr(self, 'timeline_fullscreen_btn'):
            self.timeline_fullscreen_btn.setEnabled(False)

        try:
            target_width = 500
            try:
                if hasattr(self, 'timeline_preview_label'):
                    target_width = max(64, int(self.timeline_preview_label.width()))
            except Exception:
                target_width = 500

            self._source_frame_thread = SourceFrameThread(video_path, ts_s, target_width=target_width)
            self._source_frame_thread.result.connect(self._on_source_frame_result)
            self._source_frame_thread.start()
        except Exception as e:
            if hasattr(self, 'timeline_preview_label'):
                self.timeline_preview_label.setText(f"{t('error')}: {str(e)}")

    def _on_source_frame_result(self, ok: bool, video_path: str, ts_s: int, jpg_bytes, error: str) -> None:
        current_path = self.input_path.text().strip() if hasattr(self, 'input_path') else ''
        if not current_path or os.path.normpath(current_path) != os.path.normpath(video_path):
            return

        pending = getattr(self, '_pending_source_frame_key', None)
        if pending and (os.path.normpath(pending[0]) != os.path.normpath(video_path) or int(pending[1]) != int(ts_s)):
            # Stale result - schedule a new extraction for the latest slider value.
            self._schedule_timeline_frame_update(immediate=True)
            return

        if not ok:
            if hasattr(self, 'timeline_preview_label'):
                self.timeline_preview_label.setText(f"{t('error')}: {error}")
            if hasattr(self, 'timeline_fullscreen_btn'):
                self.timeline_fullscreen_btn.setEnabled(False)
            self._last_source_frame_pixmap = None
            return

        try:
            pixmap = QPixmap()
            pixmap.loadFromData(jpg_bytes)
            if pixmap.isNull():
                raise Exception("Failed to decode frame image")
        except Exception as e:
            if hasattr(self, 'timeline_preview_label'):
                self.timeline_preview_label.setText(f"{t('error')}: {str(e)}")
            if hasattr(self, 'timeline_fullscreen_btn'):
                self.timeline_fullscreen_btn.setEnabled(False)
            self._last_source_frame_pixmap = None
            return

        self._set_timeline_pixmap(pixmap)

    def _set_timeline_pixmap(self, pixmap: QPixmap) -> None:
        self._last_source_frame_pixmap = pixmap

        if not hasattr(self, 'timeline_preview_label'):
            return

        self.timeline_preview_label.setText("")
        scaled = pixmap.scaled(
            self.timeline_preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.timeline_preview_label.setPixmap(scaled)

        if hasattr(self, 'timeline_fullscreen_btn'):
            self.timeline_fullscreen_btn.setEnabled(True)

        if (
            self._source_frame_fullscreen_dialog is not None
            and self._source_frame_fullscreen_dialog.isVisible()
        ):
            self._source_frame_fullscreen_dialog.set_pixmap(pixmap)

    def open_timeline_fullscreen(self) -> None:
        pixmap = self._last_source_frame_pixmap
        if pixmap is None or pixmap.isNull():
            pixmap = self.timeline_preview_label.pixmap() if hasattr(self, 'timeline_preview_label') else None
        if pixmap is None or pixmap.isNull():
            return

        if self._source_frame_fullscreen_dialog is None:
            self._source_frame_fullscreen_dialog = FullscreenPreviewDialog(self)

        self._source_frame_fullscreen_dialog.set_pixmap(pixmap)
        self._source_frame_fullscreen_dialog.showFullScreen()
        self._source_frame_fullscreen_dialog.raise_()
        self._source_frame_fullscreen_dialog.activateWindow()

    def _build_preview_output_path(self, input_filename: str, inpoint: float, outpoint: float) -> Path:
        out_dir = Path(__file__).resolve().parent / "output"
        out_dir.mkdir(exist_ok=True)

        stem = self._sanitize_filename_part(Path(input_filename).stem)
        start_ms = int(max(0.0, float(inpoint)) * 1000)
        dur_ms = int(max(0.0, float(outpoint) - float(inpoint)) * 1000)

        return out_dir / f"{stem}_preview_{start_ms}ms_{dur_ms}ms.mp4"

    def start_preview_segment(self) -> None:
        """Run the main pipeline on a small [inpoint, outpoint] segment and open the result."""
        if self.process_thread and self.process_thread.isRunning():
            QMessageBox.warning(self, t('error'), "Önizleme için önce mevcut işlemi durdurun.")
            return

        if self._is_youtube_tab_active():
            QMessageBox.warning(self, t('error'), t('preview_requires_local'))
            return

        input_path = self.input_path.text().strip() if hasattr(self, 'input_path') else ''
        if not input_path or not os.path.exists(input_path):
            QMessageBox.critical(self, t('error'), t('input_required'))
            return

        if getattr(self, 'benchmark_check', None) is not None and self.benchmark_check.isChecked():
            QMessageBox.warning(self, t('error'), "Kıyaslama modunda önizleme oluşturulamaz.")
            return

        duration = self._input_video_duration_s
        if not duration or duration <= 0:
            QMessageBox.warning(self, t('error'), t('preview_not_ready'))
            return

        start_s = float(self.preview_start_slider.value()) if hasattr(self, 'preview_start_slider') else 0.0
        clip_len = float(self._get_preview_duration_s())
        out_s = min(float(duration), start_s + clip_len)
        if out_s <= start_s:
            QMessageBox.warning(self, t('error'), t('preview_not_ready'))
            return

        preview_out = str(self._build_preview_output_path(input_path, start_s, out_s))
        # Segment preview clip should NOT run in live preview (--preview) mode.
        cmd = self.build_command(
            inpoint=start_s,
            outpoint=out_s,
            output_override=preview_out,
            enable_preview=False,
        )

        self._is_segment_preview_run = True
        self._segment_preview_output_path = preview_out

        # Start the processing thread (same UX as the main run).
        self.stop_requested = False
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_details_label.setText("")

        self.log_text.clear()
        self.log_text.append("[PREVIEW] Preview segment run")
        self.log_text.append(f"[PREVIEW] Input: {input_path}")
        self.log_text.append(f"[PREVIEW] Output: {preview_out}")
        self.log_text.append(f"[PREVIEW] Segment: {start_s:.2f}s → {out_s:.2f}s")
        self.log_text.append(f"{t('command')} " + " ".join(cmd) + "\n")

        # NVDEC backend mode: allow forcing compat path for stability.
        if (
            hasattr(self, 'nvdec_compat_check')
            and self.decode_method.currentText() == 'nvdec'
            and self.nvdec_compat_check.isChecked()
        ):
            os.environ['TAS_NVDEC_MODE'] = 'compat'
        else:
            os.environ.pop('TAS_NVDEC_MODE', None)

        self.process_thread = ProcessThread(cmd)
        self.process_thread.progress_update.connect(self.update_progress)
        self.process_thread.finished.connect(self.processing_finished)
        self.process_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_action_button_states(start_active=True, stop_active=False)
        self.status_label.setText(t('processing'))
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #f9e2af;")

        if hasattr(self, 'live_preview_check'):
            self.live_preview_check.setEnabled(False)

        # Segment preview clip runs without live preview (--preview) by design.
        if "--preview" in cmd:
            self.preview_timer.start(1000)
            self.preview_label.setText(t('loading_preview'))
        else:
            self.preview_timer.stop()
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText(t('live_preview_disabled'))
        self._set_segment_preview_enabled(False)

    def _build_auto_image_output_path(self, input_filename: str) -> Path:
        out_dir = Path(__file__).parent / "output"
        out_dir.mkdir(exist_ok=True)
        factor = self.upscale_factor.currentText() if hasattr(self, 'upscale_factor') else '2x'

        method = self.upscale_method.currentText() if hasattr(self, 'upscale_method') else 'upscale'
        custom_model = None
        if (
            hasattr(self, 'custom_upscale_check')
            and self.custom_upscale_check.isChecked()
            and hasattr(self, 'custom_upscale_combo')
        ):
            selected = self.custom_upscale_combo.currentText().strip()
            if selected and selected != "None":
                candidate = str(self.custom_models_dirs['upscale'] / selected)
                if os.path.exists(candidate):
                    custom_model = candidate

        # Image Upscale supports PyTorch custom models (.pth/.pt) and ONNX custom models (.onnx).
        # Other artifacts (e.g. TensorRT .engine) are not supported in the single-image CLI yet.
        if custom_model:
            ext = Path(custom_model).suffix.lower()
            if ext not in ('.pth', '.pt', '.bin', '.onnx'):
                QMessageBox.critical(
                    self,
                    t('error'),
                    f"Image Upscale custom models currently support .pth/.pt (PyTorch) and .onnx (ONNX). Selected: {ext}",
                )
                return

        # Image upscale currently runs via UniversalPytorch; strip backend suffixes.
        if not custom_model:
            for suffix in ("-tensorrt", "-directml", "-ncnn"):
                if method.endswith(suffix):
                    method = method[: -len(suffix)]
                    break
        else:
            method = Path(custom_model).stem

        stem = self._sanitize_filename_part(Path(input_filename).stem)
        method = self._sanitize_filename_part(method)
        method_part = f"_{method}" if method else ""

        ext = ".png"
        if hasattr(self, 'image_output_format'):
            fmt = (self.image_output_format.currentText() or '').strip().lower()
            if fmt.startswith('jpg') or fmt.startswith('jpeg'):
                ext = ".jpg"

        return out_dir / f"{stem}_upscaled{method_part}_{factor}{ext}"

    def on_image_output_format_changed(self, *_):
        """Sync output path extension and quality control for Image Upscale."""
        fmt = (
            self.image_output_format.currentText().strip().lower()
            if hasattr(self, 'image_output_format')
            else 'png'
        )
        is_jpg = fmt.startswith('jpg') or fmt.startswith('jpeg')

        if hasattr(self, 'image_jpeg_quality_row'):
            self.image_jpeg_quality_row.setVisible(bool(is_jpg))

        # Keep output extension aligned with the selected format.
        desired_suffix = '.jpg' if is_jpg else '.png'
        out_path = self.image_output_path.text().strip() if hasattr(self, 'image_output_path') else ''
        if out_path:
            try:
                p = Path(out_path)
                suffix = p.suffix.lower()
                if is_jpg:
                    # Accept both .jpg and .jpeg.
                    if suffix not in ('.jpg', '.jpeg'):
                        self.image_output_path.setText(str(p.with_suffix('.jpg')))
                else:
                    # PNG: normalize strictly.
                    if suffix != desired_suffix:
                        self.image_output_path.setText(str(p.with_suffix(desired_suffix)))
            except Exception:
                pass
        else:
            in_path = self.image_input_path.text().strip() if hasattr(self, 'image_input_path') else ''
            if in_path:
                try:
                    self.image_output_path.setText(str(self._build_auto_image_output_path(in_path)))
                except Exception:
                    pass

    def _clear_image_previews(self):
        if hasattr(self, 'image_input_preview_label'):
            self.image_input_preview_label.clear()
            self.image_input_preview_label.setText(t('image_input_preview_placeholder'))
        if hasattr(self, 'image_output_preview_label'):
            self.image_output_preview_label.clear()
            self.image_output_preview_label.setText(t('image_output_preview_placeholder'))

    def _set_image_preview(self, label: QLabel, image_path: str, placeholder: str):
        if label is None:
            return

        if not image_path or not os.path.exists(image_path):
            label.clear()
            if placeholder:
                label.setText(placeholder)
            return

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            label.clear()
            if placeholder:
                label.setText(placeholder)
            return

        target = label.size()
        if target.width() <= 0 or target.height() <= 0:
            target = label.maximumSize()

        scaled = pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)

    def browse_image_input(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Image",
            "",
            "Image Files (*.png *.jpg *.jpeg);;All Files (*.*)",
        )
        if filename:
            self.image_input_path.setText(filename)
            if not self.image_output_path.text().strip():
                self.image_output_path.setText(str(self._build_auto_image_output_path(filename)))
            if hasattr(self, 'image_input_preview_label'):
                self._set_image_preview(
                    self.image_input_preview_label,
                    filename,
                    t('image_input_preview_placeholder'),
                )
            if hasattr(self, 'image_output_preview_label'):
                self.image_output_preview_label.clear()
                self.image_output_preview_label.setText(t('image_output_preview_placeholder'))

    def browse_image_output(self):
        default_path = self.image_output_path.text().strip()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Output Image",
            default_path,
            "PNG Files (*.png);;JPG Files (*.jpg *.jpeg);;All Files (*.*)",
        )
        if filename:
            self.image_output_path.setText(filename)

            # Keep output format dropdown in sync with the chosen extension.
            if hasattr(self, 'image_output_format'):
                ext = Path(filename).suffix.lower()
                if ext in ('.jpg', '.jpeg'):
                    self.image_output_format.setCurrentText('JPG')
                elif ext == '.png':
                    self.image_output_format.setCurrentText('PNG')
            if hasattr(self, 'image_output_preview_label'):
                if os.path.exists(filename):
                    self._set_image_preview(
                        self.image_output_preview_label,
                        filename,
                        t('image_output_preview_placeholder'),
                    )
                else:
                    self.image_output_preview_label.clear()
                    self.image_output_preview_label.setText(t('image_output_preview_placeholder'))

    def start_image_upscale(self):
        if self.process_thread and self.process_thread.isRunning():
            QMessageBox.warning(self, t('error'), "Stop video processing before image upscaling")
            return

        if self.image_upscale_process and self.image_upscale_process.state() != QProcess.NotRunning:
            return

        input_path = self.image_input_path.text().strip()
        if not input_path:
            QMessageBox.critical(self, t('error'), t('input_required'))
            return

        output_path = self.image_output_path.text().strip()
        if not output_path:
            output_path = str(self._build_auto_image_output_path(input_path))
            self.image_output_path.setText(output_path)

        # Enforce output extension based on selected format.
        output_format = 'png'
        if hasattr(self, 'image_output_format'):
            fmt = (self.image_output_format.currentText() or '').strip().lower()
            if fmt.startswith('jpg') or fmt.startswith('jpeg'):
                output_format = 'jpg'
        desired_suffix = '.jpg' if output_format == 'jpg' else '.png'
        try:
            p = Path(output_path)
            suffix = p.suffix.lower()
            if output_format == 'jpg':
                # Accept both .jpg and .jpeg.
                if suffix not in ('.jpg', '.jpeg'):
                    output_path = str(p.with_suffix('.jpg'))
                    self.image_output_path.setText(output_path)
            else:
                # PNG selected: normalize strictly.
                if suffix != desired_suffix:
                    output_path = str(p.with_suffix(desired_suffix))
                    self.image_output_path.setText(output_path)
        except Exception:
            pass

        jpeg_quality = 95
        if hasattr(self, 'image_jpeg_quality'):
            try:
                jpeg_quality = int(self.image_jpeg_quality.value())
            except Exception:
                jpeg_quality = 95

        # Reset OOM detection for this run (used to show a helpful message on failure).
        self.seen_oom = False

        upscale_factor = int(self.upscale_factor.currentText().replace('x', ''))
        selected_method = self.upscale_method.currentText()

        custom_model = None
        if (
            hasattr(self, 'custom_upscale_check')
            and self.custom_upscale_check.isChecked()
            and hasattr(self, 'custom_upscale_combo')
        ):
            selected = self.custom_upscale_combo.currentText().strip()
            if selected and selected != "None":
                candidate = str(self.custom_models_dirs['upscale'] / selected)
                if os.path.exists(candidate):
                    custom_model = candidate

        # Image upscale should honor the selected backend suffixes (-tensorrt/-directml/-ncnn),
        # consistent with the main video pipeline.
        upscale_method = selected_method

        # If using a custom upscale model file, allow forcing a backend.
        upscale_backend = None
        if custom_model and hasattr(self, 'global_backend_combo'):
            try:
                selected_backend = self.global_backend_combo.currentText().lower()
                if selected_backend and selected_backend != 'default':
                    upscale_backend = selected_backend
            except Exception:
                upscale_backend = None

        half = bool(self.half_check.isChecked()) if hasattr(self, 'half_check') else False
        compile_mode = self.compile_mode.currentText() if hasattr(self, 'compile_mode') else 'default'

        auto_tile_reason = None
        tile_size_pref = 0
        if hasattr(self, 'tile_size_combo'):
            try:
                tile_size_pref = int(self.tile_size_combo.currentText())
            except Exception:
                tile_size_pref = 0

        tile_size = 0
        if hasattr(self, 'tile_size_check') and self.tile_size_check.isChecked():
            tile_size = tile_size_pref
        else:
            # Auto tile fallback: custom models and large images often OOM during warmup when tiles are disabled.
            if custom_model:
                auto_tile_reason = "custom model"
            else:
                try:
                    pix = QPixmap(input_path)
                    if not pix.isNull():
                        w = int(pix.width())
                        h = int(pix.height())
                        if w > 0 and h > 0 and (w * h) >= (1920 * 1080 * 2):
                            auto_tile_reason = f"large image ({w}x{h})"
                except Exception:
                    auto_tile_reason = None

            if auto_tile_reason:
                tile_size = int(tile_size_pref or 128)

        # Optional restoration chain (from the Restoration tab)
        restore_methods = []
        restore_backend = None
        if hasattr(self, 'restore_check') and self.restore_check.isChecked():
            restore_methods = self._get_restore_chain_values() if hasattr(self, '_get_restore_chain_values') else []
            if not restore_methods:
                restore_method = self.restore_method.currentText()
                if hasattr(self, 'custom_restore_check') and self.custom_restore_check.isChecked():
                    selected = self.custom_restore_combo.currentText().strip()
                    if selected and selected != "None":
                        restore_method = str(self.custom_models_dirs['restore'] / selected)
                restore_methods = [restore_method]

            # If any custom path(s) exist in the chain, allow forcing a backend.
            if any(os.path.exists(m) for m in restore_methods):
                if hasattr(self, 'custom_restore_backend_combo'):
                    selected_backend = self.custom_restore_backend_combo.currentText().lower()
                    if selected_backend and selected_backend != 'default':
                        restore_backend = selected_backend

        self.log_text.append("\n[IMAGE] Starting image upscale...")
        self.log_text.append(f"[IMAGE] Input: {input_path}")
        self.log_text.append(f"[IMAGE] Output: {output_path}")
        display_model = Path(custom_model).name if custom_model else selected_method
        self.log_text.append(f"[IMAGE] Model: {display_model} | Factor: {upscale_factor}x")
        if restore_methods:
            display_chain = [Path(m).name if os.path.exists(m) else m for m in restore_methods]
            self.log_text.append(f"[IMAGE] Restore chain: {' -> '.join(display_chain)}")
        if tile_size > 0:
            auto = " (auto)" if auto_tile_reason else ""
            self.log_text.append(f"[IMAGE] Tile rendering: {tile_size}{auto}")
        else:
            self.log_text.append("[IMAGE] Tile rendering: off")

        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        self.progress_details_label.setText("[IMAGE] running...")
        self.status_label.setText(t('processing'))
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #f9e2af;")

        self.start_btn.setEnabled(False)

        if hasattr(self, 'image_input_preview_label'):
            self._set_image_preview(
                self.image_input_preview_label,
                input_path,
                t('image_input_preview_placeholder'),
            )
        if hasattr(self, 'image_output_preview_label'):
            self.image_output_preview_label.clear()
            self.image_output_preview_label.setText(t('image_output_preview_placeholder'))

        python_exe = sys.executable
        if python_exe.lower().endswith('pythonw.exe'):
            candidate = os.path.join(os.path.dirname(python_exe), 'python.exe')
            if os.path.exists(candidate):
                python_exe = candidate

        script_path = str((Path(__file__).resolve().parent / 'image_upscale_cli.py').resolve())
        args = [
            '-u',
            script_path,
            '--input',
            input_path,
            '--output',
            output_path,
            '--jpeg_quality',
            str(int(jpeg_quality)),
            '--method',
            upscale_method,
            '--factor',
            str(upscale_factor),
            '--compile_mode',
            str(compile_mode),
            '--tile_size',
            str(int(tile_size or 0)),
        ]
        if half:
            args.append('--half')
        if custom_model:
            args.extend(['--custom_model', custom_model])
            if upscale_backend:
                args.extend(['--custom_upscale_backend', upscale_backend])
        if restore_methods:
            args.append('--restore')
            args.append('--restore_method')
            args.extend(restore_methods)
            if restore_backend:
                args.extend(['--custom_restore_backend', restore_backend])

        proc = QProcess(self)
        proc.setWorkingDirectory(str(Path(__file__).resolve().parent))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_image_upscale_process_output)
        proc.errorOccurred.connect(self._on_image_upscale_process_error)
        proc.finished.connect(self._on_image_upscale_process_finished)

        self.image_upscale_process = proc
        proc.start(python_exe, args)

    def _append_image_upscale_log(self, message: str):
        if not message:
            return
        self.log_text.append(f"[IMAGE] {message}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_image_upscale_process_output(self):
        if not self.image_upscale_process:
            return

        try:
            chunk = bytes(self.image_upscale_process.readAllStandardOutput()).decode(
                'utf-8', errors='replace'
            )
        except Exception:
            return

        for line in chunk.splitlines():
            line = (line or '').strip()
            if not line:
                continue

            clean_line = line
            lowered = clean_line.lower()
            if (
                'out of memory' in lowered
                or 'oom' in lowered
                or 'cuda oom' in lowered
                or 'cuda out of memory' in lowered
                or 'outofmemoryerror' in lowered
            ):
                self.seen_oom = True

            self._append_image_upscale_log(clean_line)

    def _on_image_upscale_process_error(self, error):
        self.image_upscale_process = None
        self.image_upscale_finished(False, f"Image upscale process error: {error}")

    def _on_image_upscale_process_finished(self, exit_code: int, exit_status):
        # Flush any remaining output.
        try:
            self._on_image_upscale_process_output()
        except Exception:
            pass

        success = exit_status == QProcess.NormalExit and exit_code == 0
        output_path = self.image_output_path.text().strip()

        if success and hasattr(self, 'image_output_preview_label') and output_path and os.path.exists(output_path):
            self._set_image_preview(
                self.image_output_preview_label,
                output_path,
                t('image_output_preview_placeholder'),
            )

        self.image_upscale_process = None

        if success:
            self.image_upscale_finished(True, f"Saved: {output_path}")
        else:
            if getattr(self, "seen_oom", False):
                self.image_upscale_finished(False, t('oom_guidance'))
            else:
                self.image_upscale_finished(False, f"Image upscale failed (exit code {exit_code})")

    def image_upscale_finished(self, success: bool, message: str):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_details_label.setText("")

        self.start_btn.setEnabled(True)

        if success:
            self.status_label.setText(f"✓ {t('complete')}")
            self.status_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #a6e3a1;")
            QMessageBox.information(self, t('success'), message)
        else:
            self.status_label.setText(f"✗ {t('failed')}")
            self.status_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #f38ba8;")
            QMessageBox.critical(self, t('error'), message)

        self.log_text.append(f"\n[IMAGE] {message}")
    
    def browse_custom_model(self, model_type):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            f"Select Custom {model_type.capitalize()} Model",
            "",
            "Model Files (*.pth *.onnx *.pt *.bin);;All Files (*.*)"
        )
        if filename:
            backend_combo = getattr(self, f'custom_{model_type}_backend_combo', None)
            if backend_combo is None and hasattr(self, 'global_backend_combo'):
                backend_combo = self.global_backend_combo
            selected_backend = backend_combo.currentText().lower() if backend_combo else 'default'
            
            import shutil
            import json
            source = Path(filename)
            dest = self.custom_models_dirs[model_type] / source.name
            
            if source != dest:
                shutil.copy2(source, dest)
                filename = str(dest)
            
            self.custom_models[model_type]['path'] = filename
            self.custom_models[model_type]['backend'] = selected_backend
            
            config_file = self.custom_models_dirs[model_type] / 'models.json'
            config = {}
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
            
            config[source.name] = selected_backend
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.load_saved_custom_models(model_type)
            
            custom_combo = getattr(self, f'custom_{model_type}_combo')
            custom_combo.setCurrentText(source.name)
    
    def load_saved_custom_models(self, model_type):
        import json
        import re
        custom_combo = getattr(self, f'custom_{model_type}_combo')
        custom_combo.blockSignals(True)
        
        custom_combo.clear()
        custom_combo.addItem("None")
        
        config_file = self.custom_models_dirs[model_type] / 'models.json'
        config = {}
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        auto_generated_pattern = re.compile(r'_(fp16|fp32)?_?op\d+(_slim)?\.onnx$')
        
        extensions = ['*.pth', '*.onnx', '*.pt', '*.bin']
        for ext in extensions:
            for model_file in self.custom_models_dirs[model_type].glob(ext):
                if not auto_generated_pattern.search(model_file.name):
                    custom_combo.addItem(model_file.name)
        
        custom_combo.blockSignals(False)
    
    def toggle_custom_model(self, model_type):
        check = getattr(self, f'custom_{model_type}_check')
        combo = getattr(self, f'{model_type}_method')
        custom_combo = getattr(self, f'custom_{model_type}_combo')
        browse_btn = getattr(self, f'custom_{model_type}_browse_btn')
        
        is_checked = check.isChecked()
        
        if model_type in ['upscale', 'depth']:
            label = getattr(self, f'custom_{model_type}_label')
            label.setVisible(is_checked)
            custom_combo.setVisible(is_checked)
            browse_btn.setVisible(is_checked)
            combo.setEnabled(not is_checked)
            if model_type == 'upscale':
                self.global_backend_label.setVisible(is_checked)
                self.global_backend_combo.setVisible(is_checked)
            if model_type == 'depth':
                backend_label = getattr(self, 'custom_depth_backend_label', None)
                backend_combo = getattr(self, 'custom_depth_backend_combo', None)
                if backend_label is not None:
                    backend_label.setVisible(is_checked)
                if backend_combo is not None:
                    backend_combo.setVisible(is_checked)
            if is_checked:
                combo.setStyleSheet("QComboBox { background-color: #1a1a1a; color: #666; }")
            else:
                combo.setStyleSheet("")
        else:
            layout = getattr(self, f'custom_{model_type}_layout')
            for i in range(layout.count()):
                widget = layout.itemAt(i).widget()
                if widget:
                    widget.setVisible(is_checked)

            # For restore chaining we allow mixing built-in + custom models, so
            # we should NOT disable the built-in combobox when custom is enabled.
            if model_type != 'restore':
                combo.setEnabled(not is_checked)
                if is_checked:
                    combo.setStyleSheet("QComboBox { background-color: #1a1a1a; color: #666; }")
                else:
                    combo.setStyleSheet("")
            else:
                combo.setEnabled(True)
                combo.setStyleSheet("")
    
    # --- Restoration chain (multi-model) helpers ---
    def _get_restore_chain_values(self):
        if not hasattr(self, 'restore_chain_list'):
            return []
        values = []
        for i in range(self.restore_chain_list.count()):
            item = self.restore_chain_list.item(i)
            if item is None:
                continue
            text = item.text().strip()
            if text:
                values.append(text)
        return values

    def _set_restore_chain_values(self, values):
        if not hasattr(self, 'restore_chain_list'):
            return
        self.restore_chain_list.clear()
        if not values:
            self.update_output_auto_name()
            return
        if isinstance(values, str):
            values = [values]
        for v in values:
            if v:
                self.restore_chain_list.addItem(str(v))
        self.update_output_auto_name()

    def add_restore_builtin(self):
        if not hasattr(self, 'restore_chain_list'):
            return
        value = self.restore_method.currentText().strip()
        if not value:
            return
        self.restore_chain_list.addItem(value)
        self.update_output_auto_name()

    def add_restore_custom(self):
        if not hasattr(self, 'restore_chain_list'):
            return
        selected = self.custom_restore_combo.currentText()
        if not selected or selected == "None":
            return
        model_path = str(self.custom_models_dirs['restore'] / selected)
        self.restore_chain_list.addItem(model_path)
        self.update_output_auto_name()

    def remove_restore_selected(self):
        if not hasattr(self, 'restore_chain_list'):
            return
        for item in self.restore_chain_list.selectedItems():
            row = self.restore_chain_list.row(item)
            self.restore_chain_list.takeItem(row)
        self.update_output_auto_name()

    def move_restore_up(self):
        if not hasattr(self, 'restore_chain_list'):
            return
        row = self.restore_chain_list.currentRow()
        if row <= 0:
            return
        item = self.restore_chain_list.takeItem(row)
        self.restore_chain_list.insertItem(row - 1, item)
        self.restore_chain_list.setCurrentRow(row - 1)
        self.update_output_auto_name()

    def move_restore_down(self):
        if not hasattr(self, 'restore_chain_list'):
            return
        row = self.restore_chain_list.currentRow()
        if row < 0 or row >= self.restore_chain_list.count() - 1:
            return
        item = self.restore_chain_list.takeItem(row)
        self.restore_chain_list.insertItem(row + 1, item)
        self.restore_chain_list.setCurrentRow(row + 1)
        self.update_output_auto_name()

    def clear_restore_list(self):
        if not hasattr(self, 'restore_chain_list'):
            return
        self.restore_chain_list.clear()
        self.update_output_auto_name()

    def get_settings(self):
        import json

        if self._is_youtube_tab_active():
            input_source = self.youtube_url.text()
        else:
            input_source = self.input_path.text()
        
        upscale_method = self.upscale_method.currentText()
        if self.custom_upscale_check.isChecked():
            selected = self.custom_upscale_combo.currentText()
            if selected != "None":
                model_name = selected
                upscale_method = str(self.custom_models_dirs['upscale'] / model_name)
                
                global_backend = self.global_backend_combo.currentText().lower()
                backend = global_backend if global_backend != 'default' else 'default'
                
                config_file = self.custom_models_dirs['upscale'] / 'models.json'
                if config_file.exists() and backend == 'default':
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        backend = config.get(model_name, 'default')
                
                self.custom_models['upscale']['backend'] = backend
                self.custom_models['upscale']['path'] = upscale_method
        
        interpolate_method = self.interpolate_method.currentText()
        if self.custom_interpolate_check.isChecked():
            selected = self.custom_interpolate_combo.currentText()
            if selected != "None":
                model_name = selected
                interpolate_method = str(self.custom_models_dirs['interpolate'] / model_name)

                backend = 'default'
                if hasattr(self, 'custom_interpolate_backend_combo'):
                    selected_backend = self.custom_interpolate_backend_combo.currentText().lower()
                    if selected_backend != 'default':
                        backend = selected_backend

                if backend == 'default':
                    config_file = self.custom_models_dirs['interpolate'] / 'models.json'
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                            backend = config.get(model_name, 'default')

                self.custom_models['interpolate']['backend'] = backend
                self.custom_models['interpolate']['path'] = interpolate_method
        
        restore_methods = self._get_restore_chain_values() if hasattr(self, '_get_restore_chain_values') else []
        if not restore_methods:
            restore_method = self.restore_method.currentText()
            if self.custom_restore_check.isChecked():
                selected = self.custom_restore_combo.currentText()
                if selected != "None":
                    model_name = selected
                    restore_method = str(self.custom_models_dirs['restore'] / model_name)

                    backend = 'default'
                    if hasattr(self, 'custom_restore_backend_combo'):
                        selected_backend = self.custom_restore_backend_combo.currentText().lower()
                        if selected_backend != 'default':
                            backend = selected_backend

                    if backend == 'default':
                        config_file = self.custom_models_dirs['restore'] / 'models.json'
                        if config_file.exists():
                            with open(config_file, 'r') as f:
                                config = json.load(f)
                                backend = config.get(model_name, 'default')

                    self.custom_models.setdefault('restore', {})
                    self.custom_models['restore']['backend'] = backend
                    self.custom_models['restore']['path'] = restore_method
            restore_methods = [restore_method]

        # If the chain includes any custom path(s), allow the user to force a single backend
        # for those custom restore models via the GUI backend dropdown.
        if any(os.path.exists(m) for m in restore_methods):
            if hasattr(self, 'custom_restore_backend_combo'):
                selected_backend = self.custom_restore_backend_combo.currentText().lower()
                if selected_backend and selected_backend != 'default':
                    self.custom_models.setdefault('restore', {})
                    self.custom_models['restore']['backend'] = selected_backend
        
        scenechange_method = self.scenechange_method.currentText()
        if self.custom_scenechange_check.isChecked():
            selected = self.custom_scenechange_combo.currentText()
            if selected != "None":
                model_name = selected
                scenechange_method = str(self.custom_models_dirs['scenechange'] / model_name)

                backend = 'default'
                if hasattr(self, 'custom_scenechange_backend_combo'):
                    selected_backend = self.custom_scenechange_backend_combo.currentText().lower()
                    if selected_backend != 'default':
                        backend = selected_backend

                if backend == 'default':
                    config_file = self.custom_models_dirs['scenechange'] / 'models.json'
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                            backend = config.get(model_name, 'default')

                self.custom_models.setdefault('scenechange', {})
                self.custom_models['scenechange']['backend'] = backend
                self.custom_models['scenechange']['path'] = scenechange_method
        
        dedup_method = self.dedup_method.currentText()
        if self.custom_dedup_check.isChecked():
            selected = self.custom_dedup_combo.currentText()
            if selected != "None":
                model_name = selected
                dedup_method = str(self.custom_models_dirs['dedup'] / model_name)
                
                config_file = self.custom_models_dirs['dedup'] / 'models.json'
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        backend = config.get(model_name, 'default')
                        self.custom_models['dedup']['backend'] = backend
                        self.custom_models['dedup']['path'] = dedup_method
        
        segment_method = self.segment_method.currentText()
        if self.custom_segment_check.isChecked():
            selected = self.custom_segment_combo.currentText()
            if selected != "None":
                model_name = selected
                segment_method = str(self.custom_models_dirs['segment'] / model_name)

                backend = 'default'
                if hasattr(self, 'custom_segment_backend_combo'):
                    selected_backend = self.custom_segment_backend_combo.currentText().lower()
                    if selected_backend != 'default':
                        backend = selected_backend

                if backend == 'default':
                    config_file = self.custom_models_dirs['segment'] / 'models.json'
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                            backend = config.get(model_name, 'default')

                self.custom_models.setdefault('segment', {})
                self.custom_models['segment']['backend'] = backend
                self.custom_models['segment']['path'] = segment_method
        
        depth_method = self.depth_method.currentText()
        if self.custom_depth_check.isChecked():
            selected = self.custom_depth_combo.currentText()
            if selected != "None":
                model_name = selected
                depth_method = str(self.custom_models_dirs['depth'] / model_name)

                backend = 'default'
                if hasattr(self, 'custom_depth_backend_combo'):
                    selected_backend = self.custom_depth_backend_combo.currentText().lower()
                    if selected_backend != 'default':
                        backend = selected_backend

                if backend == 'default':
                    config_file = self.custom_models_dirs['depth'] / 'models.json'
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                            backend = config.get(model_name, 'default')

                self.custom_models.setdefault('depth', {})
                self.custom_models['depth']['backend'] = backend
                self.custom_models['depth']['path'] = depth_method
        
        settings = {
            'input': input_source,
            'output': self.output_path.text(),
            'benchmark': self.benchmark_check.isChecked(),
            'live_preview': self.live_preview_check.isChecked() if hasattr(self, 'live_preview_check') else False,
            'obj_detect': self.obj_detect_check.isChecked(),
            'obj_detect_method': self.obj_detect_method.currentText(),
            'obj_detect_disable_annotations': self.obj_detect_disable_annotations_check.isChecked(),
            'upscale': self.upscale_check.isChecked(),
            'upscale_factor': int(self.upscale_factor.currentText()[0]),
            'upscale_method': upscale_method,
            'interpolate': self.interpolate_check.isChecked(),
            'interpolate_factor': self.interpolate_factor.value(),
            'interpolate_method': interpolate_method,
            'ensemble': self.ensemble_check.isChecked(),
            'dynamic_scale': self.dynamic_scale_check.isChecked(),
            'slowmo': self.slowmo_check.isChecked(),
            'restore': self.restore_check.isChecked(),
            'restore_method': restore_methods,
            'sharpen': self.sharpen_check.isChecked(),
            'sharpen_sens': self.sharpen_slider.value(),
            'scenechange': self.scenechange_check.isChecked(),
            'scenechange_method': scenechange_method,
            'scenechange_sens': self.scenechange_slider.value(),
            'autoclip': self.autoclip_check.isChecked(),
            'dedup': self.dedup_check.isChecked(),
            'dedup_method': dedup_method,
            'dedup_sens': self.dedup_slider.value(),
            'segment': self.segment_check.isChecked(),
            'segment_method': segment_method,
            'depth': self.depth_check.isChecked(),
            'depth_method': depth_method,
            'depth_quality': self.depth_quality.currentText(),
            'encode_method': self.encode_method.currentText(),
            'bit_depth': self.bit_depth.currentText(),
            'resize': self.resize_check.isChecked(),
            'resize_factor': self.resize_factor.value(),
            'half': self.half_check.isChecked(),
            'decode_method': self.decode_method.currentText(),
            'nvdec_compat': self.nvdec_compat_check.isChecked() if hasattr(self, 'nvdec_compat_check') else False,
            'compile_mode': self.compile_mode.currentText(),
            'static': self.static_check.isChecked(),
            'tile_size_enabled': getattr(self, 'tile_size_check', None).isChecked() if hasattr(self, 'tile_size_check') else False,
            'tile_size': (
                int(getattr(self, 'tile_size_combo', None).currentText())
                if (
                    hasattr(self, 'tile_size_check')
                    and hasattr(self, 'tile_size_combo')
                    and self.tile_size_check.isChecked()
                )
                else 0
            ),
        }
        return settings
    
    def restore_settings(self, settings):
        self.apply_settings(settings)
    
    def apply_settings(self, settings):
        # Don't load input/output paths from presets
        if 'input' in settings:
            self.input_path.setText(settings.get('input', ''))
            self.on_input_video_changed()
        if 'output' in settings:
            self.output_path.setText(settings.get('output', ''))
        
        self.benchmark_check.setChecked(settings.get('benchmark', False))

        if hasattr(self, 'live_preview_check'):
            self.live_preview_check.setChecked(settings.get('live_preview', False))
            self._on_live_preview_checkbox_changed()
        
        # Upscaling
        self.upscale_check.setChecked(settings.get('upscale', False))
        self.upscale_factor.setCurrentText(f"{settings.get('upscale_factor', 2)}x")
        method = settings.get('upscale_method', 'shufflecugan')
        idx = self.upscale_method.findText(method)
        if idx >= 0:
            self.upscale_method.setCurrentIndex(idx)
        
        # Interpolation
        self.interpolate_check.setChecked(settings.get('interpolate', False))
        self.interpolate_factor.setValue(settings.get('interpolate_factor', 2.0))
        interp_method = settings.get('interpolate_method', 'rife4.6')
        idx = self.interpolate_method.findText(interp_method)
        if idx >= 0:
            self.interpolate_method.setCurrentIndex(idx)
        self.ensemble_check.setChecked(settings.get('ensemble', False))
        self.dynamic_scale_check.setChecked(settings.get('dynamic_scale', False))
        self.slowmo_check.setChecked(settings.get('slowmo', False))
        
        # Restoration
        self.restore_check.setChecked(settings.get('restore', False))
        restore_methods = settings.get('restore_method', ['scunet'])
        if not isinstance(restore_methods, list):
            restore_methods = [restore_methods]

        if hasattr(self, 'restore_chain_list'):
            self.restore_chain_list.clear()
            for m in restore_methods:
                if m:
                    self.restore_chain_list.addItem(str(m))

        # Keep the single dropdown in sync with the first built-in method (if any)
        if restore_methods:
            first = str(restore_methods[0])
            idx = self.restore_method.findText(first)
            if idx >= 0:
                self.restore_method.setCurrentIndex(idx)
        
        # Sharpening
        self.sharpen_check.setChecked(settings.get('sharpen', False))
        self.sharpen_slider.setValue(settings.get('sharpen_sens', 50))
        
        # Scene Change
        self.scenechange_check.setChecked(settings.get('scenechange', False))
        scenechange_method = settings.get('scenechange_method', 'maxxvit-tensorrt')
        idx = self.scenechange_method.findText(scenechange_method)
        if idx >= 0:
            self.scenechange_method.setCurrentIndex(idx)
        self.scenechange_slider.setValue(settings.get('scenechange_sens', 50))
        
        # Other processing
        self.autoclip_check.setChecked(settings.get('autoclip', False))

        # Object Detection
        obj_enabled = bool(settings.get('obj_detect', False))
        self.obj_detect_check.setChecked(obj_enabled)
        obj_detect_method = settings.get('obj_detect_method', 'yolov9_small-directml')
        idx = self.obj_detect_method.findText(obj_detect_method)
        if idx >= 0:
            self.obj_detect_method.setCurrentIndex(idx)
        self.obj_detect_disable_annotations_check.setChecked(
            settings.get('obj_detect_disable_annotations', False)
        )

        self.obj_detect_method_widget.setEnabled(obj_enabled)
        self.obj_detect_disable_annotations_check.setEnabled(obj_enabled)
        
        # Deduplication
        self.dedup_check.setChecked(settings.get('dedup', False))
        dedup_method = settings.get('dedup_method', 'ssim')
        idx = self.dedup_method.findText(dedup_method)
        if idx >= 0:
            self.dedup_method.setCurrentIndex(idx)
        self.dedup_slider.setValue(settings.get('dedup_sens', 35))
        
        # Segmentation
        self.segment_check.setChecked(settings.get('segment', False))
        segment_method = settings.get('segment_method', 'anime')
        idx = self.segment_method.findText(segment_method)
        if idx >= 0:
            self.segment_method.setCurrentIndex(idx)
        
        # Depth
        self.depth_check.setChecked(settings.get('depth', False))
        depth_method = settings.get('depth_method', 'small_v2')
        idx = self.depth_method.findText(depth_method)
        if idx >= 0:
            self.depth_method.setCurrentIndex(idx)
        depth_quality = settings.get('depth_quality', 'balanced')
        idx = self.depth_quality.findText(depth_quality)
        if idx >= 0:
            self.depth_quality.setCurrentIndex(idx)
        
        # Encoding
        encode_method = settings.get('encode_method', 'x264')
        idx = self.encode_method.findText(encode_method)
        if idx >= 0:
            self.encode_method.setCurrentIndex(idx)
        bit_depth = settings.get('bit_depth', '8bit')
        idx = self.bit_depth.findText(bit_depth)
        if idx >= 0:
            self.bit_depth.setCurrentIndex(idx)
        
        # Resize
        self.resize_check.setChecked(settings.get('resize', False))
        self.resize_factor.setValue(settings.get('resize_factor', 1.0))
        
        # Performance
        self.half_check.setChecked(settings.get('half', True))
        decode_method = settings.get('decode_method', 'cpu')
        idx = self.decode_method.findText(decode_method)
        if idx >= 0:
            self.decode_method.setCurrentIndex(idx)

        if hasattr(self, 'nvdec_compat_check'):
            self.nvdec_compat_check.setChecked(settings.get('nvdec_compat', False))
            self.nvdec_compat_check.setVisible(self.decode_method.currentText() == 'nvdec')

        compile_mode = settings.get('compile_mode', 'default')

        if hasattr(self, 'compile_mode'):
            idx = self.compile_mode.findText(compile_mode)
            if idx >= 0:
                self.compile_mode.setCurrentIndex(idx)

        # Tile size
        if hasattr(self, 'tile_size_check') and hasattr(self, 'tile_size_combo'):
            enabled = settings.get('tile_size_enabled', False)
            self.tile_size_check.setChecked(enabled)
            tile_size = str(max(128, int(settings.get('tile_size', 128))))
            idx = self.tile_size_combo.findText(tile_size)
            if idx >= 0:
                self.tile_size_combo.setCurrentIndex(idx)
            self.tile_size_combo.setEnabled(enabled)
        idx = self.compile_mode.findText(compile_mode)
        if idx >= 0:
            self.compile_mode.setCurrentIndex(idx)
        self.static_check.setChecked(settings.get('static', False))

    def get_presets_dir(self) -> Path:
        # Store presets in a stable folder next to gui_app.py (project root in the Windows bundle).
        presets_dir = Path(__file__).resolve().parent / 'presets'
        presets_dir.mkdir(parents=True, exist_ok=True)
        return presets_dir

    def refresh_presets_dropdown(self, select_path: str = None):
        if not hasattr(self, 'preset_combo'):
            return

        presets_dir = self.get_presets_dir()

        if select_path is None and self.preset_combo.currentIndex() > 0:
            select_path = self.preset_combo.itemData(self.preset_combo.currentIndex())

        preset_files = sorted(presets_dir.glob('*.json'), key=lambda p: p.name.lower())

        self.preset_combo.blockSignals(True)
        try:
            self.preset_combo.clear()
            self.preset_combo.addItem(t('load_preset'), None)
            for p in preset_files:
                self.preset_combo.addItem(p.stem, str(p))

            if select_path:
                for i in range(self.preset_combo.count()):
                    if self.preset_combo.itemData(i) == select_path:
                        self.preset_combo.setCurrentIndex(i)
                        break
        finally:
            self.preset_combo.blockSignals(False)

    def on_preset_selected(self, index: int):
        if index is None or index <= 0:
            return
        if not hasattr(self, 'preset_combo'):
            return

        preset_path = self.preset_combo.itemData(index)
        if not preset_path:
            return

        try:
            with open(preset_path, 'r') as f:
                settings = json.load(f)
            self.apply_settings(settings)
            QMessageBox.information(self, "Success", f"Preset loaded: {Path(preset_path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load preset:\n{str(e)}")
    
    def save_preset(self):
        presets_dir = self.get_presets_dir()

        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok:
            return
        name = (name or '').strip()
        if not name:
            return

        safe_name = re.sub(r'[#<>:"|?*\\/]+', '_', name).strip('_')
        if not safe_name:
            QMessageBox.critical(self, "Error", "Invalid preset name")
            return

        preset_path = presets_dir / f"{safe_name}.json"
        if preset_path.exists():
            reply = QMessageBox.question(
                self,
                "Overwrite?",
                f"Preset already exists: {preset_path.name}\nOverwrite?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        try:
            settings = self.get_settings()
            # Remove input/output paths from preset
            settings.pop('input', None)
            settings.pop('output', None)
            with open(preset_path, 'w') as f:
                json.dump(settings, f, indent=2)

            self.refresh_presets_dropdown(select_path=str(preset_path))
            QMessageBox.information(self, "Success", f"Preset saved: {preset_path.name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save preset:\n{str(e)}")
    
    def build_command(self, inpoint=None, outpoint=None, output_override=None, enable_preview=None):
        settings = self.get_settings()
        
        if not settings['input']:
            raise ValueError(t('input_required'))
        
        cmd = [sys.executable, "main.py"]
        
        cmd.extend(["--input", settings['input']])

        if output_override:
            cmd.extend(["--output", str(output_override)])
        elif settings['output']:
            cmd.extend(["--output", settings['output']])

        if inpoint is not None:
            cmd.extend(["--inpoint", str(inpoint)])
        if outpoint is not None:
            cmd.extend(["--outpoint", str(outpoint)])

        if enable_preview is None:
            enable_preview = bool(settings.get('live_preview', False))

        # Preview is incompatible with benchmark mode (the CLI disables it too).
        if enable_preview and settings.get('benchmark'):
            enable_preview = False

        if enable_preview:
            cmd.append("--preview")
        
        if settings['benchmark']:
            cmd.append("--benchmark")
        
        if settings['upscale']:
            cmd.append("--upscale")
            cmd.extend(["--upscale_factor", str(settings['upscale_factor'])])
            upscale_method = settings['upscale_method']
            if upscale_method.startswith("Custom:"):
                cmd.extend(["--upscale_method", self.custom_models['upscale']['path']])
                backend = self.custom_models['upscale'].get('backend', 'default')
                if backend != 'default':
                    cmd.extend(["--custom_upscale_backend", backend])
            elif os.path.exists(upscale_method):
                cmd.extend(["--upscale_method", upscale_method])
                backend = self.custom_models['upscale'].get('backend', 'default')
                if backend != 'default':
                    cmd.extend(["--custom_upscale_backend", backend])
            else:
                cmd.extend(["--upscale_method", upscale_method])
        
        if settings['interpolate']:
            cmd.append("--interpolate")
            cmd.extend(["--interpolate_factor", str(settings['interpolate_factor'])])
            interpolate_method = settings['interpolate_method']
            if interpolate_method.startswith("Custom:"):
                cmd.extend(["--interpolate_method", self.custom_models['interpolate']['path']])
                backend = self.custom_models['interpolate'].get('backend', 'default')
                if backend != 'default':
                    cmd.extend(["--custom_interpolate_backend", backend])
            elif os.path.exists(interpolate_method):
                cmd.extend(["--interpolate_method", interpolate_method])
                backend = self.custom_models['interpolate'].get('backend', 'default')
                if backend != 'default':
                    cmd.extend(["--custom_interpolate_backend", backend])
            else:
                cmd.extend(["--interpolate_method", interpolate_method])
            
            if settings['ensemble']:
                cmd.append("--ensemble")
            
            if settings['dynamic_scale']:
                cmd.append("--dynamic_scale")
            
            if settings['slowmo']:
                cmd.append("--slowmo")
        
        if settings['restore']:
            cmd.append("--restore")
            restore_methods = settings.get('restore_method', [])
            if not isinstance(restore_methods, list):
                restore_methods = [restore_methods]

            restore_methods = [str(m) for m in restore_methods if m]
            if restore_methods:
                cmd.append("--restore_method")
                cmd.extend(restore_methods)

            # If the chain includes any custom path(s), allow the CLI to auto-detect
            # per-model backend when custom_restore_backend is left as default.
            # (If you want a single forced backend, we still honor the stored setting.)
            if any(os.path.exists(m) for m in restore_methods):
                backend = self.custom_models.get('restore', {}).get('backend', 'default')
                if backend and backend != 'default':
                    cmd.extend(["--custom_restore_backend", backend])
        
        if settings['sharpen']:
            cmd.append("--sharpen")
            cmd.extend(["--sharpen_sens", str(settings['sharpen_sens'])])
        
        if settings['scenechange']:
            cmd.append("--scenechange")
            scenechange_method = settings['scenechange_method']
            if scenechange_method.startswith("Custom:"):
                cmd.extend(["--scenechange_method", self.custom_models['scenechange']['path']])
            else:
                cmd.extend(["--scenechange_method", scenechange_method])
            cmd.extend(["--scenechange_sens", str(settings['scenechange_sens'])])
        
        if settings['autoclip']:
            cmd.append("--autoclip")

        if settings.get('obj_detect'):
            cmd.append("--obj_detect")
            cmd.extend(["--obj_detect_method", settings.get('obj_detect_method', 'yolov9_small-directml')])
            cmd.extend(
                [
                    "--obj_detect_disable_annotations",
                    str(settings.get('obj_detect_disable_annotations', False)).lower(),
                ]
            )
        
        if settings['dedup']:
            cmd.append("--dedup")
            dedup_method = settings['dedup_method']
            if dedup_method.startswith("Custom:"):
                cmd.extend(["--dedup_method", self.custom_models['dedup']['path']])
            else:
                cmd.extend(["--dedup_method", dedup_method])
            cmd.extend(["--dedup_sens", str(settings['dedup_sens'])])
        
        if settings['segment']:
            cmd.append("--segment")
            segment_method = settings['segment_method']
            if segment_method.startswith("Custom:"):
                cmd.extend(["--segment_method", self.custom_models['segment']['path']])
            else:
                cmd.extend(["--segment_method", segment_method])
        
        if settings['depth']:
            cmd.append("--depth")
            depth_method = settings['depth_method']
            if depth_method.startswith("Custom:"):
                cmd.extend(["--depth_method", self.custom_models['depth']['path']])
            else:
                cmd.extend(["--depth_method", depth_method])
            cmd.extend(["--depth_quality", settings['depth_quality']])
        
        cmd.extend(["--encode_method", settings['encode_method']])
        cmd.extend(["--bit_depth", settings['bit_depth']])
        
        if settings['resize']:
            cmd.append("--resize")
            cmd.extend(["--resize_factor", str(settings['resize_factor'])])
        
        cmd.extend(["--half", str(settings['half']).lower()])
        cmd.extend(["--decode_method", settings['decode_method']])
        cmd.extend(["--compile_mode", settings['compile_mode']])

        if settings.get('tile_size_enabled'):
            cmd.append("--tile_rendering")
            cmd.extend(["--tile_size", str(max(128, int(settings.get('tile_size', 128))))])
        
        if settings['static']:
            cmd.append("--static")
        
        return cmd
    
    def get_youtube_qualities(self, cmd):
        env = os.environ.copy()
        env['YTDLP_LIST_QUALITIES_ONLY'] = '1'
        env['PYTHONUNBUFFERED'] = '1'
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            output, _ = process.communicate(timeout=30)
            
            for line in output.split('\n'):
                if '[QUALITIES]' in line:
                    json_str = line.split('[QUALITIES]', 1)[1]
                    return json.loads(json_str)
            
            return None
        except Exception as e:
            print(f"Error getting qualities: {e}")
            return None
    
    def on_start_clicked(self):
        if getattr(self, 'active_content_name', None) == t('image_upscale'):
            self.start_image_upscale()
            return

        self.start_processing()

    def start_processing(self):
        try:
            cmd = self.build_command()
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        # Reset preview-run state (this is a full run unless the preview button started it).
        self._is_segment_preview_run = False
        self._segment_preview_output_path = None
        
        youtube_url = self.youtube_url.text().strip() if self._is_youtube_tab_active() else ""
        selected_quality_index = None
        
        if youtube_url:
            self.loading_overlay.raise_()
            self.loading_overlay.show()
            QApplication.processEvents()
            
            quality_data = self.get_youtube_qualities(cmd)
            
            self.loading_overlay.hide()
            
            if quality_data and 'qualities' in quality_data:
                dialog = QualitySelectionDialog(
                    quality_data.get('title', 'Unknown'), 
                    quality_data['qualities'], 
                    self
                )
                result = dialog.exec_()
                
                if result == QDialog.Accepted:
                    selected_quality_index = dialog.get_selected_index()
                else:
                    self.status_label.setText(t('idle'))
                    self.status_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #cdd6f4;")
                    return
        
        self.stop_requested = False
        # Reset progress UI at the start of each run (downloads can drive this first).
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_details_label.setText("")

        self.log_text.clear()
        self.log_text.append(f"{t('starting_processing')}\n")
        self.log_text.append(f"{t('command')} " + " ".join(cmd) + "\n")

        # NVDEC backend mode: allow forcing compat path for stability.
        if (
            hasattr(self, 'nvdec_compat_check')
            and self.decode_method.currentText() == 'nvdec'
            and self.nvdec_compat_check.isChecked()
        ):
            os.environ['TAS_NVDEC_MODE'] = 'compat'
        else:
            os.environ.pop('TAS_NVDEC_MODE', None)
        
        self.process_thread = ProcessThread(cmd, selected_quality_index)
        self.process_thread.progress_update.connect(self.update_progress)
        self.process_thread.finished.connect(self.processing_finished)
        self.process_thread.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_action_button_states(start_active=True, stop_active=False)
        self.status_label.setText(t('processing'))
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #f9e2af;")

        if hasattr(self, 'live_preview_check'):
            self.live_preview_check.setEnabled(False)

        if "--preview" in cmd:
            self.preview_timer.start(1000)
            self.preview_label.setText(t('loading_preview'))
        else:
            self.preview_timer.stop()
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText(t('live_preview_disabled'))
    
    def stop_processing(self):
        if self.process_thread:
            self.stop_requested = True
            self.process_thread.stop()
            self.stop_btn.setEnabled(False)
            self._set_action_button_states(start_active=False, stop_active=True)
            self.status_label.setText(t('stopping'))
            self.status_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #fab387;")

    def _set_action_button_states(self, start_active: bool, stop_active: bool):
        """Visually mark Start/Stop buttons as active by toggling a QSS property."""
        self.start_btn.setProperty('active', bool(start_active))
        self.stop_btn.setProperty('active', bool(stop_active))

        for btn in (self.start_btn, self.stop_btn):
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.update()
    
    def update_progress(self, progress, message, progress_data):
        overwrite = bool(progress_data.get('overwrite')) if progress_data else False
        clean_message = (message or "").strip()
        is_internal_progress_line = '[PROGRESS]' in clean_message

        # Drive the progress bar with the most granular signal we have:
        # - download_percent during downloads
        # - frame counts during processing (smooth)
        # - fallback to percent
        if progress_data:
            if progress_data.get('download_indeterminate'):
                # Busy/indeterminate download (unknown total size)
                self.progress_bar.setRange(0, 0)
                self.progress_bar.setFormat("İndiriliyor...")
            elif 'download_percent' in progress_data:
                dl = int(progress_data['download_percent'])
                if self.progress_bar.maximum() != 100:
                    self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(dl)
                self.progress_bar.setFormat("İndiriliyor: %p%")
            elif (
                'current_frame' in progress_data
                and 'total_frames' in progress_data
                and progress_data.get('total_frames')
            ):
                total = int(progress_data['total_frames'])
                current = int(progress_data.get('current_frame', 0))
                if total > 0:
                    if self.progress_bar.maximum() != total:
                        self.progress_bar.setRange(0, total)
                    current = min(current, total)
                    self.progress_bar.setValue(current)
                    self.progress_bar.setFormat("%v/%m (%p%)")
            elif 'progress_percent' in progress_data:
                pct = int(progress_data['progress_percent'])
                if pct > 0:
                    if self.progress_bar.maximum() != 100:
                        self.progress_bar.setRange(0, 100)
                    self.progress_bar.setValue(pct)
                    self.progress_bar.setFormat("%p%")
        elif progress > 0:
            if self.progress_bar.maximum() != 100:
                self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(progress)

        details_parts = []
        if progress_data:
            if 'current_frame' in progress_data and 'total_frames' in progress_data:
                details_parts.append(f"{t('frames')} {progress_data['current_frame']}/{progress_data['total_frames']}")
            if 'fps' in progress_data:
                details_parts.append(f"{t('fps')} {progress_data['fps']:.2f}")
            if 'elapsed' in progress_data:
                details_parts.append(f"{t('elapsed')} {progress_data['elapsed']}")
            if 'eta' in progress_data:
                details_parts.append(f"{t('eta')} {progress_data['eta']}")
        
        if overwrite and clean_message and not is_internal_progress_line:
            # Carriage-return progress (downloads / engine build / rich progress): show as live single-line status.
            self.progress_details_label.setText(clean_message)
        elif details_parts:
            self.progress_details_label.setText(" • ".join(details_parts))
        
        if clean_message and len(clean_message) > 5 and '[PROGRESS]' not in clean_message and not overwrite:
            self.log_text.append(clean_message)
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def processing_finished(self, success, message):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.preview_timer.stop()

        if hasattr(self, 'live_preview_check'):
            self.live_preview_check.setEnabled(True)

        preview_output = None
        if getattr(self, '_is_segment_preview_run', False):
            preview_output = getattr(self, '_segment_preview_output_path', None)
        self._is_segment_preview_run = False
        self._segment_preview_output_path = None

        if getattr(self, 'stop_requested', False):
            # User cancelled: reset progress UI and return to idle state.
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("%p%")
            self.progress_details_label.setText("")
            self.status_label.setText(t('idle'))
            self.status_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #cdd6f4;")
            self._set_action_button_states(start_active=False, stop_active=True)
            self.stop_requested = False
            self._update_segment_preview_slider_range()
            return
        
        if success:
            self.status_label.setText(f"✓ {t('complete')}")
            self.status_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #a6e3a1;")
            self._set_action_button_states(start_active=False, stop_active=True)
            QMessageBox.information(self, t('success'), message)

            # If this was a preview-segment run, open the generated clip.
            if preview_output and os.path.exists(preview_output):
                try:
                    if os.name == 'nt':
                        os.startfile(preview_output)
                    else:
                        subprocess.Popen(['xdg-open', preview_output])
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        t('error'),
                        f"{t('preview_open_failed')}\n{preview_output}\n{str(e)}",
                    )

            # After a successful run, reset the progress UI back to idle.
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("%p%")
            self.progress_details_label.setText("")
            self.status_label.setText(t('idle'))
            self.status_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #cdd6f4;")
        else:
            self.status_label.setText(f"✗ {t('failed')}")
            self.status_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: #f38ba8;")
            self._set_action_button_states(start_active=False, stop_active=True)
            QMessageBox.critical(self, t('error'), message)
        
        self.log_text.append(f"\n{message}")
        self._update_segment_preview_slider_range()

    def show_backend_error_dialog(self, title, message):
        """Show error dialog for backend failures (TensorRT, PyTorch, etc.)"""
        QMessageBox.critical(self, title, message)

    def open_preview_fullscreen(self):
        if self._fullscreen_preview_dialog is None:
            self._fullscreen_preview_dialog = FullscreenPreviewDialog(self)

        self._fullscreen_preview_dialog.set_pixmap(self._last_preview_pixmap)
        self._fullscreen_preview_dialog.showFullScreen()
        self._fullscreen_preview_dialog.raise_()
        self._fullscreen_preview_dialog.activateWindow()

    def _open_pixmap_fullscreen(self, pixmap: QPixmap):
        if pixmap is None or pixmap.isNull():
            return

        if self._fullscreen_preview_dialog is None:
            self._fullscreen_preview_dialog = FullscreenPreviewDialog(self)

        self._fullscreen_preview_dialog.set_pixmap(pixmap)
        self._fullscreen_preview_dialog.showFullScreen()
        self._fullscreen_preview_dialog.raise_()
        self._fullscreen_preview_dialog.activateWindow()

    def open_image_input_fullscreen(self):
        path = self.image_input_path.text().strip() if hasattr(self, 'image_input_path') else ''
        pixmap = QPixmap(path) if path and os.path.exists(path) else None
        if pixmap is None or pixmap.isNull():
            pixmap = self.image_input_preview_label.pixmap() if hasattr(self, 'image_input_preview_label') else None
        self._open_pixmap_fullscreen(pixmap)

    def open_image_output_fullscreen(self):
        path = self.image_output_path.text().strip() if hasattr(self, 'image_output_path') else ''
        pixmap = QPixmap(path) if path and os.path.exists(path) else None
        if pixmap is None or pixmap.isNull():
            pixmap = self.image_output_preview_label.pixmap() if hasattr(self, 'image_output_preview_label') else None
        self._open_pixmap_fullscreen(pixmap)

    def _set_preview_pixmap(self, pixmap):
        self._last_preview_pixmap = pixmap

        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled_pixmap)

        if (
            self._fullscreen_preview_dialog is not None
            and self._fullscreen_preview_dialog.isVisible()
        ):
            self._fullscreen_preview_dialog.set_pixmap(pixmap)

    def _on_live_preview_checkbox_changed(self, *_):
        # While processing, the running command owns whether preview is actually available.
        if getattr(self, 'process_thread', None) is not None and self.process_thread.isRunning():
            return

        if not hasattr(self, 'preview_label') or not hasattr(self, 'live_preview_check'):
            return

        if self.live_preview_check.isChecked():
            if self.preview_label.pixmap() is None or self.preview_label.pixmap().isNull():
                self.preview_label.setText(t('live_preview_ready'))
        else:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText(t('live_preview_disabled'))
    
    def refresh_preview(self):
        preview_path = getattr(self, 'preview_file_path', None)
        if preview_path and os.path.exists(preview_path):
            pixmap = QPixmap(preview_path)
            if not pixmap.isNull():
                self._set_preview_pixmap(pixmap)
                return
    
    def on_preview_loaded(self, reply):
        if reply.error() == reply.NoError:
            image_data = reply.readAll()
            pixmap = QPixmap()
            if pixmap.loadFromData(image_data):
                self._set_preview_pixmap(pixmap)
        reply.deleteLater()


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(_GUI_ICON_PATH))
    app.setStyle('Fusion')
    
    window = TheAnimeScripterGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
