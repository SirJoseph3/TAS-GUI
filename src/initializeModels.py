"""
Model Initialization and Processing Functions

This module handles the initialization and execution of various AI models
for video processing operations including object detection, auto-clipping,
segmentation, depth estimation, and the main processing pipeline.
"""

import logging
import math
import torch


class RestoreChain:
    def __init__(
        self,
        restore_processes: list,
        restore_labels: list[str] | None = None,
    ):
        self.restore_processes = restore_processes
        self.restore_labels = restore_labels
        self._logged = False
        logging.info(f"Initialized restore chain with {len(restore_processes)} models")

    @torch.inference_mode()
    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        if not self._logged:
            try:
                from src.utils.logAndPrint import logAndPrint

                labels = self.restore_labels or [
                    p.__class__.__name__ for p in self.restore_processes
                ]
                logAndPrint(
                    message=f"Applying restoration chain: {' -> '.join(labels)}",
                    colorFunc="cyan",
                )
            except Exception:
                logging.exception("Failed to log restore chain")
            self._logged = True
        for restore_process in self.restore_processes:
            frame = restore_process(frame)
        return frame


def estimate_auto_tilesize(width: int, height: int, scale: int, half: bool) -> int:
    try:
        if checker.cudaAvailable and torch.cuda.is_available():
            free_mem, _ = torch.cuda.mem_get_info()
            bytes_per_px = 2 if half else 4
            # budget ~25% of free mem for one tile (3 channels, scale^2)
            budget = int(free_mem * 0.25)
            tile_area = max(1, budget // (bytes_per_px * 3 * (scale ** 2)))
            tile_size = int(math.sqrt(tile_area)) - 64  # leave room for padding/overhead
            tile_size = max(128, min(512, (tile_size // 64) * 64))
        else:
            tile_size = 256 if width * height > 1280 * 720 else 384
    except Exception:
        tile_size = 256 if width * height > 1280 * 720 else 384

    if scale >= 4:
        tile_size = min(tile_size, 128)

    tile_size = max(64, min(tile_size, min(width, height)))
    return tile_size


def objectDetection(self):
    """
    Initialize and execute object detection processing.

    Args:
        self: VideoProcessor instance containing processing parameters
    """
    if "directml" in self.objDetectMethod or "openvino" in self.objDetectMethod:
        from src.objectDetection.objectDetection import ObjectDetectionDML

        ObjectDetectionDML(
            self.input,
            self.output,
            self.width,
            self.height,
            self.fps,
            self.inpoint,
            self.outpoint,
            self.encodeMethod,
            self.customEncoder,
            self.benchmark,
            self.half,
            self.objDetectMethod,
            self.totalFrames,
            getattr(self, "objDetectDisableAnnotations", False),
        )
    elif "tensorrt" in self.objDetectMethod:
        from src.objectDetection.objectDetection import ObjectDetectionTensorRT

        ObjectDetectionTensorRT(
            self.input,
            self.output,
            self.width,
            self.height,
            self.fps,
            self.inpoint,
            self.outpoint,
            self.encodeMethod,
            self.customEncoder,
            self.benchmark,
            self.half,
            self.objDetectMethod,
            self.totalFrames,
        )
    else:
        from src.objectDetection.objectDetection import ObjectDetection

        ObjectDetection(
            self.input,
            self.output,
            self.width,
            self.height,
            self.fps,
            self.inpoint,
            self.outpoint,
            self.encodeMethod,
            self.customEncoder,
            self.benchmark,
            self.totalFrames,
            self.half,
        )


def autoClip(self):
    """
    Initialize and execute automatic scene detection and clipping.

    Args:
        self: VideoProcessor instance containing processing parameters
    """
    from src.autoclip.autoclip import AutoClip

    AutoClip(
        self.input,
        self.autoclipSens,
        self.inpoint,
        self.outpoint,
    )


def segment(self):
    """
    Initialize and execute video segmentation processing.

    Args:
        self: VideoProcessor instance containing processing parameters

    Raises:
        NotImplementedError: If cartoon segmentation method is selected
    """
    if self.segmentMethod == "anime":
        from src.segment.animeSegment import AnimeSegment

        AnimeSegment(
            self.input,
            self.output,
            self.width,
            self.height,
            self.outputFPS,
            self.inpoint,
            self.outpoint,
            self.encodeMethod,
            self.customEncoder,
            self.benchmark,
            self.totalFrames,
        )
    elif self.segmentMethod == "anime-tensorrt":
        from src.segment.animeSegment import AnimeSegmentTensorRT

        AnimeSegmentTensorRT(
            self.input,
            self.output,
            self.width,
            self.height,
            self.outputFPS,
            self.inpoint,
            self.outpoint,
            self.encodeMethod,
            self.customEncoder,
            self.benchmark,
            self.totalFrames,
        )
    elif self.segmentMethod == "anime-directml":
        from src.segment.animeSegment import AnimeSegmentDirectML

        AnimeSegmentDirectML(
            self.input,
            self.output,
            self.width,
            self.height,
            self.outputFPS,
            self.inpoint,
            self.outpoint,
            self.encodeMethod,
            self.customEncoder,
            self.benchmark,
            self.totalFrames,
        )
    elif self.segmentMethod == "cartoon":
        raise NotImplementedError("Cartoon segment is not implemented yet")


def depth(self):
    """
    Initialize and execute depth estimation processing.

    Args:
        self: VideoProcessor instance containing processing parameters
    """
    match self.depthMethod:
        case (
            "small_v2"
            | "base_v2"
            | "large_v2"
            | "giant_v2"
            | "distill_small_v2"
            | "distill_base_v2"
            | "distill_large_v2"
        ):
            from src.depth.depth import DepthCuda

            DepthCuda(
                self.input,
                self.output,
                self.width,
                self.height,
                self.fps,
                self.half,
                self.inpoint,
                self.outpoint,
                self.encodeMethod,
                self.depthMethod,
                self.customEncoder,
                self.benchmark,
                self.totalFrames,
                self.bitDepth,
                self.depthQuality,
                compileMode=self.compileMode,
            )

        case (
            "small_v2-tensorrt"
            | "base_v2-tensorrt"
            | "large_v2-tensorrt"
            | "distill_small_v2-tensorrt"
            | "distill_base_v2-tensorrt"
            | "distill_large_v2-tensorrt"
        ):
            from src.depth.depth import DepthTensorRTV2

            DepthTensorRTV2(
                self.input,
                self.output,
                self.width,
                self.height,
                self.fps,
                self.half,
                self.inpoint,
                self.outpoint,
                self.encodeMethod,
                self.depthMethod,
                self.customEncoder,
                self.benchmark,
                self.totalFrames,
                self.bitDepth,
                self.depthQuality,
            )

        case (
            "small_v2-directml"
            | "base_v2-directml"
            | "large_v2-directml"
            | "distill_small_v2-directml"
            | "distill_base_v2-directml"
            | "distill_large_v2-directml"
        ):
            from src.depth.depth import DepthDirectMLV2

            DepthDirectMLV2(
                self.input,
                self.output,
                self.width,
                self.height,
                self.fps,
                self.half,
                self.inpoint,
                self.outpoint,
                self.encodeMethod,
                self.depthMethod,
                self.customEncoder,
                self.benchmark,
                self.totalFrames,
                self.bitDepth,
                self.depthQuality,
            )
        case (
            "og_small_v2"
            | "og_base_v2"
            | "og_large_v2"
            | "og_giant_v2"
            | "og_distill_small_v2"
            | "og_distill_base_v2"
            | "og_distill_large_v2"
        ):
            from src.depth.depth import OGDepthV2CUDA

            OGDepthV2CUDA(
                self.input,
                self.output,
                self.width,
                self.height,
                self.fps,
                self.half,
                self.inpoint,
                self.outpoint,
                self.encodeMethod,
                self.depthMethod,
                self.customEncoder,
                self.benchmark,
                self.totalFrames,
                self.bitDepth,
                self.depthQuality,
                compileMode=self.compileMode,
            )

        case "og_video_small_v2":
            from src.depth.depth import VideoDepthAnythingCUDA

            VideoDepthAnythingCUDA(
                self.input,
                self.output,
                self.width,
                self.height,
                self.fps,
                self.half,
                self.inpoint,
                self.outpoint,
                self.encodeMethod,
                self.depthMethod,
                self.customEncoder,
                self.benchmark,
                self.totalFrames,
                self.bitDepth,
                self.depthQuality,
                compileMode=self.compileMode,
            )

        case (
            "og_small_v2-tensorrt"
            | "og_base_v2-tensorrt"
            | "og_large_v2-tensorrt"
            | "og_distill_small_v2-tensorrt"
            | "og_distill_base_v2-tensorrt"
            | "og_distill_large_v2-tensorrt"
        ):
            from src.depth.depth import OGDepthV2TensorRT

            OGDepthV2TensorRT(
                self.input,
                self.output,
                self.width,
                self.height,
                self.fps,
                self.half,
                self.inpoint,
                self.outpoint,
                self.encodeMethod,
                self.depthMethod,
                self.customEncoder,
                self.benchmark,
                self.totalFrames,
                self.bitDepth,
                self.depthQuality,
            )

        case (
            "og_small_v2-directml"
            | "og_base_v2-directml"
            | "og_large_v2-directml"
        ):
            from src.depth.depth import OGDepthV2DirectML

            OGDepthV2DirectML(
                self.input,
                self.output,
                self.width,
                self.height,
                self.fps,
                self.half,
                self.inpoint,
                self.outpoint,
                self.encodeMethod,
                self.depthMethod,
                self.customEncoder,
                self.benchmark,
                self.totalFrames,
                self.bitDepth,
                self.depthQuality,
            )

        case "small_v3" | "base_v3" | "large_v3" | "giant_v3":
            from src.depth.depth import OGDepthV3CUDA

            OGDepthV3CUDA(
                self.input,
                self.output,
                self.width,
                self.height,
                self.fps,
                self.half,
                self.inpoint,
                self.outpoint,
                self.encodeMethod,
                self.depthMethod,
                self.customEncoder,
                self.benchmark,
                self.totalFrames,
                self.bitDepth,
                self.depthQuality,
                compileMode=self.compileMode,
            )

        case (
            "small_v3-directml"
            | "base_v3-directml"
            | "large_v3-directml"
            | "giant_v3-directml"
        ):
            from src.depth.depth import DepthDirectMLV3

            DepthDirectMLV3(
                self.input,
                self.output,
                self.width,
                self.height,
                self.fps,
                self.half,
                self.inpoint,
                self.outpoint,
                self.encodeMethod,
                self.depthMethod,
                self.customEncoder,
                self.benchmark,
                self.totalFrames,
                self.bitDepth,
                self.depthQuality,
            )

        case (
            "small_v3-tensorrt"
            | "base_v3-tensorrt"
            | "large_v3-tensorrt"
            | "giant_v3-tensorrt"
        ):
            from src.depth.depth import DepthTensorRTV3

            DepthTensorRTV3(
                self.input,
                self.output,
                self.width,
                self.height,
                self.fps,
                self.half,
                self.inpoint,
                self.outpoint,
                self.encodeMethod,
                self.depthMethod,
                self.customEncoder,
                self.benchmark,
                self.totalFrames,
                self.bitDepth,
                self.depthQuality,
            )


def initializeModels(self):
    """
    Initialize all AI models for the video processing pipeline.

    Args:
        self: VideoProcessor instance containing processing parameters

    Returns:
        tuple: Contains output dimensions and initialized processing functions
            - outputWidth (int): Final output video width
            - outputHeight (int): Final output video height
            - upscaleProcess: Upscaling model function or None
            - interpolateProcess: Interpolation model function or None
            - restoreProcess: Restoration model function or None
            - dedupProcess: Deduplication function or None
            - scenechangeProcess: Scene change detection function or None
    """
    from src.utils.logAndPrint import logAndPrint
    
    outputWidth = self.width
    outputHeight = self.height
    upscaleProcess = None
    interpolateProcess = None
    restoreProcess = None
    dedupProcess = None
    scenechangeProcess = None

    if self.upscale:
        from src.unifiedUpscale import UniversalPytorch

        # Tile rendering is opt-in (GUI checkbox / CLI flag).
        tile_enabled = bool(getattr(self, "tileRendering", False))
        user_tilesize = 0
        if tile_enabled:
            try:
                user_tilesize = int(getattr(self, "tileSize", 128) or 128)
            except Exception:
                user_tilesize = 128
            user_tilesize = max(128, user_tilesize)

        outputWidth *= self.upscaleFactor
        outputHeight *= self.upscaleFactor
        logging.info(f"Upscaling to {outputWidth}x{outputHeight}")
        match self.upscaleMethod:
            case (
                "shufflecugan"
                | "compact"
                | "ultracompact"
                | "superultracompact"
                | "span"
                | "open-proteus"
                | "aniscale2"
                | "shufflespan"
                | "rtmosr"
                | "saryn"
                | "fallin_soft"
                | "fallin_strong"
                | "gauss"
            ):
                try:
                    upscaleProcess = UniversalPytorch(
                        self.upscaleMethod,
                        self.upscaleFactor,
                        self.half,
                        self.width,
                        self.height,
                        self.customModel,
                        self.compileMode,
                        tilesize=user_tilesize if tile_enabled else 0,
                    )
                except torch.OutOfMemoryError:
                    # Do not auto-enable tile rendering on OOM when the user didn't opt in.
                    # Instead, surface an actionable message to the UI.
                    logAndPrint("CUDA OOM hatası!", "yellow")
                    error_msg = (
                        "OOM hatası gibi görünüyor. Performance tab'ından tile_size açabilir "
                        "veya 2x model deneyebilir ya da daha hafif model deneyebilirsiniz."
                    )
                    logAndPrint(error_msg, "red")
                    raise RuntimeError(error_msg)

            case (
                "compact-directml"
                | "ultracompact-directml"
                | "superultracompact-directml"
                | "span-directml"
                | "open-proteus-directml"
                | "aniscale2-directml"
                | "shufflespan-directml"
                | "rtmosr-directml"
                | "saryn-directml"
                | "fallin_soft-directml"
                | "fallin_strong-directml"
                | "gauss-directml"
                | "compact-openvino"
                | "ultracompact-openvino"
                | "superultracompact-openvino"
                | "span-openvino"
                | "open-proteus-openvino"
                | "aniscale2-openvino"
                | "shufflespan-openvino"
                | "rtmosr-openvino"
                | "saryn-openvino"
                | "fallin_soft-openvino"
                | "fallin_strong-openvino"
                | "gauss-openvino"
            ):
                from src.unifiedUpscale import UniversalDirectML

                upscaleProcess = UniversalDirectML(
                    self.upscaleMethod,
                    self.upscaleFactor,
                    self.half,
                    self.width,
                    self.height,
                    self.customModel,
                )

            case "shufflecugan-ncnn" | "span-ncnn":
                from src.unifiedUpscale import UniversalNCNN

                upscaleProcess = UniversalNCNN(
                    self.upscaleMethod,
                    self.upscaleFactor,
                )

            case (
                "shufflecugan-tensorrt"
                | "compact-tensorrt"
                | "ultracompact-tensorrt"
                | "superultracompact-tensorrt"
                | "span-tensorrt"
                | "open-proteus-tensorrt"
                | "aniscale2-tensorrt"
                | "shufflespan-tensorrt"
                | "rtmosr-tensorrt"
                | "saryn-tensorrt"
                | "fallin_soft-tensorrt"
                | "fallin_strong-tensorrt"
                | "gauss-tensorrt"
            ):
                from src.unifiedUpscale import UniversalTensorRT

                upscaleProcess = UniversalTensorRT(
                    self.upscaleMethod,
                    self.upscaleFactor,
                    self.half,
                    self.width,
                    self.height,
                    self.customModel,
                    self.forceStatic,
                )

            case "animesr":
                from src.unifiedUpscale import AnimeSR

                upscaleProcess = AnimeSR(
                    2,
                    self.half,
                    self.width,
                    self.height,
                    self.compileMode,
                )

            case "animesr-tensorrt":
                from src.unifiedUpscale import AnimeSRTensorRT

                upscaleProcess = AnimeSRTensorRT(
                    2,
                    self.half,
                    self.width,
                    self.height,
                )
            
            case _:
                import os
                from pathlib import Path
                
                if os.path.exists(self.upscaleMethod):
                    logAndPrint(
                        message=f"Custom model detected: {self.upscaleMethod}",
                        colorFunc="cyan",
                    )
                    
                    original_upscale_path = self.upscaleMethod
                    file_ext = Path(original_upscale_path).suffix.lower()
                    tensorrt_model_path = original_upscale_path
                    backend = self.customUpscaleBackend
                    
                    if backend == "default":
                        if file_ext == ".onnx":
                            backend = "tensorrt"
                        elif file_ext in [".pth", ".pt"]:
                            backend = "cuda"
                        else:
                            raise ValueError(
                                f"Unsupported custom model file type: {file_ext}. "
                                f"Supported types: .onnx, .pth, .pt"
                            )
                    
                    if backend == "tensorrt":
                        if file_ext in [".pth", ".pt"]:
                            from src.utils.onnxConverter import pthToOnnx
                            
                            precision = "fp16" if self.half else "fp32"
                            expectedOnnxPath = os.path.splitext(original_upscale_path)[0] + f"_{precision}_op20_slim.onnx"
                            fallbackOnnxPath = os.path.splitext(original_upscale_path)[0] + f"_{precision}_op20.onnx"
                            # Some architectures intentionally keep ONNX in FP32 even when "fp16" is
                            # requested (e.g. OmniSR). In that case the converter produces *_op20(_slim).onnx
                            # without the _fp16_ prefix. Reuse it instead of reconverting every run.
                            genericSlimPath = os.path.splitext(original_upscale_path)[0] + "_op20_slim.onnx"
                            genericPath = os.path.splitext(original_upscale_path)[0] + "_op20.onnx"
                            
                            if os.path.exists(expectedOnnxPath):
                                onnxPath = expectedOnnxPath
                                logAndPrint(
                                    message=f"Found existing ONNX model: {onnxPath}",
                                    colorFunc="green",
                                )
                            elif os.path.exists(fallbackOnnxPath):
                                onnxPath = fallbackOnnxPath
                                logAndPrint(
                                    message=f"Found existing ONNX model: {onnxPath}",
                                    colorFunc="green",
                                )
                            elif os.path.exists(genericSlimPath):
                                onnxPath = genericSlimPath
                                logAndPrint(
                                    message=f"Found existing ONNX model: {onnxPath}",
                                    colorFunc="green",
                                )
                            elif os.path.exists(genericPath):
                                onnxPath = genericPath
                                logAndPrint(
                                    message=f"Found existing ONNX model: {onnxPath}",
                                    colorFunc="green",
                                )
                            else:
                                logAndPrint(
                                    message=f"Converting PTH model to ONNX for TensorRT: {original_upscale_path}",
                                    colorFunc="yellow",
                                )
                                try:
                                    onnxPath = pthToOnnx(
                                        pthPath=original_upscale_path,
                                        inputShape=(1, 3, self.height, self.width),
                                        precision=precision,
                                        opset=20,
                                        slim=True,
                                    )
                                    logAndPrint(
                                        message=f"Conversion completed, output path: {onnxPath}",
                                        colorFunc="cyan",
                                    )
                                except Exception as e:
                                    logAndPrint(
                                        message=f"Failed to convert PTH to ONNX: {e}",
                                        colorFunc="red",
                                    )
                                    import traceback
                                    traceback.print_exc()
                                    raise RuntimeError(
                                        f"PTH to ONNX conversion failed: {e}"
                                    )
                            
                            if onnxPath and os.path.exists(onnxPath):
                                tensorrt_model_path = onnxPath
                                logAndPrint(
                                    message=f"Using converted ONNX model: {onnxPath}",
                                    colorFunc="green",
                                )
                            else:
                                raise RuntimeError(
                                    f"ONNX file not found after conversion: {onnxPath}"
                                )
                        
                        from src.unifiedUpscale import UniversalTensorRT
                        
                        logAndPrint(
                            message="Using TensorRT backend for custom model",
                            colorFunc="cyan",
                        )
                        
                        try:
                            upscaleProcess = UniversalTensorRT(
                                upscaleMethod="custom-tensorrt",
                                upscaleFactor=self.upscaleFactor,
                                half=self.half,
                                width=self.width,
                                height=self.height,
                                customModel=tensorrt_model_path,
                                forceStatic=self.forceStatic,
                            )
                        except RuntimeError as e:
                            error_msg = str(e)
                            if "TensorRT engine" in error_msg or "OutOfMemory" in error_msg:
                                logAndPrint(
                                    "TensorRT engine build hatası! PyTorch CUDA backend'e geçiliyor...",
                                    "yellow",
                                )
                                
                                from src.spandrel import ImageModelDescriptor, ModelLoader
                                original_model_path = original_upscale_path
                                if file_ext == ".onnx":
                                    pth_path = original_model_path.replace("_op17_slim.onnx", ".pth").replace("_op17.onnx", ".pth").replace(".onnx", ".pth")
                                    if not os.path.exists(pth_path):
                                        pth_path = original_model_path.replace("_fp16_op20_slim.onnx", ".pth").replace("_fp32_op20_slim.onnx", ".pth")
                                    if os.path.exists(pth_path):
                                        self.upscaleMethod = pth_path
                                        logAndPrint(f"Using original PTH model: {pth_path}", "cyan")
                                    else:
                                        raise RuntimeError("TensorRT engine build failed and original PTH model not found")
                                
                                temp_model = torch.load(self.upscaleMethod, map_location="cpu", weights_only=False)
                                if isinstance(temp_model, dict):
                                    temp_model = ModelLoader().load_from_state_dict(temp_model)
                                
                                model_arch = None
                                if isinstance(temp_model, ImageModelDescriptor):
                                    try:
                                        model_arch = temp_model.architecture.id.upper() if hasattr(temp_model, 'architecture') else None
                                    except:
                                        pass
                                
                                del temp_model
                                torch.cuda.empty_cache()
                                
                                model_size_mb = os.path.getsize(self.upscaleMethod) / (1024 * 1024)
                                tilesize = user_tilesize if tile_enabled else 0
                                if tile_enabled:
                                    logAndPrint(
                                        f"Custom model ({model_arch or 'Unknown'}, {model_size_mb:.1f}MB) - Tile rendering enabled: tile_size={tilesize}",
                                        "yellow",
                                    )
                                
                                try:
                                    upscaleProcess = UniversalPytorch(
                                        upscaleMethod="custom",
                                        upscaleFactor=self.upscaleFactor,
                                        half=self.half,
                                        width=self.width,
                                        height=self.height,
                                        customModel=self.upscaleMethod,
                                        compileMode=self.compileMode,
                                        tilesize=tilesize,
                                    )
                                except torch.OutOfMemoryError:
                                    logAndPrint("CUDA OOM hatası!", "yellow")
                                    error_msg = (
                                        "OOM gibi görünüyor. Performance tab'ından Tile Rendering'i açıp "
                                        "tile_size deneyin ( örn. 256/384; gerekirse 128). "
                                        "Alternatif: --half false veya daha hafif model."
                                    )
                                    logAndPrint(error_msg, "red")
                                    raise RuntimeError(error_msg)
                            else:
                                raise
                    
                    elif backend == "cuda":
                        if file_ext not in [".pth", ".pt"]:
                            raise ValueError(
                                f"CUDA backend requires .pth or .pt models, got: {file_ext}"
                            )
                        
                        logAndPrint(
                            message="Using PyTorch CUDA backend for custom model",
                            colorFunc="cyan",
                        )
                        
                        from src.spandrel import ImageModelDescriptor, ModelLoader
                        temp_model = torch.load(self.upscaleMethod, map_location="cpu", weights_only=False)
                        if isinstance(temp_model, dict):
                            temp_model = ModelLoader().load_from_state_dict(temp_model)
                        
                        model_arch = None
                        if isinstance(temp_model, ImageModelDescriptor):
                            try:
                                model_arch = temp_model.architecture.id.upper() if hasattr(temp_model, 'architecture') else None
                            except:
                                pass
                        
                        del temp_model
                        torch.cuda.empty_cache()
                        
                        model_size_mb = os.path.getsize(self.upscaleMethod) / (1024 * 1024)
                        
                        tilesize = user_tilesize if tile_enabled else 0
                        if tile_enabled:
                            logAndPrint(
                                f"Custom model ({model_arch or 'Unknown'}, {model_size_mb:.1f}MB) - Tile rendering enabled: tile_size={tilesize}",
                                "yellow",
                            )
                        
                        try:
                            upscaleProcess = UniversalPytorch(
                                upscaleMethod="custom",
                                upscaleFactor=self.upscaleFactor,
                                half=self.half,
                                width=self.width,
                                height=self.height,
                                customModel=self.upscaleMethod,
                                compileMode=self.compileMode,
                                tilesize=tilesize,
                            )
                        except torch.OutOfMemoryError:
                            logAndPrint("CUDA OOM hatası!", "yellow")
                            error_msg = (
                                "OOM gibi görünüyor. Performance tab'ından Tile Rendering'i açıp "
                                "tile_size deneyin ( örn. 256/384; gerekirse 128). "
                                "Alternatif: --half false veya daha hafif model."
                            )
                            logAndPrint(error_msg, "red")
                            raise RuntimeError(error_msg)
                    
                    else:
                        raise ValueError(
                            f"Unsupported backend '{backend}' for custom upscale models. "
                            f"Supported: tensorrt, cuda"
                        )
                else:
                    raise ValueError(
                        f"Invalid upscale method or model file not found: {self.upscaleMethod}"
                    )
    
    if self.interpolate:
        logging.info(
            f"Interpolating from {format(self.fps, '.3f')}fps to {format(self.fps * self.interpolateFactor, '.3f')}fps"
        )
        match self.interpolateMethod:
            case (
                "rife"
                | "rife4.6"
                | "rife4.15-lite"
                | "rife4.16-lite"
                | "rife4.17"
                | "rife4.18"
                | "rife4.20"
                | "rife4.21"
                | "rife4.22"
                | "rife4.22-lite"
                | "rife4.25"
                | "rife4.25-lite"
                | "rife_elexor"
                | "rife4.25-heavy"
            ):
                from src.unifiedInterpolate import RifeCuda

                interpolateProcess = RifeCuda(
                    self.half,
                    self.width,
                    self.height,
                    self.interpolateMethod,
                    self.ensemble,
                    self.interpolateFactor,
                    self.dynamicScale,
                    self.staticStep,
                    compileMode=self.compileMode,
                )

            case (
                "rife-ncnn"
                | "rife4.6-ncnn"
                | "rife4.15-lite-ncnn"
                | "rife4.16-lite-ncnn"
                | "rife4.17-ncnn"
                | "rife4.18-ncnn"
                | "rife4.20-ncnn"
                | "rife4.21-ncnn"
                | "rife4.22-ncnn"
                | "rife4.22-lite-ncnn"
            ):
                from src.unifiedInterpolate import RifeNCNN

                interpolateProcess = RifeNCNN(
                    self.interpolateMethod,
                    self.ensemble,
                    self.width,
                    self.height,
                    self.half,
                    self.interpolateFactor,
                )

            case (
                "rife-tensorrt"
                | "rife4.6-tensorrt"
                | "rife4.15-tensorrt"
                | "rife4.15-lite-tensorrt"
                | "rife4.17-tensorrt"
                | "rife4.18-tensorrt"
                | "rife4.20-tensorrt"
                | "rife4.21-tensorrt"
                | "rife4.22-tensorrt"
                | "rife4.22-lite-tensorrt"
                | "rife4.25-tensorrt"
                | "rife4.25-lite-tensorrt"
                | "rife_elexor-tensorrt"
                | "rife4.25-heavy-tensorrt"
            ):
                from src.unifiedInterpolate import RifeTensorRT

                interpolateProcess = RifeTensorRT(
                    self.interpolateMethod,
                    self.interpolateFactor,
                    self.width,
                    self.height,
                    self.half,
                    self.ensemble,
                )

            case "gmfss":
                from src.gmfss.gmfss import GMFSS

                interpolateProcess = GMFSS(
                    int(self.interpolateFactor),
                    self.half,
                    outputWidth,
                    outputHeight,
                    self.ensemble,
                    compileMode=self.compileMode,
                )

            case "gmfss-tensorrt":
                from src.gmfss.gmfss import GMFSSTensorRT

                interpolateProcess = GMFSSTensorRT(
                    int(self.interpolateFactor),
                    outputWidth,
                    outputHeight,
                    self.half,
                    self.ensemble,
                )

            case "rife4.6-directml":
                from src.unifiedInterpolate import RifeDirectML

                interpolateProcess = RifeDirectML(
                    self.interpolateMethod,
                    self.interpolateFactor,
                    self.width,
                    self.height,
                    self.half,
                    self.ensemble,
                )

            case "distildrba" | "distildrba-lite":
                from src.unifiedInterpolate import DistilDRBACuda

                interpolateProcess = DistilDRBACuda(
                    self.half,
                    self.width,
                    self.height,
                    self.interpolateMethod,
                    interpolateFactor=self.interpolateFactor,
                    compileMode=self.compileMode,
                )

            case "distildrba-lite-tensorrt":
                from src.unifiedInterpolate import DistilDRBATensorRT

                interpolateProcess = DistilDRBATensorRT(
                    self.half,
                    self.width,
                    self.height,
                    self.interpolateMethod,
                    interpolateFactor=self.interpolateFactor,
                )
            
            case _:
                import os
                from pathlib import Path
                
                if os.path.exists(self.interpolateMethod):
                    logAndPrint(
                        message=f"Custom interpolation model detected: {self.interpolateMethod}",
                        colorFunc="cyan",
                    )
                    
                    file_ext = Path(self.interpolateMethod).suffix.lower()
                    backend = self.customInterpolateBackend
                    
                    if backend == "default":
                        if file_ext == ".onnx":
                            backend = "tensorrt"
                        elif file_ext in [".pth", ".pt"]:
                            backend = "cuda"
                        else:
                            raise ValueError(
                                f"Unsupported custom model file type: {file_ext}. "
                                f"Supported types: .onnx, .pth, .pt"
                            )
                    
                    if backend == "tensorrt":
                        if file_ext in [".pth", ".pt"]:
                            from src.utils.onnxConverter import pthToOnnx
                            
                            logAndPrint(
                                message=f"Converting PTH interpolation model to ONNX for TensorRT: {self.interpolateMethod}",
                                colorFunc="yellow",
                            )
                            
                            precision = "fp16" if self.half else "fp32"
                            expectedOnnxPath = os.path.splitext(self.interpolateMethod)[0] + f"_{precision}_op20_slim.onnx"
                            fallbackOnnxPath = os.path.splitext(self.interpolateMethod)[0] + f"_{precision}_op20.onnx"
                            
                            if os.path.exists(expectedOnnxPath):
                                onnxPath = expectedOnnxPath
                            elif os.path.exists(fallbackOnnxPath):
                                onnxPath = fallbackOnnxPath
                            else:
                                onnxPath = pthToOnnx(
                                    pthPath=self.interpolateMethod,
                                    inputShape=(1, 3, self.height, self.width),
                                    precision=precision,
                                    opset=20,
                                    slim=True,
                                )
                            
                            if os.path.exists(onnxPath):
                                self.interpolateMethod = onnxPath
                                logAndPrint(
                                    message=f"Using converted ONNX model: {onnxPath}",
                                    colorFunc="green",
                                )
                            else:
                                raise RuntimeError(
                                    f"Failed to convert PTH to ONNX: {self.interpolateMethod}"
                                )
                        
                        from src.unifiedInterpolate import RifeTensorRT
                        
                        logAndPrint(
                            message="Using TensorRT backend for custom interpolation model",
                            colorFunc="cyan",
                        )
                        
                        interpolateProcess = RifeTensorRT(
                            interpolateMethod="custom-tensorrt",
                            interpolateFactor=self.interpolateFactor,
                            width=self.width,
                            height=self.height,
                            half=self.half,
                            ensemble=self.ensemble,
                            customModel=self.interpolateMethod,
                        )
                    
                    elif backend == "cuda":
                        if file_ext not in [".pth", ".pt"]:
                            raise ValueError(
                                f"CUDA backend requires .pth or .pt models, got: {file_ext}"
                            )
                        
                        from src.unifiedInterpolate import RifeCuda
                        
                        logAndPrint(
                            message="Using PyTorch CUDA backend for custom interpolation model",
                            colorFunc="cyan",
                        )
                        
                        interpolateProcess = RifeCuda(
                            half=self.half,
                            width=self.width,
                            height=self.height,
                            interpolateMethod="custom",
                            ensemble=self.ensemble,
                            interpolateFactor=self.interpolateFactor,
                            dynamicScale=self.dynamicScale,
                            staticStep=self.staticStep,
                            customModel=self.interpolateMethod,
                            compileMode=self.compileMode,
                        )
                    
                    else:
                        raise ValueError(
                            f"Unsupported backend '{backend}' for custom interpolation models. "
                            f"Supported: tensorrt, cuda"
                        )
                else:
                    raise ValueError(
                        f"Invalid interpolation method or model file not found: {self.interpolateMethod}"
                    )

    if self.restore:
        restoreMethods = (
            self.restoreMethod
            if isinstance(self.restoreMethod, list)
            else [self.restoreMethod]
        )
        restoreProcesses = []
        restoreLabels = []

        for method in restoreMethods:
            match method:
                case (
                    "scunet"
                    | "dpir"
                    | "nafnet"
                    | "real-plksr"
                    | "anime1080fixer"
                    | "gater3"
                    | "deh264_real"
                    | "deh264_span"
                    | "hurrdeblur"
                ):
                    from src.unifiedRestore import UnifiedRestoreCuda

                    restoreProcesses.append(
                        UnifiedRestoreCuda(
                            method,
                            self.half,
                        )
                    )
                    restoreLabels.append(method)

                case (
                    "anime1080fixer-tensorrt"
                    | "gater3-tensorrt"
                    | "scunet-tensorrt"
                    | "codeformer-tensorrt"
                    | "deh264_real-tensorrt"
                    | "deh264_span-tensorrt"
                    | "hurrdeblur-tensorrt"
                ):
                    from src.unifiedRestore import UnifiedRestoreTensorRT

                    restoreProcesses.append(
                        UnifiedRestoreTensorRT(
                            method,
                            self.half,
                            self.width,
                            self.height,
                            self.forceStatic,
                        )
                    )
                    restoreLabels.append(method)

                case (
                    "anime1080fixer-directml"
                    | "gater3-directml"
                    | "scunet-directml"
                    | "codeformer-directml"
                    | "deh264_real-directml"
                    | "deh264_span-directml"
                    | "hurrdeblur-directml"
                ):
                    from src.unifiedRestore import UnifiedRestoreDirectML

                    restoreProcesses.append(
                        UnifiedRestoreDirectML(
                            method,
                            self.half,
                            self.width,
                            self.height,
                        )
                    )
                    restoreLabels.append(method)
                case "fastlinedarken":
                    from src.fastlinedarken import FastLineDarkenWithStreams

                    restoreProcesses.append(
                        FastLineDarkenWithStreams(
                            self.half,
                        )
                    )
                    restoreLabels.append(method)
                case "fastlinedarken-tensorrt":
                    from src.fastlinedarken import FastLineDarkenTRT

                    restoreProcesses.append(
                        FastLineDarkenTRT(
                            self.half,
                            self.height,
                            self.width,
                        )
                    )
                    restoreLabels.append(method)
                
                case _:
                    import os
                    from pathlib import Path
                    
                    if os.path.exists(method):
                        logAndPrint(
                            message=f"Custom restoration model detected: {method}",
                            colorFunc="cyan",
                        )

                        display_label = Path(method).name
                        
                        file_ext = Path(method).suffix.lower()
                        backend = self.customRestoreBackend
                        
                        if backend == "default":
                            if file_ext == ".onnx":
                                backend = "tensorrt"
                            elif file_ext in [".pth", ".pt"]:
                                backend = "cuda"
                            else:
                                raise ValueError(
                                    f"Unsupported custom model file type: {file_ext}. "
                                    f"Supported types: .onnx, .pth, .pt"
                                )
                        
                        if backend == "tensorrt":
                            if file_ext in [".pth", ".pt"]:
                                from src.utils.onnxConverter import pthToOnnx
                                
                                logAndPrint(
                                    message=f"Converting PTH restoration model to ONNX for TensorRT: {method}",
                                    colorFunc="yellow",
                                )
                                
                                precision = "fp16" if self.half else "fp32"
                                expectedOnnxPath = os.path.splitext(method)[0] + f"_{precision}_op20_slim.onnx"
                                fallbackOnnxPath = os.path.splitext(method)[0] + f"_{precision}_op20.onnx"
                                genericSlimPath = os.path.splitext(method)[0] + "_op20_slim.onnx"
                                genericPath = os.path.splitext(method)[0] + "_op20.onnx"
                                
                                if os.path.exists(expectedOnnxPath):
                                    onnxPath = expectedOnnxPath
                                elif os.path.exists(fallbackOnnxPath):
                                    onnxPath = fallbackOnnxPath
                                elif os.path.exists(genericSlimPath):
                                    onnxPath = genericSlimPath
                                elif os.path.exists(genericPath):
                                    onnxPath = genericPath
                                else:
                                    onnxPath = pthToOnnx(
                                        pthPath=method,
                                        inputShape=(1, 3, self.height, self.width),
                                        precision=precision,
                                        opset=20,
                                        slim=True,
                                    )
                                
                                if os.path.exists(onnxPath):
                                    method = onnxPath
                                    logAndPrint(
                                        message=f"Using converted ONNX model: {onnxPath}",
                                        colorFunc="green",
                                    )
                                else:
                                    raise RuntimeError(
                                        f"Failed to convert PTH to ONNX: {method}"
                                    )
                            
                            from src.unifiedRestore import UnifiedRestoreTensorRT
                            
                            logAndPrint(
                                message="Using TensorRT backend for custom restoration model",
                                colorFunc="cyan",
                            )
                            
                            restoreProcesses.append(
                                UnifiedRestoreTensorRT(
                                    restoreMethod="custom-tensorrt",
                                    half=self.half,
                                    width=self.width,
                                    height=self.height,
                                    forceStatic=self.forceStatic,
                                    customModel=method,
                                )
                            )
                            restoreLabels.append(display_label)
                        
                        elif backend == "cuda":
                            if file_ext not in [".pth", ".pt"]:
                                raise ValueError(
                                    f"CUDA backend requires .pth or .pt models, got: {file_ext}"
                                )
                            
                            from src.unifiedRestore import UnifiedRestoreCuda
                            
                            logAndPrint(
                                message="Using PyTorch CUDA backend for custom restoration model",
                                colorFunc="cyan",
                            )
                            
                            restoreProcesses.append(
                                UnifiedRestoreCuda(
                                    model="custom",
                                    half=self.half,
                                    customModel=method,
                                )
                            )
                            restoreLabels.append(display_label)
                        
                        else:
                            raise ValueError(
                                f"Unsupported backend '{backend}' for custom restoration models. "
                                f"Supported: tensorrt, cuda"
                            )
                    else:
                        raise ValueError(
                            f"Invalid restoration method or model file not found: {method}"
                        )

        if len(restoreProcesses) == 1:
            restoreProcess = restoreProcesses[0]
        else:
            restoreProcess = RestoreChain(restoreProcesses, restoreLabels)

    if self.dedup:
        match self.dedupMethod:
            case "ssim":
                from src.dedup.dedup import DedupSSIM

                dedupProcess = DedupSSIM(
                    self.dedupSens,
                )

            case "mse":
                from src.dedup.dedup import DedupMSE

                dedupProcess = DedupMSE(
                    self.dedupSens,
                )

            case "ssim-cuda":
                from src.dedup.dedup import DedupSSIMCuda

                dedupProcess = DedupSSIMCuda(
                    self.dedupSens,
                    self.half,
                )

            case "vmaf" | "vmaf-cuda":
                from src.dedup.dedup import DedupVMAF

                dedupProcess = DedupVMAF(
                    dedupMethod=self.dedupMethod,
                    treshold=self.dedupSens,
                    half=self.half,
                )

            case "mse-cuda":
                from src.dedup.dedup import DedupMSECuda

                dedupProcess = DedupMSECuda(
                    self.dedupSens,
                    self.half,
                )

            case "flownets":
                from src.dedup.dedup import DedupFlownetS

                dedupProcess = DedupFlownetS(
                    half=self.half,
                    dedupSens=self.dedupSens,
                    height=self.height,
                    width=self.width,
                )

    if self.scenechange:
        match self.scenechangeMethod:
            case "maxxvit-tensorrt" | "shift_lpips-tensorrt":
                from src.scenechange import SceneChangeTensorRT

                scenechangeProcess = SceneChangeTensorRT(
                    self.half,
                    self.scenechangeSens,
                    self.scenechangeMethod,
                )
            case "maxxvit-directml":
                from src.scenechange import SceneChange

                scenechangeProcess = SceneChange(
                    self.half,
                    self.scenechangeSens,
                )
            case "differential":
                from src.scenechange import SceneChangeCPU

                scenechangeProcess = SceneChangeCPU(
                    self.scenechangeSens,
                )
            case "differential-cuda":
                from src.scenechange import SceneChangeCuda

                scenechangeProcess = SceneChangeCuda(
                    self.scenechangeSens,
                )
            case "differential-tensorrt":
                from src.scenechange import DifferentialTensorRT

                scenechangeProcess = DifferentialTensorRT(
                    self.scenechangeSens,
                    self.height,
                    self.width,
                )
            case "differential-directml":
                # from src.scenechange import DifferentialDirectML
                # scenechangeProcess = DifferentialDirectML(
                #     self.scenechangeSens,
                # )
                raise NotImplementedError(
                    "Differential DirectML is not implemented yet"
                )
            case _:
                raise ValueError(
                    f"Unknown scenechange method: {self.scenechangeMethod}"
                )

    return (
        outputWidth,
        outputHeight,
        upscaleProcess,
        interpolateProcess,
        restoreProcess,
        dedupProcess,
        scenechangeProcess,
    )
