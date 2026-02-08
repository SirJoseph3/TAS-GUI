import os
import torch
import torch.nn.functional as F
import logging
import gc

from src.utils.modelOptimizer import ModelOptimizer
from .utils.downloadModels import downloadModels, weightsDir, modelsMap
from .utils.isCudaInit import CudaChecker
from src.utils.logAndPrint import logAndPrint

checker = CudaChecker()

torch.set_float32_matmul_precision("medium")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


class UniversalPytorch:
    def __init__(
        self,
        upscaleMethod: str = "shufflecugan",
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        customModel: str = None,
        compileMode: str = "default",
        tilesize: int = 0,
    ):
        """
        Initialize the upscaler with the desired model

        Args:
            upscaleMethod (str): The method to use for upscaling
            upscaleFactor (int): The factor to upscale by
            half (bool): Whether to use half precision
            width (int): The width of the input frame
            height (int): The height of the input frame
            customModel (str): The path to a custom model file
            compileMode: (str): The compile mode to use for the model
            tilesize: (int): Tile size for memory-intensive models (0 = disabled)
        """
        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel
        self.compileMode: str = compileMode
        self.modelArchitecture = None
        self.tilesize = tilesize
        # RVE default; controls context padding around each tile during tiled inference.
        self.tile_pad = 10

        # For custom models, the model's native scale (e.g. many ESRGAN models are 4x)
        # may not match the user-requested upscaleFactor (e.g. 2x). We'll detect the
        # native scale via Spandrel and down/up-sample the output when needed.
        self.modelScale = upscaleFactor
        self.needsResize = False

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """
        from src.spandrel import (
            ImageModelDescriptor,
            ModelLoader,
            UnsupportedDtypeError,
        )

        if not self.customModel:
            self.filename = modelsMap(
                self.upscaleMethod, self.upscaleFactor, modelType="pth"
            )
            if not os.path.exists(
                os.path.join(weightsDir, self.upscaleMethod, self.filename)
            ):
                modelPath = downloadModels(
                    model=self.upscaleMethod,
                    upscaleFactor=self.upscaleFactor,
                )
            else:
                modelPath = os.path.join(weightsDir, self.upscaleMethod, self.filename)
        else:
            if os.path.isfile(self.customModel):
                modelPath = self.customModel
            else:
                raise FileNotFoundError(
                    f"Custom model file {self.customModel} not found"
                )
            
        if self.upscaleMethod == "saryn":
            from src.extraArches.RTMoSR import RTMoSR

            self.model = RTMoSR()
            self.model.load_state_dict(torch.load(modelPath))
        elif self.upscaleMethod == "gauss":
            from src.extraArches.DIS import DIS
            from safetensors.torch import load_file

            if self.upscaleFactor != 2:
                raise ValueError(
                    "gauss upscale currently only supports --upscale_factor 2"
                )

            self.model = DIS(scale=2, num_features=32, num_blocks=12)
            self.model.load_state_dict(load_file(modelPath))
        else:
            self.model = torch.load(modelPath, map_location="cpu", weights_only=False)

            if isinstance(self.model, dict):
                self.model = ModelLoader().load_from_state_dict(self.model)

            if self.customModel:
                assert isinstance(self.model, ImageModelDescriptor)

                # Detect native model scale (e.g. 4x) for correct tiling + optional resize.
                try:
                    self.modelScale = int(
                        getattr(self.model, "scale", self.upscaleFactor)
                    )
                except Exception:
                    self.modelScale = self.upscaleFactor

                self.needsResize = self.modelScale != self.upscaleFactor
                if self.needsResize:
                    logAndPrint(
                        message=(
                            f"Custom model native scale is {self.modelScale}x but {self.upscaleFactor}x requested; "
                            "output will be resized to match requested scale"
                        ),
                        colorFunc="yellow",
                    )

                try:
                    self.modelArchitecture = (
                        self.model.architecture.id.upper()
                        if hasattr(self.model, "architecture")
                        else None
                    )
                except Exception:
                    self.modelArchitecture = None

            try:
                # SPANDREL HAXX
                self.model = self.model.model
            except Exception:
                pass

        self.model = (
            self.model.eval().cuda() if checker.cudaAvailable else self.model.eval()
        )

        if self.half and checker.cudaAvailable:
            try:
                self.model = self.model.half()
            except UnsupportedDtypeError as e:
                logging.error(f"Model does not support half precision: {e}")
                self.model = self.model.float()
                self.half = False
            except Exception as e:
                logging.error(f"Error converting model to half precision: {e}")
                self.model = self.model.float()
                self.half = False

        self.model = ModelOptimizer(
            self.model,
            torch.float16 if self.half else torch.float32,
            memoryFormat=torch.channels_last,
        ).optimizeModel()

        # Keep a reference to the eager model so we can safely fall back if torch.compile
        # fails later during the first forward (torch.compile can be lazy and compile on-call).
        self.eager_model = self.model
        self.compiled = False

        if self.compileMode != "default":
            if self.customModel:
                logAndPrint(
                    message="Skipping torch.compile for custom model to avoid Triton compilation issues",
                    colorFunc="yellow",
                )
                self.compileMode = "default"
            else:
                try:
                    if self.compileMode == "max":
                        self.model = torch.compile(
                            self.eager_model, mode="max-autotune-no-cudagraphs"
                        )
                        self.compiled = True
                    elif self.compileMode == "max-graphs":
                        self.model = torch.compile(
                            self.eager_model,
                            mode="max-autotune-no-cudagraphs",
                            fullgraph=True,
                        )
                        self.compiled = True
                except Exception as e:
                    logging.error(
                        f"Error compiling model {self.upscaleMethod} with mode {self.compileMode}: {e}"
                    )
                    logAndPrint(
                        f"Error compiling model {self.upscaleMethod} with mode {self.compileMode}: {e}",
                        "red",
                    )
                    self.model = self.eager_model
                    self.compiled = False
                    self.compileMode = "default"

        self.stream = torch.cuda.Stream()
        self.normStream = torch.cuda.Stream()
        self.outputStream = torch.cuda.Stream()

        if self.tilesize == 0:
            self.dummyInput = (
                torch.zeros(
                    (1, 3, self.height, self.width),
                    device=checker.device,
                    dtype=torch.float16 if self.half else torch.float32,
                )
                .contiguous()
                .to(memory_format=torch.channels_last)
            )

            self.dummyOutput = (
                torch.zeros(
                    (
                        1,
                        3,
                        self.height * self.upscaleFactor,
                        self.width * self.upscaleFactor,
                    ),
                    device=checker.device,
                    dtype=torch.float16 if self.half else torch.float32,
                )
                .contiguous()
                .to(memory_format=torch.channels_last)
            )

            try:
                with torch.cuda.stream(self.stream):
                    for _ in range(5):
                        self.model(self.dummyInput)
                        self.stream.synchronize()
            except Exception as e:
                if self.compiled:
                    logAndPrint(
                        message=(
                            f"torch.compile failed at runtime for {self.upscaleMethod} (mode={self.compileMode}); "
                            "falling back to default (no compilation)"
                        ),
                        colorFunc="yellow",
                    )
                    logging.exception(
                        f"torch.compile runtime failure for {self.upscaleMethod} (mode={self.compileMode})"
                    )
                    self.model = self.eager_model
                    self.compiled = False
                    self.compileMode = "default"
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    with torch.cuda.stream(self.stream):
                        for _ in range(5):
                            self.model(self.dummyInput)
                            self.stream.synchronize()
                else:
                    raise
        else:
            import math
            self.dummyInput = torch.zeros(
                (1, 3, self.height, self.width),
                device=checker.device,
                dtype=torch.float16 if self.half else torch.float32,
            ).contiguous().to(memory_format=torch.channels_last)
            
            self.dummyOutput = torch.zeros(
                (1, 3, self.height * self.upscaleFactor, self.width * self.upscaleFactor),
                device=checker.device,
                dtype=torch.float16 if self.half else torch.float32,
            ).contiguous().to(memory_format=torch.channels_last)
            
            if self.upscaleFactor == 1:
                modulo = 4
            elif self.upscaleFactor == 2:
                modulo = 2
            else:
                modulo = 1
            
            self.pad_w = math.ceil(min(self.tilesize + 2 * self.tile_pad, self.width) / modulo) * modulo
            self.pad_h = math.ceil(min(self.tilesize + 2 * self.tile_pad, self.height) / modulo) * modulo
            
            warmup_tile = torch.zeros(
                (1, 3, self.pad_h, self.pad_w),
                device=checker.device,
                dtype=torch.float16 if self.half else torch.float32,
            ).contiguous().to(memory_format=torch.channels_last)
            
            try:
                with torch.cuda.stream(self.stream):
                    for _ in range(5):
                        self.model(warmup_tile)
                        self.stream.synchronize()
            except Exception as e:
                if self.compiled:
                    logAndPrint(
                        message=(
                            f"torch.compile failed at runtime for {self.upscaleMethod} (mode={self.compileMode}); "
                            "falling back to default (no compilation)"
                        ),
                        colorFunc="yellow",
                    )
                    logging.exception(
                        f"torch.compile runtime failure for {self.upscaleMethod} (mode={self.compileMode})"
                    )
                    self.model = self.eager_model
                    self.compiled = False
                    self.compileMode = "default"
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    with torch.cuda.stream(self.stream):
                        for _ in range(5):
                            self.model(warmup_tile)
                            self.stream.synchronize()
                else:
                    raise
            
            del warmup_tile
            torch.cuda.empty_cache()

        is_memory_intensive = self.modelArchitecture in ['ESRGAN', 'REALESRGAN', 'BSRGAN', 'REALPLKSR'] if self.modelArchitecture else False
        
        if self.tilesize > 0:
            logAndPrint(
                f"Tile rendering enabled: {self.tilesize}x{self.tilesize} tiles, padding={self.tile_pad}px, pad_w={self.pad_w}, pad_h={self.pad_h} (CUDA Graph disabled)",
                "cyan",
            )
        
        if self.compileMode == "default" and not is_memory_intensive and self.tilesize == 0:
            self.cudaGraph = torch.cuda.CUDAGraph()
            self.initTorchCudaGraph()

    @torch.inference_mode()
    def initTorchCudaGraph(self):
        with torch.cuda.graph(self.cudaGraph, stream=self.stream):
            self.dummyOutput = self.model(self.dummyInput)
        self.stream.synchronize()

    @torch.inference_mode()
    def processFrame(self, frame):
        with torch.cuda.stream(self.normStream):
            self.dummyInput.copy_(
                frame.to(dtype=self.dummyInput.dtype).to(
                    memory_format=torch.channels_last
                ),
                non_blocking=False,
            )
        self.normStream.synchronize()

    @torch.inference_mode()
    def renderTiledImage(self, img: torch.Tensor) -> torch.Tensor:
        import math
        scale_out = self.upscaleFactor
        scale_model = getattr(self, "modelScale", self.upscaleFactor)
        tile_size = self.tilesize
        tile_pad = self.tile_pad

        batch, channel, height, width = img.shape
        output_shape = (batch, channel, height * scale_out, width * scale_out)

        # Overlap + blending tiling:
        # - Run the model on a padded tile.
        # - Convert to requested output scale if needed.
        # - Blend overlapping padded regions with a weight mask.
        # This is more robust against visible seams, especially when the model's
        # native scale != requested output scale (per-tile resize can otherwise
        # create grid artifacts).
        output_accum = img.new_zeros(output_shape, dtype=torch.float32)
        weight_accum = img.new_zeros((batch, 1, height * scale_out, width * scale_out), dtype=torch.float32)

        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)
        
        pad_w = self.pad_w
        pad_h = self.pad_h

        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * tile_size
                ofs_y = y * tile_size

                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y

                input_tile = img[
                    :, :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]

                h, w = input_tile.shape[2:]
                input_tile = F.pad(
                    input_tile, (0, pad_w - w, 0, pad_h - h), "replicate"
                )

                output_tile = self.model(input_tile)

                # Crop away the extra padding we added to reach (pad_h, pad_w).
                # This must use the model's native scale.
                output_tile = output_tile[:, :, : h * scale_model, : w * scale_model]

                # Convert the *padded* tile output to the requested output scale.
                # We resize the full padded region to preserve interpolation context.
                if scale_out != scale_model:
                    output_tile = torch.nn.functional.interpolate(
                        output_tile.to(dtype=torch.float32),
                        size=(h * scale_out, w * scale_out),
                        mode="bicubic",
                        antialias=True,
                        align_corners=False,
                    )
                else:
                    output_tile = output_tile.to(dtype=torch.float32)

                # Placement coords in *output* for the padded region
                out_x0 = input_start_x_pad * scale_out
                out_x1 = input_end_x_pad * scale_out
                out_y0 = input_start_y_pad * scale_out
                out_y1 = input_end_y_pad * scale_out

                # Build a weight mask that ramps in the padded regions so overlaps blend.
                # Each ramp spans exactly the padding that exists on that side.
                left_pad_px = (input_start_x - input_start_x_pad) * scale_out
                right_pad_px = (input_end_x_pad - input_end_x) * scale_out
                top_pad_px = (input_start_y - input_start_y_pad) * scale_out
                bottom_pad_px = (input_end_y_pad - input_end_y) * scale_out

                h_out = h * scale_out
                w_out = w * scale_out

                wx = torch.ones((w_out,), device=img.device, dtype=torch.float32)
                wy = torch.ones((h_out,), device=img.device, dtype=torch.float32)

                if left_pad_px > 0:
                    wx[:left_pad_px] = torch.linspace(
                        0.0, 1.0, left_pad_px, device=img.device, dtype=torch.float32
                    )
                if right_pad_px > 0:
                    wx[w_out - right_pad_px :] = torch.linspace(
                        1.0, 0.0, right_pad_px, device=img.device, dtype=torch.float32
                    )
                if top_pad_px > 0:
                    wy[:top_pad_px] = torch.linspace(
                        0.0, 1.0, top_pad_px, device=img.device, dtype=torch.float32
                    )
                if bottom_pad_px > 0:
                    wy[h_out - bottom_pad_px :] = torch.linspace(
                        1.0, 0.0, bottom_pad_px, device=img.device, dtype=torch.float32
                    )

                mask = (wy[:, None] * wx[None, :]).unsqueeze(0).unsqueeze(0)

                output_accum[:, :, out_y0:out_y1, out_x0:out_x1] += output_tile * mask
                weight_accum[:, :, out_y0:out_y1, out_x0:out_x1] += mask

        output = output_accum / weight_accum.clamp(min=1e-6)
        output = output.clamp(0, 1).to(dtype=img.dtype)
        return output

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor, nextFrame: None) -> torch.tensor:
        self.processFrame(frame)
        
        if self.tilesize > 0:
            # IMPORTANT: run tiled inference on the same CUDA stream we use elsewhere and
            # synchronize before the output stream reads/clones the result.
            # Otherwise we can end up cloning a partially-written tensor, which shows up
            # as blocky/mosaic artifacts.
            with torch.cuda.stream(self.stream):
                self.dummyOutput = self.renderTiledImage(self.dummyInput)
            self.stream.synchronize()
        elif hasattr(self, 'cudaGraph'):
            with torch.cuda.stream(self.stream):
                self.cudaGraph.replay()
            self.stream.synchronize()
        else:
            with torch.cuda.stream(self.stream):
                out = self.model(self.dummyInput)

                if getattr(self, "needsResize", False) and getattr(self, "modelScale", self.upscaleFactor) != self.upscaleFactor:
                    out = torch.nn.functional.interpolate(
                        out,
                        size=(self.height * self.upscaleFactor, self.width * self.upscaleFactor),
                        mode="bicubic",
                        antialias=True,
                        align_corners=False,
                    ).clamp(0, 1)

                self.dummyOutput.copy_(out, non_blocking=True)
            self.stream.synchronize()

        with torch.cuda.stream(self.outputStream):
            output = self.dummyOutput.clone()
        self.outputStream.synchronize()

        return output

    def frameReset(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()


class UniversalTensorRT:
    def __init__(
        self,
        upscaleMethod: str = "shufflecugan-tensorrt",
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        customModel: str = None,
        forceStatic: bool = False,
    ):
        """
        Initialize the upscaler with the desired model

        Args:
            upscaleMethod (str): The method to use for upscaling
            upscaleFactor (int): The factor to upscale by
            half (bool): Whether to use half precision
            width (int): The width of the input frame
            height (int): The height of the input frame
            customModel (str): The path to a custom model file
        """
        import tensorrt as trt
        from .utils.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )

        self.trt = trt
        self.tensorRTEngineCreator = tensorRTEngineCreator
        self.tensorRTEngineLoader = tensorRTEngineLoader
        self.tensorRTEngineNameHandler = tensorRTEngineNameHandler

        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel
        self.forceStatic = forceStatic

        self.handleModel()

    def handleModel(self):
        if self.width > 1920 or self.height > 1080:
            self.forceStatic = True
            logAndPrint(
                message="Forcing static engine due to resolution higher than 1920x1080p",
                colorFunc="yellow",
            )

        if not self.customModel:
            self.filename = modelsMap(
                self.upscaleMethod,
                self.upscaleFactor,
                modelType="onnx",
                half=self.half,
            )
            folderName = self.upscaleMethod.replace("-tensorrt", "-onnx")
            if not os.path.exists(os.path.join(weightsDir, folderName, self.filename)):
                self.modelPath = downloadModels(
                    model=self.upscaleMethod,
                    upscaleFactor=self.upscaleFactor,
                    half=self.half,
                    modelType="onnx",
                )
            else:
                self.modelPath = os.path.join(weightsDir, folderName, self.filename)
        else:
            self.modelPath = self.customModel
            logAndPrint(
                message=f"Using custom TensorRT model: {self.customModel}",
                colorFunc="cyan",
            )
            if not os.path.exists(self.customModel):
                raise FileNotFoundError(
                    f"Custom model file {self.customModel} not found"
                )

        requested_fp16 = self.half
        def _align_down(v: int, multiple: int) -> int:
            return max(multiple, (v // multiple) * multiple)

        # Custom video pipelines run at a fixed resolution. For custom models,
        # using a very small MIN/OPT shape can violate model-specific constraints
        # and cause TensorRT to fail with "no implementation" even though the
        # target (actual) resolution would work.
        if self.customModel:
            self.forceStatic = True

        # Keep OPT aligned for safety. If forceStatic is enabled, we will build
        # a static engine for (opt_h, opt_w).
        opt_h = _align_down(max(8, self.height), 8)
        opt_w = _align_down(max(8, self.width), 8)
        opt_h = min(opt_h, self.height)
        opt_w = min(opt_w, self.width)

        enginePath = self.tensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=requested_fp16,
            optInputShape=[1, 3, self.height, self.width],
        )
        
        logAndPrint(
            message=f"TensorRT model path: {self.modelPath}",
            colorFunc="cyan",
        )
        logAndPrint(
            message=f"TensorRT engine path: {enginePath}",
            colorFunc="cyan",
        )

        self.engine, self.context = self.tensorRTEngineLoader(enginePath)
        if (
            self.engine is None
            or self.context is None
            or not os.path.exists(enginePath)
        ):
            logAndPrint(
                message="Engine not found or failed to load, creating new engine...",
                colorFunc="yellow",
            )
            self.engine, self.context = self.tensorRTEngineCreator(
                modelPath=self.modelPath,
                enginePath=enginePath,
                fp16=requested_fp16,
                inputsMin=[1, 3, 8, 8],
                inputsOpt=[1, 3, opt_h, opt_w],
                inputsMax=[1, 3, self.height, self.width],
                forceStatic=self.forceStatic,
            )

        # Compatibility fallback: if FP16 engine build fails, retry FP32.
        if (self.engine is None or self.context is None) and requested_fp16:
            logAndPrint(
                message="FP16 TensorRT engine build failed. Retrying with FP32 engine for compatibility...",
                colorFunc="yellow",
            )
            fp32EnginePath = self.tensorRTEngineNameHandler(
                modelPath=self.modelPath,
                fp16=False,
                optInputShape=[1, 3, self.height, self.width],
            )

            self.engine, self.context = self.tensorRTEngineLoader(fp32EnginePath)
            if (
                self.engine is None
                or self.context is None
                or not os.path.exists(fp32EnginePath)
            ):
                logAndPrint(
                    message="FP32 engine not found or failed to load, creating new FP32 engine...",
                    colorFunc="yellow",
                )
                self.engine, self.context = self.tensorRTEngineCreator(
                    modelPath=self.modelPath,
                    enginePath=fp32EnginePath,
                    fp16=False,
                    inputsMin=[1, 3, 8, 8],
                    inputsOpt=[1, 3, opt_h, opt_w],
                    inputsMax=[1, 3, self.height, self.width],
                    forceStatic=self.forceStatic,
                )

            if self.engine is not None and self.context is not None:
                self.half = False
                enginePath = fp32EnginePath
                logAndPrint(
                    message=f"Using FP32 TensorRT engine: {enginePath}",
                    colorFunc="cyan",
                )

        self.dtype = torch.float16 if self.half else torch.float32

        if self.engine is None or self.context is None:
            raise RuntimeError(
                f"Failed to create TensorRT engine for model: {self.modelPath}"
            )

        outputHeight = self.height * self.upscaleFactor
        outputWidth = self.width * self.upscaleFactor
        
        outputDetected = False
        outputTensorName = None
        outputTensorIndex = None
        for i in range(self.engine.num_io_tensors):
            tensorName = self.engine.get_tensor_name(i)
            tensorShape = self.engine.get_tensor_shape(tensorName)
            tensorMode = self.engine.get_tensor_mode(tensorName)
            
            if tensorMode == self.trt.TensorIOMode.OUTPUT:
                outputTensorName = tensorName
                outputTensorIndex = i
                if len(tensorShape) >= 4 and tensorShape[2] > 0 and tensorShape[3] > 0:
                    outputHeight = tensorShape[2]
                    outputWidth = tensorShape[3]
                    outputDetected = True
                    logAndPrint(
                        message=f"Detected output size from engine metadata: {outputHeight}x{outputWidth}",
                        colorFunc="cyan",
                    )
                    break
        
        if not outputDetected and self.customModel:
            logAndPrint(
                message=f"Custom model has dynamic output shape, will allocate buffer after first inference",
                colorFunc="yellow",
            )
            self.dynamicOutputAllocation = True
            self.outputTensorName = outputTensorName
            maxPossibleScale = 4
            outputHeight = self.height * maxPossibleScale
            outputWidth = self.width * maxPossibleScale
            logAndPrint(
                message=f"Allocating maximum buffer size for dynamic detection: {outputHeight}x{outputWidth}",
                colorFunc="yellow",
            )
        elif not outputDetected:
            logAndPrint(
                message=f"Dynamic output shape, calculating from upscale factor {self.upscaleFactor}x: {outputHeight}x{outputWidth}",
                colorFunc="yellow",
            )
            self.dynamicOutputAllocation = False
        else:
            self.dynamicOutputAllocation = False

        # IMPORTANT:
        # Do NOT assume TensorRT I/O is FP16 just because we built an FP16 engine.
        # Many ONNX models have FP32 inputs/outputs; feeding FP16 buffers to an FP32 engine
        # can cause out-of-bounds reads/writes (CUDA illegal memory access).
        def _trt_dtype_to_torch_dtype(dt: "trt.DataType") -> torch.dtype:
            if dt == self.trt.DataType.FLOAT:
                return torch.float32
            if dt == self.trt.DataType.HALF:
                return torch.float16
            if dt == self.trt.DataType.INT32:
                return torch.int32
            if dt == self.trt.DataType.BOOL:
                return torch.bool
            raise RuntimeError(f"Unsupported TensorRT tensor dtype: {dt}")

        inputTensorName = None
        inputTensorIndex = None
        for i in range(self.engine.num_io_tensors):
            tensorName = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensorName) == self.trt.TensorIOMode.INPUT:
                inputTensorName = tensorName
                inputTensorIndex = i
                break

        if inputTensorName is None or outputTensorName is None:
            raise RuntimeError(
                f"TensorRT engine I/O tensors not found (input={inputTensorName}, output={outputTensorName})."
            )

        self.inputTensorName = inputTensorName
        self.outputTensorName = outputTensorName
        self.outputTensorIndex = outputTensorIndex

        self.inputDType = _trt_dtype_to_torch_dtype(self.engine.get_tensor_dtype(self.inputTensorName))
        self.outputDType = _trt_dtype_to_torch_dtype(self.engine.get_tensor_dtype(self.outputTensorName))

        self.stream = torch.cuda.Stream()
        self.dummyInput = torch.zeros(
            (1, 3, self.height, self.width),
            device=checker.device,
            dtype=self.inputDType,
        )

        self.dummyOutput = torch.zeros(
            (1, 3, outputHeight, outputWidth),
            device=checker.device,
            dtype=self.outputDType,
        )
        # Bindings array must align with engine tensor ordering.
        self.bindings = [0] * self.engine.num_io_tensors
        if inputTensorIndex is None or outputTensorIndex is None:
            raise RuntimeError(
                f"TensorRT tensor index resolution failed (inputIndex={inputTensorIndex}, outputIndex={outputTensorIndex})."
            )
        self.bindings[inputTensorIndex] = self.dummyInput.data_ptr()
        self.bindings[outputTensorIndex] = self.dummyOutput.data_ptr()
        
        if self.dynamicOutputAllocation:
            self.outputHeight = outputHeight
            self.outputWidth = outputWidth
            self.needsResize = False
            self.targetHeight = outputHeight
            self.targetWidth = outputWidth

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(tensor_name, self.bindings[i])
            
            if self.engine.get_tensor_mode(tensor_name) == self.trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, self.dummyInput.shape)
                logAndPrint(
                    message=f"Set input tensor '{tensor_name}' shape to {self.dummyInput.shape}",
                    colorFunc="cyan",
                )
            else:
                if not self.dynamicOutputAllocation:
                    logAndPrint(
                        message=f"Set output tensor '{tensor_name}' to buffer size {self.dummyOutput.shape}",
                        colorFunc="cyan",
                    )
                else:
                    logAndPrint(
                        message=f"Output tensor '{tensor_name}' will be allocated dynamically after first inference",
                        colorFunc="yellow",
                    )

        self.normStream = torch.cuda.Stream()
        self.outputStream = torch.cuda.Stream()

        self.useCudaGraph = False
        self.needsCudaGraphInit = self.customModel
        if not self.customModel:
            torch.cuda.synchronize()
            logAndPrint(
                message="Running TensorRT warmup iterations...",
                colorFunc="cyan",
            )
            try:
                with torch.cuda.stream(self.stream):
                    for _ in range(5):
                        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                        self.stream.synchronize()
                torch.cuda.synchronize()
                logAndPrint(
                    message="TensorRT warmup completed successfully",
                    colorFunc="green",
                )
            except Exception as e:
                logAndPrint(
                    message=f"Warning: TensorRT warmup failed: {e}",
                    colorFunc="yellow",
                )
                logging.warning(f"TensorRT warmup failed: {e}")

            self.cudaGraph = torch.cuda.CUDAGraph()
            try:
                self.initTorchCudaGraph()
                self.useCudaGraph = True
                logAndPrint(
                    message="CUDA Graph initialized successfully for TensorRT",
                    colorFunc="green",
                )
            except Exception as e:
                logAndPrint(
                    message=f"CUDA Graph initialization failed, falling back to direct execution: {e}",
                    colorFunc="yellow",
                )
                logging.warning(f"CUDA Graph init failed for TensorRT: {e}")
        else:
            self.cudaGraph = torch.cuda.CUDAGraph()
            logAndPrint(
                message="CUDA Graph will be initialized after first inference for custom model",
                colorFunc="cyan",
            )

    @torch.inference_mode()
    def initTorchCudaGraph(self):
        torch.cuda.synchronize()
        with torch.cuda.graph(self.cudaGraph, stream=self.stream):
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        torch.cuda.synchronize()

    @torch.inference_mode()
    def processFrame(self, frame):
        with torch.cuda.stream(self.normStream):
            self.dummyInput.copy_(
                frame.to(dtype=self.dummyInput.dtype),
                non_blocking=True,
            )
        self.normStream.synchronize()

    @torch.inference_mode()
    def __call__(self, frame, nextFrame: None) -> torch.tensor:
        try:
            self.processFrame(frame)

            if self.dynamicOutputAllocation:
                with torch.cuda.stream(self.stream):
                    success = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                    if not success:
                        raise RuntimeError("TensorRT execute_async_v3 failed during shape inference")
                self.stream.synchronize()
                
                outputShape = self.context.get_tensor_shape(self.outputTensorName)
                logAndPrint(
                    message=f"Detected actual output shape after first inference: {outputShape}",
                    colorFunc="green",
                )
                
                actualScaleFactor = outputShape[2] / self.height
                if abs(actualScaleFactor - self.upscaleFactor) > 0.1:
                    logAndPrint(
                        message=f"Model produces {actualScaleFactor:.1f}x but {self.upscaleFactor}x requested - will resize output",
                        colorFunc="yellow",
                    )
                    self.needsResize = True
                    self.targetHeight = self.height * self.upscaleFactor
                    self.targetWidth = self.width * self.upscaleFactor
                else:
                    self.needsResize = False
                
                if tuple(outputShape) != self.dummyOutput.shape:
                    logAndPrint(
                        message=f"Re-allocating output buffer from {self.dummyOutput.shape} to {tuple(outputShape)}",
                        colorFunc="yellow",
                    )
                    self.dummyOutput = torch.zeros(
                        tuple(outputShape),
                        device=checker.device,
                        dtype=self.outputDType,
                    )
                    self.bindings[self.outputTensorIndex] = self.dummyOutput.data_ptr()
                    self.context.set_tensor_address(self.outputTensorName, self.bindings[self.outputTensorIndex])
                else:
                    logAndPrint(
                        message=f"Output shape matches expected, using existing buffer",
                        colorFunc="green",
                    )
                
                self.dynamicOutputAllocation = False
                
                if self.needsCudaGraphInit:
                    logAndPrint(
                        message="Running warmup iterations for custom model...",
                        colorFunc="cyan",
                    )
                    try:
                        with torch.cuda.stream(self.stream):
                            for _ in range(5):
                                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                                self.stream.synchronize()
                        torch.cuda.synchronize()
                        
                        try:
                            self.initTorchCudaGraph()
                            self.useCudaGraph = True
                            logAndPrint(
                                message="CUDA Graph initialized successfully for custom model - performance boost enabled!",
                                colorFunc="green",
                            )
                        except Exception as e:
                            logAndPrint(
                                message=f"CUDA Graph initialization failed, using direct execution: {e}",
                                colorFunc="yellow",
                            )
                            logging.warning(f"CUDA Graph init failed: {e}")
                        
                        self.needsCudaGraphInit = False
                    except Exception as e:
                        logAndPrint(
                            message=f"Warmup failed: {e}",
                            colorFunc="yellow",
                        )
                        self.needsCudaGraphInit = False
                
                with torch.cuda.stream(self.outputStream):
                    output = self.dummyOutput.clone()
                    if self.needsResize:
                        output = torch.nn.functional.interpolate(
                            output,
                            size=(int(self.targetHeight), int(self.targetWidth)),
                            mode='bicubic',
                            antialias=True,
                            align_corners=False,
                        ).clamp(0, 1)
                        logAndPrint(
                            message=f"Resized output from {outputShape[2]}x{outputShape[3]} to {int(self.targetHeight)}x{int(self.targetWidth)}",
                            colorFunc="cyan",
                        )
                    if output.dtype != self.dtype:
                        output = output.to(dtype=self.dtype)
                self.outputStream.synchronize()
                return output

            if self.useCudaGraph:
                with torch.cuda.stream(self.stream):
                    self.cudaGraph.replay()
                self.stream.synchronize()
            else:
                with torch.cuda.stream(self.stream):
                    success = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
                    if not success:
                        raise RuntimeError("TensorRT execute_async_v3 failed")
                self.stream.synchronize()

            with torch.cuda.stream(self.outputStream):
                output = self.dummyOutput.clone()
                if hasattr(self, 'needsResize') and self.needsResize:
                    output = torch.nn.functional.interpolate(
                        output,
                        size=(int(self.targetHeight), int(self.targetWidth)),
                        mode='bicubic',
                        antialias=True,
                        align_corners=False,
                    ).clamp(0, 1)
                if output.dtype != self.dtype:
                    output = output.to(dtype=self.dtype)
            self.outputStream.synchronize()

            return output
        except Exception as e:
            logging.error(f"TensorRT inference error: {e}", exc_info=True)
            print(f"[ERROR] TensorRT inference failed: {e}", flush=True)
            print(f"[ERROR] Input shape: {frame.shape}, Expected: {self.dummyInput.shape}", flush=True)
            if self.dummyOutput is not None:
                print(f"[ERROR] Output shape: {self.dummyOutput.shape}", flush=True)
            else:
                print(f"[ERROR] Output buffer not allocated yet", flush=True)
            raise

    def frameReset(self):
        pass


class UniversalDirectML:
    def __init__(
        self,
        upscaleMethod: str,
        upscaleFactor: int,
        half: bool,
        width: int,
        height: int,
        customModel: str,
    ):
        """
        Initialize the upscaler with the desired model

        Args:
            upscaleMethod (str): The method to use for upscaling
            upscaleFactor (int): The factor to upscale by
            half (bool): Whether to use half precision
            width (int): The width of the input frame
            height (int): The height of the input frame
            customModel (str): The path to a custom model file
        """

        import onnxruntime as ort
        import numpy as np

        self.ort = ort
        self.np = np
        self.ort.set_default_logger_severity(3)

        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.customModel = customModel

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """

        wantsOpenVINO = self.upscaleMethod.endswith("-openvino")
        method = (
            self.upscaleMethod.replace("-openvino", "-directml")
            if wantsOpenVINO
            else self.upscaleMethod
        )

        if not self.customModel:
            self.filename = modelsMap(method, self.upscaleFactor, modelType="onnx")
            folderName = method.replace("-directml", "-onnx")
            if not os.path.exists(os.path.join(weightsDir, folderName, self.filename)):
                modelPath = downloadModels(
                    model=method,
                    upscaleFactor=self.upscaleFactor,
                    modelType="onnx",
                    half=self.half,
                )
            else:
                modelPath = os.path.join(weightsDir, folderName, self.filename)
        else:
            logging.info(
                f"Using custom model: {self.customModel}, this is an experimental feature, expect potential issues"
            )
            if os.path.isfile(self.customModel) and self.customModel.endswith(".onnx"):
                modelPath = self.customModel
            else:
                if not self.customModel.endswith(".onnx"):
                    raise FileNotFoundError(
                        f"Custom model file {self.customModel} is not an ONNX file"
                    )
                else:
                    raise FileNotFoundError(
                        f"Custom model file {self.customModel} not found"
                    )

        providers = self.ort.get_available_providers()

        if wantsOpenVINO:
            # Optional: this import helps surface a clearer error if OpenVINO python
            # package is missing.
            try:
                import openvino  # noqa: F401
            except Exception as e:
                logging.warning(
                    f"OpenVINO requested but python package import failed: {e}"
                )

            if "OpenVINOExecutionProvider" in providers:
                logging.info("Using OpenVINOExecutionProvider")
                self.model = self.ort.InferenceSession(
                    modelPath, providers=["OpenVINOExecutionProvider"]
                )
            else:
                logging.info(
                    "OpenVINO provider not available, falling back to CPU. "
                    "Install OpenVINOExecutionProvider for onnxruntime or use a -directml method."
                )
                self.model = self.ort.InferenceSession(
                    modelPath, providers=["CPUExecutionProvider"]
                )
        else:
            if "DmlExecutionProvider" in providers:
                logging.info("DirectML provider available. Defaulting to DirectML")
                self.model = self.ort.InferenceSession(
                    modelPath, providers=["DmlExecutionProvider"]
                )
            else:
                logging.info(
                    "DirectML provider not available, falling back to CPU, expect significantly worse performance, ensure that your drivers are up to date and your GPU supports DirectX 12"
                )
                self.model = self.ort.InferenceSession(
                    modelPath, providers=["CPUExecutionProvider"]
                )

        self.deviceType = "cpu"
        self.device = torch.device(self.deviceType)

        if self.half:
            self.numpyDType = self.np.float16
            self.torchDType = torch.float16
        else:
            self.numpyDType = self.np.float32
            self.torchDType = torch.float32

        self.IoBinding = self.model.io_binding()
        self.dummyInput = torch.zeros(
            (1, 3, self.height, self.width),
            device=self.deviceType,
            dtype=self.torchDType,
        ).contiguous()

        self.dummyOutput = torch.zeros(
            (1, 3, self.height * self.upscaleFactor, self.width * self.upscaleFactor),
            device=self.deviceType,
            dtype=self.torchDType,
        ).contiguous()

        self.IoBinding.bind_output(
            name="output",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyOutput.shape,
            buffer_ptr=self.dummyOutput.data_ptr(),
        )

    def __call__(self, frame: torch.tensor, nextFrame: None) -> torch.tensor:
        """
        Run the model on the input frame
        """

        if self.half:
            frame = frame.half()
        else:
            frame = frame.float()

        self.dummyInput.copy_(frame.contiguous(), non_blocking=False)

        self.IoBinding.bind_input(
            name="input",
            device_type=self.deviceType,
            device_id=0,
            element_type=self.numpyDType,
            shape=self.dummyInput.shape,
            buffer_ptr=self.dummyInput.data_ptr(),
        )

        self.model.run_with_iobinding(self.IoBinding)
        frame = self.dummyOutput.contiguous()

        return frame

    def frameReset(self):
        pass


class UniversalNCNN:
    def __init__(self, upscaleMethod, upscaleFactor):
        self.upscaleMethod = upscaleMethod
        self.upscaleFactor = upscaleFactor

        from upscale_ncnn_py import UPSCALE

        self.filename = modelsMap(
            self.upscaleMethod,
            modelType="ncnn",
        )

        if self.filename.endswith("-ncnn.zip"):
            self.filename = self.filename[:-9]
        elif self.filename.endswith("-ncnn"):
            self.filename = self.filename[:-5]

        if not os.path.exists(
            os.path.join(weightsDir, self.upscaleMethod, self.filename)
        ):
            modelPath = downloadModels(
                model=self.upscaleMethod,
                modelType="ncnn",
            )
        else:
            modelPath = os.path.join(weightsDir, self.upscaleMethod, self.filename)

        if modelPath.endswith("-ncnn.zip"):
            modelPath = modelPath[:-9]
        elif modelPath.endswith("-ncnn"):
            modelPath = modelPath[:-5]

        lastSlash = modelPath.split("\\")[-1]
        modelPath = modelPath + "\\" + lastSlash

        self.model = UPSCALE(
            gpuid=0,
            tta_mode=False,
            tilesize=0,
            model_str=modelPath,
            num_threads=2,
        )

    def __call__(self, frame, nextFrame: None) -> torch.tensor:
        iniFrameDtype = frame.dtype
        frame = self.model.process_torch(
            frame.mul(255).to(torch.uint8).squeeze(0).permute(1, 2, 0).cpu()
        )

        frame = frame.to(iniFrameDtype).mul(1 / 255).permute(2, 0, 1).unsqueeze(0)
        return frame

    def frameReset(self):
        pass


class AnimeSR:
    def __init__(
        self,
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        compileMode: str = "default",
    ):
        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height
        self.compileMode: str = compileMode

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """
        self.filename = modelsMap("animesr", self.upscaleFactor, modelType="pth")
        if not os.path.exists(os.path.join(weightsDir, "animesr", self.filename)):
            modelPath = downloadModels(
                model="animesr",
                upscaleFactor=self.upscaleFactor,
            )
        else:
            modelPath = os.path.join(weightsDir, "animesr", self.filename)

        from src.extraArches.AnimeSR import MSRSWVSR

        self.model = MSRSWVSR(num_feat=64, num_block=[5, 3, 2], netscale=4)

        self.model.load_state_dict(torch.load(modelPath))

        self.model = (
            self.model.eval().cuda() if checker.cudaAvailable else self.model.eval()
        )

        if self.half and checker.cudaAvailable:
            try:
                self.model = self.model.half()
            except Exception as e:
                logging.error(f"Error converting model to half precision: {e}")
                self.model = self.model.float()
                self.half = False

        if self.compileMode != "default":
            # Keep eager reference for safe fallback.
            self.eager_model = self.model
            self.compiled = False
            try:
                if self.compileMode == "max":
                    self.model = torch.compile(
                        self.eager_model, mode="max-autotune-no-cudagraphs"
                    )
                    self.compiled = True
                elif self.compileMode == "max-graphs":
                    self.model = torch.compile(
                        self.eager_model,
                        mode="max-autotune-no-cudagraphs",
                        fullgraph=True,
                    )
                    self.compiled = True
            except Exception as e:
                logging.error(f"Error compiling model animesr with mode {self.compileMode}: {e}")
                logAndPrint(
                    f"Error compiling model animesr with mode {self.compileMode}: {e}",
                    "red",
                )
                self.model = self.eager_model
                self.compiled = False
                self.compileMode = "default"

        # padding related logic
        ph = (4 - self.height % 4) % 4
        pw = (4 - self.width % 4) % 4
        self.padding = (0, pw, 0, ph)

        # The arch requires 3 inputs, so we create dummy inputs for the other two
        self.prevFrame = torch.zeros(
            (
                1,
                3,
                self.padding[3] + self.height + self.padding[2],
                self.padding[1] + self.width + self.padding[0],
            ),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        ).to(memory_format=torch.channels_last)
        self.nextFrame = torch.zeros(
            (
                1,
                3,
                self.padding[3] + self.height + self.padding[2],
                self.padding[1] + self.width + self.padding[0],
            ),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        ).to(memory_format=torch.channels_last)

        self.dummyOutput = self.prevFrame.new_zeros(
            1, 3, self.height * 4, self.width * 4
        ).to(memory_format=torch.channels_last)

        # The model has some caching functionality that requires a state
        self.state = self.prevFrame.new_zeros(1, 64, self.height, self.width)

        self.stream = torch.cuda.Stream()
        self.normStream = torch.cuda.Stream()
        self.outputStream = torch.cuda.Stream()

        self.firstRun = True

    def padFrame(self, frame: torch.tensor) -> torch.tensor:
        return torch.nn.functional.pad(frame, self.padding, mode="reflect")

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor, nextFrame: torch.tensor) -> torch.tensor:
        if self.firstRun:
            with torch.cuda.stream(self.normStream):
                self.prevFrame.copy_(
                    frame.to(dtype=frame.dtype).to(memory_format=torch.channels_last),
                    non_blocking=False,
                )
                if nextFrame is None:
                    self.nextFrame.copy_(
                        frame.to(dtype=frame.dtype).to(
                            memory_format=torch.channels_last
                        ),
                        non_blocking=False,
                    )
                else:
                    self.nextFrame.copy_(
                        nextFrame.to(dtype=frame.dtype).to(
                            memory_format=torch.channels_last
                        ),
                        non_blocking=False,
                    )
            self.normStream.synchronize()

            self.firstRun = False
        else:
            with torch.cuda.stream(self.normStream):
                if nextFrame is None:
                    self.nextFrame.copy_(
                        frame.to(dtype=frame.dtype).to(
                            memory_format=torch.channels_last
                        ),
                        non_blocking=False,
                    )
                else:
                    self.nextFrame.copy_(
                        nextFrame.to(dtype=frame.dtype).to(
                            memory_format=torch.channels_last
                        ),
                        non_blocking=False,
                    )
            self.normStream.synchronize()

        # preparing that mofo
        with torch.cuda.stream(self.normStream):
            frame = self.padFrame(frame)
        self.normStream.synchronize()

        model_input = torch.cat((self.prevFrame, frame, self.nextFrame), dim=1)

        try:
            with torch.cuda.stream(self.outputStream):
                self.dummyOutput, state = self.model(
                    model_input,
                    self.dummyOutput,
                    self.state,
                )

                self.state = state
            self.outputStream.synchronize()
        except Exception as e:
            if getattr(self, "compiled", False):
                logAndPrint(
                    message=(
                        f"torch.compile failed at runtime for animesr (mode={self.compileMode}); "
                        "falling back to default (no compilation)"
                    ),
                    colorFunc="yellow",
                )
                logging.exception(
                    f"torch.compile runtime failure for animesr (mode={self.compileMode})"
                )
                self.model = self.eager_model
                self.compiled = False
                self.compileMode = "default"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                with torch.cuda.stream(self.outputStream):
                    self.dummyOutput, state = self.model(
                        model_input,
                        self.dummyOutput,
                        self.state,
                    )
                    self.state = state
                self.outputStream.synchronize()
            else:
                raise

        with torch.cuda.stream(self.normStream):
            self.prevFrame.copy_(frame, non_blocking=False)
        self.normStream.synchronize()

        # resize the output to self.height*2 and self.width * 2
        with torch.cuda.stream(self.outputStream):
            output = torch.nn.functional.interpolate(
                self.dummyOutput,
                size=(self.height * 2, self.width * 2),
                mode="bicubic",
                align_corners=False,
            )
        self.outputStream.synchronize()

        return output

    def frameReset(self):
        with torch.cuda.stream(self.normStream):
            self.prevFrame.zero_()
            self.nextFrame.zero_()
            self.state.zero_()
            self.dummyOutput.zero_()
        self.normStream.synchronize()
        self.firstRun = True


class AnimeSRTensorRT:
    def __init__(
        self,
        upscaleFactor: int = 2,
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
    ):
        """
        Initialize the upscaler with the desired model

        Args:
            upscaleFactor (int): The factor to upscale by
            half (bool): Whether to use half precision
            width (int): The width of the input frame
            height (int): The height of the input frame
        """
        import tensorrt as trt
        from .utils.trtHandler import (
            tensorRTEngineCreator,
            tensorRTEngineLoader,
            tensorRTEngineNameHandler,
        )

        self.trt = trt
        self.tensorRTEngineCreator = tensorRTEngineCreator
        self.tensorRTEngineLoader = tensorRTEngineLoader
        self.tensorRTEngineNameHandler = tensorRTEngineNameHandler

        self.upscaleFactor = upscaleFactor
        self.half = half
        self.width = width
        self.height = height

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """

        if self.width > 1920 or self.height > 1080:
            self.forceStatic = True
            logAndPrint(
                message="Forcing static engine due to resolution higher than 1920x1080p",
                colorFunc="yellow",
            )

        self.filename = modelsMap(
            "animesr-tensorrt",
            self.upscaleFactor,
            modelType="onnx",
            half=self.half,
        )
        folderName = "animesr-onnx"
        if not os.path.exists(os.path.join(weightsDir, folderName, self.filename)):
            self.modelPath = downloadModels(
                model="animesr-tensorrt",
                upscaleFactor=self.upscaleFactor,
                half=self.half,
                modelType="onnx",
            )
        else:
            self.modelPath = os.path.join(weightsDir, folderName, self.filename)


        self.dtype = torch.float16 if self.half else torch.float32
        enginePath = self.tensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=self.half,
            optInputShape=[1, 3, self.height, self.width],
        )

        ph = (4 - self.height % 4) % 4
        pw = (4 - self.width % 4) % 4
        self.padding = (0, pw, 0, ph)

        # Padded dimensions for x and fb
        self.paddedHeight = self.padding[3] + self.height + self.padding[2]
        self.paddedWidth = self.padding[1] + self.width + self.padding[0]

        self.engine, self.context = self.tensorRTEngineLoader(enginePath)
        if (
            self.engine is None
            or self.context is None
            or not os.path.exists(enginePath)
        ):
            inputs = [
                [
                    1,
                    9,
                    self.paddedHeight,
                    self.paddedWidth,
                ],
                [
                    1,
                    3,
                    self.paddedHeight * 4,
                    self.paddedWidth * 4,
                ],
                [
                    1,
                    64,
                    self.height,
                    self.width,
                ],
            ]

            inputsMin = inputsOpt = inputsMax = inputs
            inputNames = ["x", "fb", "state"]

            self.engine, self.context = self.tensorRTEngineCreator(
                modelPath=self.modelPath,
                enginePath=enginePath,
                fp16=self.half,
                inputsMin=inputsMin,
                inputsOpt=inputsOpt,
                inputsMax=inputsMax,
                inputName=inputNames,
                isMultiInput=True,
            )

        self.prevFrame = torch.zeros(
            (1, 3, self.paddedHeight, self.paddedWidth),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        ).contiguous()
        self.nextFrame = torch.zeros(
            (1, 3, self.paddedHeight, self.paddedWidth),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        ).contiguous()

        self.dummyOutput = torch.zeros(
            (1, 3, self.paddedHeight * 4, self.paddedWidth * 4),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        ).contiguous()

        self.state = torch.zeros(
            (1, 64, self.height, self.width),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        )
        self.stateOutput = torch.zeros(
            (1, 64, self.height, self.width),
            device=checker.device,
            dtype=torch.float16 if self.half else torch.float32,
        )

        self.dummyInput = torch.cat(
            (self.prevFrame, self.prevFrame, self.nextFrame), dim=1
        )

        self.bindings = {
            "x": self.dummyInput.data_ptr(),
            "fb": self.dummyOutput.data_ptr(),
            "state": self.state.data_ptr(),
            "out_img": self.dummyOutput.data_ptr(),
            "out_state": self.stateOutput.data_ptr(),
        }

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)

            if tensor_name == "x":
                self.context.set_tensor_address(tensor_name, self.bindings["x"])
                self.context.set_input_shape(tensor_name, self.dummyInput.shape)
            elif tensor_name == "fb":
                self.context.set_tensor_address(tensor_name, self.bindings["fb"])
                self.context.set_input_shape(tensor_name, self.dummyOutput.shape)
            elif tensor_name == "state":
                self.context.set_tensor_address(tensor_name, self.bindings["state"])
                self.context.set_input_shape(tensor_name, self.state.shape)
            elif tensor_name == "out_img":
                self.context.set_tensor_address(tensor_name, self.bindings["out_img"])
            elif tensor_name == "out_state":
                self.context.set_tensor_address(tensor_name, self.bindings["out_state"])

        self.stream = torch.cuda.Stream()
        self.normStream = torch.cuda.Stream()
        self.outputStream = torch.cuda.Stream()

        self.firstRun = True

    def padFrame(self, frame: torch.tensor) -> torch.tensor:
        return torch.nn.functional.pad(frame, self.padding, mode="reflect")

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor, nextFrame: torch.tensor) -> torch.tensor:
        if self.firstRun:
            with torch.cuda.stream(self.normStream):
                paddedFrame = self.padFrame(frame)
                self.prevFrame.copy_(
                    paddedFrame.to(dtype=self.dtype),
                    non_blocking=False,
                )
                if nextFrame is None:
                    self.nextFrame.copy_(
                        paddedFrame.to(dtype=self.dtype),
                        non_blocking=False,
                    )
                else:
                    paddedNextFrame = self.padFrame(nextFrame)
                    self.nextFrame.copy_(
                        paddedNextFrame.to(dtype=self.dtype),
                        non_blocking=False,
                    )
            self.normStream.synchronize()
            self.firstRun = False
        else:
            with torch.cuda.stream(self.normStream):
                paddedFrame = self.padFrame(frame)
                if nextFrame is None:
                    self.nextFrame.copy_(
                        paddedFrame.to(dtype=self.dtype),
                        non_blocking=False,
                    )
                else:
                    paddedNextFrame = self.padFrame(nextFrame)
                    self.nextFrame.copy_(
                        paddedNextFrame.to(dtype=self.dtype),
                        non_blocking=False,
                    )
            self.normStream.synchronize()

        with torch.cuda.stream(self.normStream):
            self.dummyInput.copy_(
                torch.cat((self.prevFrame, paddedFrame, self.nextFrame), dim=1),
                non_blocking=False,
            )
        self.normStream.synchronize()

        with torch.cuda.stream(self.outputStream):
            self.context.execute_async_v3(stream_handle=self.outputStream.cuda_stream)
        self.outputStream.synchronize()

        with torch.cuda.stream(self.normStream):
            self.state.copy_(self.stateOutput, non_blocking=False)
            self.prevFrame.copy_(paddedFrame, non_blocking=False)
        self.normStream.synchronize()

        with torch.cuda.stream(self.outputStream):
            output = torch.nn.functional.interpolate(
                self.dummyOutput,
                size=(self.height * 2, self.width * 2),
                mode="bicubic",
                align_corners=False,
            )
        self.outputStream.synchronize()

        return output

    def frameReset(self):
        with torch.cuda.stream(self.normStream):
            self.prevFrame.zero_()
            self.nextFrame.zero_()
            self.state.zero_()
            self.stateOutput.zero_()
            self.dummyOutput.zero_()
        self.normStream.synchronize()
        self.firstRun = True
