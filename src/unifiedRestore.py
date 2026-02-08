import os
import torch
import logging

from .utils.downloadModels import downloadModels, weightsDir, modelsMap
from .utils.isCudaInit import CudaChecker
from .utils.logAndPrint import logAndPrint

checker = CudaChecker()


class UnifiedRestoreCuda:
    def __init__(
        self,
        model: str = "scunet",
        half: bool = True,
        customModel: str = None,
    ):
        """
        Initialize the denoiser with the desired model

        Args:
            model (str): The model to use for denoising
            width (int): The width of the input frame
            height (int): The height of the input frame
            half (bool): Whether to use half precision
            customModel (str): The path to a custom model file
        """

        self.model = model
        self.half = half
        self.customModel = customModel
        self.CHANNELSLAST = True
        self.handleModel()

    def handleModel(self):
        """
        Load the Model
        """
        from src.spandrel import ModelLoader

        if self.customModel:
            modelPath = self.customModel
            if not os.path.exists(modelPath):
                raise FileNotFoundError(
                    f"Custom restore model file not found: {modelPath}"
                )
        elif self.model in ["nafnet"]:
            self.half = False
            print("NAFNet does not support half precision, using float32 instead")

        if not self.customModel:
            self.filename = modelsMap(self.model)
            if not os.path.exists(os.path.join(weightsDir, self.model, self.filename)):
                modelPath = downloadModels(model=self.model)
            else:
                modelPath = os.path.join(weightsDir, self.model, self.filename)

        if self.customModel or self.model not in ["gater3"]:
            try:
                self.model = ModelLoader().load_from_file(path=modelPath)
                if isinstance(self.model, dict):
                    self.model = ModelLoader().load_from_state_dict(self.model)
            except Exception as e:
                logging.error(f"Error loading model: {e}")
        else:
            from safetensors.torch import load_file

            if self.model == "gater3":
                from src.extraArches.gaterv3 import GateRV3

                self.CHANNELSLAST = False

                self.model = GateRV3()

                stateDict = load_file(modelPath)
                self.model.load_state_dict(stateDict)

        try:
            # Weird spandrel hack to bypass ModelDecriptor
            self.model = self.model.model
        except Exception:
            pass

        self.model = (
            self.model.eval().cuda() if checker.cudaAvailable else self.model.eval()
        )

        if self.half:
            self.model.half()
            self.dType = torch.float16
        else:
            self.model.float()  # Sanity check, should not be needed
            self.dType = torch.float32
        self.stream = torch.cuda.Stream()

        if self.CHANNELSLAST:
            self.model.to(memory_format=torch.channels_last)
        else:
            self.model.to(memory_format=torch.contiguous_format)

    @torch.inference_mode()
    def __call__(self, frame: torch.tensor) -> torch.tensor:
        with torch.cuda.stream(self.stream):
            frame = self.model(
                frame.to(checker.device, non_blocking=True, dtype=self.dType).to(
                    memory_format=torch.channels_last
                )
                if self.CHANNELSLAST
                else frame.to(checker.device, non_blocking=True, dtype=self.dType)
            )
        self.stream.synchronize()
        return frame


class UnifiedRestoreTensorRT:
    def __init__(
        self,
        restoreMethod: str = "anime1080fixer-tensorrt",
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
        forceStatic: bool = False,
        customModel: str = None,
    ):
        """
        Initialize the upscaler with the desired model

        Args:
            restoreMethod (str): The method to use for upscaling
            upscaleFactor (int): The factor to upscale by
            half (bool): Whether to use half precision
            width (int): The width of the input frame
            height (int): The height of the input frame
            customModel (str): The path to a custom model file
        """

        # Attempt to lazy load for faster startup

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

        self.restoreMethod = restoreMethod
        self.half = half
        self.width = width
        self.height = height
        self.forceStatic = forceStatic
        self.customModel = customModel

        self.handleModel()

    def handleModel(self):
        self.originalHeight = self.height
        self.originalWidth = self.width

        if self.restoreMethod in ["scunet-tensorrt"]:
            if self.forceStatic is not True:
                self.forceStatic = True
                logAndPrint(
                    "Forcing static engine due to SCUNET limitations.",
                    "yellow",
                )
            # padding to 64x64
            self.height = (self.height + 63) // 64 * 64
            self.width = (self.width + 63) // 64 * 64
        elif self.restoreMethod in ["codeformer-tensorrt"]:
            if self.forceStatic is not True:
                self.forceStatic = True
                logAndPrint(
                    "Forcing static engine due to Codeformer's limitations.",
                    "yellow",
                )

            self.width = 512
            self.height = 512

        if self.width >= 1920 and self.height >= 1080:
            if self.forceStatic is not True:
                self.forceStatic = True
                logAndPrint(
                    "Forcing static engine due to resolution being equal or greater than 1080p.",
                    "yellow",
                )
            if self.restoreMethod in ["scunet-tensorrt"]:
                logAndPrint(
                    "!WARNING:! SCUNET requires more than 24GB of VRAM for 1920x1080 resolutions or higher.",
                    "red",
                )

        if self.customModel:
            self.modelPath = self.customModel
            if not os.path.exists(self.modelPath):
                raise FileNotFoundError(
                    f"Custom TensorRT restore model not found: {self.modelPath}"
                )
        else:
            self.filename = modelsMap(
                self.restoreMethod,
                modelType="onnx",
                half=self.half,
            )
            folderName = self.restoreMethod.replace("-tensorrt", "-onnx")
            if not os.path.exists(os.path.join(weightsDir, folderName, self.filename)):
                self.modelPath = downloadModels(
                    model=self.restoreMethod,
                    half=self.half,
                    modelType="onnx",
                )
            else:
                self.modelPath = os.path.join(weightsDir, folderName, self.filename)

        # Desired output dtype for the rest of the pipeline.
        self.dtype = torch.float16 if self.half else torch.float32

        enginePath = self.tensorRTEngineNameHandler(
            modelPath=self.modelPath,
            fp16=self.half,
            optInputShape=[1, 3, self.height, self.width],
        )

        self.engine, self.context = self.tensorRTEngineLoader(enginePath)
        if (
            self.engine is None
            or self.context is None
            or not os.path.exists(enginePath)
        ):
            self.engine, self.context = self.tensorRTEngineCreator(
                modelPath=self.modelPath,
                enginePath=enginePath,
                fp16=self.half,
                inputsMin=[1, 3, 8, 8],
                inputsOpt=[1, 3, self.height, self.width],
                inputsMax=[1, 3, self.height, self.width],
                forceStatic=self.forceStatic,
            )

        self.stream = torch.cuda.Stream()

        # IMPORTANT:
        # Do NOT assume TensorRT I/O is FP16 just because we built an FP16 engine.
        # Many ONNX models (including our PTH->ONNX exports) have FP32 inputs/outputs;
        # feeding FP16 buffers to an FP32 engine can cause out-of-bounds reads/writes
        # (CUDA illegal memory access).
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

        io_tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        input_names = [
            n for n in io_tensor_names if self.engine.get_tensor_mode(n) == self.trt.TensorIOMode.INPUT
        ]
        output_names = [
            n for n in io_tensor_names if self.engine.get_tensor_mode(n) == self.trt.TensorIOMode.OUTPUT
        ]

        if len(input_names) != 1 or len(output_names) != 1:
            raise RuntimeError(
                f"Unsupported TensorRT restore engine I/O: inputs={input_names}, outputs={output_names}. "
                "Expected exactly 1 input and 1 output."
            )

        self.inputTensorName = input_names[0]
        self.outputTensorName = output_names[0]

        self.inputDType = _trt_dtype_to_torch_dtype(self.engine.get_tensor_dtype(self.inputTensorName))
        self.outputDType = _trt_dtype_to_torch_dtype(self.engine.get_tensor_dtype(self.outputTensorName))

        self.dummyInput = torch.zeros(
            (1, 3, self.height, self.width),
            device=checker.device,
            dtype=self.inputDType,
        )
        self.context.set_input_shape(self.inputTensorName, self.dummyInput.shape)

        out_shape = tuple(self.context.get_tensor_shape(self.outputTensorName))
        if any(d < 0 for d in out_shape):
            # Should not happen for our forced-static 1080p engines, but keep a safe fallback.
            out_shape = (1, 3, self.height, self.width)

        self.dummyOutput = torch.zeros(
            out_shape,
            device=checker.device,
            dtype=self.outputDType,
        )

        self.context.set_tensor_address(self.inputTensorName, int(self.dummyInput.data_ptr()))
        self.context.set_tensor_address(self.outputTensorName, int(self.dummyOutput.data_ptr()))

        self.normStream = torch.cuda.Stream()
        self.outputStream = torch.cuda.Stream()
        
        self.useCudaGraph = False
        torch.cuda.synchronize()
        logAndPrint(
            message="Running TensorRT warmup iterations for restore model...",
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
            # Fail fast: after a CUDA illegal memory access, continuing will often produce
            # misleading downstream errors (e.g. "Video decode failed") due to a corrupted
            # CUDA context.
            logAndPrint(
                message=f"TensorRT warmup failed for restore model: {e}",
                colorFunc="red",
            )
            logging.exception(f"TensorRT warmup failed for restore model: {e}")
            raise RuntimeError(f"TensorRT warmup failed for restore model: {e}") from e
        
        self.cudaGraph = torch.cuda.CUDAGraph()
        try:
            self.initTorchCudaGraph()
            self.useCudaGraph = True
            logAndPrint(
                message="CUDA Graph initialized successfully for restore model",
                colorFunc="green",
            )
        except Exception as e:
            logAndPrint(
                message=f"CUDA Graph initialization failed, falling back to direct execution: {e}",
                colorFunc="yellow",
            )
            logging.warning(f"CUDA Graph init failed for restore model: {e}")
            self.useCudaGraph = False

    @torch.inference_mode()
    def initTorchCudaGraph(self):
        with torch.cuda.graph(self.cudaGraph, stream=self.stream):
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

    @torch.inference_mode()
    def processFrame(self, frame):
        with torch.cuda.stream(self.normStream):
            if self.originalHeight != self.height or self.originalWidth != self.width:
                frame = torch.nn.functional.interpolate(
                    frame,
                    size=(self.height, self.width),
                    mode="bilinear",
                    align_corners=False,
                )
            self.dummyInput.copy_(
                frame.to(dtype=self.dummyInput.dtype),
                non_blocking=False,
            )
        self.normStream.synchronize()

    @torch.inference_mode()
    def processOutput(self):
        with torch.cuda.stream(self.outputStream):
            output = self.dummyOutput[
                :, :, : self.originalHeight, : self.originalWidth
            ].clamp(0, 1)

            if output.dtype != self.dtype:
                output = output.to(dtype=self.dtype)
        self.outputStream.synchronize()

        return output

    @torch.inference_mode()
    def __call__(self, frame):
        self.processFrame(frame)

        if self.useCudaGraph:
            with torch.cuda.stream(self.stream):
                self.cudaGraph.replay()
            self.stream.synchronize()
        else:
            with torch.cuda.stream(self.stream):
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            self.stream.synchronize()
        
        return self.processOutput()


class UnifiedRestoreDirectML:
    def __init__(
        self,
        restoreMethod: str = "anime1080fixer-tensorrt",
        half: bool = False,
        width: int = 1920,
        height: int = 1080,
    ):
        """
        Initialize the upscaler with the desired model

        Args:
            restoreMethod (str): The method to use for upscaling
            half (bool): Whether to use half precision
            width (int): The width of the input frame
            height (int): The height of the input frame
        """

        import onnxruntime as ort
        import numpy as np

        self.ort = ort
        self.np = np
        self.ort.set_default_logger_severity(3)

        self.restoreMethod = restoreMethod
        self.half = half
        self.width = width
        self.height = height

        self.handleModel()

    def handleModel(self):
        """
        Load the desired model
        """

        self.filename = modelsMap(self.restoreMethod, modelType="onnx")
        folderName = self.restoreMethod.replace("directml", "-onnx")
        if not os.path.exists(os.path.join(weightsDir, folderName, self.filename)):
            modelPath = downloadModels(
                model=self.restoreMethod,
                modelType="onnx",
                half=self.half,
            )
        else:
            modelPath = os.path.join(weightsDir, folderName, self.filename)

        providers = self.ort.get_available_providers()

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
            (1, 3, self.height, self.width),
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

    def __call__(self, frame: torch.tensor) -> torch.tensor:
        """
        Run the model on the input frame
        """
        if self.half:
            frame = frame.half()
        else:
            frame = frame.float()

        self.dummyInput.copy_(frame.contiguous())

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

    def getSkippedCounter(self):
        return self.skippedCounter
