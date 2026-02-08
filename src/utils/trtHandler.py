import tensorrt as trt
import os
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional, Union

from src.utils.logAndPrint import logAndPrint, coloredPrint, show_error_dialog
from src.constants import ADOBE

if ADOBE:
    from src.utils.aeComms import progressState


class _BufferedTRTLogger(trt.ILogger):
    """TensorRT logger that captures messages instead of printing to stderr.

    This prevents noisy TensorRT build-time messages (e.g. transient OOM during tactic
    selection) from spamming the console. We can still inspect the buffered records
    when a build actually fails.
    """

    def __init__(
        self,
        severity: trt.Logger.Severity = trt.Logger.ERROR,
        suppress_substrings: Optional[List[str]] = None,
    ):
        trt.ILogger.__init__(self)
        self.severity = severity
        self.suppress_substrings = suppress_substrings or []
        self.records: List[Tuple[trt.Logger.Severity, str]] = []

    def log(self, severity: trt.Logger.Severity, msg: str) -> None:
        # TensorRT severities are ordered: INTERNAL_ERROR(0) < ERROR(1) < WARNING(2) < INFO(3) < VERBOSE(4)
        # Keep only messages at or above the configured severity threshold.
        if severity > self.severity:
            return

        for s in self.suppress_substrings:
            if s in msg:
                return

        # Keep records bounded to avoid unbounded memory growth.
        if len(self.records) > 200:
            self.records.pop(0)
        self.records.append((severity, msg))


def get_optimal_workspace_size():
    """Get optimal TensorRT workspace size based on available VRAM."""
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # 6GB-class GPUs often report slightly under 6.0GB; using only 512MB workspace
            # can prevent TensorRT from finding any valid tactics for transformer-ish models.
            if vram_gb < 5:
                workspace_size = (1 << 29)
                logAndPrint(f"GPU VRAM: {vram_gb:.1f}GB - Using 512MB workspace", "cyan")
            elif vram_gb < 8:
                workspace_size = (1 << 30)
                logAndPrint(f"GPU VRAM: {vram_gb:.1f}GB - Using 1GB workspace", "cyan")
            elif vram_gb < 12:
                workspace_size = (1 << 31)
                logAndPrint(f"GPU VRAM: {vram_gb:.1f}GB - Using 2GB workspace", "cyan")
            else:
                workspace_size = (1 << 32)
                logAndPrint(f"GPU VRAM: {vram_gb:.1f}GB - Using 4GB workspace", "cyan")
            
            return workspace_size
    except:
        pass
    
    return (1 << 30)


def createNetworkAndConfig(
    builder: trt.Builder,
    maxWorkspaceSize: int,
) -> Tuple[trt.INetworkDefinition, trt.IBuilderConfig]:
    """Create TensorRT network and builder configuration."""
    # NOTE:
    # - STRONGLY_TYPED networks can make some models harder to build (no implicit casts),
    #   and on some TensorRT versions it also conflicts with enabling BuilderFlag.FP16.
    # - We prefer a regular network here and let the builder handle precision based on
    #   config flags (e.g. FP16) for better compatibility.
    network = builder.create_network(0)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, maxWorkspaceSize)
    return network, config


def parseModel(parser: trt.OnnxParser, modelPath: str) -> bool:
    """Parse ONNX model with enhanced error handling."""
    if not os.path.exists(modelPath):
        logAndPrint(f"Model file not found: {modelPath}", "red")
        return False

    try:
        with open(modelPath, "rb") as model:
            modelData = model.read()
            if not modelData:
                logAndPrint(f"Empty model file: {modelPath}", "red")
                return False

            if not parser.parse(modelData):
                logAndPrint("Failed to parse ONNX model:", "red")
                for error in range(parser.num_errors):
                    errorMSG = parser.get_error(error)
                    logAndPrint(f"  Parser error {error}: {errorMSG}", "red")
                    logging.error(f"ONNX parser error {error}: {errorMSG}")
                return False
        return True
    except Exception as e:
        logAndPrint(f"Error reading model file {modelPath}: {e}", "red")
        logging.error(f"Error reading model file {modelPath}: {e}")
        return False


def setOptimizationProfile(
    builder: trt.Builder,
    config: trt.IBuilderConfig,
    inputName: List[str],
    inputsMin: Union[List[Tuple[int, ...]], Tuple[int, ...]],
    inputsOpt: Union[List[Tuple[int, ...]], Tuple[int, ...]],
    inputsMax: Union[List[Tuple[int, ...]], Tuple[int, ...]],
    isMultiInput: bool,
    fp16: bool = False,
) -> bool:
    """Set optimization profile with improved error handling and validation."""
    try:
        profile = builder.create_optimization_profile()

        if isMultiInput:
            if not all(isinstance(x, list) for x in [inputsMin, inputsOpt, inputsMax]):
                logAndPrint("Multi-input mode requires list inputs", "red")
                return False

            if not all(
                len(x) == len(inputName) for x in [inputsMin, inputsOpt, inputsMax]
            ):
                logAndPrint("Input tensors and names must have same length", "red")
                return False

            for name, minShape, optShape, maxShape in zip(
                inputName, inputsMin, inputsOpt, inputsMax
            ):
                profile.set_shape(
                    name, tuple(minShape), tuple(optShape), tuple(maxShape)
                )
                _logInputShapes(name, minShape, optShape, maxShape, fp16)
        else:
            if len(inputName) == 0:
                logAndPrint("Input name list cannot be empty", "red")
                return False

            profile.set_shape(
                inputName[0], tuple(inputsMin), tuple(inputsOpt), tuple(inputsMax)
            )
            _logInputShapes(inputName[0], inputsMin, inputsOpt, inputsMax, fp16)

        config.add_optimization_profile(profile)
        return True

    except Exception as e:
        logAndPrint(f"Error setting optimization profile: {e}", "red")
        logging.error(f"Error setting optimization profile: {e}")
        return False


def _logInputShapes(name: str, minShape, optShape, maxShape, fp16) -> None:
    """Helper function to log input shapes consistently."""
    if not ADOBE:
        precision = "FP16" if fp16 else "FP32"
        coloredPrint(
            f"╭─ Input: {name} | {precision} \n"
            f"├─ Min: {minShape}\n"
            f"├─ Opt: {optShape}\n"
            f"╰─ Max: {maxShape}",
        )
    logging.info(f"Input: {name} - Min: {minShape}, Opt: {optShape}, Max: {maxShape}")


class TensorRTBuildMonitor:
    """Monitor TensorRT build progress with timeout detection."""
    def __init__(self, timeout_seconds=300):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
        self.completed = False
        
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.completed = False
        
    def check_timeout(self):
        """Check if build has timed out."""
        if self.start_time is None or self.completed:
            return False
        elapsed = time.time() - self.start_time
        return elapsed > self.timeout_seconds
        
    def complete(self):
        """Mark build as completed."""
        self.completed = True


def tensorRTEngineCreator(
    modelPath: str = "",
    enginePath: str = "model.engine",
    fp16: bool = False,
    inputsMin: Union[List[Tuple[int, ...]], Tuple[int, ...]] = [],
    inputsOpt: Union[List[Tuple[int, ...]], Tuple[int, ...]] = [],
    inputsMax: Union[List[Tuple[int, ...]], Tuple[int, ...]] = [],
    inputName: Optional[List[str]] = None,
    maxWorkspaceSize: Optional[int] = None,
    optimizationLevel: Optional[int] = None,
    forceStatic: bool = False,
    isMultiInput: bool = False,
    isRife: bool = False,
) -> Tuple[Optional[trt.ICudaEngine], Optional[trt.IExecutionContext]]:
    """
    Create a TensorRT engine from an ONNX model with enhanced validation and error handling.

    Parameters:
        modelPath (str): The path to the ONNX model.
        enginePath (str): The path to save the engine.
        fp16 (bool): Use half precision for the engine.
        inputsMin: The minimum shape(s) that the profile will support.
        inputsOpt: The shape(s) for which TensorRT will optimize the engine.
        inputsMax: The maximum shape(s) that the profile will support.
        inputName (List[str]): The names of the input tensors.
        maxWorkspaceSize (int): The maximum GPU memory that the engine will use.
        optimizationLevel (int): The optimization level for the engine.
        forceStatic (bool): Force static shapes for all inputs.
        isMultiInput (bool): Whether the model has multiple inputs.
        isRife (bool): Whether the model is a RIFE model.

    Returns:
        Tuple of (engine, context) or (None, None) on failure.
    """
    if maxWorkspaceSize is None:
        maxWorkspaceSize = get_optimal_workspace_size()
    
    if optimizationLevel is None:
        optimizationLevel = 3
    
    if not modelPath or not os.path.exists(modelPath):
        logAndPrint(f"Invalid model path: {modelPath}", "red")
        return None, None

    if inputName is None:
        inputName = ["input"]

    if not inputName:
        logAndPrint("Input name list cannot be empty", "red")
        return None, None

    if not all([inputsMin, inputsOpt, inputsMax]) and not forceStatic:
        logAndPrint("Input shapes must be provided unless forceStatic is True", "red")
        return None, None

    logAndPrint(
        f"Model engine not found, creating engine for model: {modelPath}",
        "yellow",
    )

    if ADOBE:
        progressState.update(
            {
                "status": f"Creating a TensorRT engine for {os.path.basename(modelPath)}.",
            }
        )

    if forceStatic:
        inputsMin = inputsOpt
        inputsMax = inputsOpt

    try:
        TRTLOGGER = _BufferedTRTLogger(
            severity=trt.Logger.ERROR,
            suppress_substrings=[
                # This message can spam many times during build even if the engine
                # is eventually built successfully.
                "virtualMemoryBuffer.cpp::nvinfer1::StdVirtualMemoryBufferImpl::resizePhysical",
            ],
        )

        # Make sure built-in plugin creators are registered (LayerNorm/Einsum/etc.).
        # Without this, the ONNX parser can produce ForeignNode placeholders that
        # fail at build time with "Could not find any implementation for node".
        trt.init_libnvinfer_plugins(TRTLOGGER, "")

        attempts: List[Tuple[bool, int, int]] = [(fp16, optimizationLevel, maxWorkspaceSize)]

        # For some transformer-ish models (e.g. OmniSR), TensorRT can fail to find any
        # tactics at FP16 + high optimization level even though a more conservative build
        # (lower opt + larger workspace) can succeed. Keep the retry minimal to avoid
        # spending many minutes on repeated builds.
        if fp16:
            fallback_workspace = max(maxWorkspaceSize, (1 << 31))  # at least 2GB
            fallback_opt = min(optimizationLevel, 1)
            if fallback_workspace != maxWorkspaceSize or fallback_opt != optimizationLevel:
                attempts.append((True, fallback_opt, fallback_workspace))

        serializedEngine = None
        last_error: Optional[str] = None

        for attempt_idx, (attempt_fp16, attempt_opt_level, attempt_ws) in enumerate(
            attempts, start=1
        ):
            builder = trt.Builder(TRTLOGGER)
            network, config = createNetworkAndConfig(builder, attempt_ws)
            config.builder_optimization_level = attempt_opt_level

            if attempt_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            parser = trt.OnnxParser(network, TRTLOGGER)
            if not parseModel(parser, modelPath):
                return None, None

            if not setOptimizationProfile(
                builder,
                config,
                inputName,
                inputsMin,
                inputsOpt,
                inputsMax,
                isMultiInput,
                attempt_fp16,
            ):
                return None, None

            prec = "FP16" if attempt_fp16 else "FP32"
            ws_gb = attempt_ws / (1024**3)
            logAndPrint(
                f"Building TensorRT engine (attempt {attempt_idx}/{len(attempts)}; {prec}; optimization level {attempt_opt_level}; workspace {ws_gb:.1f}GB). This may take several minutes...",
                "green",
            )

            # Track new TensorRT log records produced during this attempt.
            trt_log_start = len(getattr(TRTLOGGER, "records", []))

            try:
                t0 = time.time()
                serializedEngine = builder.build_serialized_network(network, config)
                dt = time.time() - t0
            except Exception as e:
                last_error = str(e)
                logAndPrint(
                    f"TensorRT build attempt {attempt_idx} failed with an exception: {last_error}",
                    "yellow",
                )
                continue

            if serializedEngine:
                break

            last_error = f"build_serialized_network returned None (elapsed {dt:.1f}s)"
            # If TensorRT logged a concrete error, surface that as the last_error.
            new_records = getattr(TRTLOGGER, "records", [])[trt_log_start:]
            if new_records:
                _, msg = new_records[-1]
                last_error = msg
            logAndPrint(
                f"TensorRT build attempt {attempt_idx} failed: {last_error}",
                "yellow",
            )

        if not serializedEngine:
            logAndPrint("", "white")
            logAndPrint("=" * 80, "red")
            logAndPrint("TENSORRT ENGINE BUILD FAILED", "red")
            logAndPrint("=" * 80, "red")
            logAndPrint("TensorRT could not build an optimized engine for this model.", "yellow")
            if last_error:
                logAndPrint(f"Last error: {last_error}", "yellow")
            logAndPrint("", "white")
            logAndPrint("This is usually caused by:", "yellow")
            logAndPrint("  - Insufficient GPU VRAM for this model/resolution", "cyan")
            logAndPrint("  - Model architecture incompatibility with TensorRT", "cyan")
            logAndPrint("", "white")
            logAndPrint("RECOMMENDED SOLUTION:", "green")
            logAndPrint("  Switch to PyTorch backend (works with all models)", "cyan")
            logAndPrint("", "white")
            logAndPrint("Alternative options:", "yellow")
            logAndPrint("  - Try a different/smaller model", "cyan")
            logAndPrint("  - Try DirectML or NCNN backend", "cyan")
            logAndPrint("=" * 80, "red")
            logAndPrint("", "white")

            details = (
                "TensorRT could not build an optimized engine for this model.\n\n"
                "Common causes:\n"
                "• Insufficient GPU VRAM for this model/resolution\n"
                "• Model architecture incompatibility with TensorRT"
            )
            if last_error:
                details += f"\n\nLast error:\n{last_error}"
            details += (
                "\n\nRECOMMENDED SOLUTION:\n"
                "Switch to PyTorch backend (works with all models)\n\n"
                "Alternative options:\n"
                "• Try a different/smaller model\n"
                "• Try DirectML or NCNN backend"
            )

            show_error_dialog("TensorRT Build Failed", details)

            return None, None

        logAndPrint("Serialized engine built successfully!", "green")

        engineDir = os.path.dirname(enginePath)
        if engineDir:
            os.makedirs(engineDir, exist_ok=True)

        with open(enginePath, "wb") as f:
            f.write(serializedEngine)

        engine, context = tensorRTEngineLoader(enginePath)
        if engine is None:
            logAndPrint("Failed to load created engine", "red")
            return None, None

        logAndPrint(f"Engine saved to {enginePath}", "yellow")
        return engine, context

    except Exception as e:
        logAndPrint("", "white")
        logAndPrint("=" * 80, "red")
        logAndPrint("TENSORRT ENGINE BUILD ERROR", "red")
        logAndPrint("=" * 80, "red")
        logAndPrint(f"An error occurred during engine creation: {str(e)}", "yellow")
        logAndPrint("", "white")
        logAndPrint("RECOMMENDED SOLUTION:", "green")
        logAndPrint("  Switch to PyTorch backend", "cyan")
        logAndPrint("", "white")
        logAndPrint("Alternative backends:", "yellow")
        logAndPrint("  - DirectML (for AMD/Intel GPUs)", "cyan")
        logAndPrint("  - NCNN (lightweight, cross-platform)", "cyan")
        logAndPrint("=" * 80, "red")
        logAndPrint("", "white")
        logging.error(f"Error creating TensorRT engine: {e}")
        
        show_error_dialog(
            "TensorRT Build Error",
            f"An error occurred during engine creation:\n{str(e)}\n\n"
            "RECOMMENDED SOLUTION:\n"
            "Switch to PyTorch backend\n\n"
            "Alternative backends:\n"
            "• DirectML (for AMD/Intel GPUs)\n"
            "• NCNN (lightweight, cross-platform)"
        )
        
        return None, None


def tensorRTEngineLoader(
    enginePath: str,
) -> Tuple[Optional[trt.ICudaEngine], Optional[trt.IExecutionContext]]:
    """
    Load a TensorRT engine from a file with enhanced error handling.

    Parameters:
        enginePath (str): The path to the engine file.

    Returns:
        Tuple of (engine, context) or (None, None) on failure.
    """
    if not enginePath:
        logAndPrint("Engine path is empty, engine needs to be created", "yellow")
        logging.warning("Engine path is empty")
        return None, None
    
    if not os.path.exists(enginePath):
        logAndPrint(f"Engine file not found at: {enginePath}", "yellow")
        logAndPrint("Will create new TensorRT engine", "yellow")
        logging.info(f"Engine file not found, will create: {enginePath}")
        return None, None

    try:
        logAndPrint(f"Loading existing TensorRT engine from: {enginePath}", "cyan")
        trtLogger = trt.Logger(trt.Logger.ERROR)
        with (
            open(enginePath, "rb") as f,
            trt.Runtime(trtLogger) as runtime,
        ):
            engineData = f.read()
            if not engineData:
                logAndPrint(f"Empty engine file: {enginePath}", "red")
                return None, None

            engine = runtime.deserialize_cuda_engine(engineData)
            if not engine:
                logAndPrint(f"Failed to deserialize engine: {enginePath}", "red")
                return None, None

            context = engine.create_execution_context()
            if not context:
                logAndPrint(f"Failed to create execution context: {enginePath}", "red")
                return None, None

            logAndPrint(f"Successfully loaded TensorRT engine!", "green")
            return engine, context

    except FileNotFoundError:
        return None, None
    except Exception as e:
        logAndPrint(
            f"Model engine is outdated due to a TensorRT Update, creating a new engine. Error: {e}",
            "yellow",
        )
        logging.warning(f"Engine loading failed: {e}")
        return None, None


def tensorRTEngineNameHandler(
    modelPath: str = "",
    fp16: bool = False,
    optInputShape: List[int] = None,
    ensemble: bool = False,
    isRife: bool = False,
) -> str:
    """
    Create a name for the TensorRT engine file with validation.

    Parameters:
        modelPath (str): The path to the ONNX / PTH model.
        fp16 (bool): Use half precision for the engine.
        optInputShape (List[int]): The shape for which TensorRT will optimize the engine.
        ensemble (bool): Whether this is an ensemble model.
        isRife (bool): Whether this is a RIFE model.

    Returns:
        str: The generated engine file path.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not modelPath:
        raise ValueError("Model path cannot be empty")

    if optInputShape is None or len(optInputShape) < 4:
        raise ValueError("optInputShape must have at least 4 dimensions")

    enginePrecision = "fp16" if fp16 else "fp32"
    height, width = optInputShape[2], optInputShape[3]

    modelPath = Path(modelPath)
    if modelPath.suffix not in [".onnx", ".pth"]:
        raise ValueError(
            f"Unsupported model file extension: {modelPath.suffix}. Only .onnx and .pth are supported."
        )

    nameParts = [f"_{enginePrecision}_{height}x{width}"]

    if isRife and ensemble:
        nameParts.append("_ensemble")

    engineName = "".join(nameParts) + ".engine"
    return str(modelPath.with_suffix("")) + engineName
