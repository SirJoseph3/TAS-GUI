import onnx
import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import onnxslim

    isOnnxSlim = True
except ImportError:
    print("onnxslim not found. Please install onnx-slim using: pip install onnxslim")
    isOnnxSlim = False

OPSET = 20


def convertToFloat16(model):
    import copy
    from onnx import TensorProto

    def convertDtype(tensor):
        if tensor.data_type == TensorProto.FLOAT:
            tensor.data_type = TensorProto.FLOAT16
            if tensor.HasField("raw_data"):
                float32_data = np.frombuffer(tensor.raw_data, dtype=np.float32)
                float16_data = float32_data.astype(np.float16)
                tensor.raw_data = float16_data.tobytes()
            elif len(tensor.float_data) > 0:
                float32_data = np.array(tensor.float_data, dtype=np.float32)
                float16_data = float32_data.astype(np.float16)
                tensor.ClearField("float_data")
                tensor.raw_data = float16_data.tobytes()

    model = copy.deepcopy(model)

    for inputTensor in model.graph.input:
        if inputTensor.type.HasField("tensor_type"):
            if inputTensor.type.tensor_type.elem_type == TensorProto.FLOAT:
                inputTensor.type.tensor_type.elem_type = TensorProto.FLOAT16

    for outputTensor in model.graph.output:
        if outputTensor.type.HasField("tensor_type"):
            if outputTensor.type.tensor_type.elem_type == TensorProto.FLOAT:
                outputTensor.type.tensor_type.elem_type = TensorProto.FLOAT16

    for initializer in model.graph.initializer:
        convertDtype(initializer)

    for node in model.graph.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                convertDtype(attr.t)
            for tensor in attr.tensors:
                convertDtype(tensor)

    return model


def convertAndSaveModel(model, modelPath, precision, opset):
    if precision == "fp16":
        model = convertToFloat16(model)
    newModelPath = modelPath.replace(".onnx", f"_{precision}_op{opset}.onnx")
    onnx.save(model, newModelPath)
    savedModel = onnx.load(newModelPath)
    print(f"Opset version for {precision}: {savedModel.opset_import[0].version}")
    print(f"IR version for {precision}: {savedModel.ir_version}")
    return newModelPath


def fixResizeForTensorRT(modelPath):
    """Fix Resize nodes for TensorRT compatibility."""
    try:
        model = onnx.load(modelPath)
        modified = False
        resize_count = 0
        
        for node in model.graph.node:
            if node.op_type == "Resize":
                resize_count += 1
                for attr in node.attribute:
                    if attr.name == "mode":
                        mode_value = attr.s.decode('utf-8') if attr.s else None
                        if mode_value in ["cubic", "bicubic"]:
                            print(f"  - TensorRT fix: Changing Resize mode '{mode_value}' -> 'linear'")
                            attr.s = b"linear"
                            modified = True
        
        if resize_count > 0:
            print(f"  - Found {resize_count} Resize node(s)")
        
        if modified:
            onnx.save(model, modelPath)
            print("  - Applied TensorRT Resize compatibility fix")
        
        return modified
    except Exception as e:
        print(f"  - Warning: Could not fix Resize nodes: {e}")
        return False


def slimModel(modelPath, slimPath):
    if isOnnxSlim:
        try:
            onnxslim.slim(modelPath, slimPath)
            if os.path.exists(slimPath):
                os.remove(modelPath)
                print(f"(*) Successfully slimmed: {slimPath}")
                return slimPath
            print(f"(!) Slimming did not produce output, keeping original: {modelPath}")
            return modelPath
        except Exception as e:
            print(f"(!) Slimming failed with error: {e}")
            print(f"  Keeping original: {modelPath}")
            return modelPath
    else:
        print(f"onnxslim not found. Skipping {modelPath} slimming")
        return modelPath


def pthToOnnx(
    pthPath,
    outputPath=None,
    inputShape=(1, 3, 256, 256),
    precision="fp32",
    opset=OPSET,
    slim=True,
):
    from src.spandrel import ModelLoader, ImageModelDescriptor

    print(f"\n{'=' * 60}")
    print(f"Converting PyTorch model to ONNX: {pthPath}")
    print(f"{'=' * 60}")

    if not os.path.exists(pthPath):
        raise FileNotFoundError(f"Model file not found: {pthPath}")

    if outputPath is None:
        outputPath = os.path.splitext(pthPath)[0] + ".onnx"

    print("Loading model with spandrel...")
    import torch
    loader = ModelLoader(device="cpu")
    
    try:
        modelDescriptor = loader.load_from_file(pthPath)
    except RuntimeError as e:
        if "size mismatch" in str(e) and ("norm.scale" in str(e) or "norm.offset" in str(e)):
            print("Detected RTMoSR shape mismatch, attempting to fix state_dict...")
            state_dict = torch.load(pthPath, map_location="cpu")
            
            if isinstance(state_dict, dict) and "params" in state_dict:
                state_dict = state_dict["params"]
            
            fixed_state_dict = {}
            for key, value in state_dict.items():
                if ("norm.scale" in key or "norm.offset" in key) and len(value.shape) == 3:
                    print(f"  Reshaping {key}: {value.shape} -> {value.squeeze().shape}")
                    fixed_state_dict[key] = value.squeeze()
                else:
                    fixed_state_dict[key] = value
            
            modelDescriptor = loader.load_from_state_dict(fixed_state_dict)
        else:
            raise

    if not isinstance(modelDescriptor, ImageModelDescriptor):
        raise ValueError(f"Model is not an image model. Got: {type(modelDescriptor)}")

    model = modelDescriptor.model
    model.eval()

    print("Model loaded successfully:")
    architecture = modelDescriptor.architecture.id if hasattr(modelDescriptor, 'architecture') else 'Unknown'
    print(f"  - Architecture: {architecture}")
    print(f"  - Scale: {modelDescriptor.scale}x")
    print(f"  - Input channels: {modelDescriptor.input_channels}")
    print(f"  - Output channels: {modelDescriptor.output_channels}")
    
    if architecture.upper() in ['ESRGAN', 'REALESRGAN', 'BSRGAN', 'REALPLKSR']:
        original_opset = opset
        opset = 17
        print(f"  - Detected {architecture} - using opset {opset} for TensorRT compatibility (original: {original_opset})")

    # FP16-converting the ONNX can make TensorRT fail to find any tactics for some
    # transformer-ish models (e.g. OmniSR). For those, keep ONNX in FP32 and let
    # TensorRT handle precision via builder flags.
    prefer_fp16_onnx = architecture.upper() in ['ESRGAN', 'REALESRGAN', 'BSRGAN', 'REALPLKSR']

    inputShape = (
        inputShape[0],
        modelDescriptor.input_channels,
        inputShape[2],
        inputShape[3],
    )
    
    exportShape = (inputShape[0], inputShape[1], 64, 64)
    print(f"  - Using compact export shape to minimize memory usage: {exportShape}")
    print(f"  - Target resolution: {inputShape[2]}x{inputShape[3]} (will be handled via dynamic axes)")

    dummyInput = torch.randn(exportShape, dtype=torch.float32)

    print(f"\nExporting to ONNX (opset {opset})...")
    print(f"  - Export shape: {exportShape}")
    print(f"  - Output path: {outputPath}")

    useDynamo = opset > 20
    if useDynamo:
        print(
            f"  - Using torch.export-based ONNX exporter (dynamo=True) for opset {opset}"
        )

    try:
        torch.onnx.export(
            model,
            dummyInput,
            outputPath,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=useDynamo,
            verbose=False,
        )
    except Exception as e:
        if useDynamo and opset > 20:
            print(f"(!) Dynamo export failed, falling back to opset 20: {e}")
            opset = 20
            torch.onnx.export(
                model,
                dummyInput,
                outputPath,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch", 2: "height", 3: "width"},
                    "output": {0: "batch", 2: "height", 3: "width"},
                },
                opset_version=opset,
                do_constant_folding=True,
                verbose=False,
                dynamo=False,
            )
        else:
            raise

    print("(*) ONNX export successful!")

    basePath = outputPath.replace(".onnx", f"_op{opset}.onnx")
    if os.path.exists(basePath):
        os.remove(basePath)
    os.rename(outputPath, basePath)
    outputPath = basePath

    if slim and isOnnxSlim:
        print("\nOptimizing with onnxslim...")
        slimPath = outputPath.replace(".onnx", "_slim.onnx")
        
        if precision == "fp16":
            if prefer_fp16_onnx:
                print("  - Converting to FP16 during optimization...")
                onnxslim.slim(outputPath, slimPath, dtype="fp16")
            else:
                print("  - Keeping FP32 ONNX for compatibility (TensorRT can still build an FP16 engine)")
                onnxslim.slim(outputPath, slimPath)
        else:
            onnxslim.slim(outputPath, slimPath)
        
        if os.path.exists(slimPath):
            os.remove(outputPath)
            outputPath = slimPath
            print("(*) Optimization successful!")
        else:
            print("(!) Optimization failed, keeping original")
    elif precision == "fp16" and prefer_fp16_onnx:
        print("\nConverting to FP16...")
        onnxModel = onnx.load(outputPath)
        onnxModel = convertToFloat16(onnxModel)
        fp16Path = outputPath.replace("_op", "_fp16_op")
        onnx.save(onnxModel, fp16Path)
        os.remove(outputPath)
        outputPath = fp16Path
        print("(*) FP16 conversion successful!")

    print("\nApplying TensorRT compatibility fixes...")
    fixResizeForTensorRT(outputPath)

    if outputPath and os.path.exists(outputPath):
        finalModel = onnx.load(outputPath)
        print("\nFinal ONNX model info:")
        print(f"  - Path: {outputPath}")
        print(f"  - Opset version: {finalModel.opset_import[0].version}")
        print(f"  - IR version: {finalModel.ir_version}")
        print(f"  - File size: {os.path.getsize(outputPath) / (1024 * 1024):.2f} MB")

    print(f"\n{'=' * 60}")
    print("Conversion complete!")
    print(f"{'=' * 60}\n")

    return outputPath


if __name__ == "__main__":
    # Local dev helper. Keep this under __main__ to avoid side effects on import.
    modelList = [r"F:\test\dasdas\AnimeSR_v2.onnx"]

    for modelPath in modelList:
        if not os.path.exists(modelPath):
            print(f"Warning: Model file not found: {modelPath}")
            continue

        if modelPath.endswith(".onnx"):
            print(f"Processing ONNX model: {modelPath}")
            model = onnx.load(modelPath)

            newModelPathFp16 = convertAndSaveModel(model, modelPath, "fp16", OPSET)
            slimPathFp16 = newModelPathFp16.replace(".onnx", "_slim.onnx")
            print(f"{newModelPathFp16} -> {slimPathFp16}")
            slimModel(newModelPathFp16, slimPathFp16)

            newModelPathFp32 = convertAndSaveModel(model, modelPath, "fp32", OPSET)
            slimPathFp32 = newModelPathFp32.replace(".onnx", "_slim.onnx")
            print(f"{newModelPathFp32} -> {slimPathFp32}")
            slimModel(newModelPathFp32, slimPathFp32)

        elif modelPath.endswith((".pth", ".pt", ".ckpt", ".safetensors")):
            try:
                pthToOnnx(modelPath, precision="fp32", opset=OPSET, slim=isOnnxSlim)
                pthToOnnx(modelPath, precision="fp16", opset=OPSET, slim=isOnnxSlim)
            except Exception as e:
                print(f"Error converting {modelPath}: {e}")
                import traceback

                traceback.print_exc()
        else:
            print(f"Warning: Unsupported file type: {modelPath}")