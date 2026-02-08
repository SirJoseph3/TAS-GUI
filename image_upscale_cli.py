import argparse
import os
import sys
from pathlib import Path


# Avoid crashes when printing paths containing characters not representable in the active console encoding.
# (e.g., cp1254 on Turkish Windows).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="backslashreplace")


# Keep CUDA allocator config for the PyTorch path (ONNX path ignores it).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args():
    p = argparse.ArgumentParser(description="Single-image upscaler for The Anime Scripter GUI")
    p.add_argument("--input", required=True, help="Input image path (.png/.jpg/.jpeg)")
    p.add_argument("--output", required=True, help="Output image path")
    p.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG quality (1-100) when saving .jpg/.jpeg outputs",
    )
    p.add_argument(
        "--method",
        required=True,
        help="Upscale method (same naming as --upscale_method; may include -tensorrt/-directml/-ncnn suffixes)",
    )
    p.add_argument("--factor", type=int, default=2, help="Upscale factor (e.g. 2)")
    p.add_argument("--half", action="store_true", help="Use half precision (FP16) when available")
    p.add_argument("--compile_mode", default="default", help="torch.compile mode: default|max|max-graphs")
    p.add_argument("--tile_size", type=int, default=0, help="Tile size (0 disables tiling)")
    p.add_argument("--custom_model", default=None, help="Optional custom model path")
    p.add_argument(
        "--custom_upscale_backend",
        type=str,
        choices=["default", "cuda", "tensorrt", "directml", "ncnn"],
        default="default",
        help="Backend to use for custom upscale models. 'default' auto-detects based on file extension / filename.",
    )
    p.add_argument("--restore", action="store_true", help="Apply restoration before upscaling")
    p.add_argument(
        "--restore_method",
        nargs="+",
        default=[],
        help="Restoration method(s), can specify multiple for chaining (builtin model name or path to custom model)",
    )
    p.add_argument(
        "--custom_restore_backend",
        type=str,
        choices=["default", "cuda", "tensorrt", "directml", "ncnn"],
        default="default",
        help="Backend to use for custom restore models. 'default' auto-detects based on file extension.",
    )
    return p.parse_args()


def _ensure_tas_mainpath() -> None:
    """Mirror main.py init so model/engine caches go to the standard TAS folder.

    Without this, the GUI image-upscale subprocess can end up caching weights/engines
    relative to the portable folder instead of %APPDATA%\\TheAnimeScripter.
    """

    from platform import system as platform_system

    import src.constants as cs

    if not getattr(cs, "SYSTEM", ""):
        cs.SYSTEM = platform_system()

    if not getattr(cs, "MAINPATH", ""):
        if cs.SYSTEM == "Windows":
            appdata = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA")
            if not appdata:
                appdata = os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
            cs.MAINPATH = os.path.join(appdata, "TheAnimeScripter")
        else:
            xdg_config = os.getenv("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
            cs.MAINPATH = os.path.join(xdg_config, "TheAnimeScripter")
        os.makedirs(cs.MAINPATH, exist_ok=True)

    if not getattr(cs, "WHEREAMIRUNFROM", ""):
        cs.WHEREAMIRUNFROM = os.path.dirname(os.path.abspath(__file__))


def _save_pil_image(out_pil, output_path: Path, *, jpeg_quality: int) -> None:
    """Save output image honoring JPEG quality when output suffix is .jpg/.jpeg."""
    import time
    import gc

    suffix = str(output_path.suffix or "").lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        for attempt in range(3):
            try:
                gc.collect()
                output_path.unlink()
                break
            except OSError:
                if attempt < 2:
                    time.sleep(0.5)
                else:
                    base = output_path.stem
                    ext = output_path.suffix
                    output_path = output_path.parent / f"{base}_{int(time.time())}{ext}"

    if suffix in (".jpg", ".jpeg"):
        q = int(jpeg_quality)
        q = max(1, min(100, q))
        out_pil.save(str(output_path), quality=q)
        return

    # PNG (and others) are saved losslessly by default.
    out_pil.save(str(output_path))


def _apply_restore_chain(
    frame,
    restore_methods: list[str],
    *,
    half: bool,
    width: int,
    height: int,
    force_static: bool,
    custom_restore_backend: str,
):
    import os

    if not restore_methods:
        return frame

    # Keep labels stable for logging.
    restore_labels: list[str] = []
    restore_processes: list = []

    for method in restore_methods:
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

                restore_processes.append(UnifiedRestoreCuda(method, half))
                restore_labels.append(method)

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

                restore_processes.append(
                    UnifiedRestoreTensorRT(
                        method,
                        half,
                        width,
                        height,
                        force_static,
                    )
                )
                restore_labels.append(method)

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

                restore_processes.append(
                    UnifiedRestoreDirectML(
                        method,
                        half,
                        width,
                        height,
                    )
                )
                restore_labels.append(method)

            case "fastlinedarken":
                from src.fastlinedarken import FastLineDarkenWithStreams

                restore_processes.append(FastLineDarkenWithStreams(half))
                restore_labels.append(method)

            case "fastlinedarken-tensorrt":
                from src.fastlinedarken import FastLineDarkenTRT

                restore_processes.append(FastLineDarkenTRT(half, height, width))
                restore_labels.append(method)

            case _:
                if not os.path.exists(method):
                    raise ValueError(
                        f"Invalid restoration method or model file not found: {method}"
                    )

                from pathlib import Path

                display_label = Path(method).name
                file_ext = Path(method).suffix.lower()

                backend = str(custom_restore_backend or "default").lower()
                if backend == "default":
                    if file_ext == ".onnx":
                        backend = "tensorrt"
                    elif file_ext in [".pth", ".pt"]:
                        backend = "cuda"
                    else:
                        raise ValueError(
                            f"Unsupported custom model file type: {file_ext}. Supported: .onnx, .pth, .pt"
                        )

                if backend == "tensorrt":
                    if file_ext in [".pth", ".pt"]:
                        from src.utils.onnxConverter import pthToOnnx

                        precision = "fp16" if half else "fp32"
                        expectedOnnxPath = (
                            os.path.splitext(method)[0]
                            + f"_{precision}_op20_slim.onnx"
                        )
                        fallbackOnnxPath = (
                            os.path.splitext(method)[0] + f"_{precision}_op20.onnx"
                        )
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
                                inputShape=(1, 3, height, width),
                                precision=precision,
                                opset=20,
                                slim=True,
                            )

                        if not os.path.exists(onnxPath):
                            raise RuntimeError(f"Failed to convert PTH to ONNX: {method}")

                        method = onnxPath
                        file_ext = ".onnx"

                    from src.unifiedRestore import UnifiedRestoreTensorRT

                    restore_processes.append(
                        UnifiedRestoreTensorRT(
                            restoreMethod="custom-tensorrt",
                            half=half,
                            width=width,
                            height=height,
                            forceStatic=force_static,
                            customModel=method,
                        )
                    )
                    restore_labels.append(display_label)

                elif backend == "cuda":
                    if file_ext not in [".pth", ".pt"]:
                        raise ValueError(
                            f"CUDA backend requires .pth or .pt models, got: {file_ext}"
                        )

                    from src.unifiedRestore import UnifiedRestoreCuda

                    restore_processes.append(
                        UnifiedRestoreCuda(
                            model="custom",
                            half=half,
                            customModel=method,
                        )
                    )
                    restore_labels.append(display_label)

                else:
                    raise ValueError(
                        f"Unsupported backend '{backend}' for custom restoration models. Supported: tensorrt, cuda"
                    )

    label_str = " -> ".join(restore_labels)
    print(f"Applying restoration chain: {label_str}", flush=True)

    for proc in restore_processes:
        frame = proc(frame)

    return frame


def _infer_scale_from_filename(model_path: Path):
    name = model_path.name.lower()
    for i in range(1, 100):
        if f"{i}x" in name or f"x{i}" in name:
            return i
    return None


def _choose_ort_providers(ort, model_path: Path):
    available = list(ort.get_available_providers())

    # Prefer CUDA when available.
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider"]

    # Empirical: some CUGAN ONNX exports produce corrupted output on DirectML.
    # CPU provider yields correct output (slower but correct).
    name = model_path.name.lower()
    if "cugan" in name and "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]

    preferred = []
    for p in ("DmlExecutionProvider", "CPUExecutionProvider"):
        if p in available:
            preferred.append(p)
    return preferred or available


def _ensure_nchw(output):
    # Accept NCHW (1,3,H,W) or NHWC (1,H,W,3)
    if output.ndim != 4:
        raise ValueError(f"Unexpected ONNX output rank: {output.shape}")

    if output.shape[1] == 3:
        return output
    if output.shape[-1] == 3:
        return output.transpose(0, 3, 1, 2)

    raise ValueError(f"Unexpected ONNX output shape (expected 3 channels): {output.shape}")


def _patch_reduce_axes_attr_for_opset18(model_path: Path) -> Path | None:
    """Best-effort fix for some broken ONNX exports:

    ONNX opset >=18 moved `axes` from attribute -> input for Reduce* ops.
    Some files declare opset 18 but still contain `axes` as an attribute.
    ORT rejects those as INVALID_GRAPH. We patch by converting the attribute
    into a constant initializer input.
    """

    try:
        import numpy as np
        import onnx
        from onnx import numpy_helper
    except Exception:
        return None

    model = onnx.load(str(model_path))

    opset = 0
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            opset = max(opset, int(imp.version))

    if opset < 18:
        return None

    reduce_ops = {
        "ReduceMean",
        "ReduceSum",
        "ReduceMax",
        "ReduceMin",
        "ReduceProd",
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceSumSquare",
    }

    used_names = {init.name for init in model.graph.initializer}
    patched = False

    for node in model.graph.node:
        if node.op_type not in reduce_ops:
            continue

        axes_attr = None
        for attr in node.attribute:
            if attr.name == "axes":
                axes_attr = attr
                break

        if axes_attr is None:
            continue

        axes = list(axes_attr.ints)

        # Remove the axes attribute.
        new_attrs = [a for a in node.attribute if a.name != "axes"]
        del node.attribute[:]
        node.attribute.extend(new_attrs)

        base_name = (node.name or (node.output[0] if node.output else "node"))
        base_name = base_name.replace("/", "_").replace(":", "_")
        axes_name = f"{base_name}_axes"
        i = 0
        while axes_name in used_names:
            i += 1
            axes_name = f"{base_name}_axes_{i}"
        used_names.add(axes_name)

        tensor = numpy_helper.from_array(np.asarray(axes, dtype=np.int64), name=axes_name)
        model.graph.initializer.append(tensor)

        # Add as 2nd input (data is input[0]).
        if len(node.input) == 0:
            return None
        if len(node.input) == 1:
            node.input.append(axes_name)
        else:
            node.input[1] = axes_name

        patched = True

    if not patched:
        return None

    out_dir = Path(__file__).resolve().parent / "output" / "onnx_patched_axes"
    out_dir.mkdir(parents=True, exist_ok=True)
    patched_path = out_dir / model_path.name
    onnx.save(model, str(patched_path))
    return patched_path


def _patch_onnx_opset_to_17(model_path: Path) -> Path | None:
    """Best-effort fix for some mislabeled models.

    If a model uses opset<=17 style nodes (e.g. ReduceMean has `axes` attribute) but declares
    opset>=18, ORT will reject it. In some cases simply downgrading the declared ai.onnx opset
    to 17 is the correct fix.
    """

    try:
        import onnx
    except Exception:
        return None

    model = onnx.load(str(model_path))
    changed = False
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx") and int(imp.version) >= 18:
            imp.version = 17
            changed = True

    if not changed:
        return None

    out_dir = Path(__file__).resolve().parent / "output" / "onnx_patched_opset17"
    out_dir.mkdir(parents=True, exist_ok=True)
    patched_path = out_dir / model_path.name
    onnx.save(model, str(patched_path))
    return patched_path


def _to_nchw_float(img_u8, dtype):
    # img_u8: HWC uint8
    arr = img_u8.astype(dtype) / dtype(255.0)
    arr = arr.transpose(2, 0, 1)[None, ...]
    return arr


def _run_onnx_image_upscale(
    img_u8,
    model_path: Path,
    out_path: Path,
    factor: int,
    tile_size: int,
    jpeg_quality: int,
    *,
    providers: list[str] | None = None,
) -> int:
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as e:
        print(f"ERROR: ONNX path requires onnxruntime + numpy: {e}", file=sys.stderr, flush=True)
        return 1

    scale_guess = _infer_scale_from_filename(model_path)
    scale = int(scale_guess or factor)
    if scale_guess and int(scale_guess) != int(factor):
        print(
            f"WARNING: ONNX model scale inferred as {scale_guess}x but factor={factor} requested; using {scale}x",
            flush=True,
        )

    if providers is None:
        providers = _choose_ort_providers(ort, model_path)
    print(f"ONNX providers: {providers}", flush=True)

    tile_size = int(tile_size or 0)
    if providers == ["CPUExecutionProvider"] and tile_size > 0:
        print("Using CPUExecutionProvider; disabling tile rendering for performance.", flush=True)
        tile_size = 0

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=providers)
    except Exception as e:
        msg = str(e)
        if "Unrecognized attribute: axes for operator ReduceMean" in msg:
            # Try opset downgrade first (often yields correct semantics), then fallback to axes->input patch.
            session = None

            patched_path = _patch_onnx_opset_to_17(model_path)
            if patched_path:
                print(f"Patched ONNX opset->17 for ORT: {patched_path}", flush=True)
                try:
                    session = ort.InferenceSession(
                        str(patched_path),
                        sess_options=session_options,
                        providers=providers,
                    )
                except Exception:
                    session = None

            if session is None:
                patched_path = _patch_reduce_axes_attr_for_opset18(model_path)
                if patched_path:
                    print(f"Patched ONNX Reduce* axes attr->input for ORT: {patched_path}", flush=True)
                    session = ort.InferenceSession(
                        str(patched_path),
                        sess_options=session_options,
                        providers=providers,
                    )
                else:
                    raise
        else:
            raise
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name

    input_type = str(getattr(input_meta, "type", ""))
    input_dtype = np.float16 if "float16" in input_type else np.float32
    print(f"ONNX input: {input_name} | {input_type} -> {input_dtype}", flush=True)

    if tile_size > 0:
        print(f"Tile rendering: {tile_size}", flush=True)

    img_f = _to_nchw_float(img_u8, input_dtype)

    print("Running upscale...", flush=True)

    if tile_size <= 0:
        out = session.run(None, {input_name: img_f})[0]
        out_nchw = _ensure_nchw(out).astype(np.float32, copy=False)
    else:
        tile_pad = 10
        h, w = int(img_u8.shape[0]), int(img_u8.shape[1])
        out_full = np.zeros((1, 3, h * scale, w * scale), dtype=np.float32)

        for top in range(0, h, tile_size):
            for left in range(0, w, tile_size):
                tile_h = min(tile_size, h - top)
                tile_w = min(tile_size, w - left)

                in_top0 = max(top - tile_pad, 0)
                in_left0 = max(left - tile_pad, 0)
                in_bot0 = min(top + tile_h + tile_pad, h)
                in_right0 = min(left + tile_w + tile_pad, w)

                patch_u8 = img_u8[in_top0:in_bot0, in_left0:in_right0, :]
                patch_f = _to_nchw_float(patch_u8, input_dtype)
                patch_out = session.run(None, {input_name: patch_f})[0]
                patch_out = _ensure_nchw(patch_out).astype(np.float32, copy=False)

                crop_top = (top - in_top0) * scale
                crop_left = (left - in_left0) * scale
                crop_bot = crop_top + tile_h * scale
                crop_right = crop_left + tile_w * scale

                dest_top = top * scale
                dest_left = left * scale

                out_full[:, :, dest_top : dest_top + tile_h * scale, dest_left : dest_left + tile_w * scale] = patch_out[
                    :, :, crop_top:crop_bot, crop_left:crop_right
                ]

        out_nchw = out_full

    out_hwc = out_nchw[0].transpose(1, 2, 0)
    out_img_bgr = (out_hwc.clip(0, 1) * 255.0).round().astype("uint8")
    # ONNX upscalers in this repo are typically fed with OpenCV (BGR) tensors.
    # Convert back to RGB for PIL saving.
    out_img_rgb = out_img_bgr[:, :, ::-1]

    from PIL import Image

    _save_pil_image(Image.fromarray(out_img_rgb), out_path, jpeg_quality=jpeg_quality)

    print(f"Saved: {out_path}", flush=True)
    return 0


def main() -> int:
    args = parse_args()

    try:
        root = Path(__file__).resolve().parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        _ensure_tas_mainpath()

        input_path = Path(args.input)
        output_path = Path(args.output)

        if not input_path.exists():
            print(f"ERROR: input file not found: {input_path}", file=sys.stderr, flush=True)
            return 2

        print("Loading image...", flush=True)

        import numpy as np
        from PIL import Image

        img = Image.open(str(input_path)).convert("RGB")
        width, height = img.size
        img_u8 = np.asarray(img).astype("uint8")

        method = str(args.method).strip()
        factor = int(args.factor)
        tile_size = int(args.tile_size or 0)
        custom_model = str(args.custom_model) if args.custom_model else None
        custom_upscale_backend = str(getattr(args, "custom_upscale_backend", "default") or "default").lower()

        if custom_model:
            print(f"Custom model: {custom_model}", flush=True)

        # --- Build a torch frame early so we can optionally run restoration before upscaling. ---
        import torch

        arr = img_u8.astype("float32") / 255.0
        frame = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

        # Optional restoration chain (same semantics as video pipeline).
        restore_methods = list(getattr(args, "restore_method", []) or [])
        if bool(getattr(args, "restore", False)):
            if not restore_methods:
                print(
                    "ERROR: --restore specified but no --restore_method provided",
                    file=sys.stderr,
                    flush=True,
                )
                return 2

            force_static = bool(width > 1920 or height > 1080)
            frame = _apply_restore_chain(
                frame,
                restore_methods,
                half=bool(args.half),
                width=int(width),
                height=int(height),
                force_static=force_static,
                custom_restore_backend=str(getattr(args, "custom_restore_backend", "default")),
            )

        if tile_size > 0:
            print(f"Tile rendering: {tile_size}", flush=True)

        out = None

        # --- Upscale ---
        if custom_model:
            model_path = Path(custom_model)
            ext = model_path.suffix.lower()

            backend = custom_upscale_backend
            if backend == "default":
                name_lower = model_path.name.lower()
                if ext == ".param" or "ncnn" in name_lower:
                    backend = "ncnn"
                elif "directml" in name_lower:
                    backend = "directml"
                elif ext == ".onnx":
                    backend = "tensorrt"
                elif ext in [".pth", ".pt"]:
                    backend = "cuda"
                else:
                    raise ValueError(
                        f"Unsupported custom model file type: {ext}. Supported: .onnx, .pth, .pt"
                    )

            print(f"Custom upscale backend: {backend}", flush=True)

            if backend == "directml":
                if ext != ".onnx":
                    raise ValueError(
                        f"DirectML backend requires an .onnx model, got: {ext}"
                    )

                # Convert restored frame back to uint8 BGR for ONNXRuntime.
                frame_cpu = (
                    frame.squeeze(0)
                    .permute(1, 2, 0)
                    .detach()
                    .to("cpu")
                    .to(dtype=torch.float32)
                    .clamp(0, 1)
                    .numpy()
                )
                img_rgb_u8 = (frame_cpu * 255.0).round().astype("uint8")
                img_bgr = img_rgb_u8[:, :, ::-1].copy()

                forced_providers = None
                try:
                    import onnxruntime as ort

                    avail = list(ort.get_available_providers())
                    if "DmlExecutionProvider" in avail:
                        forced_providers = ["DmlExecutionProvider"]
                    else:
                        forced_providers = ["CPUExecutionProvider"]
                except Exception:
                    forced_providers = None

                return _run_onnx_image_upscale(
                    img_u8=img_bgr,
                    model_path=model_path,
                    out_path=output_path,
                    factor=factor,
                    tile_size=tile_size,
                    jpeg_quality=int(getattr(args, "jpeg_quality", 95)),
                    providers=forced_providers,
                )

            if backend == "ncnn":
                raise ValueError(
                    "Custom NCNN models are not supported for Image Upscale yet; use shufflecugan-ncnn/span-ncnn built-ins"
                )

            if backend == "tensorrt":
                if ext != ".onnx":
                    raise ValueError(
                        f"TensorRT backend requires an .onnx model, got: {ext}"
                    )

                scale_guess = _infer_scale_from_filename(model_path)
                scale = int(scale_guess or factor)
                if scale_guess and int(scale_guess) != int(factor):
                    print(
                        f"WARNING: ONNX model scale inferred as {scale_guess}x but factor={factor} requested; using {scale}x",
                        flush=True,
                    )

                from src.unifiedUpscale import UniversalTensorRT

                print(f"Loading model: {model_path.stem} (TensorRT)", flush=True)
                upscaler = UniversalTensorRT(
                    upscaleMethod="custom-tensorrt",
                    upscaleFactor=scale,
                    half=bool(args.half),
                    width=int(width),
                    height=int(height),
                    customModel=str(custom_model),
                    forceStatic=True,
                )

                device = getattr(getattr(upscaler, "dummyInput", None), "device", None)
                if device is not None:
                    frame = frame.to(device=device)

                print("Running upscale...", flush=True)
                out = upscaler(frame, None)

            elif backend == "cuda":
                if ext not in [".pth", ".pt"]:
                    raise ValueError(
                        f"CUDA backend requires a .pth/.pt model, got: {ext}"
                    )

                from src.unifiedUpscale import UniversalPytorch

                print(f"Loading model: {model_path.stem} (PyTorch)", flush=True)
                upscaler = UniversalPytorch(
                    upscaleMethod="custom",
                    upscaleFactor=factor,
                    half=bool(args.half),
                    width=int(width),
                    height=int(height),
                    customModel=str(custom_model),
                    compileMode=str(args.compile_mode),
                    tilesize=tile_size,
                )

                device = getattr(getattr(upscaler, "dummyInput", None), "device", None)
                if device is not None:
                    frame = frame.to(device=device)

                print("Running upscale...", flush=True)
                out = upscaler(frame, None)
            else:
                raise ValueError(
                    f"Unsupported backend '{backend}' for custom upscale models. Supported: cuda, tensorrt, directml, ncnn"
                )

        else:
            print(f"Loading model: {method} ({factor}x)", flush=True)

            if method == "animesr":
                # AnimeSR implementation currently returns a 2x output.
                if factor != 2:
                    print("WARNING: animesr currently outputs 2x; forcing factor=2", flush=True)
                    factor = 2

                from src.unifiedUpscale import AnimeSR

                upscaler = AnimeSR(
                    upscaleFactor=factor,
                    half=bool(args.half),
                    width=int(width),
                    height=int(height),
                    compileMode=str(args.compile_mode),
                )

                frame = frame.to(
                    device=upscaler.prevFrame.device,
                    dtype=upscaler.prevFrame.dtype,
                ).to(memory_format=torch.channels_last)

                print("Running upscale...", flush=True)
                out = upscaler(frame, None)

            elif method == "animesr-tensorrt":
                if factor != 2:
                    print("WARNING: animesr-tensorrt currently outputs 2x; forcing factor=2", flush=True)
                    factor = 2

                from src.unifiedUpscale import AnimeSRTensorRT

                upscaler = AnimeSRTensorRT(
                    upscaleFactor=factor,
                    half=bool(args.half),
                    width=int(width),
                    height=int(height),
                )

                frame = frame.to(
                    device=upscaler.prevFrame.device,
                    dtype=upscaler.prevFrame.dtype,
                ).to(memory_format=torch.channels_last)

                print("Running upscale...", flush=True)
                out = upscaler(frame, None)

            elif method.endswith("-tensorrt"):
                from src.unifiedUpscale import UniversalTensorRT

                upscaler = UniversalTensorRT(
                    upscaleMethod=method,
                    upscaleFactor=factor,
                    half=bool(args.half),
                    width=int(width),
                    height=int(height),
                    customModel=None,
                    forceStatic=bool(width > 1920 or height > 1080),
                )

                device = getattr(getattr(upscaler, "dummyInput", None), "device", None)
                if device is not None:
                    frame = frame.to(device=device)

                print("Running upscale...", flush=True)
                out = upscaler(frame, None)

            elif method.endswith("-directml"):
                from src.unifiedUpscale import UniversalDirectML

                upscaler = UniversalDirectML(
                    upscaleMethod=method,
                    upscaleFactor=factor,
                    half=bool(args.half),
                    width=int(width),
                    height=int(height),
                    customModel=None,
                )

                frame = frame.to("cpu")
                print("Running upscale...", flush=True)
                out = upscaler(frame, None)

            elif method.endswith("-ncnn"):
                from src.unifiedUpscale import UniversalNCNN

                upscaler = UniversalNCNN(
                    upscaleMethod=method,
                    upscaleFactor=factor,
                )

                print("Running upscale...", flush=True)
                out = upscaler(frame, None)

            else:
                from src.unifiedUpscale import UniversalPytorch

                upscaler = UniversalPytorch(
                    upscaleMethod=method,
                    upscaleFactor=factor,
                    half=bool(args.half),
                    width=int(width),
                    height=int(height),
                    customModel=None,
                    compileMode=str(args.compile_mode),
                    tilesize=tile_size,
                )

                device = getattr(getattr(upscaler, "dummyInput", None), "device", None)
                if device is not None:
                    frame = frame.to(device=device)

                print("Running upscale...", flush=True)
                out = upscaler(frame, None)

        if out is None:
            raise RuntimeError("Upscale failed: no output")

        print("Saving output...", flush=True)
        out_np = (
            out.squeeze(0)
            .permute(1, 2, 0)
            .detach()
            .to("cpu")
            .to(dtype=torch.float32)
            .numpy()
        )
        out_img = (out_np * 255.0).round().clip(0, 255).astype("uint8")
        out_pil = Image.fromarray(out_img)

        _save_pil_image(out_pil, output_path, jpeg_quality=int(getattr(args, "jpeg_quality", 95)))

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        print(f"Saved: {output_path}", flush=True)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
