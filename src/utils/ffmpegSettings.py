import logging
import subprocess
import os
import sys
import importlib.util
import torch
import src.constants as cs
import time
import threading

from queue import Queue
from torch.nn import functional as F
from src.utils.encodingSettings import matchEncoder, getPixFMT
from .isCudaInit import CudaChecker


checker = CudaChecker()

# NOTE: On Windows, `os.add_dll_directory()` returns a handle object. If we don't keep a
# reference to it, CPython will immediately GC it and remove the directory from the DLL
# search path, making the call effectively a no-op.
_CELUX_DLL_DIR_HANDLES: list = []
_CELUX_DLL_DIRS_ADDED: set[str] = set()

# Global cache for VideoReader to support reconfigure() across batch processing.
_CACHED_READER = None
_CACHED_READER_CONFIG: tuple[str, str] | None = None

# If we detect a CUDA/PTX compatibility issue during NVDEC preprocessing, we flip this
# so subsequent NVDEC runs default to the safer compat path (backend="numpy").
_NVDEC_FORCE_COMPAT = False


def _import_celux(decode_method: str):
    # IMPORTANT: importing both `celux` and `celux_cuda` in the same process can crash
    # with pybind11 type registration errors (e.g. "LogLevel is already registered").
    # Prefer `celux_cuda` when available to keep only one CeLux module loaded.
    candidates = ["celux_cuda"] if decode_method == "nvdec" else ["celux_cuda", "celux"]

    import importlib

    last_err: Exception | None = None
    for module_name in candidates:
        # On Windows, Python's DLL search does not automatically include arbitrary
        # directories. For CeLux we may need both:
        # 1) The selected CeLux package dir (next to _celux.pyd)
        # 2) Our bundled FFmpeg dir (for gpl-shared builds that ship DLLs)
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            try:
                ffmpeg_dir = os.path.abspath(os.path.join(cs.MAINPATH, "ffmpeg"))
                if os.path.isdir(ffmpeg_dir) and ffmpeg_dir not in _CELUX_DLL_DIRS_ADDED:
                    handle = os.add_dll_directory(ffmpeg_dir)
                    _CELUX_DLL_DIR_HANDLES.append(handle)
                    _CELUX_DLL_DIRS_ADDED.add(ffmpeg_dir)

                spec = importlib.util.find_spec(module_name)
                pkg_dir = (
                    list(spec.submodule_search_locations)[0]
                    if spec and spec.submodule_search_locations
                    else None
                )
                if pkg_dir and os.path.isdir(pkg_dir):
                    if pkg_dir not in _CELUX_DLL_DIRS_ADDED:
                        handle = os.add_dll_directory(pkg_dir)
                        _CELUX_DLL_DIR_HANDLES.append(handle)
                        _CELUX_DLL_DIRS_ADDED.add(pkg_dir)
            except Exception:
                # If this fails we still try the import; the caller will surface the error.
                pass

        try:
            return importlib.import_module(module_name)
        except Exception as e:
            last_err = e
            # NVDEC requires celux_cuda; no fallback.
            if decode_method == "nvdec":
                break

    if last_err is not None:
        raise last_err

    raise ImportError(f"Failed to import CeLux module for decode_method={decode_method}")


class BuildBuffer:
    def __init__(
        self,
        videoInput: str = "",
        inpoint: float = 0.0,
        outpoint: float = 0.0,
        half: bool = True,
        resize: bool = False,
        width: int = 1920,
        height: int = 1080,
        bitDepth: str = "8bit",
        toTorch: bool = True,
        decode_method: str = "cpu",
    ):
        """

        global _NVDEC_FORCE_COMPAT, _CACHED_READER, _CACHED_READER_CONFIG
        Initializes the BuildBuffer class.

        Args:
            videoInput (str): Path to the input video file.
            inpoint (float): Start time of the segment to decode, in seconds.
            outpoint (float): End time of the segment to decode, in seconds.
            half (bool): Whether to use half precision (float16) for tensors.
            resize (bool): Whether to resize the frames.
            width (int): Width of the output frames.
            height (int): Height of the output frames.
            bitDepth (str): Bit depth of the output frames, e.g., "8bit" or "10bit".
            toTorch (bool): Whether to convert frames to torch tensors.
            decode_method (str): The backend to use for decoding, e.g., "cpu" or "nvdec".

        """
        self.decodeMethod = decode_method
        self.half = half
        self.decodeBuffer = Queue(maxsize=20)
        self.useOpenCV = False
        self.width = width
        self.height = height
        self.resize = resize
        self.isFinished = False
        self._frameAvailable = threading.Event()
        self.bitDepth = bitDepth
        self.videoInput = os.path.normpath(videoInput)
        self.toTorch = toTorch
        self.inpoint = inpoint
        self.outpoint = outpoint

        if "%" not in self.videoInput and not os.path.exists(self.videoInput):
            raise FileNotFoundError(f"Video file not found: {self.videoInput}")

        # Determine device and create CUDA stream if possible; gracefully fall back on failure
        self.cudaEnabled = False
        if checker.cudaAvailable:
            try:
                self.normStream = torch.cuda.Stream()
                self.deviceType = "cuda"
                self.cudaEnabled = True
            except Exception as e:
                logging.warning(
                    f"CUDA stream init failed, falling back to CPU. Reason: {e}"
                )
                self.deviceType = "cpu"
                self.cudaEnabled = False
        else:
            self.deviceType = "cpu"
            self.cudaEnabled = False

        # CeLux legacy API uses backend names like "pytorch"/"numpy".
        self.backend = "pytorch" if toTorch else "numpy"

        # NVDEC:
        # - fast path: backend="pytorch" -> CeLux can return CUDA torch tensors (best perf)
        # - compat path: backend="numpy" -> CPU frames + CPU preprocessing + copy to CUDA
        # Some driver/toolchain combinations can fail when running PyTorch kernels on GPU
        # (e.g. "unsupported toolchain"). We auto-fallback to compat when that happens.
        self._force_cpu_preprocess = False
        if self.decodeMethod == "nvdec":
            nvdec_mode = os.environ.get("TAS_NVDEC_MODE", "auto").strip().lower()
            if nvdec_mode not in ("auto", "fast", "compat"):
                nvdec_mode = "auto"

            if (
                nvdec_mode == "compat"
                or _NVDEC_FORCE_COMPAT
                or not (self.cudaEnabled and self.toTorch)
            ):
                self.backend = "numpy"
                self._force_cpu_preprocess = True
            else:
                # fast/auto: attempt CUDA-tensor path; processFrameToTorch will fallback on error.
                self.backend = "pytorch"

    def __call__(self):
        """
        Decodes frames from the video and stores them in the decodeBuffer.
        """
        decodedFrames = 0
        decodeError = None
        global _CACHED_READER, _CACHED_READER_CONFIG
        import logging
        logging.info(f"BuildBuffer starting: input={self.videoInput}, decode_method={self.decodeMethod}, backend={self.backend}")

        cache_key = (self.decodeMethod, self.backend)
        if _CACHED_READER is not None and _CACHED_READER_CONFIG != cache_key:
            _CACHED_READER = None
            _CACHED_READER_CONFIG = None

        try:
            celux_mod = _import_celux(self.decodeMethod)

            reader = _CACHED_READER
            if reader is not None:
                try:
                    if hasattr(reader, "reconfigure"):
                        logging.info(
                            f"Reconfiguring cached VideoReader for {self.videoInput}"
                        )
                        reader.reconfigure(self.videoInput)
                    else:
                        reader = None
                        _CACHED_READER = None
                        _CACHED_READER_CONFIG = None
                except Exception as e:
                    logging.warning(
                        f"Failed to reconfigure VideoReader: {e}. Creating new instance."
                    )
                    reader = None
                    _CACHED_READER = None
                    _CACHED_READER_CONFIG = None

            if reader is None:
                if self.inpoint > 0 or self.outpoint > 0:
                    logging.info(
                        f"Initializing VideoReader: input={self.videoInput}, decode_method={self.decodeMethod}, backend={self.backend}, inpoint={self.inpoint}, outpoint={self.outpoint}"
                    )
                else:
                    logging.info(
                        f"Initializing VideoReader: input={self.videoInput}, decode_method={self.decodeMethod}, backend={self.backend}"
                    )

                try:
                    reader = celux_mod.VideoReader(
                        self.videoInput,
                        decode_accelerator=self.decodeMethod,
                        backend=self.backend,
                    )
                except TypeError as e:
                    # New CeLux API (0.7.3+) removed these kwargs.
                    if self.decodeMethod != "cpu":
                        raise RuntimeError(
                            "This build's CeLux does not support selecting decode_method via "
                            f"decode_accelerator=... (requested decode_method={self.decodeMethod})."
                        ) from e
                    reader = celux_mod.VideoReader(self.videoInput)

                _CACHED_READER = reader
                _CACHED_READER_CONFIG = cache_key

            if self.inpoint > 0 or self.outpoint > 0:
                try:
                    reader_iter = reader([float(self.inpoint), float(self.outpoint)])
                except TypeError as e:
                    raise RuntimeError(
                        "This build's CeLux does not support inpoint/outpoint slicing via "
                        "VideoReader(...)([inpoint, outpoint]). Please set inpoint/outpoint to 0."
                    ) from e
            else:
                reader_iter = reader

            logging.info("Reader ready, starting frame iteration")
            for frame in reader_iter:
                if self.toTorch:
                    frame = self.processFrameToTorch(
                        frame, self.normStream if self.cudaEnabled else None
                    )

                self.decodeBuffer.put(frame)
                self._frameAvailable.set()
                decodedFrames += 1
                if decodedFrames == 1:
                    logging.info("First frame decoded successfully")

        except Exception as e:
            decodeError = e
            logging.error(f"Decoding error: {e}", exc_info=True)
        finally:
            logging.info(f"BuildBuffer finishing, decoded {decodedFrames} frames")
            self.decodeBuffer.put(None)
            self._frameAvailable.set()

            self.isFinished = True
            logging.info(f"Decoded {decodedFrames} frames")

        if decodeError is not None:
            hint = ""
            root = f"{type(decodeError).__name__}: {decodeError}"
            if self.decodeMethod == "nvdec":
                msg = str(decodeError)
                if "cudaErrorUnsupportedPtxVersion" in msg or "unsupported toolchain" in msg:
                    hint = (
                        " NVDEC preprocessing hit a CUDA/PTX compatibility error (your NVIDIA driver is likely too old for the bundled PyTorch/CUDA build)."
                        " Update NVIDIA drivers or use --decode_method cpu."
                    )
                elif "CeLux was not built with CUDA support" in msg:
                    hint = (
                        " NVDEC failed because CeLux has no CUDA support."
                        " This build uses CUDA-enabled CeLux via the bundled 'celux_cuda' module (and requires NVIDIA drivers)."
                        " If NVDEC isn't available on your system, use --decode_method cpu."
                    )
                elif "No module named" in msg or "ModuleNotFoundError" in msg:
                    hint = " NVDEC requires CUDA-enabled CeLux (bundled as 'celux_cuda')."
                else:
                    hint = " Try --decode_method cpu (or update NVIDIA drivers / FFmpeg)."

            raise RuntimeError(
                f"Video decode failed (decode_method={self.decodeMethod}). Root cause: {root}.{hint}"
            ) from decodeError

        if decodedFrames == 0:
            hint = ""
            if self.decodeMethod == "nvdec":
                hint = " NVDEC decoded 0 frames; try --decode_method cpu."
            raise RuntimeError(
                f"No frames decoded from video (decode_method={self.decodeMethod}).{hint}"
            )

    def processFrameToTorch(self, frame, normStream=None):
        """Convert CeLux frames to NCHW float tensors in [0, 1].

        Notes:
            - CeLux can return either torch.Tensor (backend="pytorch") or numpy.ndarray
              (backend="numpy").
            - Some systems can hit a CUDA/PTX compatibility error when casting on GPU
              (e.g. "unsupported toolchain"). In that case we fall back to CPU
              preprocessing and then copy to CUDA.
        """

        if not isinstance(frame, torch.Tensor):
            # Support CeLux numpy backend.
            import numpy as np

            if isinstance(frame, np.ndarray):
                frame = torch.from_numpy(frame)
            else:
                raise TypeError(f"Unsupported frame type: {type(frame)}")

        if frame.dtype == torch.uint8:
            norm = 1 / 255.0
        elif frame.dtype == torch.uint16:
            norm = 1 / 65535.0
        else:
            # Assume already normalized or float.
            norm = 1.0

        target_dtype = torch.float16 if self.half else torch.float32

        def _cpu_preprocess(t: torch.Tensor) -> torch.Tensor:
            if t.is_cuda:
                t = t.to(device="cpu", non_blocking=False)

            # CPU cast + normalize to avoid GPU kernels on systems with CUDA/PTX issues.
            t = t.to(device="cpu", non_blocking=False, dtype=target_dtype)
            t = t.permute(2, 0, 1).contiguous().mul(norm).clamp(0, 1)

            if self.resize:
                t = F.interpolate(
                    t.unsqueeze(0),
                    size=(self.height, self.width),
                    mode="bicubic",
                    align_corners=False,
                )
            else:
                t = t.unsqueeze(0)

            # Make contiguous on CPU before copy to CUDA to avoid any device-side packing.
            t = t.contiguous()

            if self.cudaEnabled:
                try:
                    t = t.pin_memory()
                except Exception:
                    pass
                return t.to(device="cuda", non_blocking=True)

            return t

        if self.cudaEnabled and not self._force_cpu_preprocess:
            try:
                with torch.cuda.stream(normStream):
                    # Only pin CPU tensors.
                    if not frame.is_cuda:
                        try:
                            frame = frame.pin_memory()
                        except Exception:
                            pass

                    frame = (
                        frame.to(
                            device="cuda",
                            non_blocking=True,
                            dtype=target_dtype,
                        )
                        .permute(2, 0, 1)
                        .mul(norm)
                        .clamp(0, 1)
                    )

                    if self.resize:
                        frame = F.interpolate(
                            frame.unsqueeze(0),
                            size=(self.height, self.width),
                            mode="bicubic",
                            align_corners=False,
                        )
                    else:
                        frame = frame.unsqueeze(0)

                if normStream is not None:
                    normStream.synchronize()
                return frame
            except Exception as e:
                msg = str(e)
                accel_err = getattr(torch, "AcceleratorError", None)
                if (
                    (accel_err is not None and isinstance(e, accel_err))
                    or "unsupported toolchain" in msg
                    or "cudaErrorUnsupportedPtxVersion" in msg
                ):
                    logging.warning(
                        "CUDA preprocess failed (likely NVIDIA driver / PyTorch CUDA mismatch). "
                        "Falling back to CPU preprocessing for the rest of this run, and enabling "
                        "NVDEC compat mode for subsequent runs. Consider updating your NVIDIA driver.",
                    )

                    if self.decodeMethod == "nvdec":
                        _NVDEC_FORCE_COMPAT = True
                        _CACHED_READER = None
                        _CACHED_READER_CONFIG = None

                    self._force_cpu_preprocess = True
                    return _cpu_preprocess(frame)
                raise

        return _cpu_preprocess(frame)

    def read(self):
        """
        Reads a frame from the decodeBuffer.

        Returns:
            The next frame from the decodeBuffer.
        """
        return self.decodeBuffer.get()

    def peek(self):
        """
        Peeks at the next frame in the decodeBuffer without removing it.

        Returns:
            The next frame from the decodeBuffer, or None if decoding is finished and queue is empty.
        """
        while True:
            with self.decodeBuffer.mutex:
                if len(self.decodeBuffer.queue) > 0:
                    return self.decodeBuffer.queue[0]

            if self.isFinished:
                return None

            self._frameAvailable.wait(timeout=0.1)
            self._frameAvailable.clear()

    def isReadFinished(self) -> bool:
        """
        Returns:
            Whether the decoding process is finished.
        """
        return self.isFinished

    def isQueueEmpty(self) -> bool:
        """
        Returns:
            Whether the decoding buffer is empty.
        """

        if self.decodeBuffer.empty() and self.isFinished:
            return True
        else:
            return False


class WriteBuffer:
    def __init__(
        self,
        input: str = "",
        output: str = "",
        encode_method: str = "x264",
        custom_encoder: str = "",
        width: int = 1920,
        height: int = 1080,
        fps: float = 60.0,
        sharpen: bool = False,
        sharpen_sens: float = 0.0,
        grayscale: bool = False,
        transparent: bool = False,
        benchmark: bool = False,
        bitDepth: str = "8bit",
        inpoint: float = 0.0,
        outpoint: float = 0.0,
        slowmo: bool = False,
        output_scale_width: int = None,
        output_scale_height: int = None,
        enablePreview: bool = False,
    ):
        """
        A class meant to Pipe the input to FFMPEG from a queue.

        output: str - The path to the output video file.
        encode_method: str - The method to use for encoding the video. Options include "x264", "x264_animation", "nvenc_h264", etc.
        custom_encoder: str - A custom encoder string to use for encoding the video.
        grayscale: bool - Whether to encode the video in grayscale.
        width: int - The width of the output video in pixels.
        height: int - The height of the output video in pixels.
        fps: float - The frames per second of the output video.
        sharpen: bool - Whether to apply a sharpening filter to the video.
        sharpen_sens: float - The sensitivity of the sharpening filter.
        transparent: bool - Whether to encode the video with transparency.
        audio: bool - Whether to include audio in the output video.
        benchmark: bool - Whether to benchmark the encoding process, this will not output any video.
        bitDepth: str - The bit depth of the output video. Options include "8bit" and "10bit".
        inpoint: float - The start time of the segment to encode, in seconds.
        outpoint: float - The end time of the segment to encode, in seconds.
        output_scale_width: int - The target width for output scaling (optional).
        output_scale_height: int - The target height for output scaling (optional).
        enablePreview: bool - Whether to enable FFmpeg-based preview output (optional).
        """
        self.input = input
        self.output = os.path.normpath(output)
        self.encode_method = encode_method

        if self.encode_method == "png" and "%" not in self.output:
            # If user passed a directory, write a numbered PNG sequence in that dir.
            # If user passed a file path, create a numbered sequence next to it.
            _, ext = os.path.splitext(self.output)
            if not ext:
                self.output = os.path.join(self.output, "%08d.png")
            else:
                base, _ = os.path.splitext(self.output)
                self.output = f"{base}_%08d.png"

        self.custom_encoder = custom_encoder
        self.grayscale = grayscale
        self.width = width
        self.height = height
        self.fps = fps
        self.sharpen = sharpen
        self.sharpen_sens = sharpen_sens
        self.transparent = transparent
        self.benchmark = benchmark
        self.bitDepth = bitDepth
        self.inpoint = inpoint
        self.outpoint = outpoint
        self.slowmo = slowmo
        self.output_scale_width = output_scale_width
        self.output_scale_height = output_scale_height
        self.enablePreview = enablePreview

        self.writtenFrames = 0
        self.writeBuffer = Queue(maxsize=20)

        self.previewPath = (
            os.path.join(cs.MAINPATH, "preview.jpg") if enablePreview else None
        )

    def encodeSettings(self) -> list:
        """
        Simplified structure for setting input/output pix formats
        and building FFMPEG command.
        """
        # Set environment variables
        os.environ["FFREPORT"] = "file=FFmpeg-Log.log:level=32"
        if "av1" in [self.encode_method, self.custom_encoder]:
            os.environ["SVT_LOG"] = "0"

        self.inputPixFmt, outputPixFmt, self.encode_method = getPixFMT(
            self.encode_method, self.bitDepth, self.grayscale, self.transparent
        )

        if self.benchmark:
            return self._buildBenchmarkCommand()
        else:
            return self._buildEncodingCommand(outputPixFmt)

    def _buildBenchmarkCommand(self):
        """Build FFmpeg command for benchmarking"""
        return [
            cs.FFMPEGPATH,
            "-y",
            "-hide_banner",
            "-v",
            "warning",
            "-nostats",
            "-f",
            "rawvideo",
            "-video_size",
            f"{self.width}x{self.height}",
            "-pix_fmt",
            self.inputPixFmt,
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-benchmark",
            "-f",
            "null",
            "-",
        ]

    def _isNvencEncoder(self) -> bool:
        """Check if the current encode method uses NVENC"""
        nvenc_methods = [
            "nvenc_h264",
            "slow_nvenc_h264",
            "nvenc_h265",
            "slow_nvenc_h265",
            "nvenc_h265_10bit",
            "nvenc_av1",
            "slow_nvenc_av1",
            "lossless_nvenc_h264",
        ]
        return self.encode_method in nvenc_methods

    def _buildEncodingCommand(self, outputPixFmt):
        """Build FFmpeg command for encoding"""
        useHwUpload = self._isNvencEncoder() and not self.custom_encoder and not self.enablePreview

        command = [
            cs.FFMPEGPATH,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
        ]

        # Initialize CUDA device for hwupload when using NVENC.
        if useHwUpload:
            command.extend(["-init_hw_device", "cuda=cu:0", "-filter_hw_device", "cu"])

        command.extend(
            [
                "-f",
                "rawvideo",
                "-pix_fmt",
                self.inputPixFmt,
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                str(self.fps),
            ]
        )

        if self.outpoint != 0 and not self.slowmo:
            # For MP4/MOV segment outputs, avoid using `-itsoffset` on the video stream.
            # It tends to force MP4 edit lists (edts/elst) to reconcile timestamps, and
            # some players handle those poorly (video freezes / wrong duration display).
            if os.path.splitext(self.output)[1].lower() in (".mp4", ".mov"):
                command.extend(
                    [
                        "-i",
                        "pipe:0",
                        "-ss",
                        str(self.inpoint),
                        "-to",
                        str(self.outpoint),
                    ]
                )
            else:
                command.extend(
                    [
                        "-itsoffset",
                        str(self.inpoint),
                        "-i",
                        "pipe:0",
                        "-ss",
                        str(self.inpoint),
                        "-to",
                        str(self.outpoint),
                    ]
                )
        else:
            command.extend(["-i", "pipe:0"])

        if cs.AUDIO:
            command.extend(["-i", self.input])

        filterList = self._buildFilterList()

        if self.enablePreview:
            filterComplexParts = []

            if filterList:
                baseFilters = ",".join(filterList)
                filterComplexParts.append(f"[0:v]{baseFilters},split=2[main][preview]")
            else:
                filterComplexParts.append("[0:v]split=2[main][preview]")

            filterComplexParts.append("[preview]fps=2[previewThrottled]")

            combinedFilter = ";".join(filterComplexParts)
            command.extend(["-filter_complex", combinedFilter])

            command.extend(["-map", "[main]"])

            if not self.custom_encoder:
                command.extend(matchEncoder(self.encode_method))
                command.extend(["-pix_fmt", outputPixFmt])
            else:
                customArgs = self.custom_encoder.split()
                if "-vf" in customArgs:
                    vfIdx = customArgs.index("-vf")
                    customArgs.pop(vfIdx)
                    customArgs.pop(vfIdx)
                if "-pix_fmt" not in customArgs:
                    customArgs.extend(["-pix_fmt", outputPixFmt])
                command.extend(customArgs)

            if cs.AUDIO:
                command.extend(self._buildAudioSettings())

            # Put MP4/MOV metadata (moov atom) at the beginning of the file so players
            # can start playback without needing a manual seek/scrub.
            if os.path.splitext(self.output)[1].lower() in (".mp4", ".mov"):
                command.extend(["-movflags", "+faststart"])

            command.append(self.output)

            command.extend(
                [
                    "-map",
                    "[previewThrottled]",
                    "-q:v",
                    "2",
                    "-update",
                    "1",
                    self.previewPath,
                ]
            )
        else:
            command.extend(["-map", "0:v"])

            if not self.custom_encoder:
                command.extend(matchEncoder(self.encode_method))

                if useHwUpload:
                    hwFilters = filterList.copy() if filterList else []
                    hwFmt = "p010le" if outputPixFmt == "p010le" else "nv12"
                    hwFilters.append(f"format={hwFmt}")
                    hwFilters.append("hwupload_cuda")
                    command.extend(["-vf", ",".join(hwFilters)])
                else:
                    if filterList:
                        command.extend(["-vf", ",".join(filterList)])
                    command.extend(["-pix_fmt", outputPixFmt])
            else:
                command.extend(self._buildCustomEncoder(filterList, outputPixFmt))

            if cs.AUDIO:
                command.extend(self._buildAudioSettings())

            # Put MP4/MOV metadata (moov atom) at the beginning of the file so players
            # can start playback without needing a manual seek/scrub.
            if os.path.splitext(self.output)[1].lower() in (".mp4", ".mov"):
                command.extend(["-movflags", "+faststart"])

            command.append(self.output)

        return command

    def _getOutputFormat(self):
        ext = os.path.splitext(self.output)[1].lower()
        formatMap = {
            ".mp4": "mp4",
            ".mkv": "matroska",
            ".webm": "webm",
            ".mov": "mov",
            ".avi": "avi",
        }
        return formatMap.get(ext, "mp4")

    def _buildFilterList(self):
        """Build list of video filters based on settings"""
        filterList = []

        if self.output_scale_width and self.output_scale_height:
            filterList.append(
                f"scale={self.output_scale_width}:{self.output_scale_height}:flags=bilinear"
            )

        if self.sharpen:
            filterList.append(f"cas={self.sharpen_sens}")
        if self.grayscale:
            filterList.append(
                "format=gray" if self.bitDepth == "8bit" else "format=gray16be"
            )
        if self.transparent:
            filterList.append("format=yuva420p")

        """
                "-vf",
            "zscale=matrix=709:dither=error_diffusion,format=yuv420p",
            """

        import json

        metadata = json.loads(open(cs.METADATAPATH, "r", encoding="utf-8").read())
        if not self.grayscale and not self.transparent:
            colorSPaceFilter = {
                "bt709": f"zscale=matrix=709:dither=error_diffusion,format={self.inputPixFmt}",
                "bt2020": "zscale=matrix=bt2020:norm=bt2020:dither=error_diffusion,format=yuv420p",
            }

            metadataFields = ["ColorSpace", "PixelFormat", "ColorTRT"]
            detectedColorSpace = None

            for field in metadataFields:
                colorValue = metadata["metadata"].get(field, "unknown")
                if colorValue in colorSPaceFilter:
                    detectedColorSpace = colorValue
                    break

            filterList.append(
                colorSPaceFilter.get(detectedColorSpace, colorSPaceFilter["bt709"])
            )

        return filterList

    def _buildCustomEncoder(self, filterList, outputPixFmt):
        """Apply custom encoder settings with filters"""
        customEncoderArgs = self.custom_encoder.split()

        if "-vf" in customEncoderArgs:
            vfIndex = customEncoderArgs.index("-vf")
            filterString = customEncoderArgs[vfIndex + 1]
            for filterItem in filterList:
                filterString += f",{filterItem}"
            customEncoderArgs[vfIndex + 1] = filterString
        elif filterList:
            customEncoderArgs.extend(["-vf", ",".join(filterList)])

        if "-pix_fmt" not in customEncoderArgs:
            logging.info(f"-pix_fmt was not found, adding {outputPixFmt}.")
            customEncoderArgs.extend(["-pix_fmt", outputPixFmt])

        return customEncoderArgs

    def _buildAudioSettings(self):
        """Build audio encoding settings"""
        audioSettings = ["-map", "1:a"]

        audioCodec = "copy"
        subCodec = "copy"
        if self.output.endswith(".webm"):
            audioCodec = "libopus"
            subCodec = "webvtt"
        elif (
            self.outpoint != 0
            and not self.slowmo
            and os.path.splitext(self.output)[1].lower() in (".mp4", ".mov")
        ):
            # For segment outputs, stream-copying audio can preserve non-zero timestamps
            # and force MP4 edit lists (edts/elst), which some players handle poorly.
            # Re-encode the short audio segment so the output timestamps start at 0.
            audioCodec = "aac"
        audioSettings.extend(["-c:a", audioCodec, "-map", "1:s?", "-c:s", subCodec])

        # NOTE: Segment trimming for audio is already applied as *input* options
        # ("-ss/-to" before the audio "-i ..." in _buildEncodingCommand). Adding
        # another "-ss/-to" here applies it to the *output* and can result in an
        # empty container (0 streams / ~261 bytes) for short segments.

        return audioSettings

    def __call__(self):
        writtenFrames = 0

        # Wait for at least one frame to be queued before starting encoding
        while self.writeBuffer.empty():
            try:
                time.sleep(0.01)
            except KeyboardInterrupt:
                logging.warning("Encoding interrupted by user")
                return

        ffmpegProc = None
        encodeError = None
        ffmpegStderr = b""
        try:
            initialFrame = self.writeBuffer.queue[0]

            self.channels = 1 if self.grayscale else 4 if self.transparent else 3

            isEightBit = self.bitDepth == "8bit"
            multiplier = 255 if isEightBit else 65535
            dtype = torch.uint8 if isEightBit else torch.uint16

            needsResize = (
                initialFrame.shape[2] != self.height
                or initialFrame.shape[3] != self.width
            )

            if needsResize:
                logging.info(
                    f"Frame size mismatch. Frame: {initialFrame.shape[3]}x{initialFrame.shape[2]}, Output: {self.width}x{self.height}"
                )

            command = self.encodeSettings()
            logging.info(f"Encoding with: {' '.join(map(str, command))}")

            if self.enablePreview:
                logging.info(f"Preview enabled, writing to: {self.previewPath}")
                from src.utils.logAndPrint import logAndPrint

                logAndPrint(f"Preview will be saved to: {self.previewPath}", "cyan")


            useCuda = False
            transferStream = None
            if checker.cudaAvailable:
                try:
                    transferStream = torch.cuda.Stream()
                    useCuda = True
                except Exception as e:
                    logging.warning(
                        f"CUDA init failed in writer, using CPU path. Reason: {e}"
                    )
                    useCuda = False

            ffmpegProc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=None,
                stderr=subprocess.PIPE,
                shell=False,
                cwd=cs.MAINPATH,
            )

            if useCuda:
                frameShape = (self.height, self.width, self.channels)
                pinnedBuffers = [
                    torch.empty(frameShape, dtype=dtype, pin_memory=True),
                    torch.empty(frameShape, dtype=dtype, pin_memory=True),
                ]
                transferEvents = [torch.cuda.Event(), torch.cuda.Event()]
                bufferIdx = 0
                pendingBuffer = None
                pendingEvent = None

                while True:
                    try:
                        frame = self.writeBuffer.get(timeout=1.0)
                    except Exception:
                        time.sleep(0.001)
                        continue

                    if frame is None:
                        if pendingBuffer is not None:
                            pendingEvent.synchronize()
                            ffmpegProc.stdin.write(memoryview(pendingBuffer.numpy()))
                            writtenFrames += 1
                        break

                    with torch.cuda.stream(transferStream):
                        if needsResize:
                            frame = F.interpolate(
                                frame,
                                size=(self.height, self.width),
                                mode="bicubic",
                                align_corners=False,
                            )

                        gpuTensor = (
                            frame.squeeze(0)
                            .permute(1, 2, 0)
                            .mul(multiplier)
                            .clamp(0, multiplier)
                            .to(dtype)
                            .contiguous()
                        )

                        currentBuffer = pinnedBuffers[bufferIdx]
                        currentBuffer.copy_(gpuTensor, non_blocking=True)
                        currentEvent = transferEvents[bufferIdx]
                        currentEvent.record(transferStream)

                    if pendingBuffer is not None:
                        pendingEvent.synchronize()
                        ffmpegProc.stdin.write(memoryview(pendingBuffer.numpy()))
                        writtenFrames += 1

                    pendingBuffer = currentBuffer
                    pendingEvent = currentEvent
                    bufferIdx = 1 - bufferIdx

            else:
                while True:
                    try:
                        frame = self.writeBuffer.get(timeout=1.0)
                    except Exception:
                        time.sleep(0.001)
                        continue
                    if frame is None:
                        break

                    if needsResize:
                        frame = F.interpolate(
                            frame,
                            size=(self.height, self.width),
                            mode="bicubic",
                            align_corners=False,
                        )
                    frameTensor = (
                        frame.squeeze(0)
                        .permute(1, 2, 0)
                        .mul(multiplier)
                        .clamp(0, multiplier)
                        .to(dtype)
                        .contiguous()
                    )

                    ffmpegProc.stdin.write(memoryview(frameTensor.numpy()))
                    writtenFrames += 1


            logging.info(f"Encoded {writtenFrames} frames")

            # Keep this in the instance for debugging / downstream checks.
            self.writtenFrames = writtenFrames

        except Exception as e:
            encodeError = e
        finally:
            if ffmpegProc is not None:
                try:
                    if ffmpegProc.stdin:
                        ffmpegProc.stdin.close()
                except Exception as e:
                    logging.warning(f"Cleanup error (stdin close): {e}")

                try:
                    _, ffmpegStderr = ffmpegProc.communicate(timeout=60)
                except subprocess.TimeoutExpired:
                    ffmpegProc.kill()
                    _, ffmpegStderr = ffmpegProc.communicate()
                    if encodeError is None:
                        encodeError = RuntimeError(
                            "FFmpeg did not exit within 60s after stdin close"
                        )
                except Exception as e:
                    if encodeError is None:
                        encodeError = e

                if ffmpegProc.returncode not in (0, None):
                    stderrText = ""
                    try:
                        stderrText = (ffmpegStderr or b"").decode(
                            errors="replace"
                        ).strip()
                    except Exception:
                        stderrText = ""

                    ffmpegError = RuntimeError(
                        f"FFmpeg encoder failed with exit code {ffmpegProc.returncode}.\n{stderrText[-4000:]}"
                    )
                    encodeError = ffmpegError if encodeError is None else RuntimeError(
                        f"{encodeError}\n\n{ffmpegError}"
                    )

                # Fail fast on clearly broken outputs (e.g. 0/very small bytes).
                if encodeError is None and not self.benchmark:
                    try:
                        if os.path.exists(self.output):
                            outSize = os.path.getsize(self.output)
                            if outSize < 1024:
                                encodeError = RuntimeError(
                                    f"FFmpeg produced an unexpectedly small output file ({outSize} bytes): {self.output}"
                                )
                    except Exception as e:
                        logging.warning(f"Could not stat output file: {e}")

            if encodeError is not None:
                logging.error(f"Encoding error: {encodeError}")
                raise encodeError

    def write(self, frame: torch.Tensor):
        """
        Add a frame to the queue. Must be in [B, C, H, W] format.
        """
        self.writeBuffer.put(frame)

    def put(self, frame: torch.Tensor):
        """
        Equivalent to write()
        Add a frame to the queue. Must be in [B, C, H, W] format.
        """
        self.writeBuffer.put(frame)

    def close(self):
        self.writeBuffer.put(None)

        if self.previewPath and os.path.exists(self.previewPath):
            try:
                os.remove(self.previewPath)
            except Exception as e:
                logging.warning(f"Could not remove preview file: {e}")
