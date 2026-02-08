import os
import time


def _import_celux(decode_method: str):
    # Prefer celux_cuda when available; importing both celux and celux_cuda in the same
    # process can crash with pybind11 type registration errors ("LogLevel is already registered").
    if decode_method == "nvdec":
        import celux_cuda as celux

        return celux

    try:
        import celux_cuda as celux

        return celux
    except Exception:
        import celux

        return celux


def _make_reader(celux_mod, video_path: str, decode_method: str, backend: str):
    try:
        return celux_mod.VideoReader(
            video_path,
            decode_accelerator=decode_method,
            backend=backend,
        )
    except TypeError:
        # Some CeLux builds don't expose decode_accelerator/backend kwargs.
        return celux_mod.VideoReader(video_path)


def bench(video_path: str, decode_method: str, nvdec_mode: str | None, n_frames: int = 300):
    if nvdec_mode is None:
        os.environ.pop("TAS_NVDEC_MODE", None)
    else:
        os.environ["TAS_NVDEC_MODE"] = nvdec_mode

    import torch

    from src.utils.ffmpegSettings import BuildBuffer

    bb = BuildBuffer(
        videoInput=video_path,
        decode_method=decode_method,
        resize=False,
        half=True,
        toTorch=True,
    )

    celux_mod = _import_celux(decode_method)
    reader = _make_reader(celux_mod, video_path, decode_method, bb.backend)

    t0 = time.perf_counter()
    n = 0
    for frame in reader:
        _ = bb.processFrameToTorch(frame, bb.normStream if bb.cudaEnabled else None)
        n += 1
        if n >= n_frames:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dt = time.perf_counter() - t0
    fps = n / dt if dt > 0 else float("inf")

    label = decode_method
    if decode_method == "nvdec":
        label += f" ({nvdec_mode})"

    print(
        f"{label}: {fps:.1f} FPS | backend={bb.backend} | force_cpu_preprocess={bb._force_cpu_preprocess}"
    )


def main():
    video_path = os.path.normpath(os.environ.get("TAS_BENCH_VIDEO", r"output\\TAS-YTDLP-957.mp4"))
    print("Video:", video_path)

    import torch

    print(
        "Torch:",
        torch.__version__,
        "cuda=", torch.version.cuda,
        "available=", torch.cuda.is_available(),
    )
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    bench(video_path, decode_method="nvdec", nvdec_mode="auto", n_frames=300)
    bench(video_path, decode_method="nvdec", nvdec_mode="compat", n_frames=300)
    bench(video_path, decode_method="cpu", nvdec_mode=None, n_frames=300)


if __name__ == "__main__":
    main()
