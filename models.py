"""Download the OpenCV Zoo models used by YuNet and SFace."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path


MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/"
        "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
}


def ensure_models(directory: str | Path) -> tuple[Path, Path]:
    model_dir = Path(directory)
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for filename, url in MODEL_URLS.items():
        model_path = model_dir / filename
        if not model_path.exists() or model_path.stat().st_size == 0:
            temporary_path = model_path.with_suffix(model_path.suffix + ".part")
            request = urllib.request.Request(
                url, headers={"User-Agent": "face-detection-recognition/1.0"}
            )
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    with temporary_path.open("wb") as destination:
                        shutil.copyfileobj(response, destination)
                if temporary_path.stat().st_size < 1024:
                    raise OSError("downloaded model file is unexpectedly small")
                temporary_path.replace(model_path)
            except (OSError, TimeoutError) as error:
                temporary_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Could not download OpenCV model '{filename}' from {url}: {error}"
                ) from error
        paths.append(model_path)

    return paths[0], paths[1]