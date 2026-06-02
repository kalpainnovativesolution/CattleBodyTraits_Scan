"""Download model artifacts that are intentionally excluded from Git."""

from pathlib import Path

import gdown


MODEL_URLS = {
    "Resnet101_Rcnn_Model.pth": "https://drive.google.com/uc?id=1XCto30B9evkfp31Qwl85MJa1gOHbnt3L",
    "rear_best_v2.pt": "https://drive.google.com/uc?id=1pUEQhvMgsxj9O2v2G3_yOt8b1aAiXMrG",
    "sam2_b.pt": "https://drive.google.com/uc?id=1j4XgLSlhjW0FKwJ0wzDiDjWfVbHwy13F",
    "rf_score_model.pkl": "https://drive.google.com/uc?id=1Q3f1PD46l6UiROok-Ql5n6m_U2A2SuSy",
}


def model_file_available(filename):
    """Return whether a model exists locally or has a configured Drive URL."""
    path = Path(filename)
    return path.exists() or path.name in MODEL_URLS


def ensure_model_file(filename):
    """Return a local model path, downloading a missing file from Drive."""
    path = Path(filename)
    if path.exists():
        return str(path)

    url = MODEL_URLS.get(path.name)
    if not url:
        raise FileNotFoundError(
            f"{path.name} is missing and has no configured Google Drive URL."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f"{path.name}.part")
    try:
        downloaded = gdown.download(
            url=url,
            output=str(partial_path),
            quiet=False,
        )
        if not downloaded or not partial_path.exists():
            raise RuntimeError(
                f"Google Drive did not return a downloadable file for {path.name}."
            )
        partial_path.replace(path)
    finally:
        if partial_path.exists():
            partial_path.unlink()

    return str(path)
