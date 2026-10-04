# Face Detection & Recognition

Local face detection and recognition with OpenCV YuNet and SFace. It handles multiple faces in images, video files, and webcam streams, draws a box and identity label for each face, and labels unmatched faces `Unknown`.

## Setup (Windows PowerShell)

Python 3.10 or newer is recommended. From the project folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The YuNet and SFace ONNX models download into `models/` the first time a recognition or registration command runs. Internet access is needed only for that initial download; processing and the face registry remain local.

## Register People

Register samples from one or more images. Use clear, front-facing photos; multiple images with different lighting or expressions usually improve matching.

```powershell
python run.py register-images --name "Ada Lovelace" .\photos\ada-front.jpg .\photos\ada-side.jpg
```

For live webcam capture, press Space for each sample and Q or Esc to finish. The largest detected face in each captured frame is used.

```powershell
python run.py register-camera --name "Ada Lovelace" --camera 0 --samples 5
```

Embeddings are stored in `data/known_faces.json`. They are biometric data: keep the file private and remove it when no longer needed.

## Recognize Faces

Images open in a preview by default. Use `--output` to save the annotated result and `--no-display` for headless image processing.

```powershell
python run.py recognize --source .\photos\group.jpg --output .\output\group.jpg
python run.py recognize --source .\videos\meeting.mp4 --output .\output\meeting.mp4
python run.py recognize --source 0
```

Press Q or Esc to stop a live camera or video preview. The default SFace cosine-similarity threshold is `0.363`; adjust it with `--threshold` if your data needs stricter or looser matching. A higher threshold is stricter.

Global options such as `--registry` and `--models-dir` go before the subcommand:

```powershell
python run.py --registry .\data\team.json recognize --source .\photos\group.jpg
```

## Tests

The unit tests do not require a camera or model files:

```powershell
python -m unittest discover -s tests -v
```

## Project Files

- `run.py` starts the command-line app.
- `faceid/cli.py` implements image, video, webcam, and registration commands.
- `faceid/vision.py` runs YuNet detection, SFace embeddings, and frame annotation.
- `faceid/registry.py` saves and matches local face embeddings.
- `faceid/models.py` downloads model files on demand.
- `tests/test_registry.py` checks local registry behavior.