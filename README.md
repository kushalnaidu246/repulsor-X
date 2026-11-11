# REPULSOR-X

Webcam hand tracking with an Iron Man style palm repulsor, written in Python with OpenCV and MediaPipe.

Open your palm toward the camera to charge the repulsor. Once fully charged it fires a beam in the direction your palm points. The back of your hand and a closed fist do nothing.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python repulsor.py
```

Press `q` or `Esc` to quit.

Options:

- `--camera N` choose a different camera
- `--flip-palm` use this if palm and back of hand are detected the wrong way round

On macOS, allow camera access for your terminal in System Settings, Privacy & Security, Camera.

The hand model `hand_landmarker.task` comes from Google MediaPipe and is included in this repository.
